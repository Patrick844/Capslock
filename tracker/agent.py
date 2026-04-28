"""Agent loop — phased pipeline with native tool calling.

Pipeline (one LLM call per ▼):
  ▼ Phase 1: search       → list of Source candidates
  ▼ Phase 2: fetch (×N)   → Article per candidate (parallel)
  ▼ Phase 3: summarize    → SummarizerResponse per article (parallel)
  ▼ Phase 4: cluster      → list of duplicate-story groups
  Post-process            : build items, merge clusters, truncate

Phases 2 & 3 are deterministic: every candidate is attempted for fetch,
every fetched article is attempted for summarisation. An item is only
built when BOTH succeeded — so every digest item has a real summary
and relevance, never None.

Clustering is decoupled: it runs after items are built and is applied
via a pure post-processing function.

Outer retry: up to MAX_AGENT_ATTEMPTS = 6 full agent runs. No inner
per-call retry — total budget on persistent failure is 6, not 12.
"""

import asyncio
import json
from datetime import UTC, date, datetime
from typing import cast

from openai import AsyncOpenAI
from openai.types.responses import Response, ToolParam

from tracker.config import settings
from tracker.schemas import (
    Article,
    ClusterArticlesParam,
    DigestItem,
    DigestRequest,
    DigestResponse,
    FetchArticleParam,
    SearchNews,
    Source,
    SummarizerResponse,
    TokenUsage,
)
from tracker.token_usage import get_token_usage, sum_token_usages
from tracker.tool_description import (
    CLUSTER_ARTICLES_TOOL,
    FETCH_ARTICLE_TOOL,
    SEARCH_NEWS_TOOL,
    SUMMARIZE_ARTICLE_TOOL,
)
from tracker.tools import cluster_articles, fetch_article, search_news, summarize_content

# Limit how many candidate articles are processed in one agent run.
# This protects runtime and token usage when the search tool returns many results.
MAX_ITERATIONS = 6  # cap on candidate articles processed per run

# Number of full agent attempts before giving up.
# A full attempt includes search, fetch, summarize, cluster, and post-processing.
MAX_AGENT_ATTEMPTS = 6  # total agent retries on unhandled failure

# Shared async OpenAI client used by the agent-level LLM calls.
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Tool schemas exposed to the LLM.
# These are descriptions of available tools, not the Python functions themselves.
TOOLS = cast(
    list[ToolParam],
    [SEARCH_NEWS_TOOL, FETCH_ARTICLE_TOOL, SUMMARIZE_ARTICLE_TOOL, CLUSTER_ARTICLES_TOOL],
)

# System-level behavior instructions for the LLM.
# This tells the model when to use each tool and prevents it from inventing data.
SYSTEM_PROMPT = (
    "You are a news assistant. "
    "Use search_news to find article IDs based on topic and date. "
    "Use fetch_article to get the full article content based on article_id. "
    "Use summarize_article to summarize an article content. "
    "Use cluster_articles to group articles describing the same story by title. "
    "Do not invent article data. Only answer using tool results."
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

# Ledger of token usage entries collected across all LLM calls in one digest run.
# Some token extraction calls may return None, so the ledger accepts None values.
TokenLedger = list[TokenUsage | None]


async def _call_llm(user_input: str) -> Response:
    """Single LLM call with all tools registered."""

    # The model decides whether to call one of the registered tools.
    # The returned response may contain normal text or function_call items.
    return await client.responses.create(
        model=settings.LLM_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_input,
        tools=TOOLS,
        tool_choice="auto",
    )


def _extract_tool_calls(response: Response, tool_name: str) -> list:
    """Filter the Responses API output to function_call items for *tool_name*."""

    # The Responses API output can contain different item types.
    # We only keep function calls for the specific tool we expect in the current phase.
    return [
        item for item in response.output if item.type == "function_call" and item.name == tool_name
    ]


# ---------------------------------------------------------------------------
# Phase 1 — search
# ---------------------------------------------------------------------------


async def _search(topic: str, since: date, max_items: int, usage: TokenLedger) -> list[Source]:
    # Ask the LLM to decide how to call search_news from topic/date/max_items.
    response = await _call_llm(f"topic: {topic}\nsince: {since}\nmax_items: {max_items}")

    # Store token usage from this LLM call.
    usage.append(get_token_usage(response))

    # Execute the first search_news tool call returned by the model.
    for tc in _extract_tool_calls(response, "search_news"):
        args = SearchNews.model_validate(json.loads(tc.arguments))
        print(f"    [search_news] query='{args.query}' since={args.since} limit={args.limit}")

        # This is the actual local Python tool execution.
        return search_news(args.query, args.since, args.limit)

    # If the LLM did not call search_news, treat it as no candidates found.
    return []


# ---------------------------------------------------------------------------
# Phase 2 — fetch one article via the LLM
# ---------------------------------------------------------------------------


async def _fetch(article_id: str, usage: TokenLedger) -> Article | None:
    try:
        # Ask the LLM to call fetch_article for the provided article_id.
        response = await _call_llm(f"article_id: {article_id}")
    except Exception as exc:
        # If the LLM call itself fails, skip this article but keep the run alive.
        print(f"    [SKIP] LLM failed for fetch {article_id}: {exc}")
        return None

    # Store token usage from the fetch-routing LLM call.
    usage.append(get_token_usage(response))

    # Check whether the LLM actually requested the fetch_article tool.
    calls = _extract_tool_calls(response, "fetch_article")
    if not calls:
        print(f"    [SKIP] LLM did not call fetch_article for {article_id}")
        return None

    # Validate the tool arguments before executing the local Python function.
    args = FetchArticleParam.model_validate(json.loads(calls[0].arguments))
    try:
        # Execute the deterministic local fetch tool.
        article = fetch_article(args.article_id)
    except (FileNotFoundError, ValueError) as exc:
        # Bad article IDs or malformed fixture files are skipped.
        print(f"    [SKIP] Could not fetch {args.article_id}: {exc}")
        return None

    print(f"    Fetched {args.article_id} ({len(article.body)} chars)")
    return article


# ---------------------------------------------------------------------------
# Phase 3 — summarize one article via the LLM
# ---------------------------------------------------------------------------


async def _summarize(article: Article, topic: str, usage: TokenLedger) -> SummarizerResponse | None:
    try:
        # Ask the LLM to call summarize_article for this article body.
        response = await _call_llm(f"summarize the content: {article.body}")
    except Exception as exc:
        # If the LLM routing call fails, skip this article summary.
        print(f"    [SKIP] LLM failed for summarize {article.article_id}: {exc}")
        return None

    # Store token usage from the summarize-routing LLM call.
    usage.append(get_token_usage(response))

    # Check whether the LLM actually requested the summarize_article tool.
    calls = _extract_tool_calls(response, "summarize_article")
    if not calls:
        print(f"    [SKIP] LLM did not call summarize_article for {article.article_id}")
        return None

    try:
        # Run the actual summarization/classification function.
        # This function performs structured output generation.
        summary = await summarize_content(article.body, topic)

        # Add the token usage from the inner summarization LLM call.
        usage.append(summary.token_usage)
    except Exception as exc:
        # If summarization validation or LLM parsing fails, skip the article.
        print(f"    [SKIP] Summarization failed for {article.article_id}: {exc}")
        return None

    print(f"    Summarised {article.article_id}: relevance={summary.relevance}")
    return summary


# ---------------------------------------------------------------------------
# Phase 4 — cluster (one LLM call, best-effort)
# ---------------------------------------------------------------------------


async def _cluster(fetched: dict[str, Article], usage: TokenLedger) -> list[list[str]]:
    # No clustering needed when fewer than two articles were fetched.
    if len(fetched) < 2:
        return []

    # Only titles are sent for clustering to keep the prompt small and focused.
    titles = "\n".join(f"- {aid}: {a.title}" for aid, a in fetched.items())
    try:
        # Ask the LLM to call cluster_articles with the article titles.
        response = await _call_llm(f"Cluster these articles by story:\n{titles}")
    except Exception as exc:
        # Clustering is best-effort; failure does not fail the whole digest.
        print(f"    [SKIP] LLM failed for cluster: {exc}")
        return []

    # Store token usage from the cluster-routing LLM call.
    usage.append(get_token_usage(response))

    # Check whether the LLM actually requested the cluster_articles tool.
    calls = _extract_tool_calls(response, "cluster_articles")
    if not calls:
        print("    [SKIP] LLM did not call cluster_articles")
        return []

    # Validate clustering tool arguments before running the local clustering function.
    args = ClusterArticlesParam.model_validate(json.loads(calls[0].arguments))
    try:
        # Run the actual clustering function.
        result = await cluster_articles(args.articles)

        # Add token usage from the inner clustering LLM call.
        usage.append(result.token_usage)
    except Exception as exc:
        # Clustering failure is skipped because digest items can still be returned.
        print(f"    [SKIP] Clustering failed: {exc}")
        return []

    print(f"    Found {len(result.clusters)} cluster(s)")
    return result.clusters


# ---------------------------------------------------------------------------
# Pure post-processing
# ---------------------------------------------------------------------------


def _build_digest_item(
    article: Article, summary: SummarizerResponse, source: Source
) -> DigestItem | None:
    """Build one DigestItem; return None if Pydantic validation fails."""

    try:
        # Combine fetched article content, generated summary, and original source metadata
        # into the final item shape returned by the digest API/CLI.
        return DigestItem(
            title=article.title,
            summary=summary.summary,
            relevance=summary.relevance,
            published_at=article.published_at,
            sources=[source],
            reasoning=summary.reasoning,
        )
    except Exception as exc:
        # Most common cause: summary > 280 chars.
        print(f"    [SKIP] DigestItem validation failed for {article.article_id}: {exc}")
        return None


def _merge_items_by_cluster(items: list[DigestItem], clusters: list[list[str]]) -> list[DigestItem]:
    """Collapse items that belong to the same cluster into one multi-source item."""

    # If no duplicate-story clusters were found, keep the original items unchanged.
    if not clusters:
        return items

    # Build a lookup from article_id to digest item.
    # Assumes each item has at least one source.
    by_id = {item.sources[0].article_id: item for item in items if item.sources}

    # Track all article IDs that appear inside any cluster.
    clustered: set[str] = {aid for c in clusters for aid in c}

    merged: list[DigestItem] = []

    for cluster in clusters:
        # Keep only article IDs that exist in the built digest items.
        cluster_items = [by_id[aid] for aid in cluster if aid in by_id]
        if not cluster_items:
            continue

        # Use the first item as the representative story.
        rep = cluster_items[0]

        # Preserve all sources from all duplicate items.
        all_sources = [src for it in cluster_items for src in it.sources]

        # Create one merged item with multiple sources.
        merged.append(
            DigestItem(
                title=rep.title,
                summary=rep.summary,
                relevance=rep.relevance,
                published_at=rep.published_at,
                sources=all_sources,
                reasoning=rep.reasoning,
            )
        )
        print(f"    [merge] {cluster} → 1 item with {len(all_sources)} sources")

    # Add back any non-clustered items unchanged.
    merged.extend(item for aid, item in by_id.items() if aid not in clustered)
    return merged


# ---------------------------------------------------------------------------
# Public entry + orchestrator
# ---------------------------------------------------------------------------


async def run_digest(request: DigestRequest) -> DigestResponse:
    """Run the agent with up to MAX_AGENT_ATTEMPTS retries."""

    # Store the last exception so it can be raised if all attempts fail.
    last_exc: Exception | None = None
    print(f"\n>>> Agent start (max {MAX_AGENT_ATTEMPTS} attempts) <<<")

    for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
        print(f"\n  [ATTEMPT {attempt}/{MAX_AGENT_ATTEMPTS}]")
        try:
            # Run one full search/fetch/summarize/cluster pipeline.
            result = await _run_digest_once(request)
            print(f"  [OK] succeeded on attempt {attempt}")
            return result
        except Exception as exc:
            # Retry full pipeline on unhandled exceptions.
            last_exc = exc
            print(f"  [FAIL] {type(exc).__name__}: {exc}")
            if attempt == MAX_AGENT_ATTEMPTS:
                print(f"  [FATAL] exhausted all {MAX_AGENT_ATTEMPTS} attempts")
                raise

    # Defensive fallback: this should only be reachable if the loop exits unexpectedly.
    assert last_exc is not None
    raise last_exc


async def _run_digest_once(request: DigestRequest) -> DigestResponse:
    """Single agent run through all four phases."""

    # Unpack request values once for readability.
    topic, since, max_items = request.topic, request.since, request.max_items

    # Collect token usage from all LLM calls performed during this run.
    usage: TokenLedger = []

    print(f"\n{'=' * 60}")
    print(f"  Digest: topic='{topic}', since={since}, max_items={max_items}")
    print(f"{'=' * 60}")

    # -- Phase 1 --
    print("\n  [Phase 1] search")

    # Search returns candidate Source objects, then we cap how many are processed.
    candidates = (await _search(topic, since, max_items, usage))[:MAX_ITERATIONS]
    print(f"  → {len(candidates)} candidate(s)")

    # -- Phases 2 & 3 in parallel: each candidate is fetched, then summarized.
    # An item is only built when BOTH steps succeed — guaranteeing every
    # digest item has a real summary and relevance.
    print(f"\n  [Phase 2+3] fetch + summarize {len(candidates)} article(s) in parallel")

    # Fetch and summarize all candidates concurrently.
    # return_exceptions=True prevents one failed article from cancelling the whole run.
    pairs = await asyncio.gather(
        *(_fetch_then_summarize(c, topic, usage) for c in candidates),
        return_exceptions=True,
    )

    fetched: dict[str, Article] = {}
    items: list[DigestItem] = []

    # Convert successful fetch+summary pairs into final DigestItem objects.
    for candidate, result in zip(candidates, pairs, strict=True):
        if isinstance(result, BaseException) or result is None:
            continue

        article, summary = result

        # Keep fetched articles for the clustering phase.
        fetched[article.article_id] = article

        # Build final item from article, summary, and source metadata.
        item = _build_digest_item(article, summary, candidate)
        if item is not None:
            items.append(item)

    # -- Phase 4 --
    print(f"\n  [Phase 4] cluster {len(fetched)} fetched article(s)")

    # Cluster duplicate stories, then merge matching digest items.
    clusters = await _cluster(fetched, usage)
    items = _merge_items_by_cluster(items, clusters)[:max_items]

    # Aggregate token usage from all LLM calls.
    response = DigestResponse(
        topic=topic,
        since=since,
        generated_at=datetime.now(UTC),
        items=items,
        token_usage=sum_token_usages(usage),
    )

    print(f"\n{'=' * 60}")
    print(f"  Done: {len(response.items)} digest item(s) | tokens: {response.token_usage}")
    print(f"{'=' * 60}\n")
    return response


async def _fetch_then_summarize(
    candidate: Source, topic: str, usage: TokenLedger
) -> tuple[Article, SummarizerResponse] | None:
    """Run fetch + summarize for one candidate. Returns None if either step fails."""

    # First fetch the full article content using the candidate article_id.
    article = await _fetch(candidate.article_id, usage)
    if article is None:
        return None

    # Then summarize/classify the fetched article content.
    summary = await _summarize(article, topic, usage)
    if summary is None:
        return None

    # Only return a result when both fetch and summarize succeeded.
    return article, summary
