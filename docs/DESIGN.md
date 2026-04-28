# Design Notes

## 1. Architecture Overview

A four-phase pipeline driven by native OpenAI tool calling. One LLM call per phase, with phases 2 and 3 fanning out in parallel via `asyncio.gather`:

```
search_news ─▶ fetch_article (×N parallel) ─▶ summarize_article (×N parallel) ─▶ cluster_articles
                                                                                       │
                                                          _merge_items_by_cluster ◀────┘
                                                                                       │
                                                                                  DigestResponse
```

The orchestrator (`_run_digest_once`) reads top-to-bottom as a recipe. Each phase has a single helper that wraps one LLM round-trip: prompt the LLM, validate the tool call's JSON arguments through a Pydantic model, execute the underlying function. State (fetched articles, summaries) is plain dicts passed between phases. Clustering is decoupled — the LLM's cluster output flows into a pure post-processing function `_merge_items_by_cluster(items, clusters)` that has no I/O. Phases 2 & 3 are deterministic per candidate: an item is built only when *both* fetch and summarize succeed, so every digest item carries a real summary and relevance, never `None`. Top-level retry is `MAX_AGENT_ATTEMPTS = 6` — the whole pipeline retries on unhandled failure, no nested per-call retry (total budget = 6, not 12).

## 2. LLM Provider & Framework Choices

**OpenAI Responses API** with the official `AsyncOpenAI` SDK. No agent framework — the orchestrator is ~30 lines and a framework would obscure rather than clarify it. Native SDK gives me typed JSON-schema tool calls (`responses.create(... tools=...)`) and structured-output parsing (`responses.parse(... text_format=PydanticModel)`) for the inner LLM calls. `gpt-4o-mini` (configurable via `LLM_MODEL`) — cheap, supports both function calling and structured output. Async client throughout means `asyncio.gather` gives real concurrency without thread-pool wrapping.

## 3. Custom Tools Added

Two custom tools beyond the prescribed `search_news` and `fetch_article`:

**`summarize_article(content, topic)`** — wraps an inner `responses.parse` call with the Pydantic schema `ArticleClassification(relevance, summary, reason)`. Returns a `summary` capped at 280 chars (enforced both in the schema's `max_length` and in the prompt), a `relevance` literal in `{"high", "medium", "low"}`, and free-text `reason`. It earns its place because the gold cases test ranking semantics ("directly about / partially related / unrelated") and `DigestItem.relevance` is a required field — a plain summarizer wouldn't give us that label, and asking the *outer* LLM to produce it inline would mix concerns. The structured-output schema also forces lowercase relevance values, removing a class of validation bugs.

**`cluster_articles(articles)`** — same pattern: an LLM-backed tool taking `[{article_id, title}]` and returning groups of duplicate stories via the schema `ClusterClassification(clusters: list[list[str]])`. It earns its place because gold case 2 explicitly tests `must_collapse_cluster=["art_014","art_015","art_016"]`: three outlets reporting the same GPT-5 pricing change must collapse into one `DigestItem` with three sources. A heuristic (Jaccard, keyword overlap) would miss the wording variation across "OpenAI announces new GPT-5 pricing tiers" / "GPT-5 API prices revised" / "Breaking: OpenAI raises GPT-5 pricing" — the LLM gets it instantly. The tool is best-effort: if it fails, items pass through unmerged rather than killing the run.

## 4. Deduplication & Citation Strategy

Deduplication runs *after* every article has been fetched and summarised, so the LLM has full visibility of titles when grouping. `_merge_items_by_cluster` is a pure function over `(items, clusters)`: cluster members collapse into one `DigestItem` whose `sources` field aggregates every article's `Source`; non-clustered items pass through unchanged. The representative for each merged cluster is `cluster[0]` — simple, deterministic, easy to reason about.

Citation fidelity: every `Source` field comes from `search_index.json` via `_lookup_in_index` — `fetch_article` reads HTML for body/title/published_at but pulls canonical `url` and `topics` from the index. The LLM never sees URLs in a way that lets it invent them.

## 5. Known Limitations / What You'd Do With More Time

- **Iteration cap is global, not per-phase.** A pathological run with many candidates can consume the budget on fetches before reaching cluster. In practice fixtures stay under the cap.
- **Cluster representative is `cluster[0]`.** Better would be the article with highest relevance or the most authoritative source; skipped for simplicity.
- **No fallback heuristic for clustering.** If the LLM mis-clusters (misses a 4th outlet of the GPT-5 story), there's no second-pass title-similarity check.
- **No tests.** The pure functions (`_build_digest_item`, `_merge_items_by_cluster`, `_filter_article_by_date`) are designed to be unit-testable but I haven't written tests yet — that's the next thing.
