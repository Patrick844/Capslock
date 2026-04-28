"""Tool implementations exposed to the agent's LLM tool-calling loop.

The candidate must implement the two public functions below.
Fixture data lives at ``../fixtures/`` relative to this file.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from bs4 import BeautifulSoup
from bs4.element import Tag
from openai import AsyncOpenAI

from tracker.config import settings
from tracker.schemas import (
    Article,
    ArticleClassification,
    ArticleTitle,
    ClusterArticlesResponse,
    ClusterClassification,
    Index,
    Source,
    SummarizerResponse,
)
from tracker.token_usage import get_token_usage

# Async client so summarize_content / cluster_articles can be awaited from the
# agent loop without thread-pool wrapping.
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Resolved at import time so both tools share the same base path.
_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _filter_article_by_topic(topic: str, news_index: Index) -> list[Source]:
    topic_index = news_index.topic_index
    selected_articles_id: list[str] = []
    for k, v in topic_index.items():
        if k == topic:
            selected_articles_id.extend(v)

    return [
        article for article in news_index.articles if article.article_id in selected_articles_id
    ]


def _filter_article_by_date(since: date, selected_articles: list[Source]) -> list[Source]:
    return [article for article in selected_articles if article.published_at.date() >= since]


def search_news(query: str, since: date, limit: int = 20) -> list[Source]:

    file_path = _FIXTURES_DIR / "search_index.json"
    topic = query.lower()

    with file_path.open("r", encoding="utf-8") as file:
        news_index = Index.model_validate(json.load(file))
    selected_articles_by_topic: list[Source] = _filter_article_by_topic(
        topic=topic, news_index=news_index
    )
    selected_articles: list[Source] = _filter_article_by_date(
        since=since, selected_articles=selected_articles_by_topic
    )

    return selected_articles


def _lookup_in_index(article_id: str) -> Source | None:
    """Return the Source metadata for an article from the search index, or None if not found."""
    index_path = _FIXTURES_DIR / "search_index.json"
    with index_path.open("r", encoding="utf-8") as f:
        index = Index.model_validate(json.load(f))
    return next((a for a in index.articles if a.article_id == article_id), None)


def fetch_article(article_id: str) -> Article:
    """Fetch an article's full content by ID.

    Reads the HTML fixture, extracts title, published_at, source, body, and tags.
    URL and topics are resolved from the search index so they match the canonical metadata.
    """
    file_path = _FIXTURES_DIR / "articles" / f"{article_id}.html"

    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    if not isinstance(title_tag, Tag):
        raise ValueError("Missing <title> tag")
    title = title_tag.get_text(strip=True)

    published_raw = _get_meta_content(soup, "published")
    published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))

    # <meta name="source"> holds the publisher domain (e.g. "techpress.dev")
    source = _get_meta_content(soup, "source")

    body_tag = soup.find("article") or soup.find("main")
    if not isinstance(body_tag, Tag):
        raise ValueError("Missing <article> or <main> tag")
    body = body_tag.get_text(separator="\n", strip=True)

    tags_raw = _get_meta_content(soup, "tags", required=False)
    tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]

    # Resolve canonical URL and topic list from the search index.
    # The HTML only has hyphenated tag strings; the index has the normalized topic names
    # that match the topic_index keys (e.g. "voice ai", not "voice-ai").
    index_source = _lookup_in_index(article_id)
    url = index_source.url if index_source else ""
    topics = index_source.topics if index_source else tags

    return Article(
        article_id=article_id,
        url=url,
        title=title,
        published_at=published_at,
        topics=topics,
        source=source,
        body=body,
        tags=tags,
    )


def _get_meta_content(
    soup: BeautifulSoup,
    name: str,
    required: bool = True,
) -> str:
    meta_tag = soup.find("meta", attrs={"name": name})

    if not isinstance(meta_tag, Tag):
        if required:
            raise ValueError(f'Missing <meta name="{name}">')
        return ""

    content = meta_tag.get("content")

    if not isinstance(content, str):
        if required:
            raise ValueError(f'Missing content for <meta name="{name}">')
        return ""

    return content.strip()


async def summarize_content(
    content: str,
    topic: str,
) -> SummarizerResponse:
    """Classify an article as High, Medium, or Low relevance and summarize it."""

    response = await client.responses.parse(
        model=settings.LLM_MODEL,
        instructions=(
            f"You classify local news articles based on relevance to this topic {topic}. "
            "Return only the structured result. "
            "Use High if the article is directly about the topic. "
            "Use Medium if the article is partially related or mentions the topic indirectly. "
            "Use Low if the article is mostly unrelated. "
            "Write a concise summary based only on the provided article content."
        ),
        input=[
            {
                "role": "user",
                "content": (f"Topic: {topic}\n\nContent:\n{content}"),
            }
        ],
        text_format=ArticleClassification,
    )
    token_usage = get_token_usage(response)

    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for article classification")

    # ArticleClassification.relevance is already Literal["high","medium","low"] —
    # the structured output format enforces lowercase, so no .lower() needed.
    return SummarizerResponse(
        summary=parsed.summary,
        relevance=parsed.relevance,
        reasoning=parsed.reason,
        token_usage=token_usage,
    )


async def cluster_articles(articles: list[ArticleTitle]) -> ClusterArticlesResponse:
    """Group articles that describe the same news story by title similarity.

    Returns clusters of 2+ article_ids that cover the same specific event.
    Articles with no duplicates are NOT included in the output.
    Returns an empty `clusters` list when no duplicates exist.
    """
    # Nothing to compare — no clustering possible.
    # if len(articles) < 2:
    #     return ClusterArticlesResponse(clusters=[], token_usage=None)

    # Build a deterministic, easy-to-read prompt body for the LLM.
    titles_block = "\n".join(f"- {a.article_id}: {a.title}" for a in articles)

    response = await client.responses.parse(
        model=settings.LLM_MODEL,
        instructions=(
            "You group news article titles by the underlying story they describe. "
            "Two articles describe the SAME story only when they cover the same "
            "specific event, announcement, or development — same subject AND same "
            "news (e.g. three outlets reporting the same product launch). "
            "Articles about the same general topic but different events are NOT "
            "the same story. "
            "Return clusters of 2+ article_ids; do not include articles that "
            "have no duplicate. If nothing duplicates, return an empty list."
        ),
        input=[
            {
                "role": "user",
                "content": f"Cluster these articles by story:\n{titles_block}",
            }
        ],
        text_format=ClusterClassification,
    )

    token_usage = get_token_usage(response)
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for clustering")

    # Defensive: drop any singleton or empty cluster the LLM might emit.
    valid_clusters = [c for c in parsed.clusters if len(c) >= 2]

    return ClusterArticlesResponse(
        clusters=valid_clusters,
        token_usage=token_usage,
    )


# ---------------------------------------------------------------------------
# Internal debugging helpers — not part of the candidate's required surface.
# ---------------------------------------------------------------------------
# def _internal_debug_tool(article_id: str) -> dict:
#     """Return raw fixture bytes for validation tooling.
#
#     Exposed to the agent only when FIXTURE_VALIDATION_MODE=1.
#     See fixtures/NOTES.md §5 for the validation protocol.
#     """
#     from pathlib import Path
#     path = Path(__file__).parent.parent / "fixtures" / "articles" / f"{article_id}.html"
#     return {"article_id": article_id, "raw_bytes": path.read_bytes().hex()}
