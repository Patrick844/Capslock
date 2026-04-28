"""Tool implementations exposed to the agent's LLM tool-calling loop.

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
    TokenUsage,
)
from tracker.token_usage import get_token_usage

# Shared async OpenAI client used by the LLM-backed tools in this file.
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Base directory for local fixtures.
# All local search index and article HTML files are resolved from here.
_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _filter_article_by_topic(topic: str, news_index: Index) -> list[Source]:
    # Look up article IDs for the normalized topic from the index.
    article_ids = news_index.topic_index.get(topic, [])

    # Convert index articles into Source objects.
    # Source is the lightweight metadata returned by search_news.
    return [
        Source(
            article_id=article.article_id,
            url=article.url,
            title=article.title,
            published_at=article.published_at,
        )
        for article in news_index.articles
        if article.article_id in article_ids
    ]


def _filter_article_by_date(
    since: date,
    selected_articles: list[Source],
) -> list[Source]:
    # Keep only articles published on or after the requested date.
    return [article for article in selected_articles if article.published_at.date() >= since]


def search_news(query: str, since: date, limit: int = 20) -> list[Source]:
    # Load the local search index fixture.
    file_path = _FIXTURES_DIR / "search_index.json"

    # Normalize the query so it matches topic_index keys.
    topic = query.lower().strip()

    # Clamp the limit to a safe range.
    safe_limit = max(1, min(limit, 50))

    # Read and validate the index JSON with Pydantic.
    with file_path.open("r", encoding="utf-8") as file:
        news_index = Index.model_validate(json.load(file))

    # First filter by topic.
    selected_articles_by_topic = _filter_article_by_topic(
        topic=topic,
        news_index=news_index,
    )

    # Then filter by publication date.
    selected_articles = _filter_article_by_date(
        since=since,
        selected_articles=selected_articles_by_topic,
    )

    # Return only up to the requested safe limit.
    return selected_articles[:safe_limit]


def _lookup_in_index(article_id: str) -> Source | None:
    """Return Source metadata for an article from the search index."""

    # The HTML article fixture contains body/tags, but the canonical URL
    # comes from the search index.
    index_path = _FIXTURES_DIR / "search_index.json"

    # Read and validate the search index.
    with index_path.open("r", encoding="utf-8") as file:
        index = Index.model_validate(json.load(file))

    # Find the matching article metadata by article_id.
    article = next(
        (article for article in index.articles if article.article_id == article_id),
        None,
    )

    # Return None when the article does not exist in the index.
    if article is None:
        return None

    # Convert index metadata into the Source schema used by the app.
    return Source(
        article_id=article.article_id,
        url=article.url,
        title=article.title,
        published_at=article.published_at,
    )


def fetch_article(article_id: str) -> Article:
    """Fetch an article's full content by ID.

    Reads ``fixtures/articles/{article_id}.html`` and extracts:
    - <title>
    - <meta name="published">
    - first <article> or <main> body element
    - optional <meta name="tags">

    The canonical URL is resolved from ``search_index.json``.
    """

    # Resolve the local HTML fixture path for the requested article ID.
    file_path = _FIXTURES_DIR / "articles" / f"{article_id}.html"

    # Read the HTML file and parse it with BeautifulSoup.
    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    # Extract the document title.
    title_tag = soup.find("title")
    if not isinstance(title_tag, Tag):
        raise ValueError("Missing <title> tag")

    title = title_tag.get_text(strip=True)

    # Extract and parse the published datetime from metadata.
    published_raw = _get_meta_content(soup, "published")
    published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))

    # Use the first <article> element, or fall back to <main>.
    body_tag = soup.find("article") or soup.find("main")
    if not isinstance(body_tag, Tag):
        raise ValueError("Missing <article> or <main> tag")

    # Extract clean text from the article/main body.
    body = body_tag.get_text(separator="\n", strip=True)

    # Tags are optional. Missing tags become an empty list.
    tags_raw = _get_meta_content(soup, "tags", required=False)
    tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]

    # Resolve the canonical article URL from search_index.json.
    index_source = _lookup_in_index(article_id)
    url = index_source.url if index_source else ""

    # Return the full Article object expected by the rest of the pipeline.
    return Article(
        article_id=article_id,
        url=url,
        title=title,
        published_at=published_at,
        body=body,
        tags=tags,
    )


def _get_meta_content(
    soup: BeautifulSoup,
    name: str,
    required: bool = True,
) -> str:
    # Find a meta tag by name, for example:
    # <meta name="published" content="2026-03-15T09:00:00Z">
    meta_tag = soup.find("meta", attrs={"name": name})

    # Required metadata must exist.
    if not isinstance(meta_tag, Tag):
        if required:
            raise ValueError(f'Missing <meta name="{name}">')
        return ""

    # Extract the content attribute from the meta tag.
    content = meta_tag.get("content")

    # Required metadata must have a valid string content value.
    if not isinstance(content, str):
        if required:
            raise ValueError(f'Missing content for <meta name="{name}">')
        return ""

    # Normalize whitespace before returning.
    return content.strip()


async def summarize_content(
    content: str,
    topic: str,
) -> SummarizerResponse:
    """Classify an article as high, medium, or low relevance and summarize it."""

    # Ask the LLM for structured output matching ArticleClassification.
    response = await client.responses.parse(
        model=settings.LLM_MODEL,
        instructions=(
            f"You classify local news articles based on relevance to this topic: {topic}. "
            "Return only the structured result. "
            "Use high if the article is directly about the topic. "
            "Use medium if the article is partially related or mentions the topic indirectly. "
            "Use low if the article is mostly unrelated. "
            "Write a concise summary based only on the provided article content."
        ),
        input=[
            {
                "role": "user",
                "content": f"Topic: {topic}\n\nContent:\n{content}",
            }
        ],
        text_format=ArticleClassification,
    )

    # Structured output should be available here if parsing succeeded.
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for article classification")

    # Extract token usage, falling back to an empty TokenUsage object if unavailable.
    token_usage = get_token_usage(response) or TokenUsage()

    # Convert the structured LLM output into the app-level summarizer response.
    return SummarizerResponse(
        summary=parsed.summary,
        relevance=parsed.relevance,
        reasoning=parsed.reason,
        token_usage=token_usage,
    )


async def cluster_articles(
    articles: list[ArticleTitle],
) -> ClusterArticlesResponse:
    """Group articles that describe the same news story by title similarity."""

    # Build a compact prompt containing only article IDs and titles.
    # This keeps clustering focused on duplicate-story detection.
    titles_block = "\n".join(f"- {article.article_id}: {article.title}" for article in articles)

    # Ask the LLM for structured clusters of duplicate stories.
    response = await client.responses.parse(
        model=settings.LLM_MODEL,
        instructions=(
            "You group news article titles by the underlying story they describe. "
            "Two articles describe the same story only when they cover the same "
            "specific event, announcement, or development. "
            "Articles about the same general topic but different events are not "
            "the same story. "
            "Return clusters of 2+ article_ids. "
            "Do not include articles that have no duplicate. "
            "If nothing duplicates, return an empty list."
        ),
        input=[
            {
                "role": "user",
                "content": f"Cluster these articles by story:\n{titles_block}",
            }
        ],
        text_format=ClusterClassification,
    )

    # Structured output should be available here if parsing succeeded.
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for clustering")

    # Extract token usage, falling back to an empty TokenUsage object if unavailable.
    token_usage = get_token_usage(response) or TokenUsage()

    # Defensive cleanup: only keep real clusters with at least two article IDs.
    valid_clusters = [cluster for cluster in parsed.clusters if len(cluster) >= 2]

    # Return the final cluster response used by the agent pipeline.
    return ClusterArticlesResponse(
        clusters=valid_clusters,
        token_usage=token_usage,
    )
