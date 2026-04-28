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
    SearchIndex,
    Source,
    SummarizerResponse,
    TokenUsage,
)
from tracker.token_usage import get_token_usage

# Shared async OpenAI client used by the LLM-backed tools in this file.
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# Base directory for local fixtures.
_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _filter_article_by_topic(topic: str, news_index: SearchIndex) -> list[Source]:
    article_ids = news_index.topic_index.get(topic, [])

    # Source is the lightweight metadata shape returned by search_news.
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


def _filter_article_by_date(since: date, selected_articles: list[Source]) -> list[Source]:
    return [article for article in selected_articles if article.published_at.date() >= since]


def search_news(query: str, since: date, limit: int = 20) -> list[Source]:
    file_path = _FIXTURES_DIR / "search_index.json"

    # Normalize so the query matches topic_index keys (stored lowercase).
    topic = query.lower().strip()
    safe_limit = max(1, min(limit, 50))

    with file_path.open("r", encoding="utf-8") as file:
        news_index = SearchIndex.model_validate(json.load(file))

    articles_by_topic = _filter_article_by_topic(topic=topic, news_index=news_index)
    articles_by_date = _filter_article_by_date(since=since, selected_articles=articles_by_topic)

    return articles_by_date[:safe_limit]


def _lookup_in_index(article_id: str) -> Source | None:
    """Return Source metadata for an article from the search index, or None if not found."""

    index_path = _FIXTURES_DIR / "search_index.json"

    with index_path.open("r", encoding="utf-8") as file:
        index = SearchIndex.model_validate(json.load(file))

    article = next(
        (a for a in index.articles if a.article_id == article_id),
        None,
    )

    if article is None:
        return None

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

    file_path = _FIXTURES_DIR / "articles" / f"{article_id}.html"
    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    if not isinstance(title_tag, Tag):
        raise ValueError("Missing <title> tag")
    title = title_tag.get_text(strip=True)

    published_raw = _get_meta_content(soup, "published")
    published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))

    body_tag = soup.find("article") or soup.find("main")
    if not isinstance(body_tag, Tag):
        raise ValueError("Missing <article> or <main> tag")
    body = body_tag.get_text(separator="\n", strip=True)

    # Tags are optional; missing becomes an empty list.
    tags_raw = _get_meta_content(soup, "tags", required=False)
    tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]

    index_source = _lookup_in_index(article_id)
    url = index_source.url if index_source else ""

    return Article(
        article_id=article_id,
        url=url,
        title=title,
        published_at=published_at,
        body=body,
        tags=tags,
    )


def _get_meta_content(soup: BeautifulSoup, name: str, required: bool = True) -> str:
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


async def summarize_content(content: str, topic: str) -> SummarizerResponse:
    """Classify an article as high/medium/low relevance and produce a short summary."""

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
        input=[{"role": "user", "content": f"Topic: {topic}\n\nContent:\n{content}"}],
        text_format=ArticleClassification,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for article classification")

    token_usage = get_token_usage(response) or TokenUsage()

    return SummarizerResponse(
        summary=parsed.summary,
        relevance=parsed.relevance,
        reasoning=parsed.reason,
        token_usage=token_usage,
    )


async def cluster_articles(articles: list[ArticleTitle]) -> ClusterArticlesResponse:
    """Group articles that describe the same news story by title similarity."""

    titles_block = "\n".join(f"- {a.article_id}: {a.title}" for a in articles)

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
        input=[{"role": "user", "content": f"Cluster these articles by story:\n{titles_block}"}],
        text_format=ClusterClassification,
    )

    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("LLM returned no structured output for clustering")

    token_usage = get_token_usage(response) or TokenUsage()

    # Only keep clusters with at least two article IDs; the LLM may return singletons.
    valid_clusters = [cluster for cluster in parsed.clusters if len(cluster) >= 2]

    return ClusterArticlesResponse(clusters=valid_clusters, token_usage=token_usage)
