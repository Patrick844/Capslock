from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast

import pytest
from openai.types.responses import Response

from tracker.agent import (
    _build_digest_item,
    _extract_tool_calls,
    _fetch_then_summarize,
    _merge_items_by_cluster,
)
from tracker.schemas import Article, DigestItem, Source, SummarizerResponse, TokenUsage


def make_source(article_id: str = "art_001") -> Source:
    return Source(
        article_id=article_id,
        url=f"https://example.com/{article_id}",
        title=f"Title {article_id}",
        published_at=datetime(2026, 3, 1, tzinfo=UTC),
    )


def make_article(article_id: str = "art_001") -> Article:
    return Article(
        article_id=article_id,
        url=f"https://example.com/{article_id}",
        title=f"Title {article_id}",
        published_at=datetime(2026, 3, 1, tzinfo=UTC),
        body="This is the article body.",
        tags=["api", "llm"],
    )


def make_summary() -> SummarizerResponse:
    return SummarizerResponse(
        summary="This article discusses LLM API pricing changes and their impact on developers.",
        relevance="high",
        reasoning="The article directly discusses the requested topic.",
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.00001,
        ),
    )


def make_digest_item(article_id: str) -> DigestItem:
    source = make_source(article_id)

    return DigestItem(
        title=f"Title {article_id}",
        summary="Short summary.",
        relevance="high",
        published_at=datetime(2026, 3, 1, tzinfo=UTC),
        sources=[source],
        reasoning="Relevant to the topic.",
    )


def test_extract_tool_calls_filters_by_tool_name():
    response = cast(
        Response,
        SimpleNamespace(
            output=[
                SimpleNamespace(type="function_call", name="search_news"),
                SimpleNamespace(type="function_call", name="fetch_article"),
                SimpleNamespace(type="message", name=None),
            ]
        ),
    )

    calls = _extract_tool_calls(response, "search_news")

    assert len(calls) == 1
    assert calls[0].name == "search_news"


def test_build_digest_item_returns_digest_item() -> None:
    source = make_source()
    article = make_article()
    summary = make_summary()

    item = _build_digest_item(
        article=article,
        summary=summary,
        source=source,
    )

    assert item is not None
    assert item.title == article.title
    assert item.summary == summary.summary
    assert item.relevance == summary.relevance
    assert item.sources[0].article_id == article.article_id


def test_merge_items_by_cluster_merges_sources() -> None:
    item_1 = make_digest_item("art_001")
    item_2 = make_digest_item("art_002")
    item_3 = make_digest_item("art_003")

    merged = _merge_items_by_cluster(
        items=[item_1, item_2, item_3],
        clusters=[["art_001", "art_002"]],
    )

    assert len(merged) == 2

    merged_cluster_item = merged[0]
    merged_source_ids = {source.article_id for source in merged_cluster_item.sources}

    assert merged_source_ids == {"art_001", "art_002"}


@pytest.mark.asyncio
async def test_fetch_then_summarize_returns_pair(monkeypatch) -> None:
    article = make_article("art_001")
    summary = make_summary()
    source = make_source("art_001")
    usage = []

    async def fake_fetch(article_id: str, usage):
        return article

    async def fake_summarize(article: Article, topic: str, usage):
        return summary

    monkeypatch.setattr("tracker.agent._fetch", fake_fetch)
    monkeypatch.setattr("tracker.agent._summarize", fake_summarize)

    result = await _fetch_then_summarize(
        candidate=source,
        topic="llm apis",
        usage=usage,
    )

    assert result is not None

    result_article, result_summary = result

    assert result_article.article_id == "art_001"
    assert result_summary.summary == summary.summary


@pytest.mark.asyncio
async def test_fetch_then_summarize_returns_none_if_fetch_fails(monkeypatch) -> None:
    source = make_source("art_001")
    usage = []

    async def fake_fetch(article_id: str, usage):
        return None

    async def fake_summarize(article: Article, topic: str, usage):
        raise AssertionError("summarize should not be called if fetch fails")

    monkeypatch.setattr("tracker.agent._fetch", fake_fetch)
    monkeypatch.setattr("tracker.agent._summarize", fake_summarize)

    result = await _fetch_then_summarize(
        candidate=source,
        topic="llm apis",
        usage=usage,
    )

    assert result is None
