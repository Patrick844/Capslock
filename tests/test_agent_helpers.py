"""Tests for pure helper functions in the agent: extract, build, merge, fetch+summarize."""

from types import SimpleNamespace
from typing import cast

import pytest
from openai.types.responses import Response

from tests.conftest import make_article, make_digest_item, make_source, make_summary
from tracker.agent import (
    _build_digest_item,
    _extract_tool_calls,
    _fetch_then_summarize,
    _merge_items_by_cluster,
)
from tracker.schemas import Article, SummarizerResponse, TokenUsage

# ---------------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------------


def test_extract_tool_calls_filters_by_tool_name() -> None:
    """Only function_call items with the matching name are returned."""
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


# ---------------------------------------------------------------------------
# _build_digest_item
# ---------------------------------------------------------------------------


def test_build_digest_item_returns_digest_item() -> None:
    """Fields from Article, SummarizerResponse, and Source are assembled correctly."""
    source = make_source()
    article = make_article()
    summary = make_summary()

    item = _build_digest_item(article=article, summary=summary, source=source)

    assert item is not None
    assert item.title == article.title
    assert item.summary == summary.summary
    assert item.relevance == summary.relevance
    assert item.sources[0].article_id == article.article_id


# ---------------------------------------------------------------------------
# _merge_items_by_cluster
# ---------------------------------------------------------------------------


def test_merge_items_by_cluster_merges_sources() -> None:
    """Articles in the same cluster are collapsed into one item with multiple sources."""
    item_1 = make_digest_item("art_001")
    item_2 = make_digest_item("art_002")
    item_3 = make_digest_item("art_003")  # not in any cluster

    merged = _merge_items_by_cluster(
        items=[item_1, item_2, item_3],
        clusters=[["art_001", "art_002"]],
    )

    # art_001 + art_002 → 1 merged item; art_003 stays separate → 2 total
    assert len(merged) == 2

    merged_cluster_item = merged[0]
    merged_source_ids = {source.article_id for source in merged_cluster_item.sources}
    assert merged_source_ids == {"art_001", "art_002"}


# ---------------------------------------------------------------------------
# _fetch_then_summarize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_then_summarize_returns_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """When both fetch and summarize succeed, the (article, summary) pair is returned."""
    article = make_article("art_001")
    summary = make_summary()
    source = make_source("art_001")
    usage: list[TokenUsage | None] = []

    async def fake_fetch(_article_id: str, _usage: list[TokenUsage | None]) -> Article:
        return article

    async def fake_summarize(
        _article: Article, _topic: str, _usage: list[TokenUsage | None]
    ) -> SummarizerResponse:
        return summary

    monkeypatch.setattr("tracker.agent._fetch", fake_fetch)
    monkeypatch.setattr("tracker.agent._summarize", fake_summarize)

    result = await _fetch_then_summarize(candidate=source, topic="llm apis", usage=usage)

    assert result is not None
    result_article, result_summary = result
    assert result_article.article_id == "art_001"
    assert result_summary.summary == summary.summary


@pytest.mark.asyncio
async def test_fetch_then_summarize_returns_none_if_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When fetch returns None, summarize is never called and None is returned."""
    source = make_source("art_001")
    usage: list[TokenUsage | None] = []

    async def fake_fetch(_article_id: str, _usage: list[TokenUsage | None]) -> None:
        return None

    async def fake_summarize(
        _article: Article, _topic: str, _usage: list[TokenUsage | None]
    ) -> SummarizerResponse:
        raise AssertionError("summarize should not be called if fetch fails")

    monkeypatch.setattr("tracker.agent._fetch", fake_fetch)
    monkeypatch.setattr("tracker.agent._summarize", fake_summarize)

    result = await _fetch_then_summarize(candidate=source, topic="llm apis", usage=usage)

    assert result is None
