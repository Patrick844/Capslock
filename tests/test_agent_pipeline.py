"""Tests for the agent's four-phase pipeline: search, fetch, summarize, cluster."""

import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types.responses import Response

import tracker.agent as agent
from tests.conftest import make_article, make_source, make_summary
from tracker.schemas import (
    Article,
    ArticleTitle,
    ClusterArticlesResponse,
    DigestRequest,
    DigestResponse,
    Source,
    SummarizerResponse,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Helper: build a fake LLM Response with an optional tool call
# ---------------------------------------------------------------------------


def make_response(
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> Response:
    """Return a cast Response whose output contains one item.

    If tool_name is given, the item is a function_call for that tool.
    Otherwise it is a plain message (simulating no tool call).
    """
    if tool_name is not None:
        output = [
            SimpleNamespace(
                type="function_call",
                name=tool_name,
                arguments=json.dumps(arguments or {}),
            )
        ]
    else:
        output = [SimpleNamespace(type="message", name=None, arguments="{}")]

    return cast(
        Response,
        SimpleNamespace(
            output=output,
            usage=SimpleNamespace(input_tokens=100, output_tokens=50, total_tokens=150),
        ),
    )


# ---------------------------------------------------------------------------
# Phase 1 — search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM calls search_news → the Python tool is executed and its results returned."""
    source = make_source("art_001")

    async def fake_call_llm(_user_input: str) -> Response:
        return make_response(
            tool_name="search_news",
            arguments={"query": "llm apis", "since": "2026-03-01", "limit": 2},
        )

    def fake_search_news(query: str, since: date, limit: int) -> list[Source]:
        assert query == "llm apis"
        assert since == date(2026, 3, 1)
        assert limit == 2
        return [source]

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "search_news", fake_search_news)

    usage: list[TokenUsage | None] = []
    results = await agent._search(
        topic="LLM APIs", since=date(2026, 3, 1), max_items=2, usage=usage
    )

    assert len(results) == 1
    assert results[0].article_id == "art_001"
    assert len(usage) == 1  # one LLM call recorded


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM responds with a plain message instead of a tool call → empty list."""

    async def fake_call_llm(_user_input: str) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: list[TokenUsage | None] = []
    results = await agent._search(
        topic="LLM APIs", since=date(2026, 3, 1), max_items=2, usage=usage
    )

    assert results == []
    assert len(usage) == 1


# ---------------------------------------------------------------------------
# Phase 2 — fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM calls fetch_article → the local HTML tool is executed."""
    article = make_article("art_001")

    async def fake_call_llm(_user_input: str) -> Response:
        return make_response(tool_name="fetch_article", arguments={"article_id": "art_001"})

    def fake_fetch_article(article_id: str) -> Article:
        assert article_id == "art_001"
        return article

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "fetch_article", fake_fetch_article)

    usage: list[TokenUsage | None] = []
    result = await agent._fetch(article_id="art_001", usage=usage)

    assert result is not None
    assert result.article_id == "art_001"
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_fetch_returns_none_when_no_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM does not call fetch_article → None returned, article skipped."""

    async def fake_call_llm(_user_input: str) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: list[TokenUsage | None] = []
    result = await agent._fetch(article_id="art_001", usage=usage)

    assert result is None
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_fetch_returns_none_when_local_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local fixture read raises → None returned instead of propagating the error."""

    async def fake_call_llm(_user_input: str) -> Response:
        return make_response(tool_name="fetch_article", arguments={"article_id": "art_404"})

    def fake_fetch_article(_article_id: str) -> Article:
        raise ValueError("missing article")

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "fetch_article", fake_fetch_article)

    usage: list[TokenUsage | None] = []
    result = await agent._fetch(article_id="art_404", usage=usage)

    assert result is None
    assert len(usage) == 1


# ---------------------------------------------------------------------------
# Phase 3 — summarize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM calls summarize_article → inner summarize_content is called and result returned."""
    article = make_article("art_001")
    summary = make_summary()

    async def fake_call_llm(_user_input: str, **_kwargs: object) -> Response:
        return make_response(
            tool_name="summarize_article",
            arguments={"content": article.body, "topic": "llm apis"},
        )

    async def fake_summarize_content(content: str, topic: str) -> SummarizerResponse:
        assert content == article.body
        assert topic == "llm apis"
        return summary

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "summarize_content", fake_summarize_content)

    usage: list[TokenUsage | None] = []
    result = await agent._summarize(article=article, topic="llm apis", usage=usage)

    assert result is not None
    assert result.summary == "Short summary."
    assert result.relevance == "high"
    # Two entries: one for routing LLM call, one for inner summarize_content.
    assert len(usage) == 2


@pytest.mark.asyncio
async def test_summarize_returns_none_when_no_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM does not call summarize_article → None returned, article skipped."""
    article = make_article("art_001")

    async def fake_call_llm(_user_input: str, **_kwargs: object) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: list[TokenUsage | None] = []
    result = await agent._summarize(article=article, topic="llm apis", usage=usage)

    assert result is None
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_summarize_returns_none_when_inner_summarization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """summarize_content raises → None returned instead of propagating."""
    article = make_article("art_001")

    async def fake_call_llm(_user_input: str, **_kwargs: object) -> Response:
        return make_response(
            tool_name="summarize_article",
            arguments={"content": article.body, "topic": "llm apis"},
        )

    async def fake_summarize_content(_content: str, _topic: str) -> SummarizerResponse:
        raise ValueError("bad structured output")

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "summarize_content", fake_summarize_content)

    usage: list[TokenUsage | None] = []
    result = await agent._summarize(article=article, topic="llm apis", usage=usage)

    assert result is None
    assert len(usage) == 1


# ---------------------------------------------------------------------------
# Phase 4 — cluster
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_returns_empty_when_less_than_two_articles() -> None:
    """Clustering is skipped when fewer than two articles were fetched."""
    usage: list[TokenUsage | None] = []
    result = await agent._cluster(fetched={"art_001": make_article("art_001")}, usage=usage)

    assert result == []
    assert usage == []  # no LLM call made


@pytest.mark.asyncio
async def test_cluster_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM calls cluster_articles → duplicate-story groups returned."""
    fetched = {"art_001": make_article("art_001"), "art_002": make_article("art_002")}

    async def fake_call_llm(_user_input: str) -> Response:
        return make_response(
            tool_name="cluster_articles",
            arguments={
                "articles": [
                    {"article_id": "art_001", "title": "Title art_001"},
                    {"article_id": "art_002", "title": "Title art_002"},
                ]
            },
        )

    async def fake_cluster_articles(articles: list[ArticleTitle]) -> ClusterArticlesResponse:
        assert len(articles) == 2
        return ClusterArticlesResponse(
            clusters=[["art_001", "art_002"]],
            token_usage=TokenUsage(input_tokens=20, output_tokens=10, total_tokens=30),
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "cluster_articles", fake_cluster_articles)

    usage: list[TokenUsage | None] = []
    result = await agent._cluster(fetched=fetched, usage=usage)

    assert result == [["art_001", "art_002"]]
    # Two entries: one for routing LLM call, one for inner cluster_articles.
    assert len(usage) == 2


@pytest.mark.asyncio
async def test_cluster_returns_empty_when_no_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM does not call cluster_articles → empty list, digest continues unchanged."""
    fetched = {"art_001": make_article("art_001"), "art_002": make_article("art_002")}

    async def fake_call_llm(_user_input: str) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: list[TokenUsage | None] = []
    result = await agent._cluster(fetched=fetched, usage=usage)

    assert result == []
    assert len(usage) == 1


# ---------------------------------------------------------------------------
# Full pipeline: _run_digest_once and retry logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_digest_once_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two articles sharing the same story are merged into one digest item."""
    source_1 = make_source("art_001")
    source_2 = make_source("art_002")
    article_1 = make_article("art_001")
    article_2 = make_article("art_002")
    summary = make_summary()

    async def fake_search(
        _topic: str, _since: date, _max_items: int, usage: list[TokenUsage | None]
    ) -> list[Source]:
        usage.append(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))
        return [source_1, source_2]

    async def fake_fetch_then_summarize(
        candidate: Source, _topic: str, usage: list[TokenUsage | None]
    ) -> tuple[Article, SummarizerResponse]:
        usage.append(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))
        return (article_1, summary) if candidate.article_id == "art_001" else (article_2, summary)

    async def fake_cluster(
        _fetched: dict[str, Article], usage: list[TokenUsage | None]
    ) -> list[list[str]]:
        usage.append(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))
        return [["art_001", "art_002"]]

    monkeypatch.setattr(agent, "_search", fake_search)
    monkeypatch.setattr(agent, "_fetch_then_summarize", fake_fetch_then_summarize)
    monkeypatch.setattr(agent, "_cluster", fake_cluster)

    request = DigestRequest(topic="llm apis", since=date(2026, 3, 1), max_items=5)
    response = await agent._run_digest_once(request)

    assert response.topic == "llm apis"
    assert response.since == date(2026, 3, 1)
    assert len(response.items) == 1  # two articles merged into one
    assert len(response.items[0].sources) == 2
    # 1 (search) + 2 (fetch×2) + 1 (cluster) = 4 entries × total_tokens=2 each = 8
    assert response.token_usage.total_tokens == 8


@pytest.mark.asyncio
async def test_run_digest_retries_after_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_digest retries _run_digest_once on failure and succeeds on the second attempt."""
    calls = {"count": 0}

    async def fake_run_digest_once(request: DigestRequest) -> DigestResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return DigestResponse(
            topic=request.topic,
            since=request.since,
            generated_at=datetime(2026, 3, 1, tzinfo=UTC),
            items=[],
            token_usage=TokenUsage(),
        )

    monkeypatch.setattr(agent, "_run_digest_once", fake_run_digest_once)

    request = DigestRequest(topic="llm apis", since=date(2026, 3, 1), max_items=5)
    response = await agent.run_digest(request)

    assert calls["count"] == 2
    assert response.topic == "llm apis"
