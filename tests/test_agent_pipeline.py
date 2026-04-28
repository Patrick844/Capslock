import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai.types.responses import Response

import tracker.agent as agent
from tracker.schemas import (
    Article,
    ClusterArticlesResponse,
    DigestRequest,
    DigestResponse,
    Source,
    SummarizerResponse,
    TokenUsage,
)


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
        body=f"Body for {article_id}",
        tags=["llm", "api"],
    )


def make_summary() -> SummarizerResponse:
    return SummarizerResponse(
        summary="Short summary.",
        relevance="high",
        reasoning="The article is relevant to the requested topic.",
        token_usage=TokenUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            estimated_cost_usd=0.00001,
        ),
    )


def make_response(
    tool_name: str | None = None,
    arguments: dict[str, Any] | None = None,
) -> Response:
    output = []

    if tool_name is not None:
        output.append(
            SimpleNamespace(
                type="function_call",
                name=tool_name,
                arguments=json.dumps(arguments or {}),
            )
        )
    else:
        output.append(
            SimpleNamespace(
                type="message",
                name=None,
                arguments="{}",
            )
        )

    return cast(
        Response,
        SimpleNamespace(
            output=output,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        ),
    )


@pytest.mark.asyncio
async def test_search_success(monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source("art_001")

    async def fake_call_llm(user_input: str) -> Response:
        return make_response(
            tool_name="search_news",
            arguments={
                "query": "llm apis",
                "since": "2026-03-01",
                "limit": 2,
            },
        )

    def fake_search_news(query: str, since: date, limit: int) -> list[Source]:
        assert query == "llm apis"
        assert since == date(2026, 3, 1)
        assert limit == 2
        return [source]

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "search_news", fake_search_news)

    usage: agent.TokenLedger = []

    results = await agent._search(
        topic="LLM APIs",
        since=date(2026, 3, 1),
        max_items=2,
        usage=usage,
    )

    assert len(results) == 1
    assert results[0].article_id == "art_001"
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_search_returns_empty_when_no_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_llm(user_input: str) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: agent.TokenLedger = []

    results = await agent._search(
        topic="LLM APIs",
        since=date(2026, 3, 1),
        max_items=2,
        usage=usage,
    )

    assert results == []
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_fetch_success(monkeypatch: pytest.MonkeyPatch) -> None:
    article = make_article("art_001")

    async def fake_call_llm(user_input: str) -> Response:
        return make_response(
            tool_name="fetch_article",
            arguments={"article_id": "art_001"},
        )

    def fake_fetch_article(article_id: str) -> Article:
        assert article_id == "art_001"
        return article

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "fetch_article", fake_fetch_article)

    usage: agent.TokenLedger = []

    result = await agent._fetch(
        article_id="art_001",
        usage=usage,
    )

    assert result is not None
    assert result.article_id == "art_001"
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_fetch_returns_none_when_no_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_llm(user_input: str) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: agent.TokenLedger = []

    result = await agent._fetch(
        article_id="art_001",
        usage=usage,
    )

    assert result is None
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_fetch_returns_none_when_local_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_call_llm(user_input: str) -> Response:
        return make_response(
            tool_name="fetch_article",
            arguments={"article_id": "art_404"},
        )

    def fake_fetch_article(article_id: str) -> Article:
        raise ValueError("missing article")

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "fetch_article", fake_fetch_article)

    usage: agent.TokenLedger = []

    result = await agent._fetch(
        article_id="art_404",
        usage=usage,
    )

    assert result is None
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_summarize_success(monkeypatch: pytest.MonkeyPatch) -> None:
    article = make_article("art_001")
    summary = make_summary()

    async def fake_call_llm(user_input: str) -> Response:
        return make_response(
            tool_name="summarize_article",
            arguments={
                "content": article.body,
                "topic": "llm apis",
            },
        )

    async def fake_summarize_content(content: str, topic: str) -> SummarizerResponse:
        assert content == article.body
        assert topic == "llm apis"
        return summary

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "summarize_content", fake_summarize_content)

    usage: agent.TokenLedger = []

    result = await agent._summarize(
        article=article,
        topic="llm apis",
        usage=usage,
    )

    assert result is not None
    assert result.summary == "Short summary."
    assert result.relevance == "high"
    assert len(usage) == 2


@pytest.mark.asyncio
async def test_summarize_returns_none_when_no_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = make_article("art_001")

    async def fake_call_llm(user_input: str) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: agent.TokenLedger = []

    result = await agent._summarize(
        article=article,
        topic="llm apis",
        usage=usage,
    )

    assert result is None
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_summarize_returns_none_when_inner_summarization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = make_article("art_001")

    async def fake_call_llm(user_input: str) -> Response:
        return make_response(
            tool_name="summarize_article",
            arguments={
                "content": article.body,
                "topic": "llm apis",
            },
        )

    async def fake_summarize_content(content: str, topic: str) -> SummarizerResponse:
        raise ValueError("bad structured output")

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "summarize_content", fake_summarize_content)

    usage: agent.TokenLedger = []

    result = await agent._summarize(
        article=article,
        topic="llm apis",
        usage=usage,
    )

    assert result is None
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_cluster_returns_empty_when_less_than_two_articles() -> None:
    usage: agent.TokenLedger = []

    result = await agent._cluster(
        fetched={"art_001": make_article("art_001")},
        usage=usage,
    )

    assert result == []
    assert usage == []


@pytest.mark.asyncio
async def test_cluster_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched = {
        "art_001": make_article("art_001"),
        "art_002": make_article("art_002"),
    }

    async def fake_call_llm(user_input: str) -> Response:
        return make_response(
            tool_name="cluster_articles",
            arguments={
                "articles": [
                    {"article_id": "art_001", "title": "Title art_001"},
                    {"article_id": "art_002", "title": "Title art_002"},
                ]
            },
        )

    async def fake_cluster_articles(articles) -> ClusterArticlesResponse:
        assert len(articles) == 2
        return ClusterArticlesResponse(
            clusters=[["art_001", "art_002"]],
            token_usage=TokenUsage(
                input_tokens=20,
                output_tokens=10,
                total_tokens=30,
                estimated_cost_usd=0.00002,
            ),
        )

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "cluster_articles", fake_cluster_articles)

    usage: agent.TokenLedger = []

    result = await agent._cluster(
        fetched=fetched,
        usage=usage,
    )

    assert result == [["art_001", "art_002"]]
    assert len(usage) == 2


@pytest.mark.asyncio
async def test_cluster_returns_empty_when_no_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = {
        "art_001": make_article("art_001"),
        "art_002": make_article("art_002"),
    }

    async def fake_call_llm(user_input: str) -> Response:
        return make_response()

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    usage: agent.TokenLedger = []

    result = await agent._cluster(
        fetched=fetched,
        usage=usage,
    )

    assert result == []
    assert len(usage) == 1


@pytest.mark.asyncio
async def test_run_digest_once_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    source_1 = make_source("art_001")
    source_2 = make_source("art_002")
    article_1 = make_article("art_001")
    article_2 = make_article("art_002")
    summary = make_summary()

    async def fake_search(
        topic: str,
        since: date,
        max_items: int,
        usage: agent.TokenLedger,
    ) -> list[Source]:
        usage.append(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))
        return [source_1, source_2]

    async def fake_fetch_then_summarize(
        candidate: Source,
        topic: str,
        usage: agent.TokenLedger,
    ) -> tuple[Article, SummarizerResponse]:
        usage.append(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))

        if candidate.article_id == "art_001":
            return article_1, summary

        return article_2, summary

    async def fake_cluster(
        fetched: dict[str, Article],
        usage: agent.TokenLedger,
    ) -> list[list[str]]:
        usage.append(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))
        return [["art_001", "art_002"]]

    monkeypatch.setattr(agent, "_search", fake_search)
    monkeypatch.setattr(agent, "_fetch_then_summarize", fake_fetch_then_summarize)
    monkeypatch.setattr(agent, "_cluster", fake_cluster)

    request = DigestRequest(
        topic="llm apis",
        since=date(2026, 3, 1),
        max_items=5,
    )

    response = await agent._run_digest_once(request)

    assert response.topic == "llm apis"
    assert response.since == date(2026, 3, 1)
    assert len(response.items) == 1
    assert len(response.items[0].sources) == 2
    assert response.token_usage.total_tokens == 8


@pytest.mark.asyncio
async def test_run_digest_retries_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    request = DigestRequest(
        topic="llm apis",
        since=date(2026, 3, 1),
        max_items=5,
    )

    response = await agent.run_digest(request)

    assert calls["count"] == 2
    assert response.topic == "llm apis"
