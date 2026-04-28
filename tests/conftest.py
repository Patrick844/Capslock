"""Shared test factories used across multiple test modules."""

from datetime import UTC, datetime

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


def make_digest_item(article_id: str) -> DigestItem:
    return DigestItem(
        title=f"Title {article_id}",
        summary="Short summary.",
        relevance="high",
        published_at=datetime(2026, 3, 1, tzinfo=UTC),
        sources=[make_source(article_id)],
        reasoning="Relevant to the topic.",
    )
