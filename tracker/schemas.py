from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    article_id: str = Field(..., description="Fixture article identifier, e.g., 'art_001'")
    url: str
    title: str
    published_at: datetime


class Article(Source):
    body: str
    tags: list[str] = Field(default_factory=list)


class Index(BaseModel):
    version: str
    generated_at: str
    articles: list[IndexArticles]
    topic_index: dict[str, list[str]]


class IndexArticles(Source):
    topics: list[str]
    source: str


class DigestItem(BaseModel):
    title: str
    summary: str = Field(..., max_length=280)
    relevance: Literal["high", "medium", "low"]
    published_at: datetime
    sources: list[Source] = Field(..., min_length=1)
    reasoning: str


class DigestRequest(BaseModel):
    topic: str
    since: date
    max_items: int = Field(default=10, ge=1, le=50)


class DigestResponse(BaseModel):
    topic: str
    since: date
    generated_at: datetime
    items: list[DigestItem]
    token_usage: TokenUsage


class SearchNews(BaseModel):
    query: str
    since: date
    limit: int


class FetchArticleParam(BaseModel):
    article_id: str


class SummarizerParam(BaseModel):
    content: str
    topic: str


class SummarizerResponse(BaseModel):
    summary: str
    relevance: Literal["high", "medium", "low"]
    reasoning: str
    token_usage: TokenUsage


class ArticleTitle(BaseModel):
    article_id: str = Field(..., description="Article identifier, e.g. 'art_001'")
    title: str = Field(..., description="Article title used for similarity comparison")


class ClusterArticlesParam(BaseModel):
    articles: list[ArticleTitle]


class ClusterArticlesResponse(BaseModel):
    clusters: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Groups of article_ids that describe the same story. "
            "Each inner list contains 2+ article_ids. "
            "Empty if no duplicates were found."
        ),
    )
    token_usage: TokenUsage


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


# Structured Output LLM
class ArticleClassification(BaseModel):
    relevance: Literal["high", "medium", "low"] = Field(
        ...,
        description="How relevant the article is to the requested topic.",
    )
    summary: str = Field(
        ...,
        max_length=280,
        description=(
            "Concise summary of the article in 1-2 sentences. Hard cap: 280 characters total."
        ),
    )
    reason: str = Field(
        ...,
        description="Brief explanation for the relevance rating.",
    )


class ClusterClassification(BaseModel):
    """Structured output for the inner LLM call that groups duplicate stories."""

    clusters: list[list[str]] = Field(
        ...,
        description=(
            "Groups of article_ids that describe the SAME specific news story "
            "(same event/announcement, not just the same general topic). "
            "Each inner list must contain 2 or more article_ids. "
            "Articles that have no duplicates must NOT appear in the output. "
            "Return an empty list if no duplicates exist."
        ),
    )
