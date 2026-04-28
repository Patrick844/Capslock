import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from bs4 import BeautifulSoup

import tracker.tools as tools
from tracker.schemas import ArticleTitle, Index, IndexArticles, Source


def make_source(
    article_id: str,
    published_at: datetime | None = None,
) -> Source:
    return Source(
        article_id=article_id,
        title=f"Title {article_id}",
        url=f"https://example.com/{article_id}",
        published_at=published_at or datetime(2026, 3, 15, 9, 0, tzinfo=UTC),
    )


def make_index_article(
    article_id: str,
    topic: str = "llm apis",
    published_at: datetime | None = None,
) -> IndexArticles:
    return IndexArticles(
        article_id=article_id,
        title=f"Title {article_id}",
        url=f"https://example.com/{article_id}",
        published_at=published_at or datetime(2026, 3, 15, 9, 0, tzinfo=UTC),
        topics=[topic],
        source="example.com",
    )


def make_index() -> Index:
    return Index(
        version="1.0",
        generated_at="2026-04-16T00:00:00Z",
        articles=[
            make_index_article("art_001", topic="voice ai"),
            make_index_article("art_002", topic="llm apis"),
            make_index_article(
                "art_003",
                topic="llm apis",
                published_at=datetime(2026, 2, 1, 9, 0, tzinfo=UTC),
            ),
        ],
        topic_index={
            "voice ai": ["art_001"],
            "llm apis": ["art_002", "art_003"],
        },
    )


def write_test_fixtures(tmp_path: Path) -> Path:
    fixtures_dir = tmp_path / "fixtures"
    articles_dir = fixtures_dir / "articles"
    articles_dir.mkdir(parents=True)

    index = make_index()

    (fixtures_dir / "search_index.json").write_text(
        json.dumps(index.model_dump(mode="json")),
        encoding="utf-8",
    )

    html = """
    <html>
      <head>
        <title>Test Article Title</title>
        <meta name="published" content="2026-03-15T09:00:00Z">
        <meta name="tags" content="llm, api, pricing">
      </head>
      <body>
        <article>
          <h1>Test Article Title</h1>
          <p>This is the article body.</p>
        </article>
      </body>
    </html>
    """

    (articles_dir / "art_002.html").write_text(html, encoding="utf-8")

    return fixtures_dir


def test_filter_article_by_topic_returns_matching_articles() -> None:
    index = make_index()

    articles = tools._filter_article_by_topic(
        topic="llm apis",
        news_index=index,
    )

    article_ids = [article.article_id for article in articles]

    assert article_ids == ["art_002", "art_003"]


def test_filter_article_by_topic_returns_empty_list_for_unknown_topic() -> None:
    index = make_index()

    articles = tools._filter_article_by_topic(
        topic="unknown topic",
        news_index=index,
    )

    assert articles == []


def test_filter_article_by_date_removes_old_articles() -> None:
    articles = [
        make_source(
            "art_001",
            published_at=datetime(2026, 3, 15, 9, 0, tzinfo=UTC),
        ),
        make_source(
            "art_002",
            published_at=datetime(2026, 2, 1, 9, 0, tzinfo=UTC),
        ),
    ]

    filtered = tools._filter_article_by_date(
        since=date(2026, 3, 1),
        selected_articles=articles,
    )

    assert len(filtered) == 1
    assert filtered[0].article_id == "art_001"


def test_search_news_reads_fixture_and_filters_by_topic_and_date(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    results = tools.search_news(
        query="LLM APIs",
        since=date(2026, 3, 1),
        limit=20,
    )

    assert len(results) == 1
    assert results[0].article_id == "art_002"
    assert results[0].published_at.date() >= date(2026, 3, 1)


def test_search_news_respects_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    results = tools.search_news(
        query="llm apis",
        since=date(2026, 1, 1),
        limit=1,
    )

    assert len(results) == 1


def test_search_news_returns_empty_list_when_topic_not_found(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    results = tools.search_news(
        query="unknown topic",
        since=date(2026, 3, 1),
        limit=20,
    )

    assert results == []


def test_lookup_in_index_returns_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    source = tools._lookup_in_index("art_002")

    assert source is not None
    assert source.article_id == "art_002"
    assert source.url == "https://example.com/art_002"
    assert source.title == "Title art_002"


def test_lookup_in_index_returns_none_for_unknown_article(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    source = tools._lookup_in_index("art_999")

    assert source is None


def test_fetch_article_reads_html_and_returns_article(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    article = tools.fetch_article("art_002")

    assert article.article_id == "art_002"
    assert article.title == "Test Article Title"
    assert article.published_at.date() == date(2026, 3, 15)
    assert article.url == "https://example.com/art_002"
    assert "This is the article body." in article.body
    assert article.tags == ["llm", "api", "pricing"]


def test_fetch_article_raises_when_title_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    bad_html_path = fixtures_dir / "articles" / "art_002.html"
    bad_html_path.write_text(
        """
        <html>
          <head>
            <meta name="published" content="2026-03-15T09:00:00Z">
          </head>
          <body>
            <article>Body</article>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    with pytest.raises(ValueError, match="Missing <title> tag"):
        tools.fetch_article("art_002")


def test_fetch_article_raises_when_published_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    bad_html_path = fixtures_dir / "articles" / "art_002.html"
    bad_html_path.write_text(
        """
        <html>
          <head>
            <title>Bad Article</title>
          </head>
          <body>
            <article>Body</article>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    with pytest.raises(ValueError, match='Missing <meta name="published">'):
        tools.fetch_article("art_002")


def test_fetch_article_raises_when_body_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    bad_html_path = fixtures_dir / "articles" / "art_002.html"
    bad_html_path.write_text(
        """
        <html>
          <head>
            <title>Bad Article</title>
            <meta name="published" content="2026-03-15T09:00:00Z">
          </head>
          <body>
            <section>No article or main tag</section>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    with pytest.raises(ValueError, match="Missing <article> or <main> tag"):
        tools.fetch_article("art_002")


def test_fetch_article_allows_missing_tags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures_dir = write_test_fixtures(tmp_path)

    html_path = fixtures_dir / "articles" / "art_002.html"
    html_path.write_text(
        """
        <html>
          <head>
            <title>No Tags Article</title>
            <meta name="published" content="2026-03-15T09:00:00Z">
          </head>
          <body>
            <article>Body without tags</article>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(tools, "_FIXTURES_DIR", fixtures_dir)

    article = tools.fetch_article("art_002")

    assert article.tags == []


def test_get_meta_content_returns_content() -> None:
    soup = BeautifulSoup(
        '<meta name="published" content="2026-03-15T09:00:00Z">',
        "html.parser",
    )

    value = tools._get_meta_content(soup, "published")

    assert value == "2026-03-15T09:00:00Z"


def test_get_meta_content_optional_missing_returns_empty_string() -> None:
    soup = BeautifulSoup("<html></html>", "html.parser")

    value = tools._get_meta_content(soup, "tags", required=False)

    assert value == ""


def test_get_meta_content_required_missing_raises_value_error() -> None:
    soup = BeautifulSoup("<html></html>", "html.parser")

    with pytest.raises(ValueError, match='Missing <meta name="published">'):
        tools._get_meta_content(soup, "published")


def test_get_meta_content_required_missing_content_raises_value_error() -> None:
    soup = BeautifulSoup(
        '<meta name="published">',
        "html.parser",
    )

    with pytest.raises(ValueError, match='Missing content for <meta name="published">'):
        tools._get_meta_content(soup, "published")


@pytest.mark.asyncio
async def test_summarize_content_returns_summarizer_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_parsed = SimpleNamespace(
        summary="This article is about LLM API pricing.",
        relevance="high",
        reason="The content directly discusses the requested topic.",
    )

    fake_response = SimpleNamespace(
        output_parsed=fake_parsed,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
    )

    async def fake_parse(**kwargs: Any) -> SimpleNamespace:
        return fake_response

    monkeypatch.setattr(tools.client.responses, "parse", fake_parse)

    result = await tools.summarize_content(
        content="The article discusses LLM API pricing changes.",
        topic="llm apis",
    )

    assert result.summary == "This article is about LLM API pricing."
    assert result.relevance == "high"
    assert result.reasoning == "The content directly discusses the requested topic."
    assert result.token_usage.input_tokens == 100
    assert result.token_usage.output_tokens == 50
    assert result.token_usage.total_tokens == 150


@pytest.mark.asyncio
async def test_summarize_content_raises_when_no_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = SimpleNamespace(
        output_parsed=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        ),
    )

    async def fake_parse(**kwargs: Any) -> SimpleNamespace:
        return fake_response

    monkeypatch.setattr(tools.client.responses, "parse", fake_parse)

    with pytest.raises(ValueError, match="LLM returned no structured output"):
        await tools.summarize_content(
            content="Some article content.",
            topic="llm apis",
        )


@pytest.mark.asyncio
async def test_cluster_articles_filters_singletons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_parsed = SimpleNamespace(
        clusters=[
            ["art_014", "art_015"],
            ["art_017"],
            [],
        ]
    )

    fake_response = SimpleNamespace(
        output_parsed=fake_parsed,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        ),
    )

    async def fake_parse(**kwargs: Any) -> SimpleNamespace:
        return fake_response

    monkeypatch.setattr(tools.client.responses, "parse", fake_parse)

    result = await tools.cluster_articles(
        [
            ArticleTitle(article_id="art_014", title="OpenAI announces GPT-5 pricing"),
            ArticleTitle(article_id="art_015", title="GPT-5 API prices revised"),
            ArticleTitle(article_id="art_017", title="Gemini 3 reaches GA"),
        ]
    )

    assert result.clusters == [["art_014", "art_015"]]
    assert result.token_usage.input_tokens == 100
    assert result.token_usage.output_tokens == 20
    assert result.token_usage.total_tokens == 120


@pytest.mark.asyncio
async def test_cluster_articles_raises_when_no_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_response = SimpleNamespace(
        output_parsed=None,
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
        ),
    )

    async def fake_parse(**kwargs: Any) -> SimpleNamespace:
        return fake_response

    monkeypatch.setattr(tools.client.responses, "parse", fake_parse)

    with pytest.raises(ValueError, match="LLM returned no structured output"):
        await tools.cluster_articles(
            [
                ArticleTitle(article_id="art_001", title="Article one"),
                ArticleTitle(article_id="art_002", title="Article two"),
            ]
        )
