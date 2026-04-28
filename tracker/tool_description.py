SEARCH_NEWS_TOOL = {
    "type": "function",
    "name": "search_news",
    "description": (
        "Search articles by topic and date. "
        "Returns article IDs and basic article metadata. "
        "Use this when the user asks for articles about a topic."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic to search for, for example 'llm apis' or 'voice ai'.",
            },
            "since": {
                "type": "string",
                "description": (
                    "Only return articles published on or after this date. Format: YYYY-MM-DD."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of articles to return. Use 20 if not specified.",
            },
        },
        "required": ["query", "since", "limit"],
        "additionalProperties": False,
    },
    "strict": True,
}

FETCH_ARTICLE_TOOL = {
    "type": "function",
    "name": "fetch_article",
    "description": (
        "Fetch the full article content by article ID. "
        "Use this when you already have an article_id and need the article body/content."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "article_id": {
                "type": "string",
                "description": "Article ID to fetch, for example 'art_001'.",
            }
        },
        "required": ["article_id"],
        "additionalProperties": False,
    },
    "strict": True,
}

SUMMARIZE_ARTICLE_TOOL = {
    "type": "function",
    "name": "summarize_article",
    "description": (
        "REQUIRED: call this tool every time you receive an article body to summarize. "
        "Pass the full article content and the topic being tracked. "
        "Returns a relevance label (high/medium/low) and a concise summary focused on "
        "the parts of the content relevant to that topic. "
        "Never skip this step — every fetched article must be summarized."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The full article content/body to summarize.",
            },
            "topic": {
                "type": "string",
                "description": (
                    "The topic or user interest to focus the summary on, "
                    "for example 'llm apis', 'voice ai', or 'agent frameworks'."
                ),
            },
        },
        "required": ["content", "topic"],
        "additionalProperties": False,
    },
    "strict": True,
}

CLUSTER_ARTICLES_TOOL = {
    "type": "function",
    "name": "cluster_articles",
    "description": (
        "Group articles that describe the SAME specific news story by comparing their titles. "
        "Use this AFTER fetching all candidate articles and BEFORE summarizing them, so that "
        "duplicate coverage of one event (e.g. three outlets reporting the same product launch) "
        "can be collapsed into a single digest item with multiple sources. "
        "\n\n"
        "Returns a list of clusters; each cluster is a list of 2+ article_ids that cover "
        "the same story. "
        "Returns an EMPTY list if no articles describe the same story. "
        "Articles that are unique (no duplicate coverage) are NOT included in the output — "
        "only multi-article clusters are returned. "
        "\n\n"
        "Two articles describe the same story only if they cover the same specific event, "
        "announcement, or development — same subject AND same news. "
        "Different articles about the same general topic are NOT the same story."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "articles": {
                "type": "array",
                "description": "List of {article_id, title} pairs to cluster.",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_id": {
                            "type": "string",
                            "description": "Article identifier, e.g. 'art_001'.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Article title used for similarity matching.",
                        },
                    },
                    "required": ["article_id", "title"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["articles"],
        "additionalProperties": False,
    },
    "strict": True,
}
