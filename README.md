# CapsLock AI Industry Mini-Tracker

An agentic news digester that turns a topic and a `since` date into a ranked, cited JSON digest using LLM tool-calling over local fixtures.

## Setup

Requires Python 3.11+. Using `uv` is recommended, but plain `pip` works fine.

```bash
# with uv
uv sync

# or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and fill in your LLM provider key.

## Run

**CLI:**

```bash
python -m tracker --topic "Voice AI" --since 2026-03-01
```

**API:**

```bash
uvicorn tracker.api:app --reload
curl -X POST http://localhost:8000/digest \
  -H 'content-type: application/json' \
  -d '{"topic":"LLM APIs","since":"2026-03-01","max_items":5}'
```

## Evals

```bash
python evals/run.py
```

Prints a pass/fail report across the gold cases in `evals/gold.json`.

## Tests

This project uses `pytest` for testing and `pytest-cov` for coverage reports.

```bash
# install dev dependencies
uv add --dev pytest pytest-cov pytest-asyncio

# run tests
uv run pytest

# run with coverage
uv run pytest --cov=tracker --cov-report=term-missing

# generate HTML coverage report
uv run pytest --cov=tracker --cov-report=term-missing --cov-report=html
open htmlcov/index.html
```

Tests do **not** call the real OpenAI API. LLM responses are mocked to keep tests fast, deterministic, and safe to run in CI.

**What is tested:**

- Local fixture search logic (topic filtering, date filtering, limit clamping)
- Article HTML parsing and metadata extraction
- Token usage aggregation
- Agent helper functions (`_extract_tool_calls`, `_build_digest_item`, `_merge_items_by_cluster`)
- Async summarization and clustering via mocked LLM responses

## Type Checking

```bash
uv run pyright
```

## Docker

The project includes a `Dockerfile` for running the API inside a lightweight Python container.

The Docker image:

- Uses `python:3.11-slim`
- Sets `/app` as the working directory
- Installs the project from `pyproject.toml`
- Copies the `tracker` source and `fixtures` folder into the image
- Exposes port `8000` and starts the API with Uvicorn

```bash
docker build -t capslock-tracker .
docker run -e OPENAI_API_KEY=sk-... -p 8000:8000 capslock-tracker
```

## Docker Compose

```bash
docker compose up --build
```

Docker Compose:

- Builds the API image from the local `Dockerfile`
- Runs the container as `capslock-tracker-api`
- Maps local port `8000` to container port `8000`
- Passes `OPENAI_API_KEY`, `LLM_MODEL`, and `LLM_PROVIDER` as environment variables
- Mounts the local `fixtures` folder as read-only so fixtures can be updated without rebuilding:

```yaml
volumes:
  - ./fixtures:/app/fixtures:ro
```

The API will be available at `http://localhost:8000`. Stop with `docker compose down`.

