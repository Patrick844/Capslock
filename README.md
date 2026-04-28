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

CLI:

```bash
python -m tracker --topic "Voice AI" --since 2026-03-01
```

API:

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

## Docker

The project includes a `Dockerfile` for running the API inside a lightweight Python container.

The Docker image:

- Uses `python:3.11-slim`
- Sets `/app` as the working directory
- Installs the project from `pyproject.toml`
- Copies the `tracker` source code into the image
- Copies the `fixtures` folder used by `search_news` and `fetch_article`
- Exposes port `8000`
- Starts the FastAPI API with Uvicorn

The container runs:

```bash
uvicorn tracker.api:app --host 0.0.0.0 --port 8000
```

`0.0.0.0` is used so the API is reachable from the host machine.

## Docker Compose

The project also includes a `docker-compose.yml` file to make running the API easier.

Docker Compose:

- Builds the API image from the local `Dockerfile`
- Runs the container as `capslock-tracker-api`
- Maps local port `8000` to container port `8000`
- Passes environment variables such as `OPENAI_API_KEY`, `LLM_MODEL`, and `LLM_PROVIDER`
- Mounts the local `fixtures` folder into the container as read-only

The fixtures mount is useful during development:

```yaml
volumes:
  - ./fixtures:/app/fixtures:ro
```

This means fixture files can be updated locally without rebuilding the Docker image.

### Run with Docker Compose

```bash
docker compose up --build
```

The API will be available at:

```text
http://localhost:8000
```

### Stop the container

```bash
docker compose down
```

## Tests

This project uses `pytest` for testing and `pytest-cov` for coverage reports.

### Install test dependencies

```bash
uv add --dev pytest pytest-cov pytest-asyncio
```

### Run tests

```bash
uv run pytest
```

### Run tests with coverage

```bash
uv run pytest --cov=tracker --cov-report=term-missing
```

### Generate an HTML coverage report

```bash
uv run pytest --cov=tracker --cov-report=term-missing --cov-report=html
```

Then open the report:

```bash
open htmlcov/index.html
```

### What is tested

The test suite covers:

- Local fixture search logic
- Article HTML parsing
- Metadata extraction
- Token usage aggregation
- Agent helper functions
- Async summarization and clustering logic using mocked LLM responses

Tests do **not** call the real OpenAI API. LLM responses are mocked to keep tests fast, deterministic, and safe to run locally or in CI.

### Type checking

Run Pyright with:

```bash
uv run pyright
```

Prints a pass/fail summary across the gold cases in `evals/gold.json`.

## Notes from the Candidate

_Fill this section in with any notable design decisions, trade-offs, or things you'd do differently with more time._
