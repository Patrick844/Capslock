FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
# Install dependencies first for better layer caching.
COPY pyproject.toml ./
COPY tracker ./tracker
RUN pip install --no-cache-dir .

# Copy fixtures (read at runtime by search_news / fetch_article).
COPY fixtures ./fixtures

EXPOSE 8000

# Use 0.0.0.0 so the container is reachable from the host.
CMD ["uvicorn", "tracker.api:app", "--host", "0.0.0.0", "--port", "8000"]
