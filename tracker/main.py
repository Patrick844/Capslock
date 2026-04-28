"""CLI entry point: `python -m tracker --topic "Voice AI" --since 2026-03-01`."""

import argparse
import asyncio
from datetime import date

from tracker.agent import run_digest
from tracker.schemas import DigestRequest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tracker", description="AI Industry Mini-Tracker")
    parser.add_argument("--topic", required=True, help="Topic to scan for, e.g., 'Voice AI'")
    parser.add_argument("--since", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--max-items", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request = DigestRequest(topic=args.topic, since=args.since, max_items=args.max_items)
    response = asyncio.run(run_digest(request))
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
