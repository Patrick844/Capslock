"""Evaluation harness for the AI Industry Mini-Tracker agent.

Loads `evals/gold.json`, runs the agent against each case, and reports
pass/fail for four checks:

  1. schema   — the agent returned a valid DigestResponse (no crash).
  2. includes — every id in `must_include_article_ids` appears as a source
                somewhere in the response.
  3. excludes — no id in `must_exclude_article_ids` appears as a source.
  4. cluster  — every id in `must_collapse_cluster` appears in the SAME
                DigestItem (i.e. duplicates were collapsed into one item
                with multiple sources).

Usage:
    python evals/run.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Make `tracker` importable when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tracker.agent import run_digest  # noqa: E402
from tracker.schemas import DigestRequest, DigestResponse  # noqa: E402

GOLD_PATH = Path(__file__).parent / "gold.json"

CheckResult = tuple[bool, str]


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _all_source_ids(response: DigestResponse) -> set[str]:
    """All article IDs cited anywhere in the response."""
    return {src.article_id for item in response.items for src in item.sources}


def _check_schema(response: DigestResponse | None) -> CheckResult:
    """Pass if the agent returned a valid DigestResponse object."""
    if response is None:
        return False, "agent did not return a DigestResponse"
    return True, f"valid DigestResponse with {len(response.items)} item(s)"


def _check_includes(response: DigestResponse, must_include: list[str]) -> CheckResult:
    """Every id in must_include must appear as a source somewhere."""
    if not must_include:
        return True, "nothing required"
    found = _all_source_ids(response)
    missing = [aid for aid in must_include if aid not in found]
    if missing:
        return False, f"missing required ids: {missing}"
    return True, f"all {len(must_include)} required ids present"


def _check_excludes(response: DigestResponse, must_exclude: list[str]) -> CheckResult:
    """No id in must_exclude may appear as a source."""
    if not must_exclude:
        return True, "nothing forbidden"
    found = _all_source_ids(response)
    leaked = [aid for aid in must_exclude if aid in found]
    if leaked:
        return False, f"forbidden ids present: {leaked}"
    return True, f"all {len(must_exclude)} forbidden ids absent"


def _check_cluster(response: DigestResponse, must_collapse: list[str]) -> CheckResult:
    """All ids in must_collapse must be sources of ONE single DigestItem."""
    if not must_collapse:
        return True, "no cluster expected"

    target = set(must_collapse)
    for item in response.items:
        item_ids = {src.article_id for src in item.sources}
        if target.issubset(item_ids):
            return True, f"collapsed into one item with {len(item.sources)} sources"

    # Not collapsed — report where each id ended up for debugging.
    locations: dict[str, list[int]] = {aid: [] for aid in must_collapse}
    for idx, item in enumerate(response.items):
        for src in item.sources:
            if src.article_id in locations:
                locations[src.article_id].append(idx)
    return False, f"not collapsed; item-index per id: {locations}"


# ---------------------------------------------------------------------------
# Per-case runner
# ---------------------------------------------------------------------------

async def _run_case(case: dict) -> dict:
    """Run one gold case and return a dict of check results."""
    name = case.get("name", "<unnamed>")
    print(f"\n{'#' * 70}")
    print(f"# Case: {name}")
    print(f"# Notes: {case.get('notes', '')}")
    print(f"{'#' * 70}")

    request = DigestRequest.model_validate(case["request"])

    try:
        response: DigestResponse | None = await run_digest(request)
    except Exception as exc:
        print(f"  [CRASH] Agent raised: {type(exc).__name__}: {exc}")
        # All checks fail when the agent crashes.
        return {
            "name": name,
            "schema": (False, f"agent crashed: {type(exc).__name__}: {exc}"),
            "includes": (False, "n/a (agent crashed)"),
            "excludes": (False, "n/a (agent crashed)"),
            "cluster": (False, "n/a (agent crashed)"),
        }

    return {
        "name": name,
        "schema": _check_schema(response),
        "includes": _check_includes(response, case.get("must_include_article_ids", [])),
        "excludes": _check_excludes(response, case.get("must_exclude_article_ids", [])),
        "cluster": _check_cluster(response, case.get("must_collapse_cluster", [])),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

CHECK_NAMES = ("schema", "includes", "excludes", "cluster")


def _print_report(results: list[dict]) -> int:
    """Print a per-case + overall summary. Returns the failure count."""
    failed_cases = 0
    print(f"\n{'=' * 70}")
    print("EVAL REPORT")
    print(f"{'=' * 70}")

    for r in results:
        case_failed = False
        print(f"\n[{r['name']}]")
        for check in CHECK_NAMES:
            ok, msg = r[check]
            mark = "PASS" if ok else "FAIL"
            print(f"  {check:9s} {mark}  {msg}")
            if not ok:
                case_failed = True
        if case_failed:
            failed_cases += 1

    total = len(results)
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total - failed_cases}/{total} cases fully passed")
    print(f"{'=' * 70}\n")
    return failed_cases


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> int:
    if not GOLD_PATH.exists():
        print(f"[ERROR] gold.json not found at {GOLD_PATH}")
        return 2

    with GOLD_PATH.open("r", encoding="utf-8") as f:
        gold = json.load(f)

    cases: list[dict] = gold.get("cases", [])
    if not cases:
        print("[ERROR] gold.json has no cases")
        return 2

    print(f"Running {len(cases)} eval case(s) from {GOLD_PATH}")

    # Run sequentially so logs stay readable. Switch to asyncio.gather
    # if you want parallel cases (and don't mind interleaved prints).
    results: list[dict] = []
    for case in cases:
        results.append(await _run_case(case))

    failures = _print_report(results)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
