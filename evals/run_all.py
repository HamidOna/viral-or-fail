"""Single entrypoint for the Viral or Fail eval suite.

Run with:
    python -m evals.run_all

Runs all four tests sequentially, surfaces each headline metric, and
writes JSON results + PNG plots into eval_results/.

Each test file is also runnable in isolation, e.g.:
    python -m evals.test_2_calibration
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv

from evals import test_1_consistency, test_2_calibration, test_3_rubric_adherence, test_4_workflow
from evals.client_factory import active_provider

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

TESTS = [
    ("Test 1 — Consistency", test_1_consistency.run_test_1),
    ("Test 2 — Calibration", test_2_calibration.run_test_2),
    ("Test 3 — Rubric Adherence", test_3_rubric_adherence.run_test_3),
    ("Test 4 — Per-agent workflow", test_4_workflow.run_test_4),
]


def _banner() -> None:
    print()
    print("=" * 70)
    print(" VIRAL OR FAIL — EVALUATION SUITE")
    print(" Evaluating the Algorithm Simulator (free tier, GitHub Models)")
    print("=" * 70)


async def main() -> int:
    provider = active_provider()
    if provider == "none":
        print("ERROR: No API credentials set.")
        print("Set GITHUB_TOKEN (free, GitHub Models) or OPENAI_API_KEY (paid, OpenAI direct)")
        print("in your .env file. See .env.example.")
        return 1

    _banner()
    print(f" Provider: {provider}")

    headlines: list[tuple[str, str, str]] = []  # (name, status, headline)
    started = time.time()

    for name, runner in TESTS:
        print()
        print(f"\n>>> {name}")
        t0 = time.time()
        try:
            summary = await runner(verbose=True)
            elapsed = time.time() - t0
            headlines.append((name, f"OK ({elapsed:.1f}s)", summary.get("headline", "(no headline)")))
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - t0
            headlines.append((name, f"FAILED ({elapsed:.1f}s)", str(exc)))
            print(f"\n  !!! {name} failed: {exc}")
            traceback.print_exc()

    total_elapsed = time.time() - started

    print()
    print("=" * 70)
    print(" SUITE SUMMARY")
    print("=" * 70)
    for name, status, headline in headlines:
        print(f"  {name:<32}  {status}")
        print(f"      {headline}")
    print()
    print(f"  Total wall time: {total_elapsed:.1f}s")
    print()
    print("  Outputs:")
    print(f"    {(REPO_ROOT / 'eval_results').relative_to(REPO_ROOT)}/*.json")
    print(f"    {(REPO_ROOT / 'eval_results' / 'plots').relative_to(REPO_ROOT)}/*.png")

    failed = sum(1 for _, status, _ in headlines if status.startswith("FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
