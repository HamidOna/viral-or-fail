"""Test 1 — Consistency.

Question: does the Algorithm Simulator give the same score to the same
post across multiple runs?

Method: pick 5 posts spanning viral/decent/flop, call the Simulator
10 times each with identical input, then summarise the score distribution
per post (mean, std, min, max, coefficient of variation).

Outputs:
- eval_results/test_1_consistency.json
- eval_results/plots/01_consistency.png
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from agents.algorithm import create_algorithm_simulator_agent
from config.platform_rules import PLATFORM_RULES
from evals.client_factory import get_chat_client
from evals.custom_evaluators import parse_weighted_total
from evals.harness import EvalRunner, evaluator
from evals.plot_style import (
    DPI,
    FIG_SIZE,
    PALETTE,
    annotate_metric,
    apply_dark_theme,
    color_for_label,
    save_plot,
)

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "evals" / "golden_dataset.json"
RESULTS_PATH = REPO_ROOT / "eval_results" / "test_1_consistency.json"
PLOT_PATH = REPO_ROOT / "eval_results" / "plots" / "01_consistency.png"

# 5 posts spanning the score range — Test 1 doesn't need all 10
SELECTED_POST_IDS = ["post_001", "post_002", "post_003", "post_007", "post_010"]
NUM_REPETITIONS = 10


@evaluator
def weighted_total_score(response: str) -> float:
    """Return the parsed weighted total (or 0 if unparseable). 0–100 scale."""
    score = parse_weighted_total(response)
    return score if score is not None else 0.0


def _build_simulator_prompt(post: dict) -> str:
    """Render the same prompt the live game builds, so we test the real path."""
    rubric = PLATFORM_RULES[post["platform"]]
    rubric_lines = [
        f"Platform: {post['platform']}",
        f"Description: {rubric['description']}",
        "",
        "Scoring Criteria (use these exact weights):",
    ]
    for name, info in rubric["criteria"].items():
        rubric_lines.append(f"- {name} ({int(info['weight'] * 100)}%): {info['description']}")
    rubric_text = "\n".join(rubric_lines)

    return (
        f"Evaluate this {post['platform']} post about '{post['topic']}' using the platform's "
        f"scoring rubric.\n\n"
        f"--- SCORING RUBRIC ---\n{rubric_text}\n\n"
        f"--- CONTENT TO EVALUATE ---\n{post['content']}\n\n"
        f"Score each criterion out of 100, then calculate the weighted total. "
        f"Be specific and reference platform algorithm mechanics."
    )


async def run_test_1(*, verbose: bool = True) -> dict:
    """Run the consistency test and write its JSON + PNG outputs.

    Returns a dict of summary metrics (mean CV, per-post stats) used by
    run_all.py for the final headline report.
    """
    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        posts_by_id = {p["id"]: p for p in json.load(fh)["posts"]}
    selected = [posts_by_id[pid] for pid in SELECTED_POST_IDS]

    client = get_chat_client()
    agent = create_algorithm_simulator_agent(client)

    # Map each prompt back to its post for downstream stats.
    queries = [_build_simulator_prompt(p) for p in selected]
    query_to_post = dict(zip(queries, selected))

    runner = EvalRunner(rate_limit_sleep=4.5)  # 12 RPM, under GitHub Models' 15 RPM cap

    def _on_progress(done: int, total: int, _preview: str) -> None:
        if verbose:
            print(f"  [{done:>3}/{total}] simulator runs", end="\r", flush=True)

    results = await runner.run(
        agent=agent,
        queries=queries,
        evaluators=[weighted_total_score],
        num_repetitions=NUM_REPETITIONS,
        progress=_on_progress,
    )
    if verbose:
        print()

    # Group scores per post.
    per_post: list[dict] = []
    for query, items in results.by_query().items():
        post = query_to_post[query]
        scores = [it.scores["weighted_total_score"] for it in items]
        valid = [s for s in scores if s > 0]
        if not valid:
            mean = std = cv = 0.0
        else:
            mean = statistics.mean(valid)
            std = statistics.pstdev(valid) if len(valid) > 1 else 0.0
            cv = (std / mean * 100.0) if mean else 0.0
        per_post.append(
            {
                "post_id": post["id"],
                "platform": post["platform"],
                "label": post["label"],
                "topic": post["topic"],
                "scores": scores,
                "mean": round(mean, 2),
                "std": round(std, 2),
                "min": min(valid) if valid else 0.0,
                "max": max(valid) if valid else 0.0,
                "cv_pct": round(cv, 2),
                "valid_runs": len(valid),
            }
        )

    # Preserve dataset order for the plot.
    per_post.sort(key=lambda r: SELECTED_POST_IDS.index(r["post_id"]))
    cvs = [r["cv_pct"] for r in per_post if r["valid_runs"] > 1]
    mean_cv = round(statistics.mean(cvs), 2) if cvs else 0.0

    summary = {
        "test": "consistency",
        "num_posts": len(per_post),
        "num_repetitions": NUM_REPETITIONS,
        "mean_cv_pct": mean_cv,
        "headline": (
            f"Mean coefficient of variation across {len(per_post)} posts "
            f"({NUM_REPETITIONS} reps each): {mean_cv:.2f}%"
        ),
        "per_post": per_post,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _plot(per_post, mean_cv)

    if verbose:
        print(f"  Mean CV across posts: {mean_cv:.2f}%")
        for r in per_post:
            print(
                f"  {r['post_id']:<10} {r['label']:<8} "
                f"mean={r['mean']:>5.1f}  std={r['std']:>4.1f}  CV={r['cv_pct']:>5.2f}%"
            )

    return summary


def _plot(per_post: list[dict], mean_cv: float) -> None:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    labels = [f"{r['post_id']} ({r['label']}, CV {r['cv_pct']:.1f}%)" for r in per_post]
    data = [[s for s in r["scores"] if s > 0] for r in per_post]

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        widths=0.6,
        medianprops={"color": PALETTE["text"], "linewidth": 1.5},
        flierprops={"marker": "o", "markerfacecolor": PALETTE["reference"], "markersize": 4},
    )
    for patch, row in zip(bp["boxes"], per_post):
        patch.set_facecolor(color_for_label(row["label"]))
        patch.set_edgecolor(PALETTE["text"])
        patch.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_color(PALETTE["muted"])
    for cap in bp["caps"]:
        cap.set_color(PALETTE["muted"])

    # Mean reference markers
    means = [r["mean"] for r in per_post]
    ax.scatter(
        means,
        range(1, len(per_post) + 1),
        color=PALETTE["reference"],
        marker="D",
        s=42,
        zorder=5,
        label="mean",
    )

    ax.set_yticks(range(1, len(per_post) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Weighted total score (0–100)")
    ax.set_title("Test 1 — Algorithm Simulator score consistency (10 reps per post)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    ax.legend(loc="lower right", framealpha=0.9)

    annotate_metric(ax, f"Mean CV across posts: {mean_cv:.2f}%", loc="upper right")

    save_plot(fig, PLOT_PATH)


async def main() -> None:
    print("=" * 70)
    print(" TEST 1 — CONSISTENCY")
    print("=" * 70)
    summary = await run_test_1()
    print()
    print(f"  Wrote {RESULTS_PATH.relative_to(REPO_ROOT)}")
    print(f"  Wrote {PLOT_PATH.relative_to(REPO_ROOT)}")
    print()
    print(f"  Headline: {summary['headline']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
