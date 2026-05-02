"""Test 2 — Calibration.

Question: do the Simulator's scores correlate with real-world engagement?

Method: run the Simulator once per post on the full 10-item golden
dataset, then compute Pearson and Spearman correlations plus mean
absolute error against the labeled ``engagement_score``.

Outputs:
- eval_results/test_2_calibration.json
- eval_results/plots/02_calibration.png
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

from agents.algorithm import create_algorithm_simulator_agent
from config.platform_rules import PLATFORM_RULES
from evals.client_factory import get_chat_client
from evals.custom_evaluators import correlates_with_truth, parse_weighted_total
from evals.harness import EvalRunner
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
RESULTS_PATH = REPO_ROOT / "eval_results" / "test_2_calibration.json"
PLOT_PATH = REPO_ROOT / "eval_results" / "plots" / "02_calibration.png"


def _build_simulator_prompt(post: dict) -> str:
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


async def run_test_2(*, verbose: bool = True) -> dict:
    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        posts = json.load(fh)["posts"]

    client = get_chat_client()
    agent = create_algorithm_simulator_agent(client)

    queries = [_build_simulator_prompt(p) for p in posts]
    expected = [str(p["engagement_score"]) for p in posts]
    query_to_post = dict(zip(queries, posts))

    runner = EvalRunner(rate_limit_sleep=4.5)  # 12 RPM, under GitHub Models' 15 RPM cap

    def _on_progress(done: int, total: int, _preview: str) -> None:
        if verbose:
            print(f"  [{done:>3}/{total}] simulator runs", end="\r", flush=True)

    results = await runner.run(
        agent=agent,
        queries=queries,
        evaluators=[correlates_with_truth],
        expected_output=expected,
        progress=_on_progress,
    )
    if verbose:
        print()

    rows: list[dict] = []
    for item in results.items:
        post = query_to_post[item.query]
        sim = parse_weighted_total(item.response)
        truth = float(post["engagement_score"])
        rows.append(
            {
                "post_id": post["id"],
                "platform": post["platform"],
                "label": post["label"],
                "topic": post["topic"],
                "engagement_score": truth,
                "simulator_score": sim,
                "abs_error": abs((sim or 0.0) - truth),
                "alignment": item.scores.get("correlates_with_truth", 0.0),
                "notes": post["notes"],
            }
        )

    valid = [r for r in rows if r["simulator_score"] is not None]
    truths = [r["engagement_score"] for r in valid]
    sims = [r["simulator_score"] for r in valid]

    if len(valid) >= 2:
        pearson_r, pearson_p = pearsonr(truths, sims)
        spearman_r, spearman_p = spearmanr(truths, sims)
        mae = sum(r["abs_error"] for r in valid) / len(valid)
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")
        mae = float("nan")

    summary = {
        "test": "calibration",
        "num_posts": len(posts),
        "valid_runs": len(valid),
        "pearson_r": round(pearson_r, 4),
        "pearson_p": round(pearson_p, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": round(spearman_p, 4),
        "mae": round(mae, 2),
        "headline": (
            f"Pearson r between Simulator score and real engagement: r = {pearson_r:.2f} "
            f"(MAE {mae:.1f} pts)"
        ),
        "posts": rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _plot(rows, pearson_r, spearman_r, mae)

    if verbose:
        print(f"  Pearson r = {pearson_r:.2f}, Spearman ρ = {spearman_r:.2f}, MAE = {mae:.1f}")
        for r in rows:
            sim_disp = f"{r['simulator_score']:.0f}" if r["simulator_score"] is not None else "??"
            print(
                f"  {r['post_id']:<10} {r['label']:<8} "
                f"truth={r['engagement_score']:>3.0f}  sim={sim_disp:>3}  "
                f"err={r['abs_error']:>4.1f}"
            )

    return summary


def _plot(rows: list[dict], pearson_r: float, spearman_r: float, mae: float) -> None:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    # Reference y=x line
    ax.plot([0, 100], [0, 100], color=PALETTE["reference"], linestyle="--", alpha=0.6, label="y = x")

    seen_labels: set[str] = set()
    for row in rows:
        if row["simulator_score"] is None:
            continue
        label = row["label"]
        legend_label = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.scatter(
            row["engagement_score"],
            row["simulator_score"],
            color=color_for_label(label),
            s=110,
            edgecolor=PALETTE["text"],
            linewidth=0.8,
            zorder=4,
            label=legend_label,
        )
        ax.annotate(
            row["post_id"].replace("post_", "#"),
            (row["engagement_score"], row["simulator_score"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=PALETTE["text"],
            alpha=0.85,
        )

    # Callout for the outlier post — positioned below-right of the data point
    # so it doesn't collide with the headline annotation in the upper-left.
    outliers = [r for r in rows if r["label"] == "outlier" and r["simulator_score"] is not None]
    if outliers:
        o = outliers[0]
        callout_x = min(o["engagement_score"] + 18, 78)
        callout_y = max(o["simulator_score"] - 28, 8)
        ax.annotate(
            f"{o['post_id']}: ratio'd",
            xy=(o["engagement_score"], o["simulator_score"]),
            xytext=(callout_x, callout_y),
            fontsize=10,
            color=PALETTE["outlier"],
            arrowprops={
                "arrowstyle": "->",
                "color": PALETTE["outlier"],
                "alpha": 0.7,
            },
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#1E1E1E",
                "edgecolor": PALETTE["outlier"],
                "alpha": 0.9,
            },
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Real engagement score (labeled, 0–100)")
    ax.set_ylabel("Simulator predicted score (0–100)")
    ax.set_title("Test 2 — Algorithm Simulator vs. labeled engagement")
    ax.grid(linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)

    annotate_metric(
        ax,
        f"Pearson r = {pearson_r:.2f}\nSpearman ρ = {spearman_r:.2f}\nMAE = {mae:.1f} pts",
        loc="upper left",
    )

    save_plot(fig, PLOT_PATH)


async def main() -> None:
    print("=" * 70)
    print(" TEST 2 — CALIBRATION")
    print("=" * 70)
    summary = await run_test_2()
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
