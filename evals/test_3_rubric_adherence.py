"""Test 3 — Rubric Adherence (free-tier LLM-as-judge).

Question: when the Simulator scores a post, does it actually follow its
own rubric — covering each criterion with a defensible score and respecting
the weights — or does it drift?

Method:
1. For each post in the golden dataset, run the Algorithm Simulator once.
2. Hand the (rubric, post, evaluation output) triple to a second
   ``OpenAIChatClient`` agent acting as a Rubric Adherence Judge.
3. The judge returns strict JSON: {adherence_score 1-5, missing_criteria,
   criteria_present, weight_drift, reasoning}.

Notes:
- The cloud-tier equivalent here is roughly ``FoundryEvals.TaskAdherence``.
  We're staying free, so we build the same idea ourselves.

Outputs:
- eval_results/test_3_rubric_adherence.json
- eval_results/plots/03_adherence.png
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

from agents.algorithm import create_algorithm_simulator_agent
from config.platform_rules import PLATFORM_RULES
from evals.client_factory import get_chat_client
from evals.harness import call_agent_with_retry
from evals.llm_judge import RubricAdherenceJudge
from evals.plot_style import (
    DPI,
    FIG_SIZE,
    PALETTE,
    apply_dark_theme,
    save_plot,
)

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "evals" / "golden_dataset.json"
RESULTS_PATH = REPO_ROOT / "eval_results" / "test_3_rubric_adherence.json"
PLOT_PATH = REPO_ROOT / "eval_results" / "plots" / "03_adherence.png"

RATE_LIMIT_SLEEP = 4.5  # 12 RPM, under GitHub Models' 15 RPM cap
FOCUS_PLATFORM = "Twitter/X"  # the per-criterion bar chart focuses on this rubric


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


async def run_test_3(*, verbose: bool = True) -> dict:
    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        posts = json.load(fh)["posts"]

    client = get_chat_client()
    simulator = create_algorithm_simulator_agent(client)
    judge = RubricAdherenceJudge(client)

    rows: list[dict] = []
    total = len(posts)

    for idx, post in enumerate(posts, start=1):
        if verbose:
            print(f"  [{idx:>2}/{total}] {post['id']} ({post['platform']:<10}) ", end="", flush=True)

        # Run Simulator (retry-aware: handles 429s + transient errors)
        sim_text = await call_agent_with_retry(simulator, _build_simulator_prompt(post))
        await asyncio.sleep(RATE_LIMIT_SLEEP)

        # Run Judge
        rubric = PLATFORM_RULES[post["platform"]]
        verdict = await judge.judge(
            rubric=rubric,
            post_content=post["content"],
            evaluation_output=sim_text,
        )
        await asyncio.sleep(RATE_LIMIT_SLEEP)

        rows.append(
            {
                "post_id": post["id"],
                "platform": post["platform"],
                "label": post["label"],
                "topic": post["topic"],
                "adherence_score": verdict.adherence_score,
                "math_diff": verdict.math_diff,
                "reasoning_quality": verdict.reasoning_quality,
                "criteria_present": verdict.criteria_present,
                "missing_criteria": verdict.missing_criteria,
                "weight_drift": verdict.weight_drift,
                "judge_reasoning": verdict.reasoning,
                "rubric_criteria": list(rubric["criteria"].keys()),
            }
        )

        if verbose:
            print(
                f"adherence={verdict.adherence_score}/5  "
                f"math_diff={verdict.math_diff:+.1f}  "
                f"reasoning={verdict.reasoning_quality}  "
                f"missing={len(verdict.missing_criteria)}"
            )

    # ── Aggregate stats ──────────────────────────────────────────────────
    adherence_scores = [r["adherence_score"] for r in rows]
    mean_adherence = sum(adherence_scores) / len(adherence_scores) if adherence_scores else 0.0

    # Most-skipped criterion across all posts
    skip_counter: Counter[str] = Counter()
    for r in rows:
        for crit in r["missing_criteria"]:
            skip_counter[crit] += 1
    most_skipped = skip_counter.most_common(1)[0][0] if skip_counter else "(none)"

    # Per-criterion presence stats for the focus platform
    focus_rows = [r for r in rows if r["platform"] == FOCUS_PLATFORM]
    focus_criteria = list(PLATFORM_RULES[FOCUS_PLATFORM]["criteria"].keys())
    per_criterion_pct: dict[str, float] = {}
    for crit in focus_criteria:
        if not focus_rows:
            per_criterion_pct[crit] = 0.0
            continue
        scored_count = sum(1 for r in focus_rows if crit in r["criteria_present"])
        per_criterion_pct[crit] = round(100.0 * scored_count / len(focus_rows), 1)

    # Math-fidelity and reasoning-quality stats (the new judge dimensions)
    math_diffs = [abs(r["math_diff"]) for r in rows]
    mean_abs_math_diff = sum(math_diffs) / len(math_diffs) if math_diffs else 0.0
    max_abs_math_diff = max(math_diffs) if math_diffs else 0.0
    quality_counts = Counter(r["reasoning_quality"] for r in rows)
    pct_specific = round(100.0 * quality_counts.get("specific", 0) / len(rows), 1) if rows else 0.0

    summary = {
        "test": "rubric_adherence",
        "num_posts": len(rows),
        "mean_adherence": round(mean_adherence, 2),
        "mean_abs_math_diff": round(mean_abs_math_diff, 2),
        "max_abs_math_diff": round(max_abs_math_diff, 2),
        "reasoning_quality_counts": dict(quality_counts),
        "pct_specific_reasoning": pct_specific,
        "most_skipped_criterion": most_skipped,
        "skip_counts": dict(skip_counter),
        "focus_platform": FOCUS_PLATFORM,
        "focus_per_criterion_pct_present": per_criterion_pct,
        "headline": (
            f"Mean adherence: {mean_adherence:.2f}/5. "
            f"Mean |math drift|: {mean_abs_math_diff:.1f} pts (max {max_abs_math_diff:.1f}). "
            f"{pct_specific:.0f}% of evals had specific reasoning. "
            f"Most-skipped: {most_skipped}."
        ),
        "posts": rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _plot(rows, mean_adherence, mean_abs_math_diff)

    if verbose:
        print()
        print(f"  Mean adherence:           {mean_adherence:.2f}/5")
        print(f"  Mean |math drift|:        {mean_abs_math_diff:.1f} pts (max {max_abs_math_diff:.1f})")
        print(f"  Reasoning quality counts: {dict(quality_counts)}")
        print(f"  Most-skipped:             {most_skipped}")
        print(f"  Per-criterion presence on {FOCUS_PLATFORM} ({len(focus_rows)} posts):")
        for crit, pct in per_criterion_pct.items():
            print(f"    {crit:<24} {pct:>5.1f}% explicitly scored")

    return summary


def _plot(
    rows: list[dict],
    mean_adherence: float,
    mean_abs_math_diff: float,
) -> None:
    """Render the Test 3 plot.

    Two panels surface what the tightened judge actually checked:

    - Top: per-post math drift (scatter), with the ±2-pt threshold the judge
      uses to cap the adherence score at 3. Visualises math fidelity — readers
      should see all points hovering near zero, well inside the threshold band.
    - Bottom: count of evaluations by reasoning quality (specific / mixed /
      generic). Visualises whether justifications cited platform mechanics or
      just gave vibes-based praise.

    Together these answer "did the Simulator actually deserve 5/5, or did
    the judge just rubber-stamp it?" — the central question of Test 3.
    """
    apply_dark_theme()
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=FIG_SIZE, dpi=DPI, gridspec_kw={"height_ratios": [3, 1.5]}
    )

    # ── Top: per-post math drift (scatter + threshold lines) ────────────
    post_ids = [r["post_id"] for r in rows]
    math_diffs = [r["math_diff"] for r in rows]
    threshold = 2.0  # judge caps adherence at 3 above this magnitude

    point_colors = [
        PALETTE["viral"] if abs(d) <= threshold else PALETTE["flop"]
        for d in math_diffs
    ]

    # Reference lines first so the points sit on top
    ax_top.axhspan(-threshold, threshold, color=PALETTE["viral"], alpha=0.08, zorder=0)
    ax_top.axhline(0, color=PALETTE["muted"], linewidth=1, alpha=0.7, zorder=1)
    ax_top.axhline(
        threshold,
        color=PALETTE["reference"],
        linestyle="--",
        alpha=0.7,
        label=f"±{threshold:.0f}-pt judge threshold",
        zorder=1,
    )
    ax_top.axhline(-threshold, color=PALETTE["reference"], linestyle="--", alpha=0.7, zorder=1)

    # Vertical "lollipop" lines from 0 to each non-zero drift, so zero values
    # don't look like missing data — they just sit on the baseline as dots.
    for i, d in enumerate(math_diffs):
        if d != 0:
            ax_top.vlines(i, 0, d, color=point_colors[i], linewidth=2, alpha=0.6, zorder=2)

    ax_top.scatter(
        range(len(post_ids)),
        math_diffs,
        s=120,
        c=point_colors,
        edgecolor=PALETTE["text"],
        linewidth=1,
        zorder=4,
    )

    # Annotate each point with its drift value (since 0.0/0.2 are barely visible)
    for i, d in enumerate(math_diffs):
        offset = 0.18 if d >= 0 else -0.18
        ax_top.text(
            i,
            d + offset,
            f"{d:+.1f}",
            ha="center",
            va="bottom" if d >= 0 else "top",
            color=PALETTE["text"],
            fontsize=9,
            alpha=0.9,
        )

    ax_top.set_xticks(range(len(post_ids)))
    ax_top.set_xticklabels([p.replace("post_", "#") for p in post_ids])
    ax_top.set_ylim(-3, 3)
    # Short label avoids overlap with the bottom panel's "Posts" label; the
    # title already explains what math drift means.
    ax_top.set_ylabel("Math drift (points)")
    # Headline metrics live in a subtitle (pad=22 reserves space) instead of a
    # corner annotation, so they can't overlap the ±2 threshold line.
    ax_top.set_title(
        "Test 3 — Did the Simulator earn its 5/5 under a strict judge?",
        pad=22,
    )
    ax_top.text(
        0.5,
        1.01,
        f"Mean adherence: {mean_adherence:.2f} / 5    •    "
        f"Mean |math drift|: {mean_abs_math_diff:.2f} pts",
        transform=ax_top.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        color=PALETTE["reference"],
        fontweight="bold",
    )
    ax_top.legend(loc="upper right", framealpha=0.9)
    ax_top.grid(axis="y", linestyle="--", alpha=0.25)

    # ── Bottom: reasoning quality classification ────────────────────────
    quality_order = ["specific", "mixed", "generic"]
    quality_colors = [PALETTE["viral"], PALETTE["decent"], PALETTE["flop"]]
    quality_counts = Counter(r["reasoning_quality"] for r in rows)
    counts = [quality_counts.get(q, 0) for q in quality_order]

    bars = ax_bottom.bar(
        quality_order,
        counts,
        color=quality_colors,
        edgecolor=PALETTE["text"],
        linewidth=0.6,
        width=0.55,
    )
    for bar, c in zip(bars, counts):
        if c > 0:
            ax_bottom.text(
                bar.get_x() + bar.get_width() / 2,
                c + 0.15,
                str(c),
                ha="center",
                color=PALETTE["text"],
                fontsize=11,
                fontweight="bold",
            )

    ax_bottom.set_ylabel("Posts")
    ax_bottom.set_title(
        f"Reasoning quality classification ({len(rows)} evaluations)",
        fontsize=14,
    )
    ax_bottom.set_ylim(0, max(counts) + 1.5 if any(counts) else 1)
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    save_plot(fig, PLOT_PATH)


async def main() -> None:
    print("=" * 70)
    print(" TEST 3 — RUBRIC ADHERENCE (LLM-as-judge)")
    print("=" * 70)
    summary = await run_test_3()
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
