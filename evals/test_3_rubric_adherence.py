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
    annotate_metric,
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

    _plot(per_criterion_pct, adherence_scores, mean_adherence, FOCUS_PLATFORM, len(focus_rows))

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
    per_criterion_pct: dict[str, float],
    adherence_scores: list[int],
    mean_adherence: float,
    focus_platform: str,
    n_focus: int,
) -> None:
    apply_dark_theme()
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=FIG_SIZE, dpi=DPI, gridspec_kw={"height_ratios": [3, 1.2]}
    )

    # ── Top: stacked horizontal bars per criterion ──────────────────────
    criteria = list(per_criterion_pct.keys())
    scored = [per_criterion_pct[c] for c in criteria]
    skipped = [100.0 - s for s in scored]

    y = list(range(len(criteria)))
    ax_top.barh(
        y, scored, color=PALETTE["viral"], edgecolor=PALETTE["text"], linewidth=0.6, label="explicitly scored"
    )
    ax_top.barh(
        y,
        skipped,
        left=scored,
        color=PALETTE["flop"],
        edgecolor=PALETTE["text"],
        linewidth=0.6,
        alpha=0.85,
        label="glossed over",
    )

    for i, c in enumerate(criteria):
        ax_top.text(
            max(scored[i] / 2, 4),
            i,
            f"{scored[i]:.0f}%",
            va="center",
            ha="center",
            color="#0F1110",
            fontsize=10,
            weight="bold",
        )

    ax_top.set_yticks(y)
    ax_top.set_yticklabels(criteria)
    ax_top.set_xlim(0, 100)
    ax_top.set_xlabel("% of evaluations")
    ax_top.set_title(
        f"Test 3 — Rubric adherence on {focus_platform} criteria ({n_focus} posts)"
    )
    ax_top.invert_yaxis()
    ax_top.legend(loc="lower right", framealpha=0.9)
    ax_top.grid(axis="x", linestyle="--", alpha=0.3)

    # ── Bottom: histogram of overall adherence scores ───────────────────
    counts = Counter(adherence_scores)
    bins = [1, 2, 3, 4, 5]
    bar_heights = [counts.get(b, 0) for b in bins]

    ax_bottom.bar(
        bins,
        bar_heights,
        color=PALETTE["reference"],
        edgecolor=PALETTE["text"],
        linewidth=0.6,
        width=0.7,
    )
    for b, h in zip(bins, bar_heights):
        if h:
            ax_bottom.text(b, h + 0.1, str(h), ha="center", color=PALETTE["text"], fontsize=10)

    ax_bottom.set_xticks(bins)
    ax_bottom.set_xlabel("Adherence score (1–5)")
    ax_bottom.set_ylabel("Posts")
    ax_bottom.set_title("Distribution of overall adherence (all 10 posts)")
    ax_bottom.set_ylim(0, max(bar_heights) + 1.2 if bar_heights else 1)
    ax_bottom.grid(axis="y", linestyle="--", alpha=0.3)

    annotate_metric(ax_top, f"Mean adherence: {mean_adherence:.2f} / 5", loc="upper right")

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
