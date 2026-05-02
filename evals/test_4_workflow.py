"""Test 4 — Per-agent workflow evaluation.

Question: when we run the existing 3-agent pipeline end-to-end on real
trending topics, where are the weak points?

Method (per Option A in the design discussion):
- Run the existing pipeline manually on 5 topics. We deliberately do NOT
  refactor it into a ``Workflow`` — Post 1 framed it as application-controlled
  orchestration and we keep that contract.
- For each topic, drive Creator → Algorithm → Persona once, then apply
  per-agent evaluators independently:
    Creator   — keyword check for Twitter/X-native terminology
    Algorithm — structural check (emits a parseable WEIGHTED TOTAL 0–100)
    Persona   — keyword check for the chosen persona's vocabulary
- Aggregate per-agent pass rates so the weak link is obvious in the chart.

Outputs:
- eval_results/test_4_workflow.json
- eval_results/plots/04_workflow.png
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from agents.algorithm import create_algorithm_simulator_agent
from agents.audience import PERSONAS, create_audience_persona_agent
from agents.creator import create_content_creator_agent
from config.platform_rules import PLATFORM_RULES
from evals.client_factory import get_chat_client
from evals.custom_evaluators import has_weighted_total, keyword_check
from evals.harness import call_agent_with_retry
from evals.plot_style import (
    DPI,
    FIG_SIZE,
    PALETTE,
    apply_dark_theme,
    save_plot,
)

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "eval_results" / "test_4_workflow.json"
PLOT_PATH = REPO_ROOT / "eval_results" / "plots" / "04_workflow.png"

PLATFORM = "Twitter/X"
PERSONA_KEY = "competitive_esports_fan"  # TryHard_Tyler, deterministic for repeatability
NUM_TOPICS = 5
RATE_LIMIT_SLEEP = 4.5  # 12 RPM, under GitHub Models' 15 RPM cap

# Hardcoded fallback if trendspy returns nothing or errors
FALLBACK_TOPICS = [
    "Marvel Rivals season 3",
    "Helldivers 2 patch drama",
    "Path of Exile 2 endgame meta",
    "GTA 6 release window leak",
    "Steam Deck OLED price drop",
]

# Platform-native terminology for the Creator check
PLATFORM_KEYWORDS = {
    "Twitter/X": ["thread", "ratio", "qrt", "take", "based", "cope", "🧵", "rt", "diff"],
}

# Persona-native vocabulary for the Audience check
PERSONA_KEYWORDS = {
    "competitive_esports_fan": ["diff", "cope", "clear", "fraud", "goated", "ratio", "cap", "valid"],
    "casual_mobile_gamer": ["lol", "ngl", "lowkey", "fr fr", "no cap", "vibe", "cringe"],
    "retro_indie_enthusiast": ["soul", "peak", "indie", "retro", "corporate slop", "craft"],
}


def _resolve_topics() -> list[str]:
    """Try live trends first, fall back to a curated list. Same path as the game."""
    try:
        from tools.trends_tool import fetch_gaming_trends

        trends = fetch_gaming_trends(count=NUM_TOPICS * 2)
        if trends:
            return trends[:NUM_TOPICS]
    except Exception:
        pass
    return FALLBACK_TOPICS[:NUM_TOPICS]


def _creator_prompt(topic: str) -> str:
    rules = PLATFORM_RULES[PLATFORM]
    return (
        f"Create a {PLATFORM} post about this trending gaming topic: {topic}\n\n"
        f"Platform: {PLATFORM}\n"
        f"Format hint: {rules['format_hint']}\n\n"
        f"Make it feel native to {PLATFORM}. Go hard — safe content doesn't go viral."
    )


def _algorithm_prompt(topic: str, creator_text: str) -> str:
    rules = PLATFORM_RULES[PLATFORM]
    rubric_lines = [
        f"Platform: {PLATFORM}",
        f"Description: {rules['description']}",
        "",
        "Scoring Criteria (use these exact weights):",
    ]
    for name, info in rules["criteria"].items():
        rubric_lines.append(f"- {name} ({int(info['weight'] * 100)}%): {info['description']}")
    rubric_text = "\n".join(rubric_lines)
    return (
        f"Evaluate this {PLATFORM} post about '{topic}' using the platform's scoring rubric.\n\n"
        f"--- SCORING RUBRIC ---\n{rubric_text}\n\n"
        f"--- CONTENT TO EVALUATE ---\n{creator_text}\n\n"
        f"Score each criterion out of 100, then calculate the weighted total. "
        f"Be specific and reference platform algorithm mechanics."
    )


def _persona_prompt(topic: str, creator_text: str) -> str:
    return (
        f"You just saw this on your {PLATFORM} feed. It's about '{topic}'. "
        f"React naturally as yourself.\n\n"
        f"--- THE POST ---\n{creator_text}"
    )


async def run_test_4(*, verbose: bool = True) -> dict:
    client = get_chat_client()
    creator = create_content_creator_agent(client)
    algorithm = create_algorithm_simulator_agent(client)
    persona_agent, persona = create_audience_persona_agent(client, persona=PERSONAS[PERSONA_KEY])

    topics = _resolve_topics()

    creator_check = keyword_check(PLATFORM_KEYWORDS[PLATFORM])
    persona_check = keyword_check(PERSONA_KEYWORDS[PERSONA_KEY])
    algo_check = has_weighted_total

    rows: list[dict] = []

    for idx, topic in enumerate(topics, start=1):
        if verbose:
            print(f"  [{idx}/{len(topics)}] {topic}")

        creator_text = await call_agent_with_retry(creator, _creator_prompt(topic))
        await asyncio.sleep(RATE_LIMIT_SLEEP)

        algorithm_text = await call_agent_with_retry(
            algorithm, _algorithm_prompt(topic, creator_text)
        )
        await asyncio.sleep(RATE_LIMIT_SLEEP)

        persona_text = await call_agent_with_retry(
            persona_agent, _persona_prompt(topic, creator_text)
        )
        await asyncio.sleep(RATE_LIMIT_SLEEP)

        creator_score = creator_check(response=creator_text)
        algo_score = algo_check(response=algorithm_text)
        persona_score = persona_check(response=persona_text)

        rows.append(
            {
                "topic": topic,
                "creator": {"passed": bool(creator_score), "score": creator_score},
                "algorithm": {"passed": bool(algo_score), "score": algo_score},
                "persona": {"passed": bool(persona_score), "score": persona_score},
            }
        )

        if verbose:
            mark = lambda b: "PASS" if b else "FAIL"
            print(
                f"      Creator   {mark(creator_score)}    "
                f"Algorithm {mark(algo_score)}    "
                f"Persona   {mark(persona_score)}"
            )

    n = len(rows)

    def _pct(key: str) -> float:
        return round(100.0 * sum(1 for r in rows if r[key]["passed"]) / n, 1) if n else 0.0

    creator_pct = _pct("creator")
    algorithm_pct = _pct("algorithm")
    persona_pct = _pct("persona")
    overall_pass = round(
        100.0
        * sum(1 for r in rows if all(r[k]["passed"] for k in ("creator", "algorithm", "persona")))
        / n,
        1,
    ) if n else 0.0

    summary = {
        "test": "workflow",
        "platform": PLATFORM,
        "persona": persona["name"],
        "num_topics": n,
        "overall_pass_pct": overall_pass,
        "creator_pass_pct": creator_pct,
        "algorithm_pass_pct": algorithm_pct,
        "persona_pass_pct": persona_pct,
        "headline": (
            f"Per-agent pass rates — Creator: {creator_pct}%, "
            f"Simulator: {algorithm_pct}%, Persona: {persona_pct}%"
        ),
        "topics": [r["topic"] for r in rows],
        "rows": rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _plot(creator_pct, algorithm_pct, persona_pct, overall_pass, n)

    if verbose:
        print()
        print(f"  Overall pipeline pass rate: {overall_pass}%")
        print(f"  Creator   {creator_pct}%")
        print(f"  Algorithm {algorithm_pct}%")
        print(f"  Persona   {persona_pct}%")

    return summary


def _plot(
    creator_pct: float,
    algorithm_pct: float,
    persona_pct: float,
    overall_pct: float,
    n_topics: int,
) -> None:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    agents = ["Content Creator", "Algorithm Simulator", "Audience Persona"]
    pass_rates = [creator_pct, algorithm_pct, persona_pct]
    fail_rates = [100 - p for p in pass_rates]
    x = np.arange(len(agents))
    width = 0.35

    bars_pass = ax.bar(
        x - width / 2,
        pass_rates,
        width,
        label="passed",
        color=PALETTE["viral"],
        edgecolor=PALETTE["text"],
    )
    bars_fail = ax.bar(
        x + width / 2,
        fail_rates,
        width,
        label="failed",
        color=PALETTE["flop"],
        edgecolor=PALETTE["text"],
        alpha=0.9,
    )

    for bar in list(bars_pass) + list(bars_fail):
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.0f}%",
                ha="center",
                color=PALETTE["text"],
                fontsize=11,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of topics")
    # Pad the title so we can sit a subtitle line beneath it without colliding
    # with the bar-top "100%" labels (which the annotate_metric box used to overlap).
    ax.set_title(
        f"Test 4 — Per-agent pass rates across {n_topics} topics "
        f"({PLATFORM}, {PERSONAS[PERSONA_KEY]['name']})",
        pad=22,
    )
    ax.text(
        0.5,
        1.01,
        f"Pipeline pass rate (all 3 agents): {overall_pct:.0f}%",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        color=PALETTE["reference"],
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    save_plot(fig, PLOT_PATH)


async def main() -> None:
    print("=" * 70)
    print(" TEST 4 — PER-AGENT WORKFLOW EVAL")
    print("=" * 70)
    summary = await run_test_4()
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
