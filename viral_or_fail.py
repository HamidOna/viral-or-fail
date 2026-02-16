#!/usr/bin/env python3
"""
Viral or Fail: The Gaming Content Algorithm Game

An interactive CLI game where you play as a gaming content creator.
Pick a trending topic, choose a platform, and watch three AI agents
simulate whether your content would go viral — or flop.

Built with AutoGen for multi-agent orchestration.
"""

import os
import re
import sys

import autogen
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from agents.creator import create_content_creator_agent
from agents.algorithm import create_algorithm_simulator_agent
from agents.audience import create_audience_persona_agent
from config.platform_rules import PLATFORM_RULES, PLATFORMS
from tools.trends_tool import fetch_gaming_trends

# Load environment variables from .env file
load_dotenv()

console = Console()

# ── Constants ────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 3

BANNER = r"""
[bold magenta]
 ╔══════════════════════════════════════════════════════════════╗
 ║          🎮  VIRAL OR FAIL  🎮                              ║
 ║          The Gaming Content Algorithm Game                  ║
 ║                                                             ║
 ║   Can your content crack the algorithm?                     ║
 ║   3 AI agents will judge your gaming post.                  ║
 ╚══════════════════════════════════════════════════════════════╝
[/bold magenta]"""


# ── LLM Configuration ───────────────────────────────────────────────────────

def get_llm_config() -> dict:
    """Build the AutoGen LLM configuration for GitHub Models."""
    api_key = os.getenv("GITHUB_TOKEN")
    if not api_key:
        console.print(
            "[bold red]Error: GITHUB_TOKEN not found![/bold red]\n"
            "Create a .env file with your GitHub PAT.\n"
            "Get one at: https://github.com/settings/tokens\n"
            "See .env.example for details."
        )
        sys.exit(1)

    return {
        "config_list": [
            {
                "model": "openai/gpt-4.1-mini",
                "api_key": api_key,
                "base_url": "https://models.github.ai/inference",
                "price": [0, 0],  # GitHub Models free tier
            }
        ],
        "temperature": 0.8,
        "cache_seed": None,  # Disable caching for fresh responses each run
    }


# ── User Proxy (human-in-the-loop) ──────────────────────────────────────────

def create_user_proxy() -> autogen.UserProxyAgent:
    """Create the user proxy agent that sends messages on behalf of the player."""
    return autogen.UserProxyAgent(
        name="Player",
        human_input_mode="NEVER",  # We handle input ourselves via rich
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )


# ── Display Helpers ──────────────────────────────────────────────────────────

def display_agent_response(agent_name: str, content: str, style: str) -> None:
    """Display an agent's response in a styled panel."""
    console.print()
    console.print(Panel(
        content,
        title=f"[bold]{agent_name}[/bold]",
        border_style=style,
        padding=(1, 2),
    ))


def display_scorecard(
    topic: str,
    platform: str,
    persona_name: str,
    reach: int,
    engagement: int,
    virality: int,
    weighted_total: int,
    iteration: int,
) -> None:
    """Display the final scorecard as a rich table."""
    console.print()
    console.print(Panel(
        "[bold]FINAL SCORECARD[/bold]",
        style="bold yellow",
        expand=False,
    ))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Rating", justify="center")

    def get_rating(score: int) -> str:
        if score >= 85:
            return "[bold green]VIRAL[/bold green]"
        elif score >= 70:
            return "[green]Strong[/green]"
        elif score >= 50:
            return "[yellow]Decent[/yellow]"
        elif score >= 30:
            return "[red]Weak[/red]"
        else:
            return "[bold red]FAIL[/bold red]"

    table.add_row("Reach", f"{reach}/100", get_rating(reach))
    table.add_row("Engagement", f"{engagement}/100", get_rating(engagement))
    table.add_row("Virality", f"{virality}/100", get_rating(virality))
    table.add_row("", "", "")
    table.add_row(
        "[bold]Weighted Total[/bold]",
        f"[bold]{weighted_total}/100[/bold]",
        get_rating(weighted_total),
    )

    console.print(table)

    # Summary info
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="dim")
    info_table.add_column("Value")
    info_table.add_row("Topic", topic)
    info_table.add_row("Platform", platform)
    info_table.add_row("Audience Persona", persona_name)
    info_table.add_row("Iterations Used", f"{iteration}/{MAX_ITERATIONS}")

    console.print(info_table)

    # Verdict
    if weighted_total >= 80:
        verdict = "[bold green]YOUR CONTENT IS GOING VIRAL! The algorithm gods smile upon you.[/bold green]"
    elif weighted_total >= 60:
        verdict = "[yellow]Solid content. You'll get decent reach but no blowup.[/yellow]"
    elif weighted_total >= 40:
        verdict = "[red]Mid at best. The algorithm will bury this after the first batch.[/red]"
    else:
        verdict = "[bold red]FAIL. This is getting 12 views and 3 of them are your alt accounts.[/bold red]"

    console.print()
    console.print(Panel(verdict, title="[bold]VERDICT[/bold]", border_style="bright_white"))


# ── Score Extraction ─────────────────────────────────────────────────────────

def extract_scores(algorithm_response: str) -> dict:
    """
    Extract numerical scores from the Algorithm Simulator's response.

    Handles the two formats the LLM produces:
      - Bare numbers:   "**Reach Score:** 65"
      - With /100:      "**69.25/100**"
      - With calc:      "= 25.5 + 20 + 10 + 9.75 + 4 = **69.25/100**"

    Strips markdown bold markers first, then searches for each label
    followed by the last number on that line (to skip calculation steps).
    """
    scores = {
        "reach": 50,
        "engagement": 50,
        "virality": 50,
        "weighted_total": 50,
    }

    # Strip bold markers so "**65**" becomes "65" and "**69.25/100**" becomes "69.25/100"
    clean = algorithm_response.replace("**", "")

    # For reach/engagement/virality: match "Label" then grab the FIRST number
    # that follows (these lines are simple, e.g. "Reach Score: 65" or "Reach Score: 65/100")
    simple_patterns = {
        "reach": r"Reach\s*Score\D*?(\d+(?:\.\d+)?)",
        "engagement": r"Engagement\s*Score\D*?(\d+(?:\.\d+)?)",
        "virality": r"Virality\s*Score\D*?(\d+(?:\.\d+)?)",
    }

    for key, pattern in simple_patterns.items():
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            score = round(float(match.group(1)))
            scores[key] = min(100, max(0, score))

    # For weighted total: the line often contains a full calculation like
    # "WEIGHTED TOTAL: (85*0.30) + ... = 69.25/100"
    # Strategy: look for "N/100" first (the final result). If not found,
    # grab the last number after an "=" sign.
    wt_match = re.search(
        r"Weighted\s*Total[^\n]+", clean, re.IGNORECASE
    )
    if wt_match:
        wt_line = wt_match.group(0)

        # Try 1: match "N/100" — the definitive score format
        score_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*100", wt_line)
        if score_match:
            scores["weighted_total"] = min(100, max(0, round(float(score_match.group(1)))))
        else:
            # Try 2: grab the last number after the final "=" sign
            last_eq = wt_line.rfind("=")
            if last_eq != -1:
                after_eq = wt_line[last_eq:]
                nums = re.findall(r"(\d+(?:\.\d+)?)", after_eq)
                if nums:
                    scores["weighted_total"] = min(100, max(0, round(float(nums[-1]))))
            else:
                # Try 3: just grab the first number after the label
                num_match = re.search(r"(\d+(?:\.\d+)?)", wt_line)
                if num_match:
                    scores["weighted_total"] = min(100, max(0, round(float(num_match.group(1)))))

    return scores


# ── Agent Interaction ────────────────────────────────────────────────────────

def get_agent_response(
    user_proxy: autogen.UserProxyAgent,
    agent: autogen.AssistantAgent,
    message: str,
) -> str:
    """
    Send a message to an agent and get its response.

    Uses AutoGen's initiate_chat to run a single turn of conversation.
    """
    result = user_proxy.initiate_chat(
        agent,
        message=message,
        max_turns=1,
        clear_history=False,
    )

    # Extract the agent's last reply
    if result and result.chat_history:
        for msg in reversed(result.chat_history):
            if msg.get("role") == "assistant" or msg.get("name") == agent.name:
                return msg.get("content", "")

    return "No response generated."


# ── Game Loop ────────────────────────────────────────────────────────────────

def select_trend(trends: list[str]) -> str:
    """Let the player pick a trending topic."""
    console.print("\n[bold cyan]TRENDING GAMING TOPICS:[/bold cyan]\n")

    for i, trend in enumerate(trends, 1):
        console.print(f"  [bold]{i}.[/bold] {trend}")

    console.print()
    choice = IntPrompt.ask(
        "[bold]Pick a trend",
        choices=[str(i) for i in range(1, len(trends) + 1)],
    )

    return trends[choice - 1]


def select_platform() -> str:
    """Let the player pick a social media platform."""
    console.print("\n[bold cyan]CHOOSE YOUR PLATFORM:[/bold cyan]\n")

    for i, platform in enumerate(PLATFORMS, 1):
        rules = PLATFORM_RULES[platform]
        console.print(f"  [bold]{i}.[/bold] {platform} — {rules['description']}")

    console.print()
    choice = IntPrompt.ask(
        "[bold]Pick a platform",
        choices=[str(i) for i in range(1, len(PLATFORMS) + 1)],
    )

    return PLATFORMS[choice - 1]


def build_scoring_rubric(platform: str) -> str:
    """Format the platform's scoring rubric as a string for the agent prompt."""
    rules = PLATFORM_RULES[platform]
    lines = [f"Platform: {platform}", f"Description: {rules['description']}", ""]
    lines.append("Scoring Criteria (use these exact weights):")
    for name, info in rules["criteria"].items():
        weight_pct = int(info["weight"] * 100)
        lines.append(f"- {name} ({weight_pct}%): {info['description']}")
    return "\n".join(lines)


def run_game() -> None:
    """Main game loop."""
    console.print(BANNER)

    console.print(
        "[dim]A tutorial project for building multi-agent systems with AutoGen[/dim]\n"
    )

    # ── Setup ──

    llm_config = get_llm_config()

    console.print("[bold]Setting up AI agents...[/bold]\n")

    user_proxy = create_user_proxy()
    creator = create_content_creator_agent(llm_config)
    algorithm = create_algorithm_simulator_agent(llm_config)
    audience_agent, persona = create_audience_persona_agent(llm_config)

    console.print(f"  [green]Content Creator[/green] — ready")
    console.print(f"  [blue]Algorithm Simulator[/blue] — ready")
    console.print(
        f"  [magenta]Audience Persona[/magenta] — "
        f"[bold]{persona['name']}[/bold] ({persona['description']})"
    )

    # ── Fetch Trends ──

    console.print()
    trends = fetch_gaming_trends(count=10)

    # ── Player Choices ──

    topic = select_trend(trends)
    platform = select_platform()

    console.print(
        f"\n[bold]You chose:[/bold] [cyan]{topic}[/cyan] on [cyan]{platform}[/cyan]\n"
    )
    console.print(Panel(
        f"[dim]Format hint: {PLATFORM_RULES[platform]['format_hint']}[/dim]",
        border_style="dim",
    ))

    # ── Build Prompts ──

    rubric = build_scoring_rubric(platform)

    creator_prompt = (
        f"Create a {platform} post about this trending gaming topic: {topic}\n\n"
        f"Platform: {platform}\n"
        f"Format hint: {PLATFORM_RULES[platform]['format_hint']}\n\n"
        f"Make it feel native to {platform}. Go hard — safe content doesn't go viral."
    )

    # ── Iteration Loop ──

    iteration = 0
    creator_response = ""
    algorithm_response = ""
    audience_response = ""
    scores = {}

    while iteration < MAX_ITERATIONS:
        iteration += 1

        console.print(
            f"\n{'=' * 60}",
            style="bold",
        )
        console.print(
            f"[bold yellow]  ROUND {iteration}/{MAX_ITERATIONS}[/bold yellow]"
        )
        console.print(f"{'=' * 60}\n", style="bold")

        # ── Step 1: Content Creator generates/revises ──

        console.print("[bold green]Content Creator is cooking...[/bold green]")

        if iteration == 1:
            creator_response = get_agent_response(
                user_proxy, creator, creator_prompt
            )
        else:
            # Revision prompt with feedback from previous round
            revision_prompt = (
                f"REVISION REQUEST (Round {iteration}/{MAX_ITERATIONS}):\n\n"
                f"The Algorithm Simulator and Audience Persona reviewed your "
                f"{platform} post about '{topic}'. Here's their feedback:\n\n"
                f"--- ALGORITHM FEEDBACK ---\n{algorithm_response}\n\n"
                f"--- AUDIENCE FEEDBACK ({persona['name']}) ---\n"
                f"{audience_response}\n\n"
                f"Revise your content to address their concerns. Keep what works, "
                f"fix what doesn't. Show what you changed and why."
            )
            creator_response = get_agent_response(
                user_proxy, creator, revision_prompt
            )

        display_agent_response("Content Creator", creator_response, "green")

        # ── Step 2: Algorithm Simulator scores the content ──

        console.print("\n[bold blue]Algorithm Simulator is processing...[/bold blue]")

        algorithm_prompt = (
            f"Evaluate this {platform} post about '{topic}' using the platform's "
            f"scoring rubric.\n\n"
            f"--- SCORING RUBRIC ---\n{rubric}\n\n"
            f"--- CONTENT TO EVALUATE ---\n{creator_response}\n\n"
            f"Score each criterion out of 100, then calculate the weighted total. "
            f"Be specific and reference platform algorithm mechanics."
        )

        algorithm_response = get_agent_response(
            user_proxy, algorithm, algorithm_prompt
        )

        display_agent_response("Algorithm Simulator", algorithm_response, "blue")

        # ── Step 3: Audience Persona reacts ──

        console.print(
            f"\n[bold magenta]{persona['name']} is reacting...[/bold magenta]"
        )

        audience_prompt = (
            f"You just saw this on your {platform} feed. It's about '{topic}'. "
            f"React naturally as yourself.\n\n"
            f"--- THE POST ---\n{creator_response}"
        )

        audience_response = get_agent_response(
            user_proxy, audience_agent, audience_prompt
        )

        display_agent_response(
            f"{persona['name']} ({persona['description']})",
            audience_response,
            "magenta",
        )

        # ── Extract scores ──

        scores = extract_scores(algorithm_response)

        # ── Show quick score summary ──

        console.print()
        quick_table = Table(
            title=f"Round {iteration} Quick Score",
            show_header=True,
            header_style="bold",
        )
        quick_table.add_column("Metric", style="bold")
        quick_table.add_column("Score", justify="center")
        quick_table.add_row("Reach", f"{scores['reach']}/100")
        quick_table.add_row("Engagement", f"{scores['engagement']}/100")
        quick_table.add_row("Virality", f"{scores['virality']}/100")
        quick_table.add_row(
            "[bold]Weighted Total[/bold]",
            f"[bold]{scores['weighted_total']}/100[/bold]",
        )
        console.print(quick_table)

        # ── Iterate or Lock In? ──

        if iteration < MAX_ITERATIONS:
            console.print()
            choice = Prompt.ask(
                "[bold]What do you want to do?[/bold]\n"
                f"  [cyan]1.[/cyan] ITERATE — Get the Creator to revise "
                f"(Round {iteration + 1}/{MAX_ITERATIONS})\n"
                f"  [cyan]2.[/cyan] LOCK IN — Accept this version\n\n"
                f"[bold]Your choice",
                choices=["1", "2"],
                default="1",
            )

            if choice == "2":
                console.print(
                    "\n[bold yellow]LOCKED IN![/bold yellow] "
                    "Let's see the final scorecard."
                )
                break
        else:
            console.print(
                f"\n[bold yellow]Max iterations reached ({MAX_ITERATIONS}). "
                f"Locking in final version.[/bold yellow]"
            )

    # ── Final Scorecard ──

    display_scorecard(
        topic=topic,
        platform=platform,
        persona_name=f"{persona['name']} ({persona['description']})",
        reach=scores.get("reach", 50),
        engagement=scores.get("engagement", 50),
        virality=scores.get("virality", 50),
        weighted_total=scores.get("weighted_total", 50),
        iteration=iteration,
    )

    # ── Sign-off ──

    console.print(
        "\n[dim]Thanks for playing Viral or Fail! "
        "Tweak the agents, try different platforms, and see if you can beat your score.[/dim]\n"
    )


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_game()
    except KeyboardInterrupt:
        console.print("\n\n[dim]Game interrupted. See you next time![/dim]")
        sys.exit(0)
