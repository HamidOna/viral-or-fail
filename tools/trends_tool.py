"""
Google Trends fetcher for gaming topics.

Uses trendspy to pull real-time trending searches from Google Trends,
filtered to the Games topic (topic ID 6). Falls back gracefully to
sample_trends.json if the API call fails.

Note: pytrends was archived in April 2025. trendspy is the recommended
replacement — no API key or browser dependency required.
"""

import json
from pathlib import Path

from rich.console import Console

console = Console()

# Path to fallback sample data
SAMPLE_TRENDS_PATH = Path(__file__).parent / "sample_trends.json"

# trendspy topic ID for "Games" (from trendspy.constants.TREND_TOPICS)
GAMES_TOPIC_ID = 6


def fetch_gaming_trends(count: int = 10) -> list[str]:
    """
    Fetch trending gaming topics from Google Trends.

    Fetches all trending searches via trendspy, then filters to only those
    tagged with the Games topic (ID 6). If fewer than 5 gaming trends are
    found, pads with curated sample data.

    Args:
        count: Number of trends to return (default 10).

    Returns:
        A list of trending gaming topic strings.
    """
    try:
        from trendspy import Trends

        console.print(
            "[dim]Fetching live gaming trends from Google Trends...[/dim]"
        )

        tr = Trends()
        all_trends = tr.trending_now(geo="US")

        # Filter to trends tagged with the Games topic
        gaming_trends = [
            t.keyword for t in all_trends
            if GAMES_TOPIC_ID in (t.topics or [])
        ]

        if len(gaming_trends) >= 5:
            console.print(
                f"[green]Found {len(gaming_trends)} live gaming trends![/green]"
            )
            return gaming_trends[:count]

        # Not enough Games-topic trends — also try keyword matching as backup
        gaming_keywords = [
            "game", "gaming", "gamer", "esport", "playstation", "xbox",
            "nintendo", "steam", "twitch", "fortnite", "valorant", "league",
            "minecraft", "roblox", "cod", "warzone", "apex", "zelda",
            "mario", "pokemon", "gta", "elden", "final fantasy", "ps5",
            "ps6", "switch", "gpu", "rtx", "dlc",
        ]
        keyword_matches = [
            t.keyword for t in all_trends
            if t.keyword not in gaming_trends
            and any(kw in t.keyword.lower() for kw in gaming_keywords)
        ]
        gaming_trends.extend(keyword_matches)

        if len(gaming_trends) >= 5:
            console.print(
                f"[green]Found {len(gaming_trends)} live gaming trends![/green]"
            )
            return gaming_trends[:count]

        # Still not enough — pad with sample data
        console.print(
            "[yellow]Few gaming trends found live. "
            "Mixing with sample data...[/yellow]"
        )
        sample = _load_sample_trends()
        combined = gaming_trends + [t for t in sample if t not in gaming_trends]
        return combined[:count]

    except Exception as e:
        console.print(
            f"[yellow]Could not fetch live trends: {e}[/yellow]"
        )
        console.print(
            "[yellow]Falling back to sample gaming trends...[/yellow]"
        )
        return _load_sample_trends()[:count]


def _load_sample_trends() -> list[str]:
    """Load pre-seeded gaming trends from the JSON file."""
    with open(SAMPLE_TRENDS_PATH, "r") as f:
        data = json.load(f)
    return data["trends"]
