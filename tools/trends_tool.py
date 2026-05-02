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

# trendspy topic IDs (from trendspy.constants.TREND_TOPICS)
GAMES_TOPIC_ID = 6     # Games — but Google's taxonomy bundles gambling/horse-racing here too
SPORTS_TOPIC_ID = 17   # excluded: filters out things like Kentucky Derby that get co-tagged Games

# Explicit denylist for gambling/lottery trends that get tagged Games-only (no Sports tag).
# These are real terms we've seen pass the topic filter — keep this list small and concrete.
GAMBLING_KEYWORDS = (
    "lottery", "powerball", "mega millions", "casino", "fanduel",
    "draftkings", "sportsbook", "betting", "blackjack",
)


def _is_gaming_trend(trend) -> bool:
    """True if the trend looks like actual gaming, not gambling/sports overlap.

    The Google Trends taxonomy bundles all of (video games, board games,
    horse racing, sportsbooks, lotteries) under a single "Games" topic,
    so we layer two extra filters: exclude anything also tagged Sports
    (catches horse racing, fantasy sports), and apply a small keyword
    denylist for gambling-only items that slip through.
    """
    topics = trend.topics or []
    if GAMES_TOPIC_ID not in topics:
        return False
    if SPORTS_TOPIC_ID in topics:
        return False
    keyword_lower = trend.keyword.lower()
    if any(bad in keyword_lower for bad in GAMBLING_KEYWORDS):
        return False
    return True


def fetch_gaming_trends(count: int = 10) -> list[str]:
    """
    Fetch trending gaming topics from Google Trends.

    Fetches all trending searches via trendspy, filters to actual gaming
    trends (Games-tagged, not Sports-tagged, not gambling), and pads with
    curated sample data if too few are found.

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

        # Topic-tag filter (excludes sports + gambling, see _is_gaming_trend)
        gaming_trends = [t.keyword for t in all_trends if _is_gaming_trend(t)]

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
