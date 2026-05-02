"""Shared matplotlib styling for the Viral or Fail eval suite plots.

All four plots share a dark gaming-themed palette so they read as a set
when embedded in the blog post. PNGs are 1600x900 @ DPI 200 — the right
resolution for Microsoft Tech Community embeds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

PALETTE = {
    "viral": "#00E676",      # neon green
    "decent": "#FFD600",     # amber
    "flop": "#FF5252",       # coral red
    "outlier": "#CE93D8",    # neon violet — for the ratio'd post
    "reference": "#90CAF9",  # light blue (ref lines, callouts)
    "text": "#FAFAFA",       # near-white
    "muted": "#9E9E9E",      # grey
    "accent": "#FF80AB",     # hot pink (per-agent contrast in Test 4)
    "accent2": "#80D8FF",    # cyan (per-agent contrast in Test 4)
}

FIG_SIZE = (8, 4.5)  # inches; with dpi=200 this is 1600x900 px
DPI = 200


def apply_dark_theme() -> None:
    """Apply the suite's dark theme + typography defaults to matplotlib."""
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#121212",
            "axes.facecolor": "#1E1E1E",
            "axes.edgecolor": PALETTE["muted"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "grid.color": PALETTE["muted"],
            "grid.alpha": 0.25,
            "legend.facecolor": "#1E1E1E",
            "legend.edgecolor": PALETTE["muted"],
            "legend.labelcolor": PALETTE["text"],
        }
    )


def color_for_label(label: str) -> str:
    """Map a dataset label (viral/decent/flop/outlier) to a palette colour."""
    return PALETTE.get(label, PALETTE["muted"])


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    """Save a figure as a 1600x900 PNG suitable for blog embedding."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def annotate_metric(ax, text: str, *, loc: str = "upper left") -> None:
    """Annotate the headline metric in the corner of an axis."""
    locs = {
        "upper left": (0.02, 0.97, "left", "top"),
        "upper right": (0.98, 0.97, "right", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
    }
    x, y, ha, va = locs[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=12,
        color=PALETTE["reference"],
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#1E1E1E",
            "edgecolor": PALETTE["reference"],
            "alpha": 0.85,
        },
    )
