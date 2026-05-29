"""Re-render plots from existing JSON results without re-running agents.

Useful after a layout/styling tweak — reloads each test's JSON and calls
the corresponding ``_plot()`` function with no API traffic. Run with:

    python -m evals.replot              # re-render all four
    python -m evals.replot 2 4          # re-render Tests 2 and 4 only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from evals.test_1_consistency import _plot as plot1
from evals.test_2_calibration import _plot as plot2
from evals.test_3_rubric_adherence import _plot as plot3
from evals.test_4_workflow import _plot as plot4

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "eval_results"


def replot_test_1() -> None:
    data = json.loads((RESULTS / "test_1_consistency.json").read_text(encoding="utf-8"))
    plot1(data["per_post"], data["mean_cv_pct"])
    print("  re-rendered eval_results/plots/01_consistency.png")


def replot_test_2() -> None:
    data = json.loads((RESULTS / "test_2_calibration.json").read_text(encoding="utf-8"))
    plot2(data["posts"], data["pearson_r"], data["spearman_r"], data["mae"])
    print("  re-rendered eval_results/plots/02_calibration.png")


def replot_test_3() -> None:
    data = json.loads((RESULTS / "test_3_rubric_adherence.json").read_text(encoding="utf-8"))
    plot3(data["posts"], data["mean_adherence"], data["mean_abs_math_diff"])
    print("  re-rendered eval_results/plots/03_adherence.png")


def replot_test_4() -> None:
    data = json.loads((RESULTS / "test_4_workflow.json").read_text(encoding="utf-8"))
    plot4(
        data["creator_pass_pct"],
        data["algorithm_pass_pct"],
        data["persona_pass_pct"],
        data["overall_pass_pct"],
        data["num_topics"],
    )
    print("  re-rendered eval_results/plots/04_workflow.png")


REPLOTTERS = {
    1: replot_test_1,
    2: replot_test_2,
    3: replot_test_3,
    4: replot_test_4,
}


def main() -> int:
    args = sys.argv[1:]
    if args:
        try:
            tests = [int(a) for a in args]
        except ValueError:
            print(f"usage: python -m evals.replot [1] [2] [3] [4]", file=sys.stderr)
            return 2
    else:
        tests = [1, 2, 3, 4]
    for t in tests:
        if t not in REPLOTTERS:
            print(f"unknown test {t} (valid: 1, 2, 3, 4)", file=sys.stderr)
            return 2
        REPLOTTERS[t]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
