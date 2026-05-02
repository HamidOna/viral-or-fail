"""Shared @evaluator functions and parsing utilities for the eval suite.

These are deliberately small. They illustrate the pattern readers will
recognise from the SDK's ``@evaluator`` decorator: a pure function that
takes some subset of (query, response, expected_output) and returns a
float in [0, 1] (or, for adherence, a 1–5 score normalised to [0, 1]).
"""

from __future__ import annotations

import re

from evals.harness import evaluator


# ── Score parsing ───────────────────────────────────────────────────────────

# The Algorithm Simulator's prompt asks for "**WEIGHTED TOTAL:** ... /100".
# In practice the model produces three formats interchangeably:
#   1) Same line:                "WEIGHTED TOTAL: 73/100"
#   2) Same line w/ calculation: "WEIGHTED TOTAL: = 22.5 + ... = 73.25/100"
#   3) Multi-line:               "WEIGHTED TOTAL:\n= 22.5 + ... = 73.25/100"
# We strip bold markers, locate the WEIGHTED TOTAL header, then scan a
# small window (rest of the header line + the next few non-blank lines)
# for "N/100" or a trailing "= N". Multi-line is the common case for this
# model — the original viral_or_fail.py regex misses it and silently
# falls back to 50, which masks the failure.
_WT_HEADER_RE = re.compile(r"Weighted\s*Total\s*:?", re.IGNORECASE)
_OUT_OF_100_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*100")
_ANY_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_WT_LOOKAHEAD_LINES = 4


def parse_weighted_total(response: str) -> float | None:
    """Extract the Algorithm Simulator's ``WEIGHTED TOTAL`` from its response.

    Returns the score as a float in [0, 100], or ``None`` if no parseable
    score is found. Handles same-line, calculation-style, and multi-line
    layouts.
    """
    if not response:
        return None
    clean = response.replace("**", "")
    header = _WT_HEADER_RE.search(clean)
    if not header:
        return None
    after = clean[header.end():]
    window: list[str] = []
    for raw in after.splitlines()[: _WT_LOOKAHEAD_LINES + 1]:
        line = raw.strip()
        if not line and window:
            break
        if line:
            window.append(line)
    blob = " ".join(window)
    if not blob:
        return None
    m = _OUT_OF_100_RE.search(blob)
    if m:
        return _clamp(float(m.group(1)))
    last_eq = blob.rfind("=")
    if last_eq != -1:
        nums = _ANY_NUMBER_RE.findall(blob[last_eq:])
        if nums:
            return _clamp(float(nums[-1]))
    nums = _ANY_NUMBER_RE.findall(blob)
    if nums:
        return _clamp(float(nums[0]))
    return None


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


# ── Evaluators ──────────────────────────────────────────────────────────────


@evaluator
def correlates_with_truth(response: str, expected_output: str) -> float:
    """Per-item alignment score for Test 2 (calibration).

    Returns 1.0 when the Simulator's score exactly matches the labeled
    engagement score, dropping linearly with absolute error on the 0–100
    scale. We compute the corpus-level Pearson/Spearman in the test file —
    this evaluator just attaches a per-item alignment value to each EvalItem.
    """
    sim = parse_weighted_total(response)
    if sim is None or expected_output is None:
        return 0.0
    truth = float(expected_output)
    return 1.0 - (abs(sim - truth) / 100.0)


@evaluator
def has_weighted_total(response: str) -> float:
    """Workflow check: did the Algorithm Simulator emit a parseable WEIGHTED TOTAL?

    Returns 1.0 if a 0–100 weighted total can be parsed, else 0.0. Used by
    Test 4 to check whether the Simulator stuck to its required output format.
    """
    return 1.0 if parse_weighted_total(response) is not None else 0.0


def keyword_check(keywords: list[str], *, case_insensitive: bool = True) -> object:
    """Build a keyword-presence evaluator (mirrors MAF's ``keyword_check``).

    Returns 1.0 if any keyword appears in the response, else 0.0. The
    factory pattern lets each test swap in its own keyword list.

    Args:
        keywords: List of substrings to look for in the response.
        case_insensitive: If True (default), matching ignores case.

    Returns:
        An ``@evaluator``-decorated function ready to pass to ``EvalRunner``.
    """
    needles = [k.lower() if case_insensitive else k for k in keywords]
    label = "_".join(k.lower() for k in keywords[:3]) or "any"

    @evaluator
    def _check(response: str) -> float:
        haystack = response.lower() if case_insensitive else response
        return 1.0 if any(n in haystack for n in needles) else 0.0

    _check._evaluator_name = f"keyword_check_{label}"  # type: ignore[attr-defined]
    return _check
