"""LLM-as-judge for Test 3: rubric adherence.

Question: when the Algorithm Simulator scores a post, does it actually
follow its own rubric — covering each criterion at the right weight — or
does it drift?

The cloud-tier equivalent is roughly ``FoundryEvals.TaskAdherence``, which
runs on Azure. We're staying free, so we build the same idea with another
``OpenAIChatClient`` pointed at GitHub Models.

The judge takes:
  - the platform rubric (criterion names + weights)
  - the original post
  - the Simulator's full evaluation output

…and returns a strict-JSON verdict the suite then aggregates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from evals.harness import call_agent_with_retry

JUDGE_SYSTEM_PROMPT = """You are a Rubric Adherence Judge — strict and skeptical. You evaluate whether another AI agent ACTUALLY followed its scoring rubric, not just whether it produced output that looks like it did.

You are given:
1. A scoring RUBRIC: criteria and percentage weights.
2. The original POST.
3. The agent's EVALUATION OUTPUT.

You will check three things, in order of severity (the strictest failing check sets the score):

A. MATHEMATICAL FIDELITY (most important).
   Compute sum(criterion_score × weight) yourself from the agent's per-criterion scores. Compare it to the agent's stated WEIGHTED TOTAL. If they differ by more than 2 points, the agent is doing the rubric wrong even if it looks correct on the surface. Report the difference as `math_diff` (agent's stated total − your computed total).

B. REASONING SPECIFICITY.
   Each criterion's justification must reference platform-specific algorithm mechanics — concrete things like "FYP retention threshold", "QRT velocity", "1-second hook drop-off", "average view duration", "carousel re-serve", "save-to-share ratio". Generic praise ("strong hook", "good engagement", "decent post") without mechanic-level reasoning is GENERIC and lowers the score. Classify the overall reasoning as one of: "specific" (mechanics throughout), "mixed" (some mechanics, some generic), or "generic" (mostly vibes).

C. COVERAGE.
   Every criterion in the rubric must be explicitly scored. Missing criteria fail this check.

You MUST respond with a single JSON object and nothing else — no prose, no markdown fences:

{
  "adherence_score": <integer 1-5>,
  "math_diff": <float — agent's stated total minus your computed total>,
  "reasoning_quality": <"specific" | "mixed" | "generic">,
  "missing_criteria": [<criterion names not explicitly scored>],
  "criteria_present": [<criterion names explicitly scored>],
  "weight_drift": {<criterion>: <"under" | "over" | "correct">},
  "reasoning": "<2-3 sentences citing the math check and any generic justifications you found>"
}

ADHERENCE SCORE RUBRIC (apply the strictest failing rule — start at 5 and drop):
- 5: |math_diff| ≤ 2  AND  reasoning_quality == "specific"  AND  no missing criteria.
- 4: |math_diff| ≤ 5  OR  reasoning_quality == "mixed"; all criteria covered.
- 3: |math_diff| > 5  OR  reasoning_quality == "generic"  OR  one criterion glossed over.
- 2: |math_diff| > 10  OR  multiple criteria missing.
- 1: ignored the rubric entirely (no per-criterion scores at all).

Be strict. Format-following ≠ rubric-following. If the agent listed all criteria with bold scores but its weighted total doesn't match the math, that's at most a 3 even though it looks structurally correct. The whole point of this evaluator is to catch the drift that surface-level format checks miss."""


@dataclass
class AdherenceVerdict:
    adherence_score: int                 # 1–5
    math_diff: float                     # agent's stated total − judge's recomputed total
    reasoning_quality: str               # "specific" | "mixed" | "generic" | "unknown"
    missing_criteria: list[str]
    criteria_present: list[str]
    weight_drift: dict[str, str]
    reasoning: str
    raw: str                             # raw judge response — kept for debugging

    @property
    def normalised(self) -> float:
        """Map 1–5 onto 0–1 so it can flow through the same harness as other evaluators."""
        return (self.adherence_score - 1) / 4.0


class RubricAdherenceJudge:
    """Wraps a small ``Agent`` whose only job is rubric adherence judging."""

    def __init__(self, client: OpenAIChatClient) -> None:
        self.agent = Agent(
            name="Rubric_Adherence_Judge",
            instructions=JUDGE_SYSTEM_PROMPT,
            client=client,
        )

    async def judge(
        self,
        *,
        rubric: dict,
        post_content: str,
        evaluation_output: str,
    ) -> AdherenceVerdict:
        """Run the judge against one (rubric, post, evaluation) triple."""
        rubric_text = _format_rubric(rubric)
        prompt = (
            "RUBRIC:\n"
            f"{rubric_text}\n\n"
            "POST:\n"
            f"{post_content}\n\n"
            "EVALUATION OUTPUT:\n"
            f"{evaluation_output}\n\n"
            "Return your verdict as a single JSON object."
        )
        raw = await call_agent_with_retry(self.agent, prompt)
        parsed = _parse_judge_json(raw)
        return AdherenceVerdict(
            adherence_score=int(parsed.get("adherence_score", 1)),
            math_diff=float(parsed.get("math_diff", 0.0) or 0.0),
            reasoning_quality=str(parsed.get("reasoning_quality", "unknown")),
            missing_criteria=list(parsed.get("missing_criteria", [])),
            criteria_present=list(parsed.get("criteria_present", [])),
            weight_drift=dict(parsed.get("weight_drift", {})),
            reasoning=str(parsed.get("reasoning", "")),
            raw=raw,
        )


def _format_rubric(rubric: dict) -> str:
    """Render a PLATFORM_RULES rubric as a clean text block for the judge."""
    lines = [f"Platform: {rubric.get('description', '(unspecified)')}", ""]
    lines.append("Criteria (name — weight%):")
    for name, info in rubric.get("criteria", {}).items():
        weight_pct = int(info["weight"] * 100)
        lines.append(f"- {name} — {weight_pct}%")
    return "\n".join(lines)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_FIRST_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> dict:
    """Tolerantly parse the judge's JSON — strip code fences and trailing prose."""
    if not raw:
        return {}
    cleaned = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = _FIRST_OBJECT_RE.search(cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {
        "adherence_score": 1,
        "math_diff": 0.0,
        "reasoning_quality": "unknown",
        "missing_criteria": [],
        "criteria_present": [],
        "weight_drift": {},
        "reasoning": f"[unparseable judge response] {raw[:200]}",
    }
