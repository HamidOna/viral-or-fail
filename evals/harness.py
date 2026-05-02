"""Tiny in-house evaluation harness for the Viral or Fail eval suite.

Why this file exists
--------------------
Microsoft Agent Framework's evaluation surface (``evaluate_agent``,
``LocalEvaluator``, ``@evaluator``, ``EvalItem``, ``EvalResults``) pairs
most natively with Azure AI Foundry. To stay on the same free-tier
footing as the rest of Viral or Fail (GitHub Models + ``OpenAIChatClient``),
we built this small harness using the same conceptual primitives.

What you get on Azure for free, you can build for yourself in ~150 lines
on GitHub Models. The patterns transfer directly when you upgrade.

Public API mirrors ``evaluate_agent``::

    runner = EvalRunner()
    results = await runner.run(
        agent=algorithm_simulator,
        queries=[post["content"] for post in posts],
        evaluators=[my_evaluator],
        expected_output=[str(post["engagement_score"]) for post in posts],
        num_repetitions=10,
    )

If you upgrade to Azure later, swap ``runner.run(...)`` for
``evaluate_agent(...)`` and the rest of the codebase is unchanged.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

# ── SDK primitives with graceful fallback ───────────────────────────────────
# We try to use the SDK's data types and decorator if they import cleanly.
# In agent-framework-core==1.0.0rc1 they don't exist yet, so we fall back to
# locally defined equivalents with the same shape. This keeps the upgrade
# path obvious.
_USING_SDK_PRIMITIVES = False
try:
    from agent_framework import EvalItem as _SDKEvalItem  # type: ignore
    from agent_framework import EvalResults as _SDKEvalResults  # type: ignore
    from agent_framework import evaluator as _sdk_evaluator  # type: ignore

    EvalItem = _SDKEvalItem  # noqa: F401  re-export
    EvalResults = _SDKEvalResults  # noqa: F401  re-export
    evaluator = _sdk_evaluator  # noqa: F401  re-export
    _USING_SDK_PRIMITIVES = True
except ImportError:

    @dataclass
    class EvalItem:
        """One agent run + its evaluator scores. Shape matches the MAF type."""

        query: str
        response: str
        expected_output: str | None = None
        scores: dict[str, float] = field(default_factory=dict)
        repetition: int = 0

    @dataclass
    class EvalResults:
        """Aggregate of EvalItems across queries and repetitions."""

        items: list[EvalItem] = field(default_factory=list)

        def per_evaluator(self) -> dict[str, list[float]]:
            """Group scores by evaluator name, in run order."""
            agg: dict[str, list[float]] = {}
            for item in self.items:
                for name, value in item.scores.items():
                    agg.setdefault(name, []).append(value)
            return agg

        def by_query(self) -> dict[str, list[EvalItem]]:
            """Group items by query string. Useful for repetition analysis."""
            grouped: dict[str, list[EvalItem]] = {}
            for item in self.items:
                grouped.setdefault(item.query, []).append(item)
            return grouped

    def evaluator(fn: Callable[..., float]) -> Callable[..., float]:
        """Mark a function as an evaluator. Mirrors agent_framework.evaluator.

        Supported parameter names (the runner only passes the ones the
        function asks for): ``query``, ``response``, ``expected_output``.
        """

        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> float:
            return fn(*args, **kwargs)

        wrapper._is_evaluator = True  # type: ignore[attr-defined]
        wrapper._evaluator_name = fn.__name__  # type: ignore[attr-defined]
        return wrapper


# ── Runner ──────────────────────────────────────────────────────────────────


class EvalRunner:
    """Async harness that calls an agent and applies evaluators per response.

    Args:
        rate_limit_sleep: Seconds to sleep between agent calls. GitHub Models'
            free tier limits this model to ~15 requests/minute, so 4.5s is
            the safe default (12 RPM, well under the cap).
        max_retries: Retry count for transient failures.
        rate_limit_wait_seconds: Base wait when a 429/"too many requests"
            error is detected; multiplied by attempt number. The per-minute
            window resets in 60s so 30s × N gives us a buffer.
    """

    def __init__(
        self,
        *,
        rate_limit_sleep: float = 4.5,
        max_retries: int = 5,
        rate_limit_wait_seconds: float = 30.0,
    ) -> None:
        self.rate_limit_sleep = rate_limit_sleep
        self.max_retries = max_retries
        self.rate_limit_wait_seconds = rate_limit_wait_seconds

    async def run(
        self,
        *,
        agent: Any,
        queries: list[str],
        evaluators: list[Callable[..., float]],
        expected_output: list[str] | None = None,
        num_repetitions: int = 1,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> EvalResults:
        """Run ``agent`` on each query ``num_repetitions`` times and score each response.

        Args:
            agent: An ``agent_framework.Agent`` (or anything with an async
                ``run(query) -> result.text`` interface).
            queries: List of input strings to feed the agent.
            evaluators: List of ``@evaluator``-decorated functions.
            expected_output: Optional ground-truth string per query
                (paired by index). Passed to evaluators that ask for it.
            num_repetitions: How many times to call the agent per query.
                Use >1 for consistency tests.
            progress: Optional callback ``(done, total, query_preview)`` for
                live progress reporting.
        """
        if expected_output is not None and len(expected_output) != len(queries):
            raise ValueError("expected_output must match queries length")

        total = len(queries) * num_repetitions
        done = 0
        items: list[EvalItem] = []

        for q_idx, query in enumerate(queries):
            expected = expected_output[q_idx] if expected_output else None
            for rep in range(num_repetitions):
                response = await self._call_agent_with_retry(agent, query)
                scores = await self._apply_evaluators(
                    evaluators, query=query, response=response, expected_output=expected
                )
                items.append(
                    EvalItem(
                        query=query,
                        response=response,
                        expected_output=expected,
                        scores=scores,
                        repetition=rep,
                    )
                )
                done += 1
                if progress is not None:
                    progress(done, total, query[:60])
                if self.rate_limit_sleep > 0:
                    await asyncio.sleep(self.rate_limit_sleep)

        return EvalResults(items=items)

    async def _call_agent_with_retry(self, agent: Any, query: str) -> str:
        """Delegate to the module-level helper so manual orchestration in
        Tests 3/4 (which don't go through EvalRunner) gets the same retry."""
        return await call_agent_with_retry(
            agent,
            query,
            max_retries=self.max_retries,
            rate_limit_wait_seconds=self.rate_limit_wait_seconds,
        )

    @staticmethod
    async def _apply_evaluators(
        evaluators: list[Callable[..., float]],
        *,
        query: str,
        response: str,
        expected_output: str | None,
    ) -> dict[str, float]:
        """Pass only the parameters each evaluator asks for. Mirrors MAF's signature dispatch."""
        scores: dict[str, float] = {}
        for ev in evaluators:
            sig = inspect.signature(ev)
            kwargs: dict[str, Any] = {}
            if "query" in sig.parameters:
                kwargs["query"] = query
            if "response" in sig.parameters:
                kwargs["response"] = response
            if "expected_output" in sig.parameters:
                kwargs["expected_output"] = expected_output
            value = ev(**kwargs)
            if asyncio.iscoroutine(value):
                value = await value
            name = getattr(ev, "_evaluator_name", ev.__name__)
            scores[name] = float(value)
        return scores


# ── Module-level retry helper ───────────────────────────────────────────────
# Test 3 and Test 4 orchestrate multi-agent chains by hand instead of going
# through EvalRunner.run, so they need this directly. EvalRunner internally
# delegates to it as well — single source of truth for retry behaviour.


async def call_agent_with_retry(
    agent: Any,
    query: str,
    *,
    max_retries: int = 5,
    rate_limit_wait_seconds: float = 30.0,
) -> str:
    """Call ``agent.run(query)`` with adaptive backoff.

    Distinguishes rate-limit errors (long wait — per-minute window needs to
    reset) from generic transient errors (short exponential backoff). On
    rate-limit detection, prints a status line to stderr so the user knows
    why the run is paused.
    """
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            result = await agent.run(query)
            return result.text or ""
        except Exception as err:  # noqa: BLE001 — retry on anything transient
            last_err = err
            err_str = str(err).lower()
            is_rate_limit = (
                "too many requests" in err_str
                or "rate limit" in err_str
                or "ratelimit" in err_str
                or "429" in err_str
            )
            if is_rate_limit:
                wait = rate_limit_wait_seconds * (attempt + 1)
                sys.stderr.write(
                    f"\n  [rate-limited, waiting {wait:.0f}s before retry "
                    f"{attempt + 1}/{max_retries}]\n"
                )
                sys.stderr.flush()
            else:
                wait = 2**attempt
            await asyncio.sleep(wait)
    raise RuntimeError(f"agent.run failed after {max_retries} retries: {last_err}")


__all__ = [
    "EvalItem",
    "EvalResults",
    "EvalRunner",
    "call_agent_with_retry",
    "evaluator",
]
