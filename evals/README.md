# Viral or Fail — Evaluation Suite

A four-test suite that pressure-tests the **Algorithm Simulator** agent from
[Viral or Fail](../README.md). Output backs the follow-up blog post
*"Evaluating the Evaluator: How to Test an LLM Judge with Microsoft Agent Framework"*.

## 30-second pitch

Post 1 shipped a multi-agent system where one agent role-plays as a
recommendation algorithm and scores gaming content out of 100. That post
ended with a fair question: *how do we know the Simulator itself is any good?*
This suite answers it empirically. It runs four tests against the existing
agent, all on the free tier (GitHub Models — no Azure, no paid APIs):

1. **Consistency** — same input, same score, run after run?
2. **Calibration** — does the score correlate with real engagement?
3. **Rubric adherence** — does it actually follow its own rubric, or drift?
4. **Per-agent workflow eval** — when the whole pipeline runs, where is the weak link?

## How to run

```bash
# from the repo root, with your venv activated and GITHUB_TOKEN set
python -m evals.run_all
```

Each test is also runnable in isolation:

```bash
python -m evals.test_1_consistency
python -m evals.test_2_calibration
python -m evals.test_3_rubric_adherence
python -m evals.test_4_workflow
```

Outputs land in `eval_results/`:

```
eval_results/
├── test_1_consistency.json
├── test_2_calibration.json
├── test_3_rubric_adherence.json
├── test_4_workflow.json
└── plots/
    ├── 01_consistency.png
    ├── 02_calibration.png
    ├── 03_adherence.png
    └── 04_workflow.png
```

Total wall time is typically 6–8 minutes. The suite respects a 0.5s sleep
between calls to stay within GitHub Models' free-tier rate limits.

## What each test does

### Test 1 — Consistency (`test_1_consistency.py`)
Picks 5 posts spanning viral / decent / flop / outlier from the golden
dataset, calls the Simulator 10 times each with identical input, then
reports per-post mean, std, min, max, and coefficient of variation.

**Headline metric:** mean coefficient of variation across all posts.
A high CV means the Simulator is meaningfully non-deterministic at the
same temperature — a real problem for any test that depends on reproducible
scores.

### Test 2 — Calibration (`test_2_calibration.py`)
Runs the Simulator once per post on the full 10-post golden dataset, then
computes Pearson and Spearman correlations and MAE against each post's
labeled `engagement_score`.

**Headline metric:** Pearson r between Simulator score and real engagement.
This is the suite's most important empirical finding — if r is low, the
Simulator's scores can't be used as a directional signal at all.

### Test 3 — Rubric Adherence (`test_3_rubric_adherence.py`)
Custom LLM-as-judge built with another `OpenAIChatClient` pointed at
GitHub Models. The judge takes (rubric, post, evaluation output) and
returns a strict-JSON verdict: 1–5 adherence score, criteria covered,
criteria skipped, weight drift, and short reasoning.

The cloud-tier equivalent is roughly `FoundryEvals.TaskAdherence`. We
build the same pattern ourselves so the post can show readers what
those evaluators actually do under the hood.

**Headline metric:** mean adherence score and most-skipped criterion.

### Test 4 — Per-agent workflow eval (`test_4_workflow.py`)
Runs the existing 3-agent pipeline (Creator → Algorithm → Persona) on
5 trending topics, then evaluates each agent's output independently:

- **Creator** — keyword check for Twitter/X-native terminology
- **Algorithm** — structural check (emits a parseable WEIGHTED TOTAL)
- **Persona** — keyword check for the chosen persona's vocabulary

We deliberately keep the pipeline as application-controlled orchestration
(matching Post 1's framing) instead of refactoring it into a `Workflow`
graph. The per-agent breakdown is what surfaces weak links — exactly what
a `evaluate_workflow()` call would tell you on the cloud tier.

**Headline metric:** per-agent pass rates and overall pipeline pass rate.

## Why we built our own harness

The Microsoft Agent Framework's evaluation surface (`evaluate_agent`,
`LocalEvaluator`, `@evaluator`, `EvalItem`, `EvalResults`) is
provider-agnostic in principle but pairs most natively with Azure AI Foundry.
To stay on the same free-tier footing as Post 1 (GitHub Models +
`OpenAIChatClient`), we built [`harness.py`](harness.py) — a small
in-house `EvalRunner` that mirrors `evaluate_agent`'s call shape.

What you get on Azure for free, you can build for yourself in ~150 lines
on GitHub Models. The patterns transfer directly when you upgrade — only
the import line changes.

## What's next

This suite stops where the free tier stops. Two natural follow-ups for
the cloud-tier sequel post:

**Foundry-tier evaluators.** `FoundryEvals.TaskAdherence`, `Groundedness`,
and `RelevanceEvaluator` give you what we built in Test 3 (and a lot more)
without writing the judge yourself. The Foundry SDK also wires straight
into Azure AI projects so traces and scores live next to your model
deployments. The conceptual jump is small — same `@evaluator` surface,
same `EvalResults` shape — and our harness is structured so the swap is
mostly an import change.

**AI Red Teaming Agent.** The next layer up from "does the agent score
correctly?" is "can the agent be tricked into scoring incorrectly?" The
[AI Red Teaming Agent](https://learn.microsoft.com/en-us/azure/ai-foundry/concepts/ai-red-teaming-agent)
runs adversarial prompts through your agents at scale, classifies harms,
and reports which categories your system is weakest against. That's the
right tool when you take a project like this one beyond a tutorial — and
it's a clean third post for anyone following along.

## Files

```
evals/
├── README.md                    # this file
├── golden_dataset.json          # 10 labeled gaming posts
├── harness.py                   # in-house EvalRunner (~150 lines)
├── plot_style.py                # shared matplotlib styling
├── custom_evaluators.py         # @evaluator functions + score parsers
├── llm_judge.py                 # RubricAdherenceJudge for Test 3
├── test_1_consistency.py
├── test_2_calibration.py
├── test_3_rubric_adherence.py
├── test_4_workflow.py
└── run_all.py
```
