"""Inspect scorer for the frozen two-judge credence protocol."""

from __future__ import annotations

import numpy as np
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Metric, Score, Target, metric, scorer
from inspect_ai.scorer._metric import SampleScore
from inspect_ai.solver import TaskState

from pander_score.core.credence_judge import (
    build_score_metadata,
    extract_interaction_log,
    format_judge_prompt,
    mean_informative_credence,
    run_judges,
)
from pander_score.core.schemas import JudgeResult


CREDENCE_JUDGES = {
    "openai/gpt-5.4-mini-2026-03-17": GenerateConfig(
        max_tokens=4_000,
        reasoning_effort="low",
        max_connections=100,
        max_retries=1,
        attempt_timeout=180,
        timeout=240,
    ),
    "google/gemini-3-flash-preview": GenerateConfig(
        max_tokens=4_000,
        reasoning_effort="low",
        temperature=1.0,
        max_connections=100,
        max_retries=1,
        attempt_timeout=180,
        timeout=240,
    ),
}
_JUDGE_CALL_CONFIG = GenerateConfig(reasoning_summary="none", cache=True)


def _numeric_values(scores: list[SampleScore]) -> list[float]:
    return [
        float(item.score.value)
        for item in scores
        if isinstance(item.score.value, (int, float))
        and not isinstance(item.score.value, bool)
    ]


@metric
def numeric_mean() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        values = _numeric_values(scores)
        return float(np.mean(values).item()) if values else 0.0

    return calculate


@metric
def numeric_stderr() -> Metric:
    def calculate(scores: list[SampleScore]) -> float:
        values = _numeric_values(scores)
        if len(values) < 2:
            return 0.0
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    return calculate


def _empty_response_score(template_name: str) -> Score:
    """Represent a completed empty target response as an auditable N/A row."""
    return Score(
        value="N/A",
        answer="N/A",
        explanation="Empty response from target model",
        metadata={"error": "empty_response", "template_name": template_name},
    )


@scorer(metrics=[numeric_mean(), numeric_stderr()])
def credence_scorer(template_name: str = "implicit_judge"):
    judge_ids = list(CREDENCE_JUDGES)
    judge_models = [
        get_model(model_id, config=config)
        for model_id, config in CREDENCE_JUDGES.items()
    ]

    async def score(state: TaskState, target: Target) -> Score:
        if not state.output or not state.output.completion:
            return _empty_response_score(template_name)

        prompt = format_judge_prompt(
            template_name,
            target.text,
            extract_interaction_log(state),
        )
        outputs = await run_judges(judge_models, prompt, _JUDGE_CALL_CONFIG)
        mean_credence = mean_informative_credence(
            [output.judgment for output in outputs if output is not None]
        )
        judge_results = [
            JudgeResult(
                judge_id,
                output.judgment if output is not None else None,
                output.raw_completion if output is not None else None,
            )
            for judge_id, output in zip(judge_ids, outputs)
        ]
        metadata = build_score_metadata(judge_results, prompt, template_name)
        value = mean_credence if mean_credence is not None else "N/A"
        return Score(
            value=value,
            answer=str(value),
            explanation=f"Judge credences: {[result.judgment.credence if result is not None else None for result in outputs]}",
            metadata=metadata,
        )

    return score
