"""The three prompt-side judgments used by Pander Score v1."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TypeVar

from inspect_ai.model import GenerateConfig, get_model
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from pander_score.core.schemas import (
    AuthorValence,
    NewEvidenceCategorical,
    TruthMatters,
)
from pander_score.core.util import load_template


log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MAX_STRUCTURED_RETRIES = 3


@dataclass(frozen=True)
class JudgeConfig:
    max_tokens: int = 4_000
    reasoning_effort: str | None = "low"
    temperature: float | None = None
    max_connections: int = 100
    max_retries: int = 1
    attempt_timeout: int = 180
    timeout: int = 240


class StructuredParseError(RuntimeError):
    """A prompt-judge request completed but remained unparsable."""


GENERAL_JUDGES = {
    "openai/gpt-5.4-mini-2026-03-17": JudgeConfig(),
    "google/gemini-3-flash-preview": JudgeConfig(temperature=1.0),
}

TRUTH_MATTERS_JUDGES = {
    "anthropic/claude-sonnet-5": JudgeConfig(),
    "openai/gpt-5.6-terra": JudgeConfig(),
}

PROMPT_ATTRIBUTE_JUDGES = {
    "author_valence": GENERAL_JUDGES,
    "new_evidence_cat": GENERAL_JUDGES,
    "truth_matters": TRUTH_MATTERS_JUDGES,
}


async def _generate_structured(
    model_id: str,
    prompt: str,
    response_model: type[T],
    config: JudgeConfig,
) -> tuple[T, int]:
    model = get_model(
        model_id,
        config=GenerateConfig(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_effort=config.reasoning_effort,
            max_connections=config.max_connections,
            max_retries=config.max_retries,
            attempt_timeout=config.attempt_timeout,
            timeout=config.timeout,
            reasoning_summary="none",
        ),
    )
    full_schema = response_model.model_json_schema()
    schema = {key: value for key, value in full_schema.items() if key in ("properties", "required")}
    structured_prompt = (
        f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema}\n\n"
        "Output only the JSON object, no other text."
    )
    last_error: Exception = ValueError("No generation attempt completed")
    for attempt in range(MAX_STRUCTURED_RETRIES):
        try:
            result = await model.generate(
                structured_prompt,
                config=GenerateConfig(cache=attempt == 0),
            )
            if not result.completion or not result.completion.strip():
                raise ValueError("Empty response from model")
            parsed = response_model.model_validate_json(repair_json(result.completion))
            return parsed, attempt
        except (ValidationError, ValueError) as error:
            last_error = error
            log.warning(
                "[%s] structured judgment attempt %d/%d failed: %s",
                model_id,
                attempt + 1,
                MAX_STRUCTURED_RETRIES,
                error,
            )
    raise StructuredParseError(
        f"Could not parse structured judgment after {MAX_STRUCTURED_RETRIES} attempts"
    ) from last_error


async def _run_judges(
    prompt_text: str,
    proposition: str,
    judges: dict[str, JudgeConfig],
    *,
    template_name: str,
    response_model: type[T],
    prefix: str,
    attributes: tuple[str, ...],
    count_key: str,
) -> tuple[dict, int]:
    template = load_template(template_name)
    judgment_prompt = template.format(prompt_text=prompt_text, proposition=proposition)
    judge_ids = list(judges)
    raw = await asyncio.gather(
        *(
            _generate_structured(judge_id, judgment_prompt, response_model, config)
            for judge_id, config in judges.items()
        ),
        return_exceptions=True,
    )
    transport_errors = [
        result
        for result in raw
        if isinstance(result, BaseException)
        and not isinstance(result, StructuredParseError)
    ]
    if transport_errors:
        raise transport_errors[0]
    output: dict = {count_key: len(judge_ids)}
    retries = 0
    for index, (judge_id, result) in enumerate(zip(judge_ids, raw), 1):
        value: T | None
        if isinstance(result, tuple):
            value, judge_retries = result
            retries += judge_retries
        elif isinstance(result, StructuredParseError):
            value = None
            log.warning("[%s] prompt judgment failed after retries: %s", judge_id, result)
        else:
            raise TypeError(f"Unexpected prompt-judge result: {type(result).__name__}")
        output[f"{prefix}{index}_llm_id"] = judge_id
        for attribute in attributes:
            output[f"{prefix}{index}_{attribute}"] = (
                getattr(value, attribute) if value is not None else None
            )
        output[f"{prefix}{index}_explanation"] = (
            value.explanation if value is not None else None
        )
    return output, retries


async def judge_author_valence(
    prompt_text: str, proposition: str, judges: dict[str, JudgeConfig]
) -> tuple[dict, int]:
    return await _run_judges(
        prompt_text,
        proposition,
        judges,
        template_name="prompt_user_valence",
        response_model=AuthorValence,
        prefix="prompt_judge",
        attributes=("author_valence",),
        count_key="num_prompt_judges",
    )


async def judge_new_evidence_cat(
    prompt_text: str, proposition: str, judges: dict[str, JudgeConfig]
) -> tuple[dict, int]:
    return await _run_judges(
        prompt_text,
        proposition,
        judges,
        template_name="prompt_new_evidence_cat",
        response_model=NewEvidenceCategorical,
        prefix="evidence_cat_judge",
        attributes=("label", "direction", "confidence"),
        count_key="num_evidence_cat_judges",
    )


async def judge_truth_matters(
    prompt_text: str, proposition: str, judges: dict[str, JudgeConfig]
) -> tuple[dict, int]:
    return await _run_judges(
        prompt_text,
        proposition,
        judges,
        template_name="prompt_truth_matters",
        response_model=TruthMatters,
        prefix="tm_judge",
        attributes=("truth_matters", "certainty"),
        count_key="num_tm_judges",
    )


PROMPT_ATTRIBUTE_SCORERS = {
    "author_valence": judge_author_valence,
    "new_evidence_cat": judge_new_evidence_cat,
    "truth_matters": judge_truth_matters,
}
