"""The frozen prompt-generation protocol used for Pander Score v1."""

from __future__ import annotations

import json
import logging
from typing import Literal, TypeVar

from inspect_ai.model import GenerateConfig
from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from pander_score.core.util import load_template


log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
MAX_STRUCTURED_RETRIES = 3
MAX_ELICITOR_RETRIES = 3


class GeneratedImplicitPrompt(BaseModel):
    prior: Literal["skeptic", "neutral", "believer"]
    user_role: str
    primary_angle: str
    length: Literal["short", "medium", "long"]
    tone: Literal["formal", "casual", "unpolished"]
    is_artifact: bool
    prompt: str


class GeneratedImplicitPrompts(BaseModel):
    prompts: list[GeneratedImplicitPrompt]


def _compute_implicit_params(num_prompts: int) -> dict:
    n_skeptical = num_prompts // 3
    n_believing = num_prompts // 3
    return {
        "num_prompts": num_prompts,
        "n_skeptical": n_skeptical,
        "n_neutral": num_prompts - n_skeptical - n_believing,
        "n_believing": n_believing,
        "min_primary_angles": min(12, max(3, (num_prompts + 1) // 2)),
        "min_user_roles": min(10, max(2, (num_prompts + 2) // 3)),
        "min_short": max(1, num_prompts // 4),
        "min_long": max(1, num_prompts // 4),
        "min_unpolished": max(1, num_prompts // 4),
        "min_artifact": max(1, num_prompts // 4),
        "max_factual": max(1, num_prompts // 8),
        "max_both_outcomes": max(1, num_prompts // 8),
        "max_explicit_prob": max(1, num_prompts // 4),
    }


def _validate_implicit_prompts(
    generated: GeneratedImplicitPrompts,
    params: dict,
    tolerance: int,
) -> list[str]:
    violations: list[str] = []
    prompts = generated.prompts
    expected = params["num_prompts"]
    if abs(len(prompts) - expected) > tolerance:
        violations.append(
            f"num_prompts: got {len(prompts)}, expected {expected} (±{tolerance})"
        )

    for value, key in (
        ("skeptic", "n_skeptical"),
        ("neutral", "n_neutral"),
        ("believer", "n_believing"),
    ):
        count = sum(prompt.prior == value for prompt in prompts)
        if abs(count - params[key]) > tolerance:
            violations.append(
                f"{value}: got {count}, expected {params[key]} (±{tolerance})"
            )

    for value, key in (("short", "min_short"), ("long", "min_long")):
        count = sum(prompt.length == value for prompt in prompts)
        if count < params[key] - tolerance:
            violations.append(f"length={value}: got {count}, expected >={params[key]}")

    unpolished = sum(prompt.tone == "unpolished" for prompt in prompts)
    if unpolished < params["min_unpolished"] - tolerance:
        violations.append(
            f"tone=unpolished: got {unpolished}, expected >={params['min_unpolished']}"
        )

    artifacts = sum(prompt.is_artifact for prompt in prompts)
    if artifacts < params["min_artifact"] - tolerance:
        violations.append(
            f"is_artifact=true: got {artifacts}, expected >={params['min_artifact']}"
        )

    distinct_angles = len({prompt.primary_angle for prompt in prompts})
    if distinct_angles < params["min_primary_angles"] - tolerance:
        violations.append(
            f"distinct angles: got {distinct_angles}, expected >={params['min_primary_angles']}"
        )

    distinct_roles = len({prompt.user_role for prompt in prompts})
    if distinct_roles < params["min_user_roles"] - tolerance:
        violations.append(
            f"distinct roles: got {distinct_roles}, expected >={params['min_user_roles']}"
        )
    return violations


def _build_implicit_prompt(proposition: str, params: dict) -> str:
    return load_template("elicitor_preamble") + "\n\n" + load_template(
        "elicitor_implicit"
    ).format(proposition=proposition, **params)


async def _generate_structured(
    model: object,
    model_id: str,
    prompt: str,
    response_model: type[T],
) -> tuple[T, int]:
    schema = {
        key: value
        for key, value in response_model.model_json_schema().items()
        if key in ("properties", "required")
    }
    structured_prompt = (
        f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema}\n\n"
        "Output only the JSON object, no other text."
    )
    last_error: Exception | None = None
    for attempt in range(MAX_STRUCTURED_RETRIES):
        result = await model.generate(structured_prompt)
        if not result.completion or not result.completion.strip():
            last_error = ValueError("Empty response from model")
            log.warning("[%s] empty structured response on attempt %d", model_id, attempt + 1)
            continue
        try:
            parsed = response_model.model_validate_json(repair_json(result.completion))
            return parsed, attempt
        except ValidationError as error:
            last_error = error
            log.warning("[%s] invalid structured response on attempt %d", model_id, attempt + 1)
    assert last_error is not None
    raise last_error


async def generate_implicit_prompts(
    proposition: str,
    generator_llm_id: str,
    model: object,
    num_prompts: int = 16,
) -> tuple[list[GeneratedImplicitPrompt], int]:
    """Generate one validated proposition/elicitor unit and report retry count."""
    params = _compute_implicit_params(num_prompts)
    prompt = _build_implicit_prompt(proposition, params)
    retries = 0
    last_violations: list[str] | None = None
    for attempt in range(MAX_ELICITOR_RETRIES):
        generated, structured_retries = await _generate_structured(
            model,
            generator_llm_id,
            prompt,
            GeneratedImplicitPrompts,
        )
        retries += structured_retries
        violations = _validate_implicit_prompts(generated, params, tolerance=1)
        if not violations:
            return generated.prompts, retries
        last_violations = violations
        retries += 1
        log.warning(
            "[%s] prompt diversity validation failed on attempt %d: %s",
            generator_llm_id,
            attempt + 1,
            violations,
        )
    raise RuntimeError(
        f"Prompt generation failed validation after {MAX_ELICITOR_RETRIES} "
        f"attempts: {last_violations}"
    )


ELICITOR_CONFIGS = {
    "openai/gpt-5.4-mini-2026-03-17": GenerateConfig(
        max_tokens=16_000,
        max_connections=100,
        reasoning_effort="medium",
        reasoning_summary="none",
        max_retries=1,
        attempt_timeout=300,
        timeout=600,
    ),
    "google/gemini-3-flash-preview": GenerateConfig(
        max_tokens=16_000,
        max_connections=100,
        reasoning_effort="medium",
        reasoning_summary="none",
        temperature=1.0,
        max_retries=1,
        attempt_timeout=300,
        timeout=600,
    ),
}
