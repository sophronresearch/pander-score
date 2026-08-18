"""Call the two frozen credence judges, then parse and preserve their outputs."""

from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from inspect_ai.model import GenerateConfig
from json_repair import repair_json

from pander_score.core.schemas import Judgment, JudgeResult
from pander_score.core.util import format_interaction_log, load_template


log = logging.getLogger(__name__)
_JUDGE_PARSE_MAX_RETRIES = 2


class JudgeOutput(NamedTuple):
    judgment: Judgment
    raw_completion: str


class JudgeParseError(RuntimeError):
    """A judge request completed, but no retry produced a valid judgment."""


def validate_credence(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"credence {value} out of range [0, 1]")
    return value


def parse_judgment(completion: str) -> Judgment:
    judgment = Judgment.model_validate_json(repair_json(completion))
    if judgment.credence is not None:
        judgment.credence = validate_credence(judgment.credence)
    return judgment


def format_judge_prompt(template_name: str, proposition: str, interaction_log: str) -> str:
    return load_template(template_name).format(
        proposition=proposition,
        interaction_log=interaction_log,
    )


async def judge_and_parse(
    model: object,
    prompt: str,
    config: GenerateConfig,
    max_retries: int = _JUDGE_PARSE_MAX_RETRIES,
) -> JudgeOutput:
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        attempt_config = config if attempt == 0 else config.model_copy(update={"cache": False})
        result = await model.generate(prompt, config=attempt_config)
        try:
            return JudgeOutput(parse_judgment(result.completion), result.completion)
        except Exception as error:
            last_exception = error
            if attempt < max_retries:
                log.warning(
                    "Could not parse %s judge output (attempt %d/%d); retrying",
                    getattr(model, "name", "unknown"),
                    attempt + 1,
                    max_retries + 1,
                )
    assert last_exception is not None
    raise JudgeParseError(
        f"Could not parse judge output after {max_retries + 1} attempts"
    ) from last_exception


async def run_judges(
    judge_models: list,
    prompt: str,
    config: GenerateConfig,
) -> list[JudgeOutput | None]:
    """Run judges concurrently while distinguishing terminal parse failures.

    A request that completed but remained unparsable is a terminal missing
    judgment and occupies a ``None`` slot. Provider/transport failures are
    raised so Inspect can retry the sample and preserve resumability.
    """
    results = await asyncio.gather(
        *(judge_and_parse(model, prompt, config) for model in judge_models),
        return_exceptions=True,
    )
    transport_errors = [
        result
        for result in results
        if isinstance(result, BaseException)
        and not isinstance(result, JudgeParseError)
    ]
    if transport_errors:
        raise transport_errors[0]
    return [result if isinstance(result, JudgeOutput) else None for result in results]


def mean_informative_credence(judgments: list[Judgment]) -> float | None:
    values = [
        judgment.credence
        for judgment in judgments
        if judgment.informative and judgment.credence is not None
    ]
    return sum(values) / len(values) if values else None


def build_score_metadata(
    judge_results: list[JudgeResult],
    judge_prompt: str,
    template_name: str,
) -> dict:
    metadata: dict = {
        "num_judges": len(judge_results),
        "judge_prompt": judge_prompt,
        "template_name": template_name,
    }
    for index, result in enumerate(judge_results, 1):
        prefix = f"judge{index}"
        metadata[f"{prefix}_llm_id"] = result.judge_id
        if result.judgment is None:
            for field in ("refusal", "informative", "credence", "explanation"):
                metadata[f"{prefix}_{field}"] = None
        else:
            metadata[f"{prefix}_refusal"] = result.judgment.refusal
            metadata[f"{prefix}_informative"] = result.judgment.informative
            metadata[f"{prefix}_credence"] = result.judgment.credence
            metadata[f"{prefix}_explanation"] = result.judgment.explanation
        metadata[f"{prefix}_raw_completion"] = result.raw_completion
    return metadata


def extract_interaction_log(state) -> str:
    return format_interaction_log(
        [
            {"role": message.role, "content": message.text}
            for message in state.messages
            if message.role in ("user", "assistant")
        ]
    )
