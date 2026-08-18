from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from inspect_ai.model import GenerateConfig

from pander_score.core.credence_judge import parse_judgment, run_judges


def test_parse_judgment_repairs_json_and_accepts_known_misspelling() -> None:
    judgment = parse_judgment(
        "{'refusal': false, 'informative': true, 'credance': 0.72, 'explanation': 'clear'}"
    )
    assert judgment.credence == 0.72


def test_parse_judgment_rejects_out_of_range_credence() -> None:
    with pytest.raises(ValueError, match="out of range"):
        parse_judgment(
            '{"refusal": false, "informative": true, "credence": 1.2, "explanation": "bad"}'
        )


class _FakeModel:
    name = "fake/judge"

    def __init__(
        self,
        completion: str | None = None,
        error: Exception | None = None,
    ):
        self.completion = completion
        self.error = error
        self.calls = 0

    async def generate(self, prompt: str, config: GenerateConfig):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(completion=self.completion)


def test_terminal_parse_failure_becomes_a_missing_judge_slot() -> None:
    bad = _FakeModel(completion="not valid judgment JSON")
    good = _FakeModel(
        completion=(
            '{"refusal": false, "informative": true, "credence": 0.7, '
            '"explanation": "clear"}'
        )
    )

    results = asyncio.run(
        run_judges([bad, good], "prompt", GenerateConfig(cache=True))
    )

    assert results[0] is None
    assert results[1] is not None and results[1].judgment.credence == 0.7
    assert bad.calls == 3


def test_transport_failure_remains_retryable() -> None:
    failing = _FakeModel(error=ConnectionError("provider unavailable"))

    with pytest.raises(ConnectionError, match="provider unavailable"):
        asyncio.run(run_judges([failing], "prompt", GenerateConfig(cache=True)))
