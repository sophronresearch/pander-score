from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pander_score import prompt_grading
from pander_score.core.prompt_judge import JudgeConfig


def _write_prompts(path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps({"id": sample_id, "input": sample_id, "target": "p"}) + "\n"
            for sample_id in ("s1", "s2")
        ),
        encoding="utf-8",
    )


def test_prompt_grading_checkpoints_and_rerun_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "attributes"
    _write_prompts(prompts)
    calls: list[str] = []

    async def scorer(prompt: str, proposition: str, judges: dict) -> tuple[dict, int]:
        calls.append(prompt)
        return {"judge1_value": 0.8}, 1

    monkeypatch.setattr(prompt_grading, "PROMPT_ATTRIBUTE_SCORERS", {"author_valence": scorer})
    monkeypatch.setattr(
        prompt_grading,
        "PROMPT_ATTRIBUTE_JUDGES",
        {"author_valence": {"judge": JudgeConfig(max_connections=1)}},
    )

    asyncio.run(prompt_grading.grade_prompts(prompts, output))
    asyncio.run(prompt_grading.grade_prompts(prompts, output))

    assert sorted(calls) == ["s1", "s2"]
    rows = [json.loads(line) for line in (output / "author_valence.jsonl").read_text().splitlines()]
    assert {row["sample_id"] for row in rows} == {"s1", "s2"}
    assert all(row["_complete"] for row in rows)


def test_prompt_grading_repairs_interrupted_final_append(tmp_path: Path) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_bytes(b'{"sample_id":"s1","_complete":true}\n{"sample_id":')

    rows = prompt_grading._read_repairable_jsonl(path)

    assert rows == [{"sample_id": "s1", "_complete": True}]
    assert path.read_bytes() == b'{"sample_id":"s1","_complete":true}\n'


def test_terminal_parse_failure_is_checkpointed_as_null(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "attributes"
    _write_prompts(prompts)
    calls: list[str] = []

    async def scorer(prompt: str, proposition: str, judges: dict) -> tuple[dict, int]:
        calls.append(prompt)
        return {"judge1_value": None}, 2

    monkeypatch.setattr(
        prompt_grading,
        "PROMPT_ATTRIBUTE_SCORERS",
        {"author_valence": scorer},
    )
    monkeypatch.setattr(
        prompt_grading,
        "PROMPT_ATTRIBUTE_JUDGES",
        {"author_valence": {"judge": JudgeConfig(max_connections=1)}},
    )

    asyncio.run(prompt_grading.grade_prompts(prompts, output))
    asyncio.run(prompt_grading.grade_prompts(prompts, output))

    assert sorted(calls) == ["s1", "s2"]
    rows = [
        json.loads(line)
        for line in (output / "author_valence.jsonl").read_text().splitlines()
    ]
    assert all(
        row["_complete"] is True and row["judge1_value"] is None
        for row in rows
    )


def test_prompt_grading_refuses_source_drift(tmp_path: Path, monkeypatch) -> None:
    prompts = tmp_path / "prompts.jsonl"
    output = tmp_path / "attributes"
    _write_prompts(prompts)

    async def scorer(prompt: str, proposition: str, judges: dict) -> tuple[dict, int]:
        return {"judge1_value": 0.8}, 0

    monkeypatch.setattr(prompt_grading, "PROMPT_ATTRIBUTE_SCORERS", {"author_valence": scorer})
    monkeypatch.setattr(
        prompt_grading,
        "PROMPT_ATTRIBUTE_JUDGES",
        {"author_valence": {"judge": JudgeConfig(max_connections=1)}},
    )
    asyncio.run(prompt_grading.grade_prompts(prompts, output))
    prompts.write_text(
        prompts.read_text(encoding="utf-8").replace('"input": "s1"', '"input": "changed"'),
        encoding="utf-8",
    )

    try:
        asyncio.run(prompt_grading.grade_prompts(prompts, output))
    except ValueError as error:
        assert "configuration differs" in str(error)
    else:
        raise AssertionError("changed prompt source was accepted")
