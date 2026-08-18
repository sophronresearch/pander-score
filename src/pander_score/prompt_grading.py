"""Incremental prompt grading with per-family JSONL checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from pander_score.core.prompt_judge import (
    PROMPT_ATTRIBUTE_JUDGES,
    PROMPT_ATTRIBUTE_SCORERS,
)
from pander_score.core.util import TEMPLATES_DIR


SCORER_TEMPLATES = {
    "author_valence": "prompt_user_valence",
    "new_evidence_cat": "prompt_new_evidence_cat",
    "truth_matters": "prompt_truth_matters",
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_repairable_jsonl(path: Path) -> list[dict]:
    """Read JSONL and discard only an invalid, unterminated final write."""
    if not path.exists():
        return []
    data = path.read_bytes()
    rows: list[dict] = []
    valid_bytes = 0
    for index, line in enumerate(data.splitlines(keepends=True)):
        terminated = line.endswith((b"\n", b"\r"))
        try:
            rows.append(json.loads(line))
            valid_bytes += len(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            is_last = valid_bytes + len(line) == len(data)
            if not is_last or terminated:
                raise ValueError(f"Malformed JSONL record {index + 1}: {path}") from error
            temporary = path.with_suffix(path.suffix + ".repair")
            temporary.write_bytes(data[:valid_bytes])
            temporary.replace(path)
            print(f"Recovered incomplete final checkpoint from {path}", flush=True)
    if data and not data.endswith(b"\n") and len(rows) == len(data.splitlines()):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    return rows


def _append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _grading_manifest(prompts_path: Path) -> dict:
    return {
        "benchmark_version": "v1",
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "judgments": {
            scorer: {
                "judges": {
                    judge_id: {
                        "reasoning_effort": config.reasoning_effort,
                        "temperature": config.temperature,
                        "max_tokens": config.max_tokens,
                    }
                    for judge_id, config in PROMPT_ATTRIBUTE_JUDGES[scorer].items()
                },
                "template_sha256": hashlib.sha256(
                    (TEMPLATES_DIR / f"{SCORER_TEMPLATES[scorer]}.txt").read_bytes()
                ).hexdigest(),
            }
            for scorer in PROMPT_ATTRIBUTE_SCORERS
        },
    }


def _completed_ids(path: Path, judge_ids: list[str]) -> set[str]:
    latest = {
        str(row["sample_id"]): row
        for row in _read_repairable_jsonl(path)
        if row.get("sample_id")
    }
    return {
        sample_id
        for sample_id, row in latest.items()
        if row.get("_judge_ids") == judge_ids and row.get("_complete") is True
    }


async def grade_prompts(prompts_path: Path, output_dir: Path) -> None:
    prompts = _read_repairable_jsonl(prompts_path)
    sample_ids = [str(row.get("id", "")) for row in prompts]
    if not prompts or any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"Prompt file is empty or contains a missing id: {prompts_path}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Prompt IDs are not unique: {prompts_path}")

    manifest = _grading_manifest(prompts_path)
    manifest_path = output_dir / "grading_config.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(
                f"Existing grading configuration differs: {manifest_path}. "
                "Use a fresh output directory rather than mixing conditions."
            )
    else:
        _atomic_json(manifest_path, manifest)

    total_failures = 0
    for scorer_name, scorer in PROMPT_ATTRIBUTE_SCORERS.items():
        judges = PROMPT_ATTRIBUTE_JUDGES[scorer_name]
        judge_ids = list(judges)
        output_path = output_dir / f"{scorer_name}.jsonl"
        completed = _completed_ids(output_path, judge_ids)
        remaining = [row for row in prompts if str(row["id"]) not in completed]
        print(
            f"[{scorer_name}] completed={len(completed)}/{len(prompts)}; "
            f"remaining={len(remaining)}",
            flush=True,
        )
        if not remaining:
            continue

        semaphore = asyncio.Semaphore(max(config.max_connections for config in judges.values()))

        async def score_one(sample: dict) -> tuple[dict, int]:
            async with semaphore:
                values, retries = await scorer(
                    str(sample.get("input", "")),
                    str(sample.get("target", "")),
                    judges,
                )
            return {
                "sample_id": str(sample["id"]),
                "_judge_ids": judge_ids,
                # Returning from the scorer means every judge call reached a
                # terminal state. Unparsable outputs remain visible as nulls;
                # provider failures raise and do not create a checkpoint.
                "_complete": True,
                **values,
            }, retries

        tasks = [asyncio.create_task(score_one(sample)) for sample in remaining]
        successes = 0
        failures = 0
        retries = 0
        for task in asyncio.as_completed(tasks):
            try:
                row, row_retries = await task
                retries += row_retries
                if row["_complete"]:
                    _append_checkpoint(output_path, row)
                    successes += 1
                else:
                    failures += 1
            except Exception as error:
                failures += 1
                print(f"[{scorer_name}] judgment failed: {error}", flush=True)
            processed = successes + failures
            if processed % 25 == 0 or processed == len(remaining):
                print(
                    f"[{scorer_name}] progress={len(completed) + successes}/{len(prompts)} "
                    f"successes={successes} retries={retries} failures={failures}",
                    flush=True,
                )
        total_failures += failures

    if total_failures:
        raise RuntimeError(
            f"{total_failures} prompt judgments failed; completed checkpoints were preserved"
        )
