"""Resume-safe prompt generation over the fixed 349 public propositions."""

from __future__ import annotations

import asyncio
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from inspect_ai.model import get_model

from pander_score.core.elicitor import ELICITOR_CONFIGS, generate_implicit_prompts
from pander_score.data import BENCHMARK_PROPOSITIONS


@dataclass(frozen=True)
class Proposition:
    id: str
    text: str


def _content_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def load_fixed_propositions(path: Path) -> list[Proposition]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    propositions = [Proposition(row["id"], row["proposition"]) for row in rows]
    if len(propositions) != BENCHMARK_PROPOSITIONS:
        raise ValueError(
            f"Expected {BENCHMARK_PROPOSITIONS} propositions, found {len(propositions)}"
        )
    if len({proposition.id for proposition in propositions}) != BENCHMARK_PROPOSITIONS:
        raise ValueError("Fixed proposition IDs are not unique")
    return propositions


def _unit_prefix(proposition: Proposition, elicitor_id: str) -> str:
    return f"{_content_hash(proposition.text)}__{_content_hash(elicitor_id)}__run0"


def _checkpoint_path(checkpoint_dir: Path, prefix: str) -> Path:
    return checkpoint_dir / f"{prefix}.json"


def _samples_for_unit(
    proposition: Proposition,
    elicitor_id: str,
    prompts: list,
) -> list[dict]:
    prefix = _unit_prefix(proposition, elicitor_id)
    return [
        {
            "input": prompt.prompt,
            "id": f"{prefix}__{index:03d}",
            "target": proposition.text,
            "metadata": {
                "proposition": proposition.text,
                "proposition_id": proposition.id,
                "run_id": 0,
                "generator_llm_id": elicitor_id,
                "prompter_type": "implicit",
                "prior": prompt.prior,
                "user_role": prompt.user_role,
                "primary_angle": prompt.primary_angle,
                "prompt_shape": None,
                "length": prompt.length,
                "tone": prompt.tone,
                "is_artifact": prompt.is_artifact,
            },
        }
        for index, prompt in enumerate(prompts)
    ]


def _write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _generation_manifest(propositions_path: Path) -> dict:
    return {
        "benchmark_version": "v1",
        "propositions_sha256": hashlib.sha256(propositions_path.read_bytes()).hexdigest(),
        "proposition_count": BENCHMARK_PROPOSITIONS,
        "prompts_per_elicitor_per_proposition": 16,
        "elicitors": {
            model_id: {
                "reasoning_effort": config.reasoning_effort,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            }
            for model_id, config in ELICITOR_CONFIGS.items()
        },
        "templates": {
            name: hashlib.sha256(
                (Path(__file__).parent / "core/templates" / f"{name}.txt").read_bytes()
            ).hexdigest()
            for name in ("elicitor_preamble", "elicitor_implicit")
        },
    }


def _read_checkpoint(path: Path, proposition_id: str, elicitor_id: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("proposition_id") != proposition_id or payload.get("elicitor_id") != elicitor_id:
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Checkpoint has no samples: {path}")
    return samples


def _assemble(
    propositions: list[Proposition],
    output_dir: Path,
    *,
    complete: bool,
) -> tuple[Path, int]:
    checkpoint_dir = output_dir / "checkpoints"
    samples: list[dict] = []
    for proposition in propositions:
        for elicitor_id in ELICITOR_CONFIGS:
            checkpoint = _checkpoint_path(
                checkpoint_dir,
                _unit_prefix(proposition, elicitor_id),
            )
            if checkpoint.exists():
                samples.extend(_read_checkpoint(checkpoint, proposition.id, elicitor_id))

    destination = output_dir / ("prompts.jsonl" if complete else "prompts.partial.jsonl")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        "".join(
            json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
            for sample in samples
        ),
        encoding="utf-8",
    )
    temporary.replace(destination)
    if complete:
        partial = output_dir / "prompts.partial.jsonl"
        if partial.exists():
            partial.unlink()
    return destination, len(samples)


async def generate_fixed_prompts(
    propositions_path: Path,
    output_dir: Path,
) -> tuple[Path, int, int]:
    """Generate the original two-elicitor protocol with atomic unit checkpoints."""
    propositions = load_fixed_propositions(propositions_path)
    manifest = _generation_manifest(propositions_path)
    manifest_path = output_dir / "generation_config.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                f"Existing generation configuration differs: {manifest_path}. "
                "Use a fresh output directory rather than mixing conditions."
            )
    else:
        _write_checkpoint(manifest_path, manifest)
    checkpoint_dir = output_dir / "checkpoints"
    units = [
        (proposition, elicitor_id)
        for proposition in propositions
        for elicitor_id in ELICITOR_CONFIGS
    ]
    total = len(units)
    pending = [
        unit
        for unit in units
        if not _checkpoint_path(checkpoint_dir, _unit_prefix(*unit)).exists()
    ]
    completed_before = total - len(pending)
    print(
        f"Prompt generation: {len(propositions)} propositions × "
        f"{len(ELICITOR_CONFIGS)} elicitors = {total} units",
        flush=True,
    )
    print(f"Completed: {completed_before}/{total}; pending: {len(pending)}", flush=True)

    models = {
        model_id: get_model(model_id, config=config)
        for model_id, config in ELICITOR_CONFIGS.items()
    }
    semaphore = asyncio.Semaphore(100)

    async def run_unit(proposition: Proposition, elicitor_id: str) -> int:
        async with semaphore:
            prompts, retries = await generate_implicit_prompts(
                proposition.text,
                elicitor_id,
                models[elicitor_id],
            )
        samples = _samples_for_unit(proposition, elicitor_id, prompts)
        checkpoint = _checkpoint_path(
            checkpoint_dir,
            _unit_prefix(proposition, elicitor_id),
        )
        _write_checkpoint(
            checkpoint,
            {
                "proposition_id": proposition.id,
                "elicitor_id": elicitor_id,
                "samples": samples,
            },
        )
        return retries

    tasks = [asyncio.create_task(run_unit(*unit)) for unit in pending]
    successes = completed_before
    retries = 0
    failures: list[Exception] = []
    for task in asyncio.as_completed(tasks):
        try:
            retries += await task
            successes += 1
        except Exception as error:
            failures.append(error)
        finished = successes + len(failures)
        if finished % 10 == 0 or finished == total or failures:
            print(
                f"Progress: {finished}/{total}; successes={successes}; "
                f"retries={retries}; failures={len(failures)}",
                flush=True,
            )

    complete = successes == total and not failures
    path, sample_count = _assemble(propositions, output_dir, complete=complete)
    if failures:
        raise RuntimeError(
            f"{len(failures)} generation units failed; completed checkpoints were "
            f"preserved. First error: {failures[0]}"
        )
    return path, sample_count, retries
