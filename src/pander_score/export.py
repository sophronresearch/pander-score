"""Stream completed Inspect logs into one compact per-model parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
from inspect_ai.log import EvalLog, EvalSample, read_eval_log, read_eval_log_samples

from pander_score.data import safe_model_filename


_DROP_TOPLEVEL = {"messages", "events", "store", "attachments", "timelines"}
_JUDGE_FIELDS = (
    "llm_id",
    "refusal",
    "informative",
    "credence",
    "explanation",
    "raw_completion",
)


def read_eval_header(path: Path) -> EvalLog:
    """Read a log through Inspect's format-aware public API."""
    header = read_eval_log(path, header_only=True)
    if not header.eval.model:
        raise ValueError(f"{path.name}: missing eval.model")
    return header


def stream_eval_samples(path: Path) -> Iterator[EvalSample]:
    """Read available samples without loading large message/event fields."""
    yield from read_eval_log_samples(
        path,
        all_samples_required=False,
        exclude_fields=_DROP_TOPLEVEL,
    )


def _sample_is_complete(sample: EvalSample) -> bool:
    """A complete sample has a terminal scorer result, including missing data."""
    if not sample.scores:
        return False
    for score in sample.scores.values():
        metadata = score.metadata or {}
        if metadata.get("error") == "empty_response":
            return True
        if metadata.get("num_judges") == 2 and all(
            metadata.get(f"judge{index}_llm_id") is not None
            and f"judge{index}_raw_completion" in metadata
            for index in (1, 2)
        ):
            return True
    return False


def _row_is_complete(row: dict) -> bool:
    if not row.get("scorer"):
        return False
    if row.get("error") == "empty_response":
        return True
    return row.get("num_judges") == 2 and all(
        row.get(f"judge{index}_llm_id") is not None for index in (1, 2)
    )


def _existing_completed(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    schema = set(pl.read_parquet_schema(path))
    required = {"sample_id", "epoch", "scorer"}
    if not required.issubset(schema):
        return
    yield from (
        row
        for row in pl.read_parquet(path).iter_rows(named=True)
        if _row_is_complete(row)
    )


def completed_sample_ids(log_dir: Path, model_id: str, existing_path: Path) -> set[str]:
    completed = {
        str(row["sample_id"])
        for row in _existing_completed(existing_path)
        if row.get("sample_id") is not None
    }
    for path in sorted(log_dir.glob("*.eval")):
        if read_eval_header(path).eval.model != model_id:
            continue
        completed.update(
            str(sample.id)
            for sample in stream_eval_samples(path)
            if sample.id and _sample_is_complete(sample)
        )
    return completed


def _flatten(sample: EvalSample, model_id: str, eval_run_id: str, config: dict) -> dict:
    row: dict = {
        "sample_id": str(sample.id),
        "epoch": sample.epoch,
        "schema_version": 1,
        "eval_run_id": eval_run_id,
        "target_model": model_id,
        "proposition_id": sample.metadata.get("proposition_id"),
        "response_text": sample.output.completion if sample.output else None,
        "target_reasoning_effort": config["reasoning_effort"],
        "target_temperature": config["temperature"],
        "target_max_tokens": config["max_tokens"],
        "published_condition": config["published_condition"],
        "scorer": None,
        "score_value": None,
        "score_explanation": None,
        "num_judges": None,
        "template_name": None,
        "error": None,
    }
    for index in (1, 2):
        for field in _JUDGE_FIELDS:
            row[f"judge{index}_{field}"] = None
    for scorer_name, score in (sample.scores or {}).items():
        row["scorer"] = scorer_name
        row["score_value"] = None if score.value == "N/A" else score.value
        row["score_explanation"] = score.explanation
        if score.metadata:
            row.update(score.metadata)
    row.pop("judge_prompt", None)
    return row


def export_model_results(log_dir: Path, result_root: Path, model_id: str, config: dict) -> Path:
    output = result_root / "per_model" / safe_model_filename(model_id)
    records: dict[tuple[str, int], dict] = {}
    for row in _existing_completed(output):
        records[(str(row["sample_id"]), int(row["epoch"]))] = row
    for path in sorted(log_dir.glob("*.eval")):
        header = read_eval_header(path)
        if header.eval.model != model_id:
            continue
        for sample in stream_eval_samples(path):
            if not sample.id or not _sample_is_complete(sample):
                continue
            records[(str(sample.id), int(sample.epoch))] = _flatten(
                sample,
                model_id,
                path.stem,
                config,
            )
    if not records:
        raise ValueError(f"No completed samples found for {model_id}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    pl.DataFrame(list(records.values()), infer_schema_length=None).write_parquet(
        temporary,
        compression="zstd",
    )
    temporary.replace(output)
    return output
