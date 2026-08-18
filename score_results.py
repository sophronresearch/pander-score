#!/usr/bin/env python3
"""Reproduce published Pander Scores from the public result bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from huggingface_hub import snapshot_download

from pander_score.core.metrics import compute_prompt_type_scores
from pander_score.data import load_model_results, safe_model_filename


HF_DATASET = "sophronresearch/pander-score"
ROOT = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Compute the exact published estimator from saved model outputs."
    )
    result.add_argument("--model", help="Score one model ID (default: every model in the bundle)")
    result.add_argument("--results", type=Path, help="Local v1 result-bundle directory")
    return result


def _bundle_root(path: Path) -> Path:
    """Accept either the benchmark table directory or its version parent."""
    if (path / "benchmark/per_model").exists():
        return path / "benchmark"
    return path


def _downloaded_result_root() -> Path:
    downloaded = Path(
        snapshot_download(
            repo_id=HF_DATASET,
            repo_type="dataset",
            allow_patterns=["v1/benchmark/*"],
        )
    )
    return downloaded / "v1/benchmark"


def _result_root(explicit: Path | None, model_id: str | None) -> Path:
    if explicit is not None:
        return _bundle_root(explicit)
    local = ROOT / "results/v1"
    if model_id and (local / "per_model" / safe_model_filename(model_id)).exists():
        return local
    return _downloaded_result_root()


def _available_models(result_root: Path) -> list[str]:
    models = []
    for path in sorted((result_root / "per_model").glob("*.parquet")):
        frame = pl.read_parquet(path, columns=["target_model"])
        unique = frame["target_model"].unique().to_list()
        if len(unique) != 1:
            raise ValueError(f"{path}: expected exactly one target_model")
        models.append(str(unique[0]))
    return models


def main() -> None:
    args = parser().parse_args()
    result_root = _result_root(args.results, args.model)
    models = [args.model] if args.model else _available_models(result_root)
    if not models:
        raise SystemExit(f"No per-model parquets found under {result_root}")

    print("Model\tConversational (95% CI)\tInstructional (95% CI)\tPrompts")
    for model_id in models:
        data_root = None
        if not (result_root / "prompt_attributes.parquet").exists():
            data_root = ROOT / "data/v1"
        frame = load_model_results(result_root, model_id, data_root=data_root)
        scores = compute_prompt_type_scores(frame)
        conversational = scores.conversational
        instructional = scores.instructional
        print(
            f"{model_id}\t"
            f"{100 * conversational.score:+.1f} "
            f"[{100 * conversational.ci_low:+.1f}, {100 * conversational.ci_high:+.1f}]\t"
            f"{100 * instructional.score:+.1f} "
            f"[{100 * instructional.ci_low:+.1f}, {100 * instructional.ci_high:+.1f}]\t"
            f"{frame['sample_id'].n_unique()}"
        )


if __name__ == "__main__":
    main()
