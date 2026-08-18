from __future__ import annotations

import importlib.util
from pathlib import Path

from pander_score.data import safe_model_filename


SPEC = importlib.util.spec_from_file_location(
    "score_results", Path(__file__).resolve().parents[1] / "score_results.py"
)
assert SPEC and SPEC.loader
score_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_results)


def test_explicit_version_root_resolves_benchmark_directory(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark"
    (benchmark / "per_model").mkdir(parents=True)
    assert score_results._result_root(tmp_path, None) == benchmark


def test_missing_local_model_falls_back_to_download(
    tmp_path: Path, monkeypatch
) -> None:
    local = tmp_path / "results/v1"
    (local / "per_model").mkdir(parents=True)
    (local / "per_model" / safe_model_filename("provider/local")).touch()
    downloaded = tmp_path / "downloaded/v1/benchmark"

    monkeypatch.setattr(score_results, "ROOT", tmp_path)
    monkeypatch.setattr(score_results, "_downloaded_result_root", lambda: downloaded)

    assert score_results._result_root(None, "provider/missing") == downloaded
    assert score_results._result_root(None, "provider/local") == local


def test_no_model_uses_complete_download_even_with_local_results(
    tmp_path: Path, monkeypatch
) -> None:
    local = tmp_path / "results/v1/per_model"
    local.mkdir(parents=True)
    (local / safe_model_filename("provider/local")).touch()
    downloaded = tmp_path / "downloaded/v1/benchmark"

    monkeypatch.setattr(score_results, "ROOT", tmp_path)
    monkeypatch.setattr(score_results, "_downloaded_result_root", lambda: downloaded)

    assert score_results._result_root(None, None) == downloaded
