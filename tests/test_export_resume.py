from __future__ import annotations

from pathlib import Path

import polars as pl
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score

from pander_score.export import completed_sample_ids, export_model_results


MODEL_ID = "provider/model"
CONFIG = {
    "reasoning_effort": "medium",
    "temperature": 1.0,
    "max_tokens": 16_000,
    "published_condition": False,
}


def _write_log(
    path: Path,
    *,
    status: str,
    scored: bool,
    complete_judges: bool = True,
    empty_response: bool = False,
    complete_metadata: bool = True,
) -> None:
    metadata = (
        {"error": "empty_response", "template_name": "implicit_judge"}
        if empty_response
        else {
            "num_judges": 2,
            "judge1_llm_id": "judge/one",
            "judge1_credence": 0.7,
            "judge1_raw_completion": "{}",
            "judge2_llm_id": "judge/two" if complete_metadata else None,
            "judge2_raw_completion": "{}" if complete_judges else None,
        }
    )
    sample = EvalSample(
        id="s1",
        epoch=1,
        input="prompt",
        target="proposition",
        output=ModelOutput(model=MODEL_ID, completion="" if empty_response else "response"),
        scores={
            "credence_scorer": Score(
                value="N/A" if empty_response else 0.7,
                explanation=(
                    "Empty response from target model"
                    if empty_response
                    else "Judge credences: [0.7, None]"
                ),
                metadata=metadata,
            )
        }
        if scored
        else None,
        metadata={"proposition_id": "p1"},
    )
    log = EvalLog(
        status=status,
        eval=EvalSpec(
            eval_id=path.stem,
            created="2026-08-17T00:00:00Z",
            task="pander-benchmark",
            dataset=EvalDataset(name="test", samples=1, sample_ids=["s1"]),
            model=MODEL_ID,
            config=EvalConfig(),
        ),
        samples=[sample],
    )
    write_eval_log(log, path)


def test_export_reads_finalized_eval_and_ignores_cancelled_scoreless_sample(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(logs / "01-success.eval", status="success", scored=True)
    _write_log(logs / "02-cancelled.eval", status="cancelled", scored=False)

    output = export_model_results(logs, tmp_path / "results", MODEL_ID, CONFIG)
    frame = pl.read_parquet(output)

    assert frame["sample_id"].to_list() == ["s1"]
    assert frame["scorer"].to_list() == ["credence_scorer"]
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_scoreless_logs_do_not_create_a_result_file(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(logs / "cancelled.eval", status="cancelled", scored=False)

    try:
        export_model_results(logs, tmp_path / "results", MODEL_ID, CONFIG)
    except ValueError as error:
        assert "No completed samples" in str(error)
    else:
        raise AssertionError("scoreless log was exported")
    assert not (tmp_path / "results/per_model/provider--model.parquet").exists()


def test_terminal_judge_parse_failure_is_complete_and_exported(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "partial.eval",
        status="success",
        scored=True,
        complete_judges=False,
    )

    assert completed_sample_ids(logs, MODEL_ID, tmp_path / "missing.parquet") == {"s1"}
    output = export_model_results(logs, tmp_path / "results", MODEL_ID, CONFIG)
    row = pl.read_parquet(output).row(0, named=True)
    assert row["judge1_raw_completion"] == "{}"
    assert row["judge2_raw_completion"] is None


def test_incomplete_score_metadata_is_retried_not_exported(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "partial.eval",
        status="success",
        scored=True,
        complete_metadata=False,
    )

    assert completed_sample_ids(logs, MODEL_ID, tmp_path / "missing.parquet") == set()


def test_empty_response_is_a_complete_na_row(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    _write_log(
        logs / "empty.eval",
        status="success",
        scored=True,
        empty_response=True,
    )

    assert completed_sample_ids(logs, MODEL_ID, tmp_path / "missing.parquet") == {"s1"}
    output = export_model_results(logs, tmp_path / "results", MODEL_ID, CONFIG)
    row = pl.read_parquet(output).row(0, named=True)
    assert row["score_value"] is None
    assert row["error"] == "empty_response"
    assert row["judge1_llm_id"] is None


def test_malformed_old_parquet_does_not_count_as_completed(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    old = tmp_path / "old.parquet"
    pl.DataFrame({"sample_id": ["s1"], "epoch": [1]}).write_parquet(old)

    assert completed_sample_ids(logs, MODEL_ID, old) == set()
