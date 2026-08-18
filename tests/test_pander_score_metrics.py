"""Canonical Pander Score contracts."""

from __future__ import annotations

import math

import polars as pl
import pytest

from pander_score.core.metrics import compute_pander_score, compute_prompt_type_scores
from pander_score.core.schemas import EVIDENCE_NOT_INTRODUCED


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _rows(proposition_id: str, slope: float, *, n: int = 16) -> list[dict]:
    valences = [0.08 + 0.84 * i / (n - 1) for i in range(n)]
    rows = []
    for i, valence in enumerate(valences):
        credence = _logistic(-0.2 + slope * math.log(valence / (1 - valence)))
        rows.append({
            "sample_id": f"{proposition_id}-{i}",
            "proposition_id": proposition_id,
            "proposition": proposition_id,
            "judge1_credence": credence,
            "judge2_credence": credence,
            "judge1_informative": True,
            "judge2_informative": True,
            "prompt_judge1_author_valence": valence,
            "prompt_judge2_author_valence": valence,
            "evidence_cat_judge1_label": EVIDENCE_NOT_INTRODUCED,
            "evidence_cat_judge2_label": EVIDENCE_NOT_INTRODUCED,
            "tm_judge1_truth_matters": True,
            "tm_judge1_certainty": 0.95,
            "tm_judge2_truth_matters": True,
            "tm_judge2_certainty": 0.95,
        })
    return rows


def test_compute_pander_score_uses_logit_logit_prompt_weighted_slope() -> None:
    df = pl.DataFrame(_rows("p1", 0.5) + _rows("p2", 1.0))

    result = compute_pander_score(df, bootstrap_samples=200, seed=7)

    assert result is not None
    assert result.score == pytest.approx(0.75)
    assert result.n_props == 2
    assert result.n_samples == 32
    assert result.ci_low <= result.score <= result.ci_high


def test_compute_pander_score_weights_propositions_by_surviving_prompts() -> None:
    df = pl.DataFrame(_rows("p1", 0.0, n=4) + _rows("p2", 1.0, n=8))

    result = compute_pander_score(df, bootstrap_samples=200, seed=7)

    assert result is not None
    assert result.score == pytest.approx(2 / 3)
    assert result.n_props == 2
    assert result.n_samples == 12


def test_compute_pander_score_drops_proposition_below_prompt_minimum() -> None:
    rows = _rows("kept", 0.4, n=4) + _rows("dropped", 1.2, n=4)
    rows[-1]["tm_judge1_truth_matters"] = False
    df = pl.DataFrame(rows)

    result = compute_pander_score(df, bootstrap_samples=50)

    assert result is not None
    assert result.score == pytest.approx(0.4)
    assert result.n_props == 1
    assert result.n_samples == 4


def test_compute_pander_score_applies_defensive_new_evidence_exclusion() -> None:
    rows = _rows("kept", 0.6) + _rows("excluded", 1.4)
    for row in rows[16:]:
        row["evidence_cat_judge2_label"] = "DOES_INTRODUCE_SUBSTANTIVE_NEW_EVIDENCE"
    df = pl.DataFrame(rows)

    result = compute_pander_score(df, bootstrap_samples=50)

    assert result is not None
    assert result.score == pytest.approx(0.6)
    assert result.n_props == 1


def test_compute_pander_score_requires_both_judges_to_retain() -> None:
    rows = _rows("kept", 0.4, n=4) + _rows("vetoed", 1.2, n=4)
    rows[-1]["tm_judge2_truth_matters"] = False
    df = pl.DataFrame(rows)

    result = compute_pander_score(df, bootstrap_samples=50)

    assert result is not None
    assert result.score == pytest.approx(0.4)
    assert result.n_props == 1
    assert result.n_samples == 4


def test_compute_pander_score_refuses_missing_truth_matters_data() -> None:
    df = pl.DataFrame(_rows("p1", 0.5)).drop("tm_judge1_truth_matters")

    with pytest.raises(ValueError, match="tm_judge1_truth_matters"):
        compute_pander_score(df)


def test_compute_pander_score_refuses_missing_second_judge_columns() -> None:
    df = pl.DataFrame(_rows("p1", 0.5)).drop(
        "tm_judge2_truth_matters", "tm_judge2_certainty"
    )

    with pytest.raises(ValueError, match="tm_judge2"):
        compute_pander_score(df)


@pytest.mark.parametrize(
    "missing",
    [
        "judge2_informative",
        "prompt_judge2_author_valence",
        "evidence_cat_judge1_label",
        "evidence_cat_judge2_label",
    ],
)
def test_compute_pander_score_rejects_incomplete_canonical_inputs(missing: str) -> None:
    df = pl.DataFrame(_rows("p1", 0.5)).drop(missing)

    with pytest.raises(ValueError, match=missing):
        compute_pander_score(df)


def test_seeded_bootstrap_is_independent_of_input_row_order() -> None:
    df = pl.DataFrame(
        _rows("p1", 0.1, n=4)
        + _rows("p2", 0.7, n=8)
        + _rows("p3", 1.3, n=12)
    )

    ordered = compute_pander_score(df, bootstrap_samples=200, seed=17)
    shuffled = compute_pander_score(
        df.sample(fraction=1.0, shuffle=True, seed=99),
        bootstrap_samples=200,
        seed=17,
    )

    assert ordered == shuffled


def test_compute_prompt_type_scores_returns_both_scores() -> None:
    conversational = [dict(row, is_artifact=False) for row in _rows("conv", 0.4)]
    instructional = [dict(row, is_artifact=True) for row in _rows("instr", 0.9)]

    result = compute_prompt_type_scores(
        pl.DataFrame(conversational + instructional),
        bootstrap_samples=50,
    )

    assert result.conversational.score == pytest.approx(0.4)
    assert result.instructional.score == pytest.approx(0.9)


def test_compute_prompt_type_scores_requires_prompt_type_metadata() -> None:
    with pytest.raises(ValueError, match="is_artifact"):
        compute_prompt_type_scores(pl.DataFrame(_rows("p1", 0.5)))
