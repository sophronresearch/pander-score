from __future__ import annotations

import csv
from pathlib import Path

from pander_score.data import BENCHMARK_PROMPTS, BENCHMARK_PROPOSITIONS, load_prompt_attributes


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_benchmark_has_exact_public_scope() -> None:
    with (ROOT / "data/v1/propositions.csv").open(newline="", encoding="utf-8") as handle:
        propositions = list(csv.DictReader(handle))

    assert len(propositions) == BENCHMARK_PROPOSITIONS
    assert len({row["id"] for row in propositions}) == BENCHMARK_PROPOSITIONS
    assert {row["domain"] for row in propositions} == {
        "contested_social_science",
        "frontier_natural_science",
        "historical_facts",
        "moral_claims",
        "nutrition_health",
        "paranormal_claims",
        "politically_polarizing",
    }

    attributes = load_prompt_attributes(ROOT / "data/v1")
    assert attributes.height == BENCHMARK_PROMPTS
    assert attributes["sample_id"].n_unique() == BENCHMARK_PROMPTS
    assert attributes["proposition_id"].n_unique() == BENCHMARK_PROPOSITIONS
    assert attributes["domain"].null_count() == 0
    assert set(attributes["domain"]) == {row["domain"] for row in propositions}


def test_only_canonical_prompt_attribute_files_ship() -> None:
    names = {
        path.name
        for path in (ROOT / "data/v1/prompt_attributes").iterdir()
        if path.is_file()
    }
    assert names == {
        "author_valence.jsonl",
        "new_evidence_cat.jsonl",
        "truth_matters.jsonl",
    }
