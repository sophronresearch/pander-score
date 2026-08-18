from __future__ import annotations

import json
from pathlib import Path

from pander_score.core.elicitor import (
    GeneratedImplicitPrompt,
    GeneratedImplicitPrompts,
    _build_implicit_prompt,
    _compute_implicit_params,
    _validate_implicit_prompts,
)
from pander_score.prompt_generation import (
    Proposition,
    _read_checkpoint,
    _samples_for_unit,
    _write_checkpoint,
    _generation_manifest,
    load_fixed_propositions,
)


ROOT = Path(__file__).resolve().parents[1]


def _valid_prompts() -> GeneratedImplicitPrompts:
    return GeneratedImplicitPrompts(
        prompts=[
            GeneratedImplicitPrompt(
                prior=("skeptic", "neutral", "believer")[index % 3],
                user_role=f"role_{index % 4}",
                primary_angle=f"angle_{index % 5}",
                length=("short", "medium", "long")[index % 3],
                tone=("formal", "casual", "unpolished")[index % 3],
                is_artifact=index % 4 == 0,
                prompt=f"Prompt {index}",
            )
            for index in range(8)
        ]
    )


def test_generator_is_bound_to_all_349_public_propositions() -> None:
    propositions = load_fixed_propositions(ROOT / "data/v1/propositions.csv")
    assert len(propositions) == 349
    assert len({proposition.id for proposition in propositions}) == 349


def test_original_implicit_prompt_constraints_and_template_ship() -> None:
    params = _compute_implicit_params(8)
    assert params["n_skeptical"] == 2
    assert params["n_neutral"] == 4
    assert params["n_believing"] == 2
    assert not _validate_implicit_prompts(_valid_prompts(), params, tolerance=1)
    rendered = _build_implicit_prompt("The Earth is flat", params)
    assert "The Earth is flat" in rendered
    assert '"prompts"' in rendered


def test_generated_samples_keep_fixed_proposition_identity() -> None:
    proposition = Proposition("historical_facts__1", "A proposition")
    samples = _samples_for_unit(
        proposition,
        "openai/gpt-5.4-mini-2026-03-17",
        _valid_prompts().prompts,
    )
    assert len(samples) == 8
    assert {sample["metadata"]["proposition_id"] for sample in samples} == {
        proposition.id
    }
    assert len({sample["id"] for sample in samples}) == 8


def test_unit_checkpoint_is_atomic_and_identity_checked(tmp_path: Path) -> None:
    checkpoint = tmp_path / "unit.json"
    payload = {
        "proposition_id": "p1",
        "elicitor_id": "m1",
        "samples": [{"id": "s1"}],
    }
    _write_checkpoint(checkpoint, payload)
    assert json.loads(checkpoint.read_text()) == payload
    assert _read_checkpoint(checkpoint, "p1", "m1") == [{"id": "s1"}]
    assert not checkpoint.with_suffix(".tmp").exists()


def test_generation_manifest_freezes_inputs_models_and_templates() -> None:
    manifest = _generation_manifest(ROOT / "data/v1/propositions.csv")
    assert manifest["proposition_count"] == 349
    assert manifest["prompts_per_elicitor_per_proposition"] == 16
    assert set(manifest["elicitors"]) == {
        "openai/gpt-5.4-mini-2026-03-17",
        "google/gemini-3-flash-preview",
    }
    assert set(manifest["templates"]) == {"elicitor_preamble", "elicitor_implicit"}
