from __future__ import annotations

import pytest

from pander_score.core.config import PUBLISHED_MODEL_CONFIGS, resolve_model_config


def test_registry_is_exactly_the_published_18() -> None:
    assert len(PUBLISHED_MODEL_CONFIGS) == 18
    assert "google/gemini-3.7-flash" in PUBLISHED_MODEL_CONFIGS
    assert "grok/grok-4.6" in PUBLISHED_MODEL_CONFIGS
    assert "google/gemini-3.6-flash" not in PUBLISHED_MODEL_CONFIGS
    assert "grok/grok-4.5" not in PUBLISHED_MODEL_CONFIGS


def test_published_conditions_preserve_non_default_models() -> None:
    assert resolve_model_config("anthropic/claude-fable-5").reasoning_effort == "medium"
    assert resolve_model_config("google/gemini-3.7-flash").temperature == 1.0
    assert resolve_model_config(
        "openai-api/fireworks/accounts/fireworks/models/kimi-k3"
    ).reasoning_effort == "high"
    assert resolve_model_config("grok/grok-4-1-fast-reasoning").reasoning_effort is None


def test_unknown_models_use_neutral_defaults_and_accept_overrides() -> None:
    neutral = resolve_model_config("provider/new-model")
    assert neutral.reasoning_effort is None
    assert neutral.temperature is None
    assert not neutral.published_condition

    overridden = resolve_model_config(
        "provider/new-model", reasoning_effort="high", temperature=0.7
    )
    assert overridden.reasoning_effort == "high"
    assert overridden.temperature == 0.7


def test_published_override_warns_and_marks_non_comparable() -> None:
    with pytest.warns(UserWarning, match="not directly comparable"):
        resolved = resolve_model_config(
            "google/gemini-3.7-flash",
            reasoning_effort="high",
        )
    assert not resolved.published_condition
