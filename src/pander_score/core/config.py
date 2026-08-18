"""Frozen model conditions for the published Pander Score runs."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass


DEFAULT_MAX_TOKENS = 16_000
DEFAULT_MAX_CONNECTIONS = 50


@dataclass(frozen=True)
class PublishedModelConfig:
    reasoning_effort: str | None
    temperature: float | None


@dataclass(frozen=True)
class ResolvedModelConfig:
    reasoning_effort: str | None
    temperature: float | None
    max_tokens: int
    max_connections: int
    published_condition: bool

    def to_dict(self) -> dict:
        return asdict(self)


# This is the complete machine-readable record of the target-model conditions
# used for the 18 models in the paper. Provider tuning and superseded/pilot
# models intentionally do not belong in the public registry.
PUBLISHED_MODEL_CONFIGS = {
    "anthropic/claude-sonnet-4-6": PublishedModelConfig("medium", None),
    "anthropic/claude-opus-4-6": PublishedModelConfig("medium", None),
    "anthropic/claude-fable-5": PublishedModelConfig("medium", None),
    "openai-api/meta/muse-spark-1.1": PublishedModelConfig("medium", None),
    "openai-api/zai/glm-5.2": PublishedModelConfig("medium", 1.0),
    "openai-api/fireworks/accounts/fireworks/models/kimi-k3": PublishedModelConfig("high", None),
    "openai-api/fireworks/accounts/fireworks/models/inkling": PublishedModelConfig("medium", None),
    "openai/gpt-5.4-2026-03-05": PublishedModelConfig("medium", None),
    "openai/gpt-5.4-mini-2026-03-17": PublishedModelConfig("medium", None),
    "openai/gpt-5.6-terra": PublishedModelConfig("medium", None),
    "openai/gpt-5.6-sol": PublishedModelConfig("medium", None),
    "google/gemini-3-flash-preview": PublishedModelConfig("medium", 1.0),
    "google/gemini-3.1-pro-preview": PublishedModelConfig("medium", 1.0),
    "google/gemini-3.5-flash": PublishedModelConfig("medium", 1.0),
    "google/gemini-3.7-flash": PublishedModelConfig("medium", 1.0),
    "grok/grok-4.20-0309-reasoning": PublishedModelConfig(None, None),
    "grok/grok-4-1-fast-reasoning": PublishedModelConfig(None, None),
    "grok/grok-4.6": PublishedModelConfig("high", None),
}


def resolve_model_config(
    model_id: str,
    *,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
) -> ResolvedModelConfig:
    """Resolve published conditions or neutral defaults plus explicit overrides."""
    published = PUBLISHED_MODEL_CONFIGS.get(model_id)
    base_effort = published.reasoning_effort if published else None
    base_temperature = published.temperature if published else None
    resolved_effort = reasoning_effort if reasoning_effort is not None else base_effort
    resolved_temperature = temperature if temperature is not None else base_temperature

    overridden = bool(
        published
        and (
            (reasoning_effort is not None and reasoning_effort != base_effort)
            or (temperature is not None and temperature != base_temperature)
            or max_tokens != DEFAULT_MAX_TOKENS
        )
    )
    if overridden:
        warnings.warn(
            f"{model_id} is a published model, but its generation condition was "
            "overridden. This run is not directly comparable to the published score.",
            stacklevel=2,
        )

    return ResolvedModelConfig(
        reasoning_effort=resolved_effort,
        temperature=resolved_temperature,
        max_tokens=max_tokens,
        max_connections=max_connections,
        published_condition=published is not None and not overridden,
    )
