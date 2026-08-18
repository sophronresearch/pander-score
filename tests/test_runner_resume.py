from __future__ import annotations

import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "run_benchmark", Path(__file__).resolve().parents[1] / "run_benchmark.py"
)
assert SPEC and SPEC.loader
run_benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_benchmark)
_scientific_manifest = run_benchmark._scientific_manifest


def _manifest() -> dict:
    return {
        "benchmark_version": "v1",
        "target_model": "provider/model",
        "target_config": {
            "reasoning_effort": "medium",
            "temperature": 1.0,
            "max_tokens": 16_000,
            "max_connections": 50,
            "published_condition": False,
        },
        "credence_judges": {},
        "prompts_sha256": "abc",
        "prompt_count": 11_172,
    }


def test_resume_comparison_ignores_concurrency() -> None:
    first = _manifest()
    resumed = _manifest()
    resumed["target_config"]["max_connections"] = 10
    assert _scientific_manifest(first) == _scientific_manifest(resumed)


def test_resume_comparison_keeps_output_shaping_settings() -> None:
    first = _manifest()
    changed = _manifest()
    changed["target_config"]["max_tokens"] = 8_000
    assert _scientific_manifest(first) != _scientific_manifest(changed)
