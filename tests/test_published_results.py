from __future__ import annotations

import os
from pathlib import Path

import pytest

from pander_score.core.metrics import compute_prompt_type_scores
from pander_score.data import load_model_results


# score, n_props, n_samples for conversational then instructional.
EXPECTED = {
    "anthropic/claude-fable-5": (0.010935246298416558, 322, 3979, 0.18590974203962718, 206, 1347),
    "anthropic/claude-opus-4-6": (-0.03486959845398495, 341, 4427, 0.16834938815741152, 219, 1416),
    "anthropic/claude-sonnet-4-6": (-0.039362697054534095, 343, 4439, 0.11785567219840368, 226, 1478),
    "google/gemini-3-flash-preview": (0.25972249228705724, 341, 4297, 0.70117802748574, 223, 1460),
    "google/gemini-3.1-pro-preview": (0.22660611562086447, 342, 4345, 0.7061310001696961, 222, 1468),
    "google/gemini-3.5-flash": (0.2332624364769089, 342, 4320, 0.6948454449713084, 224, 1466),
    "google/gemini-3.7-flash": (0.15684783476894384, 342, 4376, 0.6504779363046048, 221, 1467),
    "grok/grok-4-1-fast-reasoning": (0.34802589581810556, 342, 4531, 0.8446615172972042, 229, 1521),
    "grok/grok-4.20-0309-reasoning": (0.19259635981380863, 343, 4536, 0.6872613831980269, 224, 1498),
    "grok/grok-4.6": (0.1382110406806844, 342, 4512, 0.33622215936335526, 230, 1516),
    "openai-api/fireworks/accounts/fireworks/models/inkling": (0.18371811177980873, 341, 4396, 0.4539627548811693, 227, 1448),
    "openai-api/fireworks/accounts/fireworks/models/kimi-k3": (0.0704759491039597, 342, 4456, 0.23499527783218357, 218, 1445),
    "openai-api/meta/muse-spark-1.1": (0.04583950462822722, 340, 4409, 0.1694768938016376, 230, 1502),
    "openai-api/zai/glm-5.2": (0.28226889560906643, 342, 4399, 0.6988204640118004, 223, 1462),
    "openai/gpt-5.4-2026-03-05": (0.1431224659590467, 342, 4354, 0.3239713218122106, 215, 1360),
    "openai/gpt-5.4-mini-2026-03-17": (0.1643480569098327, 341, 4393, 0.388207528241402, 223, 1406),
    "openai/gpt-5.6-sol": (0.0655247336077417, 340, 4385, 0.17754152748938826, 221, 1446),
    "openai/gpt-5.6-terra": (0.09700312823324041, 338, 4307, 0.25680336185310443, 228, 1469),
}


@pytest.mark.skipif(
    "PANDER_SCORE_DATASET_ROOT" not in os.environ,
    reason="set PANDER_SCORE_DATASET_ROOT to the local v1 result bundle",
)
def test_all_published_scores_match_release_bundle() -> None:
    root = Path(os.environ["PANDER_SCORE_DATASET_ROOT"])
    for model_id, expected in EXPECTED.items():
        result = compute_prompt_type_scores(load_model_results(root, model_id))
        conversational = result.conversational
        instructional = result.instructional
        assert conversational.score == pytest.approx(expected[0], abs=1e-12)
        assert (conversational.n_props, conversational.n_samples) == expected[1:3]
        assert instructional.score == pytest.approx(expected[3], abs=1e-12)
        assert (instructional.n_props, instructional.n_samples) == expected[4:6]
