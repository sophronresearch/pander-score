from pander_score.core.scorers import CREDENCE_JUDGES, _empty_response_score


def test_empty_response_is_an_auditable_na_score() -> None:
    score = _empty_response_score("implicit_judge")

    assert score.value == "N/A"
    assert score.answer == "N/A"
    assert score.metadata == {
        "error": "empty_response",
        "template_name": "implicit_judge",
    }


def test_credence_judges_allow_longer_requests() -> None:
    assert all(config.attempt_timeout == 180 for config in CREDENCE_JUDGES.values())
    assert all(config.timeout == 240 for config in CREDENCE_JUDGES.values())
