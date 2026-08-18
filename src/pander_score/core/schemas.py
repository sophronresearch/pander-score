"""Small data contracts used by the public benchmark and estimator."""

from typing import Literal, NamedTuple

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class Judgment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    refusal: bool
    informative: bool
    credence: float | None = Field(
        default=None,
        validation_alias=AliasChoices("credence", "credance"),
    )
    explanation: str

    @model_validator(mode="after")
    def _check_consistency(self) -> "Judgment":
        if self.refusal and self.informative:
            raise ValueError("refusal=True requires informative=False")
        if self.informative and self.credence is None:
            raise ValueError("informative=True requires a non-null credence")
        return self


class JudgeResult(NamedTuple):
    judge_id: str
    judgment: Judgment | None
    raw_completion: str | None


EVIDENCE_INTRODUCED = "DOES_INTRODUCE_SUBSTANTIVE_NEW_EVIDENCE"
EVIDENCE_NOT_INTRODUCED = "DOES_NOT_INTRODUCE_SUBSTANTIVE_NEW_EVIDENCE"


class AuthorValence(BaseModel):
    author_valence: float = Field(ge=0.0, le=1.0)
    explanation: str


class NewEvidenceCategorical(BaseModel):
    label: Literal[
        "DOES_INTRODUCE_SUBSTANTIVE_NEW_EVIDENCE",
        "DOES_NOT_INTRODUCE_SUBSTANTIVE_NEW_EVIDENCE",
    ]
    direction: Literal["supports", "undermines", "unclear", "none"]
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

    @model_validator(mode="after")
    def _check_coupling(self) -> "NewEvidenceCategorical":
        if self.label == EVIDENCE_NOT_INTRODUCED and self.direction != "none":
            raise ValueError("DOES_NOT_INTRODUCE requires direction='none'")
        if self.label == EVIDENCE_INTRODUCED and self.direction == "none":
            raise ValueError("DOES_INTRODUCE requires a non-'none' direction")
        return self


class TruthMatters(BaseModel):
    explanation: str
    truth_matters: bool
    certainty: float = Field(ge=0.5, le=1.0)
