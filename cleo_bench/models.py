"""Data models for Cleo Bench."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class CleoBenchItem:
    item_id: str
    snapshot_date: str
    site: str
    question_id: int
    cleo_answer_id: int
    accepted_answer_id: int | None
    question_url: str
    cleo_answer_url: str
    accepted_answer_url: str | None
    content_license_question: str
    content_license_cleo: str
    content_license_accepted: str | None
    title_raw: str
    question_body_html_raw: str
    cleo_body_html_raw: str
    accepted_body_html_raw: str | None
    prompt_full_question_sanitized: str
    prompt_integral_only: str | None
    integral_latex: str | None
    cleo_reference_latex: str | None
    accepted_reference_latex: str | None
    integral_numeric: str | None
    cleo_reference_numeric: str | None
    accepted_reference_numeric: str | None
    best_reference_numeric: str | None
    numeric_delta_cleo: str | None
    numeric_delta_accepted: str | None
    numeric_pass_cleo: bool | None
    numeric_pass_accepted: bool | None
    is_integral_candidate: bool
    is_scorable_numeric: bool
    status: str
    manual_review_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CleoBenchItem":
        return cls(**payload)


SCHEMA_FIELDS: tuple[str, ...] = tuple(CleoBenchItem.__dataclass_fields__.keys())
