"""LLM-judge prompt and execution helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class JudgeResult:
    verdict: bool | None
    total_score: int | None
    criteria: dict[str, int] | None
    rationale: str | None
    raw_response: str | None
    error: str | None


RUBRIC_KEYS = (
    "equivalence",
    "constants_and_branches",
    "task_completeness",
    "expression_validity",
)


def build_judge_prompt(
    problem_statement: str,
    candidate_expression_latex: str,
    integral_latex: str | None,
    cleo_reference_latex: str | None,
    accepted_reference_latex: str | None,
) -> str:
    refs = []
    if cleo_reference_latex:
        refs.append(f"- Cleo reference: {cleo_reference_latex}")
    if accepted_reference_latex:
        refs.append(f"- Accepted reference: {accepted_reference_latex}")
    refs_text = "\n".join(refs) if refs else "- No reference expression available"

    integral_text = integral_latex or "<unknown integral>"

    return (
        "You are grading a symbolic integration answer.\n"
        "Score each criterion from 0 to 5 (integers only).\n"
        "Criteria: equivalence, constants_and_branches, task_completeness, expression_validity.\n"
        "Set pass=true only if equivalence >= 4 and total_score >= 14.\n"
        "Return JSON only with keys: pass, total_score, criteria, rationale.\n\n"
        f"Problem:\n{problem_statement}\n\n"
        f"Integral target:\n{integral_text}\n\n"
        f"References:\n{refs_text}\n\n"
        f"Candidate answer LaTeX:\n{candidate_expression_latex}\n"
    )


def _json_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_judge_response(raw_text: str) -> JudgeResult:
    payload = _json_from_text(raw_text)
    if payload is None:
        return JudgeResult(
            verdict=None,
            total_score=None,
            criteria=None,
            rationale=None,
            raw_response=raw_text,
            error="judge_response_not_json",
        )

    criteria = payload.get("criteria")
    if not isinstance(criteria, dict):
        return JudgeResult(None, None, None, None, raw_text, "judge_missing_criteria")

    parsed_criteria: dict[str, int] = {}
    for key in RUBRIC_KEYS:
        value = criteria.get(key)
        if not isinstance(value, int):
            return JudgeResult(None, None, None, None, raw_text, f"judge_invalid_{key}")
        parsed_criteria[key] = value

    total = payload.get("total_score")
    if not isinstance(total, int):
        total = sum(parsed_criteria.values())

    verdict = payload.get("pass")
    if not isinstance(verdict, bool):
        verdict = bool(parsed_criteria["equivalence"] >= 4 and total >= 14)

    rationale = payload.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        rationale = str(rationale)

    return JudgeResult(
        verdict=verdict,
        total_score=total,
        criteria=parsed_criteria,
        rationale=rationale,
        raw_response=raw_text,
        error=None,
    )


async def run_judge_with_inspect(
    prompt: str,
    model_name: str | None = None,
    model_role: str = "grader",
) -> JudgeResult:
    """Run judge using Inspect's model provider abstraction.

    If model_name is None, it resolves by role (default: grader).
    """
    try:
        from inspect_ai.model import get_model
    except Exception as ex:  # noqa: BLE001
        return JudgeResult(None, None, None, None, None, f"inspect_import_error: {ex}")

    try:
        model = get_model(model_name) if model_name else get_model(role=model_role)
    except Exception as ex:  # noqa: BLE001
        return JudgeResult(None, None, None, None, None, f"judge_model_init_error: {ex}")

    try:
        response = await model.generate(prompt)
        return parse_judge_response(response.completion)
    except Exception as ex:  # noqa: BLE001
        return JudgeResult(None, None, None, None, None, f"judge_generate_error: {ex}")
