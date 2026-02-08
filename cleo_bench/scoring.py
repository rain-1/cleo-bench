"""Deterministic and judge-assisted scoring helpers."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import mpmath as mp

from .constants import DEFAULT_DPS, DEFAULT_TOLERANCE
from .judge import build_judge_prompt, run_judge_with_inspect
from .numeric import abs_rel_close, evaluate_expression_numeric


@dataclass
class DeterministicScore:
    deterministic_status: str
    numeric_error: str | None
    candidate_expression_latex: str | None
    final_pass: bool | None
    final_score: float | None
    explanation: str


def extract_candidate_expression(output_text: str) -> str | None:
    if not output_text:
        return None
    raw = output_text.strip()

    # Strict JSON first.
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            value = payload.get("final_expression_latex")
            if isinstance(value, str) and value.strip():
                return value.strip()
    except json.JSONDecodeError:
        pass

    # Best effort for JSON embedded in text.
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            payload = json.loads(match.group(0))
            value = payload.get("final_expression_latex") if isinstance(payload, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        except json.JSONDecodeError:
            pass

    # Last fallback: parse boxed LaTeX.
    box = re.search(r"\\boxed\{([\s\S]+)\}", raw)
    if box:
        return box.group(1).strip()
    return None


def _reference_value_from_metadata(metadata: dict[str, Any]) -> mp.mpf | None:
    direct = metadata.get("best_reference_numeric")
    if isinstance(direct, str) and direct:
        try:
            return mp.mpf(direct)
        except Exception:  # noqa: BLE001
            pass

    candidates = [metadata.get("cleo_reference_numeric"), metadata.get("accepted_reference_numeric")]
    parsed: list[mp.mpf] = []
    for c in candidates:
        if isinstance(c, str) and c:
            try:
                parsed.append(mp.mpf(c))
            except Exception:  # noqa: BLE001
                continue
    if not parsed:
        return None
    return parsed[0]


def deterministic_score(
    output_text: str,
    metadata: dict[str, Any],
    dps: int = DEFAULT_DPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> DeterministicScore:
    candidate = extract_candidate_expression(output_text)
    if not candidate:
        return DeterministicScore(
            deterministic_status="unresolved",
            numeric_error=None,
            candidate_expression_latex=None,
            final_pass=None,
            final_score=None,
            explanation="No `final_expression_latex` found in model output.",
        )

    candidate_eval = evaluate_expression_numeric(candidate, dps=dps)
    if candidate_eval.value is None:
        return DeterministicScore(
            deterministic_status="unresolved",
            numeric_error=None,
            candidate_expression_latex=candidate,
            final_pass=None,
            final_score=None,
            explanation=f"Candidate expression not numerically evaluable: {candidate_eval.error}",
        )

    reference = _reference_value_from_metadata(metadata)
    if reference is None:
        return DeterministicScore(
            deterministic_status="unresolved",
            numeric_error=None,
            candidate_expression_latex=candidate,
            final_pass=None,
            final_score=None,
            explanation="No reference numeric value available.",
        )

    delta, passed = abs_rel_close(candidate_eval.value, reference, tolerance=tolerance)
    if passed is None:
        return DeterministicScore(
            deterministic_status="unresolved",
            numeric_error=None,
            candidate_expression_latex=candidate,
            final_pass=None,
            final_score=None,
            explanation="Numeric comparison failed.",
        )

    return DeterministicScore(
        deterministic_status="pass" if passed else "fail",
        numeric_error=delta,
        candidate_expression_latex=candidate,
        final_pass=bool(passed),
        final_score=1.0 if passed else 0.0,
        explanation="Deterministic numeric comparison.",
    )


async def score_with_optional_judge(
    output_text: str,
    metadata: dict[str, Any],
    use_judge_when_unresolved: bool,
    judge_model: str | None,
    dps: int = DEFAULT_DPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    det = deterministic_score(output_text, metadata, dps=dps, tolerance=tolerance)
    result: dict[str, Any] = {
        "deterministic_status": det.deterministic_status,
        "numeric_error": det.numeric_error,
        "judge_used": False,
        "judge_scores": None,
        "final_pass": det.final_pass,
        "final_score": det.final_score,
        "candidate_expression_latex": det.candidate_expression_latex,
        "explanation": det.explanation,
    }

    if det.deterministic_status != "unresolved" or not use_judge_when_unresolved:
        return result

    candidate_expr = det.candidate_expression_latex
    if not candidate_expr:
        return result

    prompt = build_judge_prompt(
        problem_statement=str(metadata.get("prompt_full_question_sanitized", "")),
        candidate_expression_latex=candidate_expr,
        integral_latex=metadata.get("integral_latex"),
        cleo_reference_latex=metadata.get("cleo_reference_latex"),
        accepted_reference_latex=metadata.get("accepted_reference_latex"),
    )
    judge = await run_judge_with_inspect(prompt, model_name=judge_model)
    result["judge_used"] = True

    if judge.error is not None:
        result["explanation"] = f"Judge fallback failed: {judge.error}"
        return result

    criteria = judge.criteria or {}
    total = judge.total_score if judge.total_score is not None else sum(criteria.values())
    result["judge_scores"] = criteria
    result["final_pass"] = bool(judge.verdict)
    result["final_score"] = float(total) / 20.0
    result["explanation"] = judge.rationale or "Judge-based decision for unresolved deterministic case."
    return result


def score_with_optional_judge_sync(
    output_text: str,
    metadata: dict[str, Any],
    use_judge_when_unresolved: bool,
    judge_model: str | None,
    dps: int = DEFAULT_DPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    return asyncio.run(
        score_with_optional_judge(
            output_text=output_text,
            metadata=metadata,
            use_judge_when_unresolved=use_judge_when_unresolved,
            judge_model=judge_model,
            dps=dps,
            tolerance=tolerance,
        )
    )
