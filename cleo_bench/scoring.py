"""Deterministic and judge-assisted scoring helpers."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import mpmath as mp

from .constants import DEFAULT_DPS, DEFAULT_TOLERANCE
from .judge import (
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    build_judge_prompt,
    run_judge_with_inspect,
)
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

    def _candidate_from_obj(obj_text: str) -> str | None:
        try:
            payload = json.loads(obj_text)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            value = payload.get("final_expression_latex")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _json_object_spans(text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start: int | None = None
        depth = 0
        in_str = False
        escape = False

        for idx, ch in enumerate(text):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
                continue

            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
                continue

            if ch == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, idx + 1))
                    start = None
        return spans

    # Parse JSON code fences first if present.
    for fence in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE):
        fenced = fence.group(1).strip()
        candidate = _candidate_from_obj(fenced)
        if candidate:
            return candidate
        for start, end in _json_object_spans(fenced):
            if "final_expression_latex" not in fenced[start:end]:
                continue
            candidate = _candidate_from_obj(fenced[start:end])
            if candidate:
                return candidate

    # Best effort for JSON objects embedded in prose.
    for start, end in _json_object_spans(raw):
        chunk = raw[start:end]
        if "final_expression_latex" not in chunk:
            continue
        candidate = _candidate_from_obj(chunk)
        if candidate:
            return candidate

    # Fallback: extract value from a key-value pair even if JSON is malformed.
    kv_match = re.search(
        r'"final_expression_latex"\s*:\s*"((?:\\.|[^"\\])*)"',
        raw,
        flags=re.DOTALL,
    )
    if kv_match:
        encoded = kv_match.group(1)
        try:
            decoded = json.loads(f'"{encoded}"')
            if isinstance(decoded, str) and decoded.strip():
                return decoded.strip()
        except json.JSONDecodeError:
            if encoded.strip():
                return encoded.strip()

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
    judge_api_key: str | None = None,
    judge_api_key_env: str | None = "OPENROUTER_API_KEY_JUDGE",
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    judge_reasoning_effort: str | None = None,
    judge_use_sagemath_mcp: bool = False,
    judge_sagemath_mcp_command: str | None = None,
    judge_sagemath_mcp_args: str | list[str] | tuple[str, ...] | None = None,
    judge_sagemath_mcp_cwd: str | None = None,
    judge_sagemath_mcp_tools: str | list[str] | tuple[str, ...] | None = None,
    judge_sagemath_mcp_allow_imports: bool = False,
    judge_sagemath_mcp_allowed_imports: str | list[str] | tuple[str, ...] | None = None,
    judge_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    dps: int = DEFAULT_DPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    det = deterministic_score(output_text, metadata, dps=dps, tolerance=tolerance)
    result: dict[str, Any] = {
        "deterministic_status": det.deterministic_status,
        "numeric_error": det.numeric_error,
        "judge_used": False,
        "judge_scores": None,
        "judge_error": None,
        "judge_stop_reason": None,
        "judge_tool_calls": None,
        "judge_used_sagemath_mcp": None,
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
        allow_sagemath_tools=judge_use_sagemath_mcp,
    )
    judge = await run_judge_with_inspect(
        prompt,
        model_name=judge_model,
        api_key=judge_api_key,
        api_key_env=judge_api_key_env,
        max_tokens=judge_max_tokens,
        reasoning_effort=judge_reasoning_effort,
        use_sagemath_mcp=judge_use_sagemath_mcp,
        sagemath_mcp_command=judge_sagemath_mcp_command,
        sagemath_mcp_args=judge_sagemath_mcp_args,
        sagemath_mcp_cwd=judge_sagemath_mcp_cwd,
        sagemath_mcp_tools=judge_sagemath_mcp_tools,
        sagemath_mcp_allow_imports=judge_sagemath_mcp_allow_imports,
        sagemath_mcp_allowed_imports=judge_sagemath_mcp_allowed_imports,
        sagemath_mcp_eval_timeout_seconds=judge_sagemath_mcp_eval_timeout_seconds,
    )
    result["judge_used"] = True
    result["judge_stop_reason"] = judge.stop_reason
    result["judge_tool_calls"] = judge.tool_calls
    result["judge_used_sagemath_mcp"] = judge.used_sagemath_mcp

    if judge.error is not None:
        result["judge_error"] = judge.error
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
    judge_api_key: str | None = None,
    judge_api_key_env: str | None = "OPENROUTER_API_KEY_JUDGE",
    judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    judge_reasoning_effort: str | None = None,
    judge_use_sagemath_mcp: bool = False,
    judge_sagemath_mcp_command: str | None = None,
    judge_sagemath_mcp_args: str | list[str] | tuple[str, ...] | None = None,
    judge_sagemath_mcp_cwd: str | None = None,
    judge_sagemath_mcp_tools: str | list[str] | tuple[str, ...] | None = None,
    judge_sagemath_mcp_allow_imports: bool = False,
    judge_sagemath_mcp_allowed_imports: str | list[str] | tuple[str, ...] | None = None,
    judge_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    dps: int = DEFAULT_DPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    return asyncio.run(
        score_with_optional_judge(
            output_text=output_text,
            metadata=metadata,
            use_judge_when_unresolved=use_judge_when_unresolved,
            judge_model=judge_model,
            judge_api_key=judge_api_key,
            judge_api_key_env=judge_api_key_env,
            judge_max_tokens=judge_max_tokens,
            judge_reasoning_effort=judge_reasoning_effort,
            judge_use_sagemath_mcp=judge_use_sagemath_mcp,
            judge_sagemath_mcp_command=judge_sagemath_mcp_command,
            judge_sagemath_mcp_args=judge_sagemath_mcp_args,
            judge_sagemath_mcp_cwd=judge_sagemath_mcp_cwd,
            judge_sagemath_mcp_tools=judge_sagemath_mcp_tools,
            judge_sagemath_mcp_allow_imports=judge_sagemath_mcp_allow_imports,
            judge_sagemath_mcp_allowed_imports=judge_sagemath_mcp_allowed_imports,
            judge_sagemath_mcp_eval_timeout_seconds=judge_sagemath_mcp_eval_timeout_seconds,
            dps=dps,
            tolerance=tolerance,
        )
    )
