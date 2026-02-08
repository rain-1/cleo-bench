"""LLM-judge prompt and execution helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DEFAULT_JUDGE_MAX_TOKENS = 8192
DEFAULT_SAGEMATH_MCP_ARGS: tuple[str, ...] = ("run", "sagemath-mcp")
DEFAULT_SAGEMATH_MCP_TOOLS: tuple[str, ...] = (
    "evaluate_sage",
    "calculate_expression",
    "solve_equation",
    "differentiate_expression",
    "integrate_expression",
)
DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS = 300.0
MAX_JUDGE_RETRY_TOKENS = 65535


@dataclass
class JudgeResult:
    verdict: bool | None
    total_score: int | None
    criteria: dict[str, int] | None
    rationale: str | None
    raw_response: str | None
    error: str | None
    tool_calls: int | None = None
    stop_reason: str | None = None
    used_sagemath_mcp: bool | None = None


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
    allow_sagemath_tools: bool = False,
) -> str:
    refs = []
    if cleo_reference_latex:
        refs.append(f"- Cleo reference: {cleo_reference_latex}")
    if accepted_reference_latex:
        refs.append(f"- Accepted reference: {accepted_reference_latex}")
    refs_text = "\n".join(refs) if refs else "- No reference expression available"

    integral_text = integral_latex or "<unknown integral>"

    tool_guidance = ""
    if allow_sagemath_tools:
        tool_guidance = (
            "SageMath tools are available.\n"
            "Call at least one SageMath tool to verify symbolic/numeric equivalence before grading.\n"
            "When writing Sage/Python tool code, use ** for exponentiation (never ^).\n"
            "Use explicit multiplication (`a*b`) and declare free symbols with `var('a b ...')` in evaluate_sage.\n"
            "Then return the final grading JSON.\n"
        )

    return (
        "You are grading a symbolic integration answer.\n"
        f"{tool_guidance}"
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


def _normalize_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            trimmed = item.strip()
            if trimmed:
                out.append(trimmed)
    return out


def _count_tool_calls(messages: Sequence[Any]) -> int:
    calls = 0
    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "tool":
            calls += 1
            continue

        msg_tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(msg_tool_calls, Sequence):
            calls += len(msg_tool_calls)
    return calls


async def run_judge_with_inspect(
    prompt: str,
    model_name: str | None = None,
    model_role: str = "grader",
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
    reasoning_effort: str | None = None,
    api_key: str | None = None,
    api_key_env: str | None = None,
    use_sagemath_mcp: bool = False,
    sagemath_mcp_command: str | None = None,
    sagemath_mcp_args: str | Sequence[str] | None = None,
    sagemath_mcp_cwd: str | None = None,
    sagemath_mcp_tools: str | Sequence[str] | None = None,
    sagemath_mcp_allow_imports: bool = False,
    sagemath_mcp_allowed_imports: str | Sequence[str] | None = None,
    sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
) -> JudgeResult:
    """Run judge using Inspect's model provider abstraction.

    If model_name is None, it resolves by role (default: grader).
    """
    try:
        from inspect_ai.model import GenerateConfig, get_model
    except Exception as ex:  # noqa: BLE001
        return JudgeResult(None, None, None, None, None, f"inspect_import_error: {ex}")

    resolved_api_key = api_key
    if not resolved_api_key and api_key_env:
        resolved_api_key = os.getenv(api_key_env)

    try:
        model = (
            get_model(model_name, api_key=resolved_api_key)
            if model_name
            else get_model(role=model_role, api_key=resolved_api_key)
        )
    except Exception as ex:  # noqa: BLE001
        return JudgeResult(None, None, None, None, None, f"judge_model_init_error: {ex}")

    config = GenerateConfig(
        max_tokens=(int(max_tokens) if max_tokens and max_tokens > 0 else None),
        reasoning_effort=reasoning_effort,
    )

    async def run_once(tokens: int | None, active_prompt: str) -> tuple[str, str, int]:
        active_cfg = config.model_copy(
            update={"max_tokens": (tokens if tokens and tokens > 0 else None)}
        )

        if not use_sagemath_mcp:
            response = await model.generate(active_prompt, config=active_cfg)
            return response.completion, response.stop_reason, 0

        try:
            from inspect_ai.tool import mcp_connection, mcp_server_stdio, mcp_tools
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError(f"judge_mcp_import_error: {ex}") from ex

        command = (sagemath_mcp_command or "").strip()
        if not command:
            raise RuntimeError("judge_mcp_missing_command")

        if sagemath_mcp_args is None:
            args = list(DEFAULT_SAGEMATH_MCP_ARGS) if command == "uv" else []
        else:
            args = _normalize_list(sagemath_mcp_args)

        tool_filters = _normalize_list(sagemath_mcp_tools)
        if not tool_filters:
            tool_filters = list(DEFAULT_SAGEMATH_MCP_TOOLS)

        server_env: dict[str, str] = {}
        if sagemath_mcp_eval_timeout_seconds is not None and sagemath_mcp_eval_timeout_seconds > 0:
            server_env["SAGEMATH_MCP_EVAL_TIMEOUT"] = str(float(sagemath_mcp_eval_timeout_seconds))
        if sagemath_mcp_allow_imports:
            server_env["SAGEMATH_MCP_SECURITY_ALLOW_IMPORTS"] = "1"
            allowed_imports = _normalize_list(sagemath_mcp_allowed_imports)
            if allowed_imports:
                server_env["SAGEMATH_MCP_SECURITY_ALLOWED_IMPORTS"] = ",".join(allowed_imports)

        server = mcp_server_stdio(
            name="sagemath",
            command=command,
            args=args,
            cwd=Path(sagemath_mcp_cwd) if sagemath_mcp_cwd else None,
            env=(server_env or None),
        )
        tool_source = mcp_tools(server, tools=tool_filters)
        async with mcp_connection([tool_source]):
            messages, response = await model.generate_loop(active_prompt, tools=[tool_source], config=active_cfg)
        tool_call_count = _count_tool_calls(messages)
        return response.completion, response.stop_reason, tool_call_count

    try:
        completion, stop_reason, tool_call_count = await run_once(config.max_tokens, prompt)

        # If MCP is enabled but no tool call occurred, force one explicit tool-use attempt.
        if use_sagemath_mcp and tool_call_count == 0:
            forced_prompt = (
                f"{prompt}\n\n"
                "MANDATORY STEP: Before grading, call at least one SageMath tool to verify the candidate. "
                "After tool use, return the required grading JSON."
            )
            completion, stop_reason, tool_call_count = await run_once(config.max_tokens, forced_prompt)

        parsed = parse_judge_response(completion)

        # Retry once with a larger budget if the judge output was truncated.
        if (
            parsed.error is not None
            and stop_reason in ("max_tokens", "model_length")
            and config.max_tokens is not None
        ):
            retry_max_tokens = min(max(config.max_tokens * 2, config.max_tokens + 1024), MAX_JUDGE_RETRY_TOKENS)
            if retry_max_tokens > config.max_tokens:
                completion, stop_reason, tool_call_count = await run_once(retry_max_tokens, prompt)
                parsed = parse_judge_response(completion)

        if parsed.error is not None:
            parsed.error = f"{parsed.error} (stop_reason={stop_reason})"
        parsed.stop_reason = stop_reason
        parsed.tool_calls = tool_call_count
        parsed.used_sagemath_mcp = use_sagemath_mcp
        return parsed
    except Exception as ex:  # noqa: BLE001
        return JudgeResult(None, None, None, None, None, f"judge_generate_error: {ex}")
