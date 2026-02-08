"""Inspect task definitions for Cleo Bench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

from cleo_bench.judge import (
    DEFAULT_SAGEMATH_MCP_ARGS,
    DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    DEFAULT_SAGEMATH_MCP_TOOLS,
)
from cleo_bench.scoring import score_with_optional_judge

SYSTEM_PROMPT_BASE = (
    "You are solving hard symbolic integration tasks. "
    "Prioritize algebraic/symbolic derivation of a closed form. "
    "Use numerical computation mainly to verify or discriminate symbolic candidates. "
    "Avoid guessing by fitting random combinations of constants to a decimal value. "
    "Return JSON only with key `final_expression_latex` and optional key `notes`."
)


def _resolve_dataset_path(dataset_file: str) -> str:
    path = Path(dataset_file)
    if path.is_absolute():
        return path.as_posix()
    repo_root = Path(__file__).resolve().parents[1]
    return (repo_root / path).as_posix()


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


def _tool_choice(value: str) -> Literal["auto", "any", "none"]:
    normalized = value.strip().lower()
    if normalized in ("auto", "any", "none"):
        return normalized
    raise ValueError(
        f"Invalid solver_sagemath_tool_choice={value!r}. "
        "Use one of: auto, any, none."
    )


def _solver_tool_calls(messages: Sequence[Any]) -> int:
    count = 0
    for message in messages:
        if getattr(message, "role", None) == "tool":
            count += 1
    return count


def _solver_tool_errors(messages: Sequence[Any]) -> list[str]:
    errors: list[str] = []
    for message in messages:
        if getattr(message, "role", None) != "tool":
            continue
        err = getattr(message, "error", None)
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg.strip():
                errors.append(msg.strip())
    return errors


def _tool_errors_need_retry(errors: Sequence[str]) -> bool:
    patterns = (
        "unsupported operand type(s) for ^",
        "invalid syntax",
        "syntax error",
        "unexpected eof",
        "not defined",
        "missing",
        "typeerror",
        "nameerror",
    )
    for err in errors:
        low = err.lower()
        if any(p in low for p in patterns):
            return True
    return False


def _empty_output_fallback_json(messages: Sequence[Any]) -> str:
    note = "Model returned no visible answer text after retries."
    errors = _solver_tool_errors(messages)
    if errors:
        sampled = "; ".join(errors[:2])
        note += f" Tool errors: {sampled}"
    payload = {
        "final_expression_latex": r"\text{UNRESOLVED}",
        "notes": note,
    }
    return json.dumps(payload, ensure_ascii=False)


def _assistant_visible_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, Sequence):
        return ""
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _sync_completion_from_output_message(state: TaskState) -> None:
    completion = (state.output.completion or "").strip()
    if completion:
        return
    message = getattr(state.output, "message", None)
    text = _assistant_visible_text(message)
    if text:
        state.output.completion = text


def _solver_system_prompt(
    *,
    use_sagemath_mcp: bool,
    require_sagemath_tool_call: bool,
) -> str:
    if not use_sagemath_mcp:
        return SYSTEM_PROMPT_BASE

    usage_line = (
        "Call at least one SageMath tool before finalizing your answer."
        if require_sagemath_tool_call
        else "Use SageMath tools whenever they help verification."
    )
    return (
        "You are solving hard symbolic integration tasks. "
        "Prioritize algebraic/symbolic derivation of a closed form. "
        "Use numerical computation mainly to verify or discriminate symbolic candidates. "
        "Avoid guessing by fitting random combinations of constants to a decimal value. "
        "SageMath tools are available for numerical calculations, symbolic algebra, "
        "simplification, differentiation/integration checks, and branch/constant checks. "
        "Your final JSON must be in visible assistant text, not only in reasoning content. "
        "When writing Sage/Python tool code, use `**` for exponentiation (never `^`). "
        "For symbolic parameters/free symbols, prefer `evaluate_sage` and declare them explicitly "
        "with `var('a b ...')` before use. "
        "`calculate_expression` is best for expressions in built-in symbols like x,y,z,t. "
        "Never emit pseudo tool-call JSON/text in your answer; call tools through the tool interface only. "
        "If a tool fails or times out, avoid repeating the same failing call and continue with best-effort math. "
        f"{usage_line} "
        "Return JSON only with key `final_expression_latex` and optional key `notes`."
    )


@solver
def generate_with_sagemath_mcp(
    *,
    mcp_command: str,
    mcp_args: str | Sequence[str] | None = None,
    mcp_cwd: str | None = None,
    mcp_tools: str | Sequence[str] | None = None,
    mcp_allow_imports: bool = False,
    mcp_allowed_imports: str | Sequence[str] | None = None,
    mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    require_tool_call: bool = False,
    tool_choice: Literal["auto", "any", "none"] = "auto",
) -> Solver:
    async def solve(state: TaskState, model_generate: Generate) -> TaskState:
        try:
            from inspect_ai.tool import mcp_connection, mcp_server_stdio, mcp_tools as mcp_tools_fn
        except Exception as ex:  # noqa: BLE001
            raise RuntimeError(
                "solver_mcp_import_error: install optional MCP deps with "
                "`pip install -e .[mcp]` and install the SageMath MCP server "
                "(e.g., `pip install \"git+https://github.com/XBP-Europe/sagemath-mcp.git\"`)."
            ) from ex

        command = mcp_command.strip()
        if not command:
            raise ValueError(
                "solver_use_sagemath_mcp=true requires solver_sagemath_mcp_command."
            )

        if mcp_args is None:
            args = list(DEFAULT_SAGEMATH_MCP_ARGS) if command == "uv" else []
        else:
            args = _normalize_list(mcp_args)

        tools_filter = _normalize_list(mcp_tools)
        if not tools_filter:
            tools_filter = list(DEFAULT_SAGEMATH_MCP_TOOLS)

        server_env: dict[str, str] = {}
        if mcp_eval_timeout_seconds is not None and mcp_eval_timeout_seconds > 0:
            server_env["SAGEMATH_MCP_EVAL_TIMEOUT"] = str(float(mcp_eval_timeout_seconds))
        if mcp_allow_imports:
            server_env["SAGEMATH_MCP_SECURITY_ALLOW_IMPORTS"] = "1"
            allowed_imports = _normalize_list(mcp_allowed_imports)
            if allowed_imports:
                server_env["SAGEMATH_MCP_SECURITY_ALLOWED_IMPORTS"] = ",".join(allowed_imports)

        server = mcp_server_stdio(
            name="sagemath",
            command=command,
            args=args,
            cwd=Path(mcp_cwd) if mcp_cwd else None,
            env=(server_env or None),
        )
        tool_source = mcp_tools_fn(server, tools=tools_filter)
        state.tools = await tool_source.tools()
        state.tool_choice = tool_choice

        async with mcp_connection([tool_source]):
            state = await model_generate(state, tool_calls="loop")
            _sync_completion_from_output_message(state)

            if require_tool_call and _solver_tool_calls(state.messages) == 0:
                state.messages.append(
                    ChatMessageUser(
                        content=(
                            "Tool-use requirement not met. Call at least one SageMath tool now, "
                            "then continue and finish with JSON."
                        )
                    )
                )
                previous_choice = state.tool_choice
                state.tool_choice = "any"
                state = await model_generate(state, tool_calls="loop")
                state.tool_choice = previous_choice
                _sync_completion_from_output_message(state)

            if not (state.output.completion or "").strip():
                tool_errors = _solver_tool_errors(state.messages)
                if tool_errors and _tool_errors_need_retry(tool_errors):
                    sampled = "; ".join(tool_errors[:2])
                    state.messages.append(
                        ChatMessageUser(
                            content=(
                                "Previous SageMath tool calls failed. "
                                f"Errors: {sampled}. "
                                "Run one corrected SageMath tool call now, then return final JSON. "
                                "Use explicit multiplication (`a*b`), use `**` for powers, "
                                "and declare free symbols in `evaluate_sage` using "
                                "`var('a b t ...')` before using them."
                            )
                        )
                    )
                    previous_choice = state.tool_choice
                    state.tool_choice = "any"
                    state = await model_generate(state, tool_calls="loop")
                    state.tool_choice = previous_choice
                    _sync_completion_from_output_message(state)

            if not (state.output.completion or "").strip():
                tool_errors = _solver_tool_errors(state.messages)
                tool_error_note = ""
                if tool_errors:
                    sampled = "; ".join(tool_errors[:2])
                    tool_error_note = (
                        f" Prior tool calls failed or timed out ({sampled}). "
                        "Do not repeat those failing calls."
                    )
                    if any("not defined" in e for e in tool_errors):
                        tool_error_note += (
                            " If a symbol is undefined, use `evaluate_sage` and declare symbols "
                            "first (e.g., `var('a b t')`)."
                        )
                state.messages.append(
                    ChatMessageUser(
                        content=(
                            "Your prior reply had no visible final answer."
                            f"{tool_error_note} "
                            "Return JSON only now with this schema: "
                            '{"final_expression_latex":"<latex expression>","notes":"<optional>"} '
                            "Do not include markdown or reasoning."
                        )
                    )
                )
                previous_choice = state.tool_choice
                state.tool_choice = "none"
                state = await model_generate(state, tool_calls="none")
                state.tool_choice = previous_choice
                _sync_completion_from_output_message(state)

            if not (state.output.completion or "").strip():
                state.messages.append(
                    ChatMessageUser(
                        content=(
                            "FINAL REQUIRED: output exactly one JSON object now, no extra text. "
                            'Example: {"final_expression_latex":"\\\\pi","notes":"optional"}'
                        )
                    )
                )
                previous_choice = state.tool_choice
                state.tool_choice = "none"
                state = await model_generate(state, tool_calls="none")
                state.tool_choice = previous_choice
                _sync_completion_from_output_message(state)

            if not (state.output.completion or "").strip():
                state.output.completion = _empty_output_fallback_json(state.messages)

            return state

    return solve


def _build_solver(
    *,
    solver_use_sagemath_mcp: bool,
    solver_require_sagemath_tool_call: bool,
    solver_sagemath_mcp_command: str | None,
    solver_sagemath_mcp_args: str | Sequence[str] | None,
    solver_sagemath_mcp_cwd: str | None,
    solver_sagemath_mcp_tools: str | Sequence[str] | None,
    solver_sagemath_mcp_allow_imports: bool,
    solver_sagemath_mcp_allowed_imports: str | Sequence[str] | None,
    solver_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    solver_sagemath_tool_choice: str = "auto",
) -> list[Solver]:
    prompt = _solver_system_prompt(
        use_sagemath_mcp=solver_use_sagemath_mcp,
        require_sagemath_tool_call=solver_require_sagemath_tool_call,
    )

    if not solver_use_sagemath_mcp:
        return [system_message(prompt), generate()]

    command = (solver_sagemath_mcp_command or "").strip()
    if not command:
        raise ValueError(
            "solver_use_sagemath_mcp=true requires solver_sagemath_mcp_command."
        )

    return [
        system_message(prompt),
        generate_with_sagemath_mcp(
            mcp_command=command,
            mcp_args=solver_sagemath_mcp_args,
            mcp_cwd=solver_sagemath_mcp_cwd,
            mcp_tools=solver_sagemath_mcp_tools,
            mcp_allow_imports=solver_sagemath_mcp_allow_imports,
            mcp_allowed_imports=solver_sagemath_mcp_allowed_imports,
            mcp_eval_timeout_seconds=solver_sagemath_mcp_eval_timeout_seconds,
            require_tool_call=solver_require_sagemath_tool_call,
            tool_choice=_tool_choice(solver_sagemath_tool_choice),
        ),
    ]


@scorer(metrics=[mean(), stderr()])
def cleo_bench_scorer(
    use_judge_when_unresolved: bool = True,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    judge_api_key_env: str | None = "OPENROUTER_API_KEY_JUDGE",
    judge_max_tokens: int = 8192,
    judge_reasoning_effort: str | None = None,
    judge_use_sagemath_mcp: bool = False,
    judge_sagemath_mcp_command: str | None = "sagemath-mcp",
    judge_sagemath_mcp_args: str | list[str] | None = None,
    judge_sagemath_mcp_cwd: str | None = None,
    judge_sagemath_mcp_tools: str | list[str] | None = None,
    judge_sagemath_mcp_allow_imports: bool = False,
    judge_sagemath_mcp_allowed_imports: str | list[str] | None = None,
    judge_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    solver_use_sagemath_mcp: bool = False,
    dps: int = 80,
    tolerance: float = 1e-6,
):
    async def score(state: TaskState, target: Target) -> Score:
        _ = target  # Not used; metadata carries references.
        metadata = state.metadata or {}
        result = await score_with_optional_judge(
            output_text=state.output.completion,
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
        result["solver_tool_calls"] = _solver_tool_calls(state.messages)
        result["solver_used_sagemath_mcp"] = solver_use_sagemath_mcp

        value = result.get("final_score")
        if value is None:
            value = 0.0

        return Score(
            value=float(value),
            answer=state.output.completion,
            explanation=result.get("explanation"),
            metadata=result,
        )

    return score


@task(name="cleo_bench_full_question")
def cleo_bench_full_question(
    dataset_file: str = "data/inspect/cleo_bench_full_question.jsonl",
    use_judge_when_unresolved: bool = True,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    judge_api_key_env: str | None = "OPENROUTER_API_KEY_JUDGE",
    judge_max_tokens: int = 8192,
    judge_reasoning_effort: str | None = None,
    judge_use_sagemath_mcp: bool = False,
    judge_sagemath_mcp_command: str | None = "sagemath-mcp",
    judge_sagemath_mcp_args: str | list[str] | None = None,
    judge_sagemath_mcp_cwd: str | None = None,
    judge_sagemath_mcp_tools: str | list[str] | None = None,
    judge_sagemath_mcp_allow_imports: bool = False,
    judge_sagemath_mcp_allowed_imports: str | list[str] | None = None,
    judge_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    solver_use_sagemath_mcp: bool = False,
    solver_require_sagemath_tool_call: bool = True,
    solver_sagemath_mcp_command: str | None = "sagemath-mcp",
    solver_sagemath_mcp_args: str | list[str] | None = None,
    solver_sagemath_mcp_cwd: str | None = None,
    solver_sagemath_mcp_tools: str | list[str] | None = None,
    solver_sagemath_mcp_allow_imports: bool = False,
    solver_sagemath_mcp_allowed_imports: str | list[str] | None = None,
    solver_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    solver_sagemath_tool_choice: str = "auto",
    dps: int = 80,
    tolerance: float = 1e-6,
) -> Task:
    dataset_path = _resolve_dataset_path(dataset_file)
    solver_steps = _build_solver(
        solver_use_sagemath_mcp=solver_use_sagemath_mcp,
        solver_require_sagemath_tool_call=solver_require_sagemath_tool_call,
        solver_sagemath_mcp_command=solver_sagemath_mcp_command,
        solver_sagemath_mcp_args=solver_sagemath_mcp_args,
        solver_sagemath_mcp_cwd=solver_sagemath_mcp_cwd,
        solver_sagemath_mcp_tools=solver_sagemath_mcp_tools,
        solver_sagemath_mcp_allow_imports=solver_sagemath_mcp_allow_imports,
        solver_sagemath_mcp_allowed_imports=solver_sagemath_mcp_allowed_imports,
        solver_sagemath_mcp_eval_timeout_seconds=solver_sagemath_mcp_eval_timeout_seconds,
        solver_sagemath_tool_choice=solver_sagemath_tool_choice,
    )

    return Task(
        dataset=json_dataset(dataset_path),
        solver=solver_steps,
        scorer=cleo_bench_scorer(
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
            solver_use_sagemath_mcp=solver_use_sagemath_mcp,
            dps=dps,
            tolerance=tolerance,
        ),
    )


@task(name="cleo_bench_integral_only")
def cleo_bench_integral_only(
    dataset_file: str = "data/inspect/cleo_bench_integral_only.jsonl",
    use_judge_when_unresolved: bool = True,
    judge_model: str | None = None,
    judge_api_key: str | None = None,
    judge_api_key_env: str | None = "OPENROUTER_API_KEY_JUDGE",
    judge_max_tokens: int = 8192,
    judge_reasoning_effort: str | None = None,
    judge_use_sagemath_mcp: bool = False,
    judge_sagemath_mcp_command: str | None = "sagemath-mcp",
    judge_sagemath_mcp_args: str | list[str] | None = None,
    judge_sagemath_mcp_cwd: str | None = None,
    judge_sagemath_mcp_tools: str | list[str] | None = None,
    judge_sagemath_mcp_allow_imports: bool = False,
    judge_sagemath_mcp_allowed_imports: str | list[str] | None = None,
    judge_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    solver_use_sagemath_mcp: bool = False,
    solver_require_sagemath_tool_call: bool = True,
    solver_sagemath_mcp_command: str | None = "sagemath-mcp",
    solver_sagemath_mcp_args: str | list[str] | None = None,
    solver_sagemath_mcp_cwd: str | None = None,
    solver_sagemath_mcp_tools: str | list[str] | None = None,
    solver_sagemath_mcp_allow_imports: bool = False,
    solver_sagemath_mcp_allowed_imports: str | list[str] | None = None,
    solver_sagemath_mcp_eval_timeout_seconds: float | None = DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    solver_sagemath_tool_choice: str = "auto",
    dps: int = 80,
    tolerance: float = 1e-6,
) -> Task:
    dataset_path = _resolve_dataset_path(dataset_file)
    solver_steps = _build_solver(
        solver_use_sagemath_mcp=solver_use_sagemath_mcp,
        solver_require_sagemath_tool_call=solver_require_sagemath_tool_call,
        solver_sagemath_mcp_command=solver_sagemath_mcp_command,
        solver_sagemath_mcp_args=solver_sagemath_mcp_args,
        solver_sagemath_mcp_cwd=solver_sagemath_mcp_cwd,
        solver_sagemath_mcp_tools=solver_sagemath_mcp_tools,
        solver_sagemath_mcp_allow_imports=solver_sagemath_mcp_allow_imports,
        solver_sagemath_mcp_allowed_imports=solver_sagemath_mcp_allowed_imports,
        solver_sagemath_mcp_eval_timeout_seconds=solver_sagemath_mcp_eval_timeout_seconds,
        solver_sagemath_tool_choice=solver_sagemath_tool_choice,
    )

    return Task(
        dataset=json_dataset(dataset_path),
        solver=solver_steps,
        scorer=cleo_bench_scorer(
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
            solver_use_sagemath_mcp=solver_use_sagemath_mcp,
            dps=dps,
            tolerance=tolerance,
        ),
    )
