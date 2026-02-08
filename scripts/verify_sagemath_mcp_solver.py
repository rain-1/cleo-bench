#!/usr/bin/env python3
"""Smoke test solver-side SageMath MCP tool usage with a live model."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from inspect_ai.model import GenerateConfig, get_model

from cleo_bench.judge import (
    DEFAULT_SAGEMATH_MCP_ARGS,
    DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
    DEFAULT_SAGEMATH_MCP_TOOLS,
)

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v3.2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY_MATH"
DEFAULT_MCP_COMMAND = "sagemath-mcp"
DEFAULT_PROMPT = (
    "You have SageMath tools. Mandatory: call at least one tool before answering.\n"
    "Compute this integral in a concise way: int(1/(1+x^2), x, 0, +oo).\n"
    "Return JSON only with keys `final_expression_latex` and optional `notes`."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Call a model with SageMath MCP tools and verify at least one tool call "
            "appears in the generated conversation."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Inspect model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Model provider base URL.")
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="Env var containing model API key.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max completion tokens.")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Reasoning effort passed to Inspect model config.",
    )
    parser.add_argument(
        "--mcp-command",
        default=DEFAULT_MCP_COMMAND,
        help="Command used to launch SageMath MCP server.",
    )
    parser.add_argument(
        "--mcp-args",
        default=None,
        help="Comma-separated args for the MCP command (default: run,sagemath-mcp when command=uv).",
    )
    parser.add_argument(
        "--mcp-cwd",
        default=None,
        help="Optional working directory for MCP server process.",
    )
    parser.add_argument(
        "--mcp-tools",
        default=",".join(DEFAULT_SAGEMATH_MCP_TOOLS),
        help="Comma-separated MCP tool filters to expose.",
    )
    parser.add_argument(
        "--eval-timeout-seconds",
        type=float,
        default=DEFAULT_SAGEMATH_MCP_EVAL_TIMEOUT_SECONDS,
        help="Per Sage tool execution timeout in seconds (set <=0 to disable override).",
    )
    parser.add_argument(
        "--allow-imports",
        action="store_true",
        help="Enable import statements inside Sage tool executions.",
    )
    parser.add_argument(
        "--allowed-imports",
        default=None,
        help="Optional comma-separated allowed import modules/prefixes when imports are enabled.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for the tool-call smoke test.",
    )
    parser.add_argument(
        "--min-tool-calls",
        type=int,
        default=1,
        help="Minimum required tool calls for success.",
    )
    return parser.parse_args()


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
        if getattr(msg, "role", None) == "tool":
            calls += 1
            continue
        msg_tool_calls = getattr(msg, "tool_calls", None)
        if isinstance(msg_tool_calls, Sequence):
            calls += len(msg_tool_calls)
    return calls


async def _run(args: argparse.Namespace) -> int:
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Missing API key env: {args.api_key_env}", file=sys.stderr)
        return 2

    try:
        from inspect_ai.tool import mcp_connection, mcp_server_stdio, mcp_tools
    except Exception as ex:  # noqa: BLE001
        print(
            "MCP dependencies are missing. Install with:\n"
            "  pip install -e .[mcp]\n"
            "  pip install \"git+https://github.com/XBP-Europe/sagemath-mcp.git\"",
            file=sys.stderr,
        )
        print(f"Import error: {ex}", file=sys.stderr)
        return 2

    model = get_model(args.model, base_url=args.base_url, api_key=api_key)
    cfg = GenerateConfig(
        max_tokens=(args.max_tokens if args.max_tokens > 0 else None),
        reasoning_effort=(args.reasoning_effort if args.reasoning_effort else None),
    )

    if args.mcp_args is None:
        mcp_args = list(DEFAULT_SAGEMATH_MCP_ARGS) if args.mcp_command == "uv" else []
    else:
        mcp_args = _normalize_list(args.mcp_args)

    mcp_tool_filters = _normalize_list(args.mcp_tools)
    if not mcp_tool_filters:
        mcp_tool_filters = list(DEFAULT_SAGEMATH_MCP_TOOLS)

    server_env: dict[str, str] = {}
    if args.eval_timeout_seconds > 0:
        server_env["SAGEMATH_MCP_EVAL_TIMEOUT"] = str(float(args.eval_timeout_seconds))
    if args.allow_imports:
        server_env["SAGEMATH_MCP_SECURITY_ALLOW_IMPORTS"] = "1"
        allowed_imports = _normalize_list(args.allowed_imports)
        if allowed_imports:
            server_env["SAGEMATH_MCP_SECURITY_ALLOWED_IMPORTS"] = ",".join(allowed_imports)

    server = mcp_server_stdio(
        name="sagemath",
        command=args.mcp_command,
        args=mcp_args,
        cwd=Path(args.mcp_cwd) if args.mcp_cwd else None,
        env=(server_env or None),
    )
    tool_source = mcp_tools(server, tools=mcp_tool_filters)

    try:
        async with mcp_connection([tool_source]):
            messages, output = await model.generate_loop(args.prompt, tools=[tool_source], config=cfg)
    except FileNotFoundError as ex:
        print(
            f"MCP command not found: {args.mcp_command}\n"
            "Install/configure the SageMath MCP server command first.",
            file=sys.stderr,
        )
        print(f"Details: {ex}", file=sys.stderr)
        return 2
    except Exception as ex:  # noqa: BLE001
        message = str(ex)
        if "Annotated" in message and "not defined" in message:
            print(
                "MCP server failed with a known FastMCP compatibility issue.\n"
                "Try pinning FastMCP to 2.13.3 in this venv:\n"
                "  pip install 'fastmcp==2.13.3'",
                file=sys.stderr,
            )
        print(f"MCP smoke test failed: {ex}", file=sys.stderr)
        return 2

    tool_calls = _count_tool_calls(messages)
    print(f"tool_calls: {tool_calls}")
    print(f"stop_reason: {output.stop_reason}")
    print("completion:")
    print(output.completion)

    if tool_calls < args.min_tool_calls:
        print(
            f"FAIL: expected at least {args.min_tool_calls} tool call(s), got {tool_calls}.",
            file=sys.stderr,
        )
        return 1

    print("PASS: MCP tool call path is working.")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
