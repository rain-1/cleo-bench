#!/usr/bin/env python3
"""Create a random Inspect subset of Cleo questions not seen in prior eval logs."""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("data/inspect/cleo_bench_full_question.jsonl")
DEFAULT_LOGS_DIR = Path("logs")
DEFAULT_OUTPUT_DIR = Path("data/inspect/subsets")
DEFAULT_TASK_FULL = "cleo_bench_full_question"
DEFAULT_TASK_INTEGRAL = "cleo_bench_integral_only"

DEFAULT_MODEL = "openrouter/deepseek/deepseek-v3.2"
DEFAULT_JUDGE_MODEL = "openrouter/openai/gpt-oss-120b:free"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MATH_KEY_ENV = "OPENROUTER_API_KEY_MATH"
DEFAULT_JUDGE_KEY_ENV = "OPENROUTER_API_KEY_JUDGE"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_MAX_TOKENS = 65535
DEFAULT_JUDGE_MAX_TOKENS = 8192
DEFAULT_SAGEMATH_EVAL_TIMEOUT_SECONDS = 300.0
DEFAULT_SOLVER_SAGEMATH_MCP_ARGS = ""
DEFAULT_SOLVER_SAGEMATH_MCP_TOOLS = "evaluate_sage,calculate_expression"
INTEGRAL_SUFFIX = "_integral"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pick random Cleo questions not present in prior Inspect .eval logs, "
            "write a subset JSONL, and print an inspect eval command."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Inspect JSONL dataset path.")
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS_DIR, help="Directory with .eval logs.")
    parser.add_argument("--count", type=int, default=5, help="Number of unseen items to sample.")
    parser.add_argument(
        "--pool",
        choices=("scorable", "all"),
        default="all",
        help="Sample only metadata.is_scorable_numeric=true, or all items.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output subset JSONL path. Defaults to data/inspect/subsets/<timestamp>.jsonl",
    )
    parser.add_argument(
        "--task",
        choices=(DEFAULT_TASK_FULL, DEFAULT_TASK_INTEGRAL, "auto"),
        default="auto",
        help="Inspect task to print. Default: infer from dataset filename.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Inspect model for solving.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Inspect grader model.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Model provider base URL.")
    parser.add_argument(
        "--math-key-env",
        default=DEFAULT_MATH_KEY_ENV,
        help="Env var name containing solver API key.",
    )
    parser.add_argument(
        "--judge-key-env",
        default=DEFAULT_JUDGE_KEY_ENV,
        help="Env var name containing judge API key.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=DEFAULT_REASONING_EFFORT,
        help="Inspect reasoning effort. Set empty string to skip reasoning flags.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max completion tokens for the solver model (set <=0 to omit).",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=DEFAULT_JUDGE_MAX_TOKENS,
        help="Max completion tokens for the judge model (set <=0 to omit).",
    )
    parser.add_argument(
        "--judge-use-sagemath-mcp",
        action="store_true",
        help="Enable SageMath MCP tools for judge fallback scoring.",
    )
    parser.add_argument(
        "--judge-sagemath-mcp-command",
        default="sagemath-mcp",
        help="Command used to launch the SageMath MCP server (when enabled).",
    )
    parser.add_argument(
        "--judge-sagemath-mcp-args",
        default="",
        help="Comma-separated args for the SageMath MCP command.",
    )
    parser.add_argument(
        "--solver-use-sagemath-mcp",
        action="store_true",
        help="Enable SageMath MCP tools for solver generation.",
    )
    parser.add_argument(
        "--solver-require-sagemath-tool-call",
        action="store_true",
        help="Prompt solver to call at least one SageMath tool before final answer.",
    )
    parser.add_argument(
        "--solver-sagemath-mcp-command",
        default="sagemath-mcp",
        help="Command used to launch the solver SageMath MCP server.",
    )
    parser.add_argument(
        "--solver-sagemath-mcp-args",
        default=DEFAULT_SOLVER_SAGEMATH_MCP_ARGS,
        help="Comma-separated args for the solver SageMath MCP command.",
    )
    parser.add_argument(
        "--solver-sagemath-mcp-tools",
        default=DEFAULT_SOLVER_SAGEMATH_MCP_TOOLS,
        help="Comma-separated SageMath tool names exposed to the solver.",
    )
    parser.add_argument(
        "--solver-sagemath-tool-choice",
        choices=("auto", "any", "none"),
        default="auto",
        help="Inspect tool_choice for solver-side SageMath tools.",
    )
    parser.add_argument(
        "--sagemath-allow-imports",
        action="store_true",
        help="Enable import statements for both solver/judge SageMath MCP server processes.",
    )
    parser.add_argument(
        "--sagemath-allowed-imports",
        default="",
        help="Optional comma-separated allowed imports for SageMath MCP when imports are enabled.",
    )
    parser.add_argument(
        "--sagemath-eval-timeout-seconds",
        type=float,
        default=DEFAULT_SAGEMATH_EVAL_TIMEOUT_SECONDS,
        help="Per Sage tool execution timeout in seconds for solver/judge MCP servers (set <=0 to disable override).",
    )
    parser.add_argument(
        "--show-tried",
        action="store_true",
        help="Print all normalized tried sample IDs seen in logs.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object.")
            rows.append(payload)
    return rows


def _normalize_sample_id(sample_id: str) -> str:
    if sample_id.endswith(INTEGRAL_SUFFIX):
        return sample_id[: -len(INTEGRAL_SUFFIX)]
    return sample_id


def collect_tried_ids(logs_dir: Path) -> tuple[set[str], list[str]]:
    tried: set[str] = set()
    warnings: list[str] = []

    if not logs_dir.exists():
        return tried, [f"logs directory not found: {logs_dir}"]

    for archive_path in sorted(logs_dir.glob("*.eval")):
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = set(archive.namelist())

                if "header.json" in members:
                    try:
                        header = json.loads(archive.read("header.json").decode("utf-8"))
                        dataset_ids = (
                            header.get("eval", {}).get("dataset", {}).get("sample_ids", [])
                        )
                        if isinstance(dataset_ids, list):
                            for sample_id in dataset_ids:
                                if isinstance(sample_id, str):
                                    tried.add(_normalize_sample_id(sample_id))
                    except Exception as ex:  # noqa: BLE001
                        warnings.append(f"{archive_path.name}: could not parse header.json ({ex})")

                for member in members:
                    if not member.startswith("samples/") or not member.endswith(".json"):
                        continue
                    try:
                        sample = json.loads(archive.read(member).decode("utf-8"))
                    except Exception as ex:  # noqa: BLE001
                        warnings.append(f"{archive_path.name}: could not parse {member} ({ex})")
                        continue

                    sample_id = sample.get("id")
                    if isinstance(sample_id, str):
                        tried.add(_normalize_sample_id(sample_id))

        except zipfile.BadZipFile:
            warnings.append(f"{archive_path.name}: invalid zip archive")
        except Exception as ex:  # noqa: BLE001
            warnings.append(f"{archive_path.name}: unexpected error ({ex})")

    return tried, warnings


def _infer_task(dataset_path: Path) -> str:
    stem = dataset_path.stem.lower()
    if "integral_only" in stem:
        return DEFAULT_TASK_INTEGRAL
    return DEFAULT_TASK_FULL


def _default_output_path(dataset_path: Path, output_dir: Path, count: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"random_untried_{dataset_path.stem}_{count}_{timestamp}.jsonl"


def _build_inspect_command(
    *,
    task_name: str,
    dataset_path: Path,
    model: str,
    judge_model: str,
    base_url: str,
    math_key_env: str,
    judge_key_env: str,
    reasoning_effort: str,
    max_tokens: int,
    judge_max_tokens: int,
    judge_use_sagemath_mcp: bool,
    judge_sagemath_mcp_command: str,
    judge_sagemath_mcp_args: str,
    solver_use_sagemath_mcp: bool,
    solver_require_sagemath_tool_call: bool,
    solver_sagemath_mcp_command: str,
    solver_sagemath_mcp_args: str,
    solver_sagemath_mcp_tools: str,
    solver_sagemath_tool_choice: str,
    sagemath_allow_imports: bool,
    sagemath_allowed_imports: str,
    sagemath_eval_timeout_seconds: float,
) -> str:
    lines = [
        f"OPENROUTER_API_KEY=\"${{{math_key_env}}}\" \\",
        f"inspect eval inspect_tasks/cleo_bench_eval.py@{task_name} \\",
        f"  --model {model} \\",
        f"  --model-base-url {base_url} \\",
        f"  -T judge_model={judge_model} \\",
        f"  -T judge_api_key_env={judge_key_env} \\",
        ]

    if reasoning_effort.strip():
        lines.extend(
            [
                "  -M reasoning_enabled=true \\",
                f"  --reasoning-effort {reasoning_effort} \\",
            ]
        )

    if max_tokens > 0:
        lines.append(f"  --max-tokens {max_tokens} \\")

    if judge_max_tokens > 0:
        lines.append(f"  -T judge_max_tokens={judge_max_tokens} \\")

    if judge_use_sagemath_mcp:
        lines.extend(
            [
                "  -T judge_use_sagemath_mcp=true \\",
                f"  -T judge_sagemath_mcp_command={judge_sagemath_mcp_command} \\",
            ]
        )
        if judge_sagemath_mcp_args.strip():
            lines.append(f"  -T judge_sagemath_mcp_args={judge_sagemath_mcp_args} \\")
        if sagemath_eval_timeout_seconds > 0:
            lines.append(
                f"  -T judge_sagemath_mcp_eval_timeout_seconds={float(sagemath_eval_timeout_seconds)} \\"
            )
        if sagemath_allow_imports:
            lines.append("  -T judge_sagemath_mcp_allow_imports=true \\")
            if sagemath_allowed_imports.strip():
                lines.append(f"  -T judge_sagemath_mcp_allowed_imports={sagemath_allowed_imports} \\")

    if solver_use_sagemath_mcp:
        lines.extend(
            [
                "  -T solver_use_sagemath_mcp=true \\",
                f"  -T solver_sagemath_mcp_command={solver_sagemath_mcp_command} \\",
                f"  -T solver_sagemath_mcp_tools={solver_sagemath_mcp_tools} \\",
                f"  -T solver_sagemath_tool_choice={solver_sagemath_tool_choice} \\",
            ]
        )
        if solver_sagemath_mcp_args.strip():
            lines.append(f"  -T solver_sagemath_mcp_args={solver_sagemath_mcp_args} \\")
        if sagemath_eval_timeout_seconds > 0:
            lines.append(
                f"  -T solver_sagemath_mcp_eval_timeout_seconds={float(sagemath_eval_timeout_seconds)} \\"
            )
        if solver_require_sagemath_tool_call:
            lines.append("  -T solver_require_sagemath_tool_call=true \\")
        if sagemath_allow_imports:
            lines.append("  -T solver_sagemath_mcp_allow_imports=true \\")
            if sagemath_allowed_imports.strip():
                lines.append(f"  -T solver_sagemath_mcp_allowed_imports={sagemath_allowed_imports} \\")

    lines.append(f"  -T dataset_file={dataset_path.as_posix()}")
    return "\n".join(lines)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        print("--count must be >= 1", file=sys.stderr)
        return 2
    if not args.dataset.exists():
        print(f"Dataset does not exist: {args.dataset}", file=sys.stderr)
        return 2

    try:
        dataset_rows = _read_jsonl(args.dataset)
    except Exception as ex:  # noqa: BLE001
        print(f"Failed to load dataset {args.dataset}: {ex}", file=sys.stderr)
        return 2

    tried_ids, warnings = collect_tried_ids(args.logs_dir)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    candidate_rows: list[dict[str, Any]] = []
    for row in dataset_rows:
        sample_id = row.get("id")
        if not isinstance(sample_id, str):
            continue
        normalized_id = _normalize_sample_id(sample_id)
        if normalized_id in tried_ids:
            continue
        if args.pool == "scorable":
            metadata = row.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("is_scorable_numeric") is not True:
                continue
        candidate_rows.append(row)

    if len(candidate_rows) < args.count:
        print(
            (
                f"Not enough untried items in pool={args.pool}. "
                f"Requested {args.count}, available {len(candidate_rows)}."
            ),
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.seed)
    selected_rows = rng.sample(candidate_rows, args.count)

    output_path = args.output or _default_output_path(args.dataset, DEFAULT_OUTPUT_DIR, args.count)
    _write_jsonl(output_path, selected_rows)

    task_name = _infer_task(args.dataset) if args.task == "auto" else args.task

    print(f"dataset: {args.dataset}")
    print(f"logs_dir: {args.logs_dir}")
    print(f"pool: {args.pool}")
    print(f"total_dataset_rows: {len(dataset_rows)}")
    print(f"tried_ids_in_logs: {len(tried_ids)}")
    print(f"eligible_untried: {len(candidate_rows)}")
    print(f"selected: {len(selected_rows)}")
    print(f"subset_file: {output_path.as_posix()}")

    print("\nselected_ids:")
    for row in selected_rows:
        sample_id = str(row.get("id", ""))
        metadata = row.get("metadata")
        question_url = metadata.get("question_url", "") if isinstance(metadata, dict) else ""
        print(f"- {sample_id} {question_url}".rstrip())

    if args.show_tried:
        print("\ntried_ids:")
        for sample_id in sorted(tried_ids):
            print(f"- {sample_id}")

    cmd = _build_inspect_command(
        task_name=task_name,
        dataset_path=output_path,
        model=args.model,
        judge_model=args.judge_model,
        base_url=args.base_url,
        math_key_env=args.math_key_env,
        judge_key_env=args.judge_key_env,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        judge_use_sagemath_mcp=args.judge_use_sagemath_mcp,
        judge_sagemath_mcp_command=args.judge_sagemath_mcp_command,
        judge_sagemath_mcp_args=args.judge_sagemath_mcp_args,
        solver_use_sagemath_mcp=args.solver_use_sagemath_mcp,
        solver_require_sagemath_tool_call=args.solver_require_sagemath_tool_call,
        solver_sagemath_mcp_command=args.solver_sagemath_mcp_command,
        solver_sagemath_mcp_args=args.solver_sagemath_mcp_args,
        solver_sagemath_mcp_tools=args.solver_sagemath_mcp_tools,
        solver_sagemath_tool_choice=args.solver_sagemath_tool_choice,
        sagemath_allow_imports=args.sagemath_allow_imports,
        sagemath_allowed_imports=args.sagemath_allowed_imports,
        sagemath_eval_timeout_seconds=args.sagemath_eval_timeout_seconds,
    )
    print("\nrun_command:")
    print(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
