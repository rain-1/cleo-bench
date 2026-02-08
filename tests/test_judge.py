from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import unittest
from unittest.mock import AsyncMock, patch

from inspect_ai.model import ModelOutput

from cleo_bench.judge import build_judge_prompt, run_judge_with_inspect


class JudgeTests(unittest.TestCase):
    def test_prompt_includes_sagemath_guidance(self) -> None:
        prompt = build_judge_prompt(
            problem_statement="Evaluate integral",
            candidate_expression_latex=r"\frac{1}{2}",
            integral_latex=r"\int_0^1 x\,dx",
            cleo_reference_latex=r"\frac{1}{2}",
            accepted_reference_latex=None,
            allow_sagemath_tools=True,
        )
        self.assertIn("SageMath tools are available.", prompt)
        self.assertIn("use ** for exponentiation", prompt)

    def test_judge_retries_after_truncation(self) -> None:
        truncated = ModelOutput.from_content(
            model="fake/model",
            content='{"pass": true, "total_score":',
            stop_reason="max_tokens",
        )
        complete = ModelOutput.from_content(
            model="fake/model",
            content=(
                '{"pass": true, "total_score": 16, "criteria": '
                '{"equivalence": 4, "constants_and_branches": 4, '
                '"task_completeness": 4, "expression_validity": 4}, '
                '"rationale": "OK"}'
            ),
            stop_reason="stop",
        )

        fake_model = AsyncMock()
        fake_model.generate = AsyncMock(side_effect=[truncated, complete])

        with patch("inspect_ai.model.get_model", return_value=fake_model):
            result = asyncio.run(
                run_judge_with_inspect(
                    prompt="grade this",
                    model_name="fake/model",
                    max_tokens=10,
                )
            )

        self.assertIsNone(result.error)
        self.assertTrue(result.verdict)
        self.assertEqual(result.total_score, 16)
        self.assertEqual(fake_model.generate.await_count, 2)

    def test_judge_mcp_forces_second_attempt_when_no_tool_call(self) -> None:
        @asynccontextmanager
        async def fake_connection(_tools):
            yield

        class Msg:
            def __init__(self, role: str, tool_calls=None) -> None:
                self.role = role
                self.tool_calls = tool_calls

        first = ModelOutput.from_content(
            model="fake/model",
            content=(
                '{"pass": true, "total_score": 15, "criteria": '
                '{"equivalence": 4, "constants_and_branches": 4, '
                '"task_completeness": 4, "expression_validity": 3}, '
                '"rationale": "OK"}'
            ),
            stop_reason="stop",
        )
        second = ModelOutput.from_content(
            model="fake/model",
            content=(
                '{"pass": true, "total_score": 16, "criteria": '
                '{"equivalence": 4, "constants_and_branches": 4, '
                '"task_completeness": 4, "expression_validity": 4}, '
                '"rationale": "Used tools"}'
            ),
            stop_reason="stop",
        )

        fake_model = AsyncMock()
        fake_model.generate_loop = AsyncMock(
            side_effect=[
                ([Msg("assistant", [])], first),
                ([Msg("assistant", [{"id": "call_1"}]), Msg("tool")], second),
            ]
        )

        with (
            patch("inspect_ai.model.get_model", return_value=fake_model),
            patch("inspect_ai.tool.mcp_server_stdio", return_value=object()) as patched_server_stdio,
            patch("inspect_ai.tool.mcp_tools", return_value=object()),
            patch("inspect_ai.tool.mcp_connection", side_effect=fake_connection),
        ):
            result = asyncio.run(
                run_judge_with_inspect(
                    prompt="grade this",
                    model_name="fake/model",
                    use_sagemath_mcp=True,
                    sagemath_mcp_command="uv",
                    sagemath_mcp_args=["run", "sagemath-mcp"],
                    sagemath_mcp_eval_timeout_seconds=300.0,
                )
            )

        self.assertIsNone(result.error)
        self.assertEqual(fake_model.generate_loop.await_count, 2)
        self.assertEqual(result.tool_calls, 2)
        self.assertEqual(result.stop_reason, "stop")
        self.assertTrue(result.used_sagemath_mcp)
        server_env = patched_server_stdio.call_args.kwargs.get("env") or {}
        self.assertEqual(server_env.get("SAGEMATH_MCP_EVAL_TIMEOUT"), "300.0")


if __name__ == "__main__":
    unittest.main()
