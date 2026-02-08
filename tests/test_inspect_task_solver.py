from __future__ import annotations

import json
import unittest

from inspect_tasks.cleo_bench_eval import (
    _build_solver,
    _empty_output_fallback_json,
    _solver_system_prompt,
    _tool_errors_need_retry,
    _solver_tool_calls,
    _tool_choice,
)


class _Msg:
    def __init__(self, role: str) -> None:
        self.role = role
        self.error = None


class InspectTaskSolverTests(unittest.TestCase):
    def test_solver_prompt_with_sagemath_mentions_tool_usage(self) -> None:
        prompt = _solver_system_prompt(
            use_sagemath_mcp=True,
            require_sagemath_tool_call=True,
        )
        self.assertIn("SageMath tools are available", prompt)
        self.assertIn("Call at least one SageMath tool", prompt)
        self.assertIn("use `**` for exponentiation", prompt)
        self.assertIn("visible assistant text", prompt)
        self.assertIn("Prioritize algebraic/symbolic derivation", prompt)
        self.assertIn("Avoid guessing by fitting random combinations of constants", prompt)

    def test_tool_error_retry_classifier(self) -> None:
        self.assertTrue(_tool_errors_need_retry(["unsupported operand type(s) for ^"]))
        self.assertTrue(_tool_errors_need_retry(["name 'a' is not defined"]))
        self.assertFalse(_tool_errors_need_retry(["Command timed out before completing."]))

    def test_tool_choice_validation(self) -> None:
        self.assertEqual(_tool_choice("auto"), "auto")
        self.assertEqual(_tool_choice("ANY"), "any")
        self.assertEqual(_tool_choice(" none "), "none")
        with self.assertRaises(ValueError):
            _tool_choice("required")

    def test_solver_tool_call_counter(self) -> None:
        calls = _solver_tool_calls([_Msg("assistant"), _Msg("tool"), _Msg("tool")])
        self.assertEqual(calls, 2)

    def test_build_solver_without_sagemath(self) -> None:
        steps = _build_solver(
            solver_use_sagemath_mcp=False,
            solver_require_sagemath_tool_call=True,
            solver_sagemath_mcp_command="uv",
            solver_sagemath_mcp_args="run,sagemath-mcp",
            solver_sagemath_mcp_cwd=None,
            solver_sagemath_mcp_tools="evaluate_sage",
            solver_sagemath_mcp_allow_imports=False,
            solver_sagemath_mcp_allowed_imports=None,
            solver_sagemath_tool_choice="auto",
        )
        self.assertEqual(len(steps), 2)

    def test_empty_output_fallback_json_includes_unresolved(self) -> None:
        payload = json.loads(_empty_output_fallback_json([]))
        self.assertEqual(payload["final_expression_latex"], r"\text{UNRESOLVED}")
        self.assertIn("no visible answer text", payload["notes"].lower())

    def test_empty_output_fallback_json_includes_tool_error_context(self) -> None:
        msg = _Msg("tool")
        msg.error = {"message": "name 'a' is not defined"}
        payload = json.loads(_empty_output_fallback_json([msg]))
        self.assertIn("Tool errors:", payload["notes"])
        self.assertIn("not defined", payload["notes"])


if __name__ == "__main__":
    unittest.main()
