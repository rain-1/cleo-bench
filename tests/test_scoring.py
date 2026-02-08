from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from cleo_bench.judge import JudgeResult
from cleo_bench.scoring import (
    deterministic_score,
    extract_candidate_expression,
    score_with_optional_judge,
)


class ScoringTests(unittest.TestCase):
    def test_extract_candidate_expression_from_mixed_prose_and_fenced_json(self) -> None:
        output = (
            "Given the numerical value and transformations, final answer follows.\n\n"
            "```json\n"
            "{\n"
            '  "final_expression_latex": "I = \\\\int_{0}^{\\\\infty} '
            '\\\\frac{t}{e^{t} - 1} \\\\ln\\\\left(1 + \\\\frac{1}{t^{2}}\\\\right) dt",\n'
            '  "notes": "transformed form"\n'
            "}\n"
            "```\n"
        )
        candidate = extract_candidate_expression(output)
        self.assertEqual(
            candidate,
            r"I = \int_{0}^{\infty} \frac{t}{e^{t} - 1} \ln\left(1 + \frac{1}{t^{2}}\right) dt",
        )

    def test_extract_candidate_expression_key_value_fallback(self) -> None:
        output = (
            "Result summary with malformed payload:\n"
            '{"final_expression_latex": "\\\\pi \\\\ln 2", "notes": "ok",}\n'
        )
        candidate = extract_candidate_expression(output)
        self.assertEqual(candidate, r"\pi \ln 2")

    def test_deterministic_pass(self) -> None:
        output = '{"final_expression_latex": "\\\\frac{1}{2}"}'
        metadata = {"best_reference_numeric": "0.5"}
        score = deterministic_score(output, metadata)
        self.assertEqual(score.deterministic_status, "pass")
        self.assertTrue(score.final_pass)
        self.assertEqual(score.final_score, 1.0)

    def test_judge_fallback_when_unresolved(self) -> None:
        output = '{"final_expression_latex": "\\\\notARealFunction{z}"}'
        metadata = {
            "best_reference_numeric": "0.5",
            "prompt_full_question_sanitized": "Evaluate integral",
            "integral_latex": "\\int_0^1 x dx",
            "cleo_reference_latex": "\\frac{1}{2}",
            "accepted_reference_latex": None,
        }

        mocked = AsyncMock(
            return_value=JudgeResult(
                verdict=True,
                total_score=16,
                criteria={
                    "equivalence": 4,
                    "constants_and_branches": 4,
                    "task_completeness": 4,
                    "expression_validity": 4,
                },
                rationale="Looks equivalent",
                raw_response='{"pass": true}',
                error=None,
            )
        )

        with patch("cleo_bench.scoring.run_judge_with_inspect", mocked):
            result = asyncio.run(
                score_with_optional_judge(
                    output_text=output,
                    metadata=metadata,
                    use_judge_when_unresolved=True,
                    judge_model="fake/model",
                )
            )

        self.assertEqual(result["deterministic_status"], "unresolved")
        self.assertTrue(result["judge_used"])
        self.assertTrue(result["final_pass"])
        self.assertAlmostEqual(result["final_score"], 0.8)
        self.assertIsNone(result["judge_error"])
        self.assertIsNone(result["judge_stop_reason"])
        self.assertIsNone(result["judge_tool_calls"])

    def test_judge_receives_sagemath_options(self) -> None:
        output = '{"final_expression_latex": "\\\\notARealFunction{z}"}'
        metadata = {
            "best_reference_numeric": "0.5",
            "prompt_full_question_sanitized": "Evaluate integral",
            "integral_latex": "\\int_0^1 x dx",
            "cleo_reference_latex": "\\frac{1}{2}",
            "accepted_reference_latex": None,
        }

        mocked = AsyncMock(
            return_value=JudgeResult(
                verdict=False,
                total_score=8,
                criteria={
                    "equivalence": 1,
                    "constants_and_branches": 2,
                    "task_completeness": 3,
                    "expression_validity": 2,
                },
                rationale="Not equivalent",
                raw_response='{"pass": false}',
                error=None,
                tool_calls=2,
                stop_reason="stop",
                used_sagemath_mcp=True,
            )
        )

        with patch("cleo_bench.scoring.run_judge_with_inspect", mocked):
            _ = asyncio.run(
                score_with_optional_judge(
                    output_text=output,
                    metadata=metadata,
                    use_judge_when_unresolved=True,
                    judge_model="fake/model",
                    judge_max_tokens=12345,
                    judge_reasoning_effort="medium",
                    judge_use_sagemath_mcp=True,
                    judge_sagemath_mcp_command="uv",
                    judge_sagemath_mcp_args=["run", "sagemath-mcp"],
                    judge_sagemath_mcp_cwd="/tmp",
                    judge_sagemath_mcp_tools=["evaluate_sage"],
                    judge_sagemath_mcp_eval_timeout_seconds=301.5,
                )
            )

        kwargs = mocked.await_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 12345)
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertTrue(kwargs["use_sagemath_mcp"])
        self.assertEqual(kwargs["sagemath_mcp_command"], "uv")
        self.assertEqual(kwargs["sagemath_mcp_args"], ["run", "sagemath-mcp"])
        self.assertEqual(kwargs["sagemath_mcp_cwd"], "/tmp")
        self.assertEqual(kwargs["sagemath_mcp_tools"], ["evaluate_sage"])
        self.assertEqual(kwargs["sagemath_mcp_eval_timeout_seconds"], 301.5)

    def test_judge_error_metadata_is_propagated(self) -> None:
        output = '{"final_expression_latex": "\\\\notARealFunction{z}"}'
        metadata = {
            "best_reference_numeric": "0.5",
            "prompt_full_question_sanitized": "Evaluate integral",
            "integral_latex": "\\int_0^1 x dx",
            "cleo_reference_latex": "\\frac{1}{2}",
            "accepted_reference_latex": None,
        }

        mocked = AsyncMock(
            return_value=JudgeResult(
                verdict=None,
                total_score=None,
                criteria=None,
                rationale=None,
                raw_response=None,
                error="judge_response_not_json (stop_reason=max_tokens)",
                tool_calls=0,
                stop_reason="max_tokens",
                used_sagemath_mcp=True,
            )
        )

        with patch("cleo_bench.scoring.run_judge_with_inspect", mocked):
            result = asyncio.run(
                score_with_optional_judge(
                    output_text=output,
                    metadata=metadata,
                    use_judge_when_unresolved=True,
                    judge_model="fake/model",
                    judge_use_sagemath_mcp=True,
                )
            )

        self.assertTrue(result["judge_used"])
        self.assertEqual(result["judge_error"], "judge_response_not_json (stop_reason=max_tokens)")
        self.assertEqual(result["judge_stop_reason"], "max_tokens")
        self.assertEqual(result["judge_tool_calls"], 0)
        self.assertEqual(result["judge_used_sagemath_mcp"], True)
        self.assertIn("Judge fallback failed", result["explanation"])


if __name__ == "__main__":
    unittest.main()
