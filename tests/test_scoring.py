from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from cleo_bench.judge import JudgeResult
from cleo_bench.scoring import deterministic_score, score_with_optional_judge


class ScoringTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
