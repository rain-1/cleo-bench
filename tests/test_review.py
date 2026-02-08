from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cleo_bench.io_utils import read_jsonl, write_jsonl
from cleo_bench.review import run_review


class ReviewToolTests(unittest.TestCase):
    def test_review_edits_and_writes_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            unresolved = tmp / "unresolved.jsonl"
            overrides = tmp / "overrides.jsonl"

            write_jsonl(
                unresolved,
                [
                    {
                        "item_id": "q1_a1",
                        "status": "needs_manual_review",
                        "title_raw": "Sample",
                        "manual_review_reason": "bad parse",
                        "question_url": "https://example.com/q1",
                        "cleo_answer_url": "https://example.com/a1",
                        "integral_latex": r"\int_0^1 x dx",
                        "cleo_reference_latex": None,
                        "accepted_reference_latex": None,
                        "is_integral_candidate": True,
                    }
                ],
            )

            commands = iter([
                "c",  # edit cleo reference
                r"\frac{1}{2}",
                "q",  # quit
            ])

            summary = run_review(
                unresolved_path=unresolved,
                overrides_path=overrides,
                input_fn=lambda _p: next(commands),
                output_fn=lambda _msg: None,
                auto_save=True,
            )

            self.assertEqual(summary["num_overrides"], 1)
            rows = read_jsonl(overrides)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["item_id"], "q1_a1")
            self.assertEqual(rows[0]["cleo_reference_latex"], r"\frac{1}{2}")

    def test_review_reset_removes_override(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            unresolved = tmp / "unresolved.jsonl"
            overrides = tmp / "overrides.jsonl"

            write_jsonl(
                unresolved,
                [
                    {
                        "item_id": "q2_a2",
                        "status": "needs_manual_review",
                        "title_raw": "Sample2",
                        "manual_review_reason": "bad parse",
                        "question_url": "https://example.com/q2",
                        "cleo_answer_url": "https://example.com/a2",
                        "integral_latex": r"\int_0^1 x dx",
                        "cleo_reference_latex": r"\frac{1}{2}",
                        "accepted_reference_latex": None,
                        "is_integral_candidate": True,
                    }
                ],
            )
            write_jsonl(
                overrides,
                [{"item_id": "q2_a2", "is_integral_candidate": False}],
            )

            commands = iter([
                "r",  # clear override
                "q",
            ])

            summary = run_review(
                unresolved_path=unresolved,
                overrides_path=overrides,
                input_fn=lambda _p: next(commands),
                output_fn=lambda _msg: None,
                auto_save=True,
            )

            self.assertEqual(summary["num_overrides"], 0)
            rows = read_jsonl(overrides)
            self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
