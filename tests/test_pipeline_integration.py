from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cleo_bench.io_utils import read_jsonl, write_json
from cleo_bench.pipeline import build_dataset, export_inspect, validate_dataset


class PipelineIntegrationTests(unittest.TestCase):
    def test_end_to_end_fixture_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            raw_bundle = tmp / "bundle.json"
            processed = tmp / "cleo_bench.jsonl"
            scorable = tmp / "cleo_bench_scorable.jsonl"
            unresolved = tmp / "unresolved.jsonl"
            inspect_dir = tmp / "inspect"

            bundle = {
                "metadata": {
                    "snapshot_date": "2026-02-07",
                    "site": "math.stackexchange",
                },
                "answers": [
                    {
                        "answer_id": 1001,
                        "question_id": 2001,
                        "body": "<p>$$I=\\frac{1}{2}$$</p>",
                        "content_license": "CC BY-SA 4.0",
                    },
                    {
                        "answer_id": 1002,
                        "question_id": 2002,
                        "body": "<p>Use generating functions.</p>",
                        "content_license": "CC BY-SA 4.0",
                    },
                    {
                        "answer_id": 1003,
                        "question_id": 2003,
                        "body": "<p>$$I=\\frac{1}{2}$$</p>",
                        "content_license": "CC BY-SA 4.0",
                    },
                ],
                "questions": [
                    {
                        "question_id": 2001,
                        "title": "Simple integral",
                        "body": "<p>Evaluate $$\\int_0^1 x\\,dx$$</p>",
                        "content_license": "CC BY-SA 4.0",
                        "link": "https://math.stackexchange.com/questions/2001/simple-integral",
                    },
                    {
                        "question_id": 2002,
                        "title": "Infinite series question",
                        "body": "<p>Find the closed form of this series.</p>",
                        "content_license": "CC BY-SA 4.0",
                        "link": "https://math.stackexchange.com/questions/2002/series",
                    },
                    {
                        "question_id": 2003,
                        "title": "Malformed integral text",
                        "body": "<p>Evaluate $$\\int_0^1 \\foo(x)\\,dx$$. Numeric value is 0.5</p>",
                        "content_license": "CC BY-SA 4.0",
                        "link": "https://math.stackexchange.com/questions/2003/malformed",
                    },
                ],
                "accepted_answers": [],
                "records": [
                    {"question_id": 2001, "cleo_answer_id": 1001, "accepted_answer_id": 1001},
                    {"question_id": 2002, "cleo_answer_id": 1002, "accepted_answer_id": None},
                    {"question_id": 2003, "cleo_answer_id": 1003, "accepted_answer_id": 1003},
                ],
            }
            write_json(raw_bundle, bundle)

            build_summary = build_dataset(
                snapshot_date="2026-02-07",
                raw_bundle_path=raw_bundle,
                output_path=processed,
            )
            self.assertEqual(build_summary["total_items"], 3)

            validate_summary = validate_dataset(
                input_path=processed,
                output_path=processed,
                scorable_path=scorable,
                unresolved_path=unresolved,
            )
            self.assertEqual(validate_summary["total_items"], 3)

            rows = read_jsonl(processed)
            integral_item = next(r for r in rows if r["question_id"] == 2001)
            self.assertTrue(integral_item["is_scorable_numeric"])
            self.assertEqual(integral_item["status"], "ok")

            non_integral_item = next(r for r in rows if r["question_id"] == 2002)
            self.assertFalse(non_integral_item["is_integral_candidate"])
            self.assertEqual(non_integral_item["status"], "non_integral")

            hint_fallback_item = next(r for r in rows if r["question_id"] == 2003)
            self.assertTrue(hint_fallback_item["is_scorable_numeric"])
            self.assertEqual(hint_fallback_item["status"], "ok")
            self.assertEqual(hint_fallback_item["integral_numeric"], "0.5")

            export_summary = export_inspect(input_path=processed, output_dir=inspect_dir)
            self.assertEqual(export_summary["full_question_samples"], 3)
            self.assertEqual(export_summary["integral_only_samples"], 2)


if __name__ == "__main__":
    unittest.main()
