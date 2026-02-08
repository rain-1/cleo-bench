from __future__ import annotations

import unittest
from unittest.mock import patch

from cleo_bench.stackexchange import StackExchangeClient


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, params: dict, timeout: int):
        self.calls.append((url, params))
        page = params.get("page", 1)
        if page == 1:
            return FakeResponse(
                {
                    "items": [{"answer_id": 1, "question_id": 101}],
                    "has_more": True,
                    "backoff": 0,
                }
            )
        return FakeResponse(
            {
                "items": [{"answer_id": 2, "question_id": 102}],
                "has_more": False,
                "backoff": 0,
            }
        )


class StackExchangeClientTests(unittest.TestCase):
    def test_paged_answers_pagination(self) -> None:
        session = FakeSession()
        client = StackExchangeClient(session=session)
        with patch("time.sleep") as mocked_sleep:
            pages = client.paged_user_answers(user_id=97378, site="math.stackexchange")

        self.assertEqual(len(pages), 2)
        self.assertEqual(len(session.calls), 2)
        self.assertGreaterEqual(mocked_sleep.call_count, 2)


if __name__ == "__main__":
    unittest.main()
