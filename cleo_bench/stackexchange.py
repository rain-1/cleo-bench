"""Stack Exchange API client and snapshot fetch logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests

API_BASE = "https://api.stackexchange.com/2.3"
WITH_BODY_FILTER = "withbody"


class StackExchangeError(RuntimeError):
    """Raised for Stack Exchange API failures."""


@dataclass
class StackExchangeClient:
    key: str | None = None
    session: requests.Session | None = None
    timeout: int = 30
    min_sleep: float = 0.05

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.session is not None
        query = dict(params)
        if self.key:
            query["key"] = self.key
        url = f"{API_BASE}{path}"
        resp = self.session.get(url, params=query, timeout=self.timeout)
        if resp.status_code != 200:
            raise StackExchangeError(
                f"StackExchange API error {resp.status_code} on {url}: {resp.text[:200]}"
            )
        data = resp.json()
        if "error_id" in data:
            raise StackExchangeError(
                f"StackExchange API error {data.get('error_id')}: {data.get('error_message')}"
            )
        backoff = data.get("backoff")
        if backoff:
            time.sleep(float(backoff))
        else:
            time.sleep(self.min_sleep)
        return data

    def associated(self, account_id: int) -> dict[str, Any]:
        return self._request(f"/users/{account_id}/associated", {"pagesize": 100})

    def paged_user_answers(
        self,
        user_id: int,
        site: str,
        order: str = "asc",
        sort: str = "creation",
    ) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._request(
                f"/users/{user_id}/answers",
                {
                    "site": site,
                    "pagesize": 100,
                    "page": page,
                    "order": order,
                    "sort": sort,
                    "filter": WITH_BODY_FILTER,
                },
            )
            pages.append(data)
            if not data.get("has_more"):
                break
            page += 1
        return pages

    def fetch_questions(self, ids: list[int], site: str) -> list[dict[str, Any]]:
        if not ids:
            return []
        pages: list[dict[str, Any]] = []
        for start in range(0, len(ids), 100):
            batch = ids[start : start + 100]
            semis = ";".join(str(i) for i in batch)
            data = self._request(
                f"/questions/{semis}",
                {"site": site, "pagesize": 100, "filter": WITH_BODY_FILTER},
            )
            pages.append(data)
        return pages

    def fetch_answers(self, ids: list[int], site: str) -> list[dict[str, Any]]:
        if not ids:
            return []
        pages: list[dict[str, Any]] = []
        for start in range(0, len(ids), 100):
            batch = ids[start : start + 100]
            semis = ";".join(str(i) for i in batch)
            data = self._request(
                f"/answers/{semis}",
                {"site": site, "pagesize": 100, "filter": WITH_BODY_FILTER},
            )
            pages.append(data)
        return pages


def _flatten_items(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in pages:
        items.extend(p.get("items", []))
    return items


def resolve_site_user_id(associated_response: dict[str, Any], site: str) -> int:
    target = site.replace("https://", "").replace("http://", "").rstrip("/")
    if "." not in target.split(".")[-1]:
        target = f"{target}.com"
    if not target.endswith((".com", ".net")):
        target = f"{target}.com"
    for item in associated_response.get("items", []):
        site_url = item.get("site_url", "")
        host = site_url.replace("https://", "").replace("http://", "").rstrip("/")
        if host == target:
            user_id = item.get("user_id")
            if isinstance(user_id, int):
                return user_id
    raise StackExchangeError(f"Could not resolve site user id for site '{site}'.")


def build_snapshot_bundle(
    account_id: int,
    site: str,
    client: StackExchangeClient,
    snapshot_date: str,
) -> dict[str, Any]:
    associated = client.associated(account_id)
    site_user_id = resolve_site_user_id(associated, site)

    answer_pages = client.paged_user_answers(site_user_id, site)
    answers = _flatten_items(answer_pages)

    question_ids = sorted({int(a["question_id"]) for a in answers if "question_id" in a})
    question_pages = client.fetch_questions(question_ids, site)
    questions = _flatten_items(question_pages)

    accepted_ids: set[int] = set()
    for q in questions:
        accepted = q.get("accepted_answer_id")
        if isinstance(accepted, int):
            accepted_ids.add(accepted)

    existing_answer_ids = {int(a["answer_id"]) for a in answers if "answer_id" in a}
    accepted_external_ids = sorted(accepted_ids - existing_answer_ids)
    accepted_pages = client.fetch_answers(accepted_external_ids, site)
    accepted_answers = _flatten_items(accepted_pages)

    question_by_id = {int(q["question_id"]): q for q in questions if "question_id" in q}
    records: list[dict[str, Any]] = []
    for answer in answers:
        qid = int(answer["question_id"])
        question = question_by_id.get(qid)
        accepted_id = question.get("accepted_answer_id") if question else None
        records.append(
            {
                "question_id": qid,
                "cleo_answer_id": int(answer["answer_id"]),
                "accepted_answer_id": int(accepted_id) if isinstance(accepted_id, int) else None,
            }
        )

    quota_remaining = None
    if answer_pages:
        quota_remaining = answer_pages[-1].get("quota_remaining")

    return {
        "metadata": {
            "snapshot_date": snapshot_date,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "account_id": account_id,
            "site": site,
            "site_user_id": site_user_id,
            "quota_remaining": quota_remaining,
        },
        "associated_response": associated,
        "answer_pages": answer_pages,
        "question_pages": question_pages,
        "accepted_answer_pages": accepted_pages,
        "answers": answers,
        "questions": questions,
        "accepted_answers": accepted_answers,
        "records": records,
    }
