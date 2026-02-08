"""CLI pipeline implementation for Cleo Bench."""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mpmath as mp

from .constants import (
    DATA_INSPECT,
    DATA_MANUAL,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_DPS,
    DEFAULT_SITE,
    DEFAULT_SITE_URL,
    DEFAULT_TOLERANCE,
    REPORTS,
    STATUS_FETCH_ERROR,
    STATUS_NEEDS_REVIEW,
    STATUS_NON_INTEGRAL,
    STATUS_OK,
)
from .extract import extract_item_fields
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .models import CleoBenchItem
from .numeric import abs_rel_close, evaluate_expression_numeric, evaluate_integral_numeric
from .stackexchange import StackExchangeClient, build_snapshot_bundle

HINT_VALUE_RE = re.compile(
    r"(?:\\approx|≈|approx(?:imately)?|numeric value is|value is)\s*"
    r"([+-]?(?:\d+\.\d+|\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
LONG_DECIMAL_RE = re.compile(r"[+-]?\d+\.\d{8,}")


def _ensure_dirs() -> None:
    for path in (DATA_RAW, DATA_PROCESSED, DATA_INSPECT, DATA_MANUAL, REPORTS):
        path.mkdir(parents=True, exist_ok=True)


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item.get("status") for item in items)
    return {
        "total_items": len(items),
        "status_counts": dict(counts),
        "integral_candidates": sum(1 for i in items if i.get("is_integral_candidate")),
        "scorable_numeric": sum(1 for i in items if i.get("is_scorable_numeric")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _answer_url(answer_id: int, link: str | None) -> str:
    if link:
        return link
    return f"{DEFAULT_SITE_URL}/a/{answer_id}"


def fetch_snapshot(
    account_id: int,
    site: str,
    snapshot_date: str,
    stackexchange_key: str | None = None,
) -> dict[str, Any]:
    _ensure_dirs()
    target_dir = DATA_RAW / snapshot_date
    target_dir.mkdir(parents=True, exist_ok=True)

    client = StackExchangeClient(key=stackexchange_key)
    bundle = build_snapshot_bundle(
        account_id=account_id,
        site=site,
        client=client,
        snapshot_date=snapshot_date,
    )

    write_json(target_dir / "bundle.json", bundle)
    index = {
        "metadata": bundle.get("metadata", {}),
        "num_answers": len(bundle.get("answers", [])),
        "num_questions": len(bundle.get("questions", [])),
        "num_records": len(bundle.get("records", [])),
    }
    write_json(target_dir / "index.json", index)
    return index


def build_dataset(
    snapshot_date: str,
    raw_bundle_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    _ensure_dirs()
    raw_bundle_path = raw_bundle_path or (DATA_RAW / snapshot_date / "bundle.json")
    output_path = output_path or (DATA_PROCESSED / "cleo_bench.jsonl")

    bundle = read_json(raw_bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError("Invalid bundle format")

    metadata = bundle.get("metadata", {})
    answers = bundle.get("answers", [])
    questions = bundle.get("questions", [])
    accepted_answers = bundle.get("accepted_answers", [])
    records = bundle.get("records", [])

    answer_by_id = {int(a["answer_id"]): a for a in answers if isinstance(a, dict) and "answer_id" in a}
    question_by_id = {
        int(q["question_id"]): q for q in questions if isinstance(q, dict) and "question_id" in q
    }
    accepted_by_id = {
        int(a["answer_id"]): a
        for a in accepted_answers
        if isinstance(a, dict) and "answer_id" in a
    }

    items: list[dict[str, Any]] = []

    for record in records:
        qid = int(record["question_id"])
        cleo_aid = int(record["cleo_answer_id"])
        accepted_aid = record.get("accepted_answer_id")

        question = question_by_id.get(qid)
        cleo_answer = answer_by_id.get(cleo_aid)

        if question is None or cleo_answer is None:
            item = CleoBenchItem(
                item_id=f"q{qid}_a{cleo_aid}",
                snapshot_date=str(metadata.get("snapshot_date", snapshot_date)),
                site=f"{site_to_host(str(metadata.get('site', DEFAULT_SITE)))}",
                question_id=qid,
                cleo_answer_id=cleo_aid,
                accepted_answer_id=int(accepted_aid) if isinstance(accepted_aid, int) else None,
                question_url=f"{DEFAULT_SITE_URL}/questions/{qid}",
                cleo_answer_url=f"{DEFAULT_SITE_URL}/a/{cleo_aid}",
                accepted_answer_url=(f"{DEFAULT_SITE_URL}/a/{accepted_aid}" if accepted_aid else None),
                content_license_question="",
                content_license_cleo="",
                content_license_accepted=None,
                title_raw="",
                question_body_html_raw="",
                cleo_body_html_raw="",
                accepted_body_html_raw=None,
                prompt_full_question_sanitized="",
                prompt_integral_only=None,
                integral_latex=None,
                cleo_reference_latex=None,
                accepted_reference_latex=None,
                integral_numeric=None,
                cleo_reference_numeric=None,
                accepted_reference_numeric=None,
                best_reference_numeric=None,
                numeric_delta_cleo=None,
                numeric_delta_accepted=None,
                numeric_pass_cleo=None,
                numeric_pass_accepted=None,
                is_integral_candidate=False,
                is_scorable_numeric=False,
                status=STATUS_FETCH_ERROR,
                manual_review_reason="Missing question or answer data in snapshot bundle.",
            )
            items.append(item.to_dict())
            continue

        accepted_answer = None
        if isinstance(accepted_aid, int):
            accepted_answer = answer_by_id.get(accepted_aid) or accepted_by_id.get(accepted_aid)

        extraction = extract_item_fields(
            title_raw=str(question.get("title", "")),
            question_body_html=str(question.get("body", "")),
            cleo_body_html=str(cleo_answer.get("body", "")),
            accepted_body_html=(str(accepted_answer.get("body", "")) if accepted_answer else None),
        )

        status = STATUS_OK
        reason: str | None = None
        if not extraction.is_integral_candidate:
            status = STATUS_NON_INTEGRAL
            reason = "No integral expression extracted from question content."
        elif not extraction.cleo_reference_latex and not extraction.accepted_reference_latex:
            status = STATUS_NEEDS_REVIEW
            reason = "No reference expression extracted from Cleo or accepted answer."

        item = CleoBenchItem(
            item_id=f"q{qid}_a{cleo_aid}",
            snapshot_date=str(metadata.get("snapshot_date", snapshot_date)),
            site=site_to_host(str(metadata.get("site", DEFAULT_SITE))),
            question_id=qid,
            cleo_answer_id=cleo_aid,
            accepted_answer_id=(int(accepted_aid) if isinstance(accepted_aid, int) else None),
            question_url=str(question.get("link") or f"{DEFAULT_SITE_URL}/questions/{qid}"),
            cleo_answer_url=_answer_url(cleo_aid, cleo_answer.get("link")),
            accepted_answer_url=(
                _answer_url(int(accepted_aid), accepted_answer.get("link") if accepted_answer else None)
                if isinstance(accepted_aid, int)
                else None
            ),
            content_license_question=str(question.get("content_license", "")),
            content_license_cleo=str(cleo_answer.get("content_license", "")),
            content_license_accepted=(
                str(accepted_answer.get("content_license", "")) if accepted_answer else None
            ),
            title_raw=str(question.get("title", "")),
            question_body_html_raw=str(question.get("body", "")),
            cleo_body_html_raw=str(cleo_answer.get("body", "")),
            accepted_body_html_raw=(str(accepted_answer.get("body", "")) if accepted_answer else None),
            prompt_full_question_sanitized=extraction.prompt_full_question_sanitized,
            prompt_integral_only=extraction.prompt_integral_only,
            integral_latex=extraction.integral_latex,
            cleo_reference_latex=extraction.cleo_reference_latex,
            accepted_reference_latex=extraction.accepted_reference_latex,
            integral_numeric=None,
            cleo_reference_numeric=None,
            accepted_reference_numeric=None,
            best_reference_numeric=None,
            numeric_delta_cleo=None,
            numeric_delta_accepted=None,
            numeric_pass_cleo=None,
            numeric_pass_accepted=None,
            is_integral_candidate=extraction.is_integral_candidate,
            is_scorable_numeric=False,
            status=status,
            manual_review_reason=reason,
        )
        items.append(asdict(item))

    write_jsonl(output_path, items)
    summary = _summary(items)
    write_json(REPORTS / "build_summary.json", summary)
    return summary


def site_to_host(site: str) -> str:
    site = site.replace("https://", "").replace("http://", "")
    if not site.endswith(".com") and not site.endswith(".net"):
        return f"{site}.com"
    return site


def _load_overrides(overrides_path: Path | None) -> dict[str, dict[str, Any]]:
    if overrides_path is None or not overrides_path.exists():
        return {}
    records = read_jsonl(overrides_path)
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = record.get("item_id")
        if isinstance(item_id, str):
            out[item_id] = record
    return out


def _extract_question_numeric_hint(question_body_html_raw: str) -> mp.mpf | None:
    # Local import avoids expanding the dependency surface of modules that do not need bs4.
    from .extract import html_to_text

    text = html_to_text(question_body_html_raw or "")
    m = HINT_VALUE_RE.search(text)
    if m:
        try:
            return mp.mpf(m.group(1))
        except Exception:  # noqa: BLE001
            pass

    for m2 in LONG_DECIMAL_RE.finditer(text):
        try:
            return mp.mpf(m2.group(0))
        except Exception:  # noqa: BLE001
            continue
    return None


async def _repair_latex_with_model(
    expression: str,
    error_hint: str,
    model_name: str,
) -> str | None:
    try:
        from inspect_ai.model import get_model
    except Exception:
        return None

    try:
        model = get_model(model_name)
    except Exception:
        return None

    prompt = (
        "Normalize the following LaTeX mathematical expression for strict parsing.\n"
        "Return JSON only: {\"repaired_latex\": \"...\"}.\n"
        "Do not solve the math; preserve semantic meaning.\n\n"
        f"Original:\n{expression}\n\n"
        f"Parser error hint:\n{error_hint}\n"
    )
    try:
        response = await model.generate(prompt)
        text = response.completion.strip()
        payload = json.loads(text) if text.startswith("{") else None
        if payload is None:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                payload = json.loads(m.group(0))
        if isinstance(payload, dict):
            repaired = payload.get("repaired_latex")
            if isinstance(repaired, str) and repaired.strip():
                return repaired.strip()
    except Exception:
        return None
    return None


def _maybe_repair_sync(expression: str, error_hint: str, model_name: str | None) -> str | None:
    if not model_name:
        return None
    try:
        return asyncio.run(_repair_latex_with_model(expression, error_hint, model_name))
    except Exception:
        return None


def _pick_best_reference(item: dict[str, Any], cleo_value: mp.mpf | None, accepted_value: mp.mpf | None) -> str | None:
    integral_value = item.get("_integral_value")
    if not isinstance(integral_value, mp.mpf):
        return None

    candidates: list[tuple[mp.mpf, str]] = []
    if cleo_value is not None:
        candidates.append((abs(integral_value - cleo_value), mp.nstr(cleo_value, n=25)))
    if accepted_value is not None:
        candidates.append((abs(integral_value - accepted_value), mp.nstr(accepted_value, n=25)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def validate_dataset(
    input_path: Path | None = None,
    output_path: Path | None = None,
    scorable_path: Path | None = None,
    unresolved_path: Path | None = None,
    overrides_path: Path | None = None,
    parser_repair_model: str | None = None,
    dps: int = DEFAULT_DPS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    _ensure_dirs()
    input_path = input_path or (DATA_PROCESSED / "cleo_bench.jsonl")
    output_path = output_path or input_path
    scorable_path = scorable_path or (DATA_PROCESSED / "cleo_bench_scorable.jsonl")
    unresolved_path = unresolved_path or (DATA_MANUAL / "unresolved.jsonl")

    items = read_jsonl(input_path)
    overrides = _load_overrides(overrides_path)

    updated: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    scorable: list[dict[str, Any]] = []

    for item in items:
        item_id = item.get("item_id")
        if isinstance(item_id, str) and item_id in overrides:
            patch = overrides[item_id]
            for key, value in patch.items():
                if key == "item_id":
                    continue
                item[key] = value

        if item.get("status") == STATUS_FETCH_ERROR:
            updated.append(item)
            unresolved.append(item)
            continue

        if not item.get("is_integral_candidate"):
            item["is_scorable_numeric"] = False
            item["status"] = STATUS_NON_INTEGRAL
            updated.append(item)
            continue

        errors: list[str] = []

        integral_expr = item.get("integral_latex")
        integral_eval = evaluate_integral_numeric(integral_expr, dps=dps)
        if integral_eval.value is None and parser_repair_model and isinstance(integral_expr, str):
            repaired = _maybe_repair_sync(integral_expr, integral_eval.error or "", parser_repair_model)
            if repaired:
                item["integral_latex"] = repaired
                integral_eval = evaluate_integral_numeric(repaired, dps=dps)

        integral_value = integral_eval.value
        integral_numeric = integral_eval.value_str

        if integral_value is None:
            hint_value = _extract_question_numeric_hint(str(item.get("question_body_html_raw", "")))
            if hint_value is not None:
                integral_value = hint_value
                integral_numeric = mp.nstr(hint_value, n=dps)

        item["integral_numeric"] = integral_numeric
        item["_integral_value"] = integral_value
        if integral_value is None:
            errors.append(integral_eval.error or "integral_eval_failed")

        cleo_expr = item.get("cleo_reference_latex")
        cleo_eval = evaluate_expression_numeric(cleo_expr, dps=dps)
        if cleo_eval.value is None and parser_repair_model and isinstance(cleo_expr, str):
            repaired = _maybe_repair_sync(cleo_expr, cleo_eval.error or "", parser_repair_model)
            if repaired:
                item["cleo_reference_latex"] = repaired
                cleo_eval = evaluate_expression_numeric(repaired, dps=dps)
        item["cleo_reference_numeric"] = cleo_eval.value_str
        if cleo_expr and cleo_eval.value is None:
            errors.append(f"cleo_reference: {cleo_eval.error}")

        accepted_expr = item.get("accepted_reference_latex")
        accepted_eval = evaluate_expression_numeric(accepted_expr, dps=dps)
        if accepted_eval.value is None and parser_repair_model and isinstance(accepted_expr, str):
            repaired = _maybe_repair_sync(accepted_expr, accepted_eval.error or "", parser_repair_model)
            if repaired:
                item["accepted_reference_latex"] = repaired
                accepted_eval = evaluate_expression_numeric(repaired, dps=dps)
        item["accepted_reference_numeric"] = accepted_eval.value_str
        if accepted_expr and accepted_eval.value is None:
            errors.append(f"accepted_reference: {accepted_eval.error}")

        delta_cleo, pass_cleo = abs_rel_close(integral_value, cleo_eval.value, tolerance=tolerance)
        delta_acc, pass_acc = abs_rel_close(integral_value, accepted_eval.value, tolerance=tolerance)
        item["numeric_delta_cleo"] = delta_cleo
        item["numeric_delta_accepted"] = delta_acc
        item["numeric_pass_cleo"] = pass_cleo
        item["numeric_pass_accepted"] = pass_acc

        best_ref = _pick_best_reference(item, cleo_eval.value, accepted_eval.value)
        item["best_reference_numeric"] = best_ref

        scorable_now = integral_value is not None and (cleo_eval.value is not None or accepted_eval.value is not None)
        item["is_scorable_numeric"] = scorable_now

        if scorable_now:
            item["status"] = STATUS_OK
            item["manual_review_reason"] = None
            scorable.append(item)
        else:
            item["status"] = STATUS_NEEDS_REVIEW
            reason = "; ".join(e for e in errors if e)
            item["manual_review_reason"] = reason or "Unable to evaluate integral/reference numerically."
            unresolved.append(item)

        item.pop("_integral_value", None)
        updated.append(item)

    write_jsonl(output_path, updated)
    write_jsonl(scorable_path, scorable)
    write_jsonl(unresolved_path, unresolved)

    summary = _summary(updated)
    summary["scorable_file"] = str(scorable_path)
    summary["unresolved_file"] = str(unresolved_path)
    write_json(REPORTS / "build_summary.json", summary)
    return summary


def export_inspect(
    input_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    _ensure_dirs()
    input_path = input_path or (DATA_PROCESSED / "cleo_bench.jsonl")
    output_dir = output_dir or DATA_INSPECT
    output_dir.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(input_path)
    full_samples: list[dict[str, Any]] = []
    integral_samples: list[dict[str, Any]] = []

    for item in items:
        metadata = dict(item)
        sample_base = {
            "id": item["item_id"],
            "target": "",
            "metadata": metadata,
        }

        full = dict(sample_base)
        full["input"] = item.get("prompt_full_question_sanitized", "")
        full_samples.append(full)

        if item.get("prompt_integral_only"):
            integ = dict(sample_base)
            integ["id"] = f"{item['item_id']}_integral"
            integ["input"] = item["prompt_integral_only"]
            integral_samples.append(integ)

    full_path = output_dir / "cleo_bench_full_question.jsonl"
    integral_path = output_dir / "cleo_bench_integral_only.jsonl"
    write_jsonl(full_path, full_samples)
    write_jsonl(integral_path, integral_samples)

    task_config = {
        "full_question_dataset": str(full_path),
        "integral_only_dataset": str(integral_path),
        "task_module": "inspect_tasks/cleo_bench_eval.py",
        "tasks": ["cleo_bench_full_question", "cleo_bench_integral_only"],
        "notes": "Run with inspect eval inspect_tasks/cleo_bench_eval.py@cleo_bench_full_question",
    }
    write_json(output_dir / "task_config.json", task_config)
    return {
        "full_question_samples": len(full_samples),
        "integral_only_samples": len(integral_samples),
        "full_question_path": str(full_path),
        "integral_only_path": str(integral_path),
    }


def summarize_eval_logs(log_dir: Path | None = None, output_path: Path | None = None) -> dict[str, Any]:
    """Create reports/eval_summary.json from Inspect logs."""
    output_path = output_path or (REPORTS / "eval_summary.json")

    try:
        from inspect_ai.log import list_eval_logs, read_eval_log
    except Exception as ex:  # noqa: BLE001
        summary = {
            "error": f"inspect_log_import_error: {ex}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_path, summary)
        return summary

    log_dir = log_dir or Path("logs")
    if not log_dir.exists():
        summary = {
            "error": f"log_dir_not_found: {log_dir}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output_path, summary)
        return summary

    logs = list_eval_logs(log_dir.as_posix())
    rows: list[dict[str, Any]] = []
    for info in logs:
        try:
            log = read_eval_log(info.name)
            rows.append(
                {
                    "name": info.name,
                    "status": getattr(log, "status", None),
                    "task": getattr(getattr(log, "eval", None), "task", None),
                    "model": getattr(getattr(log, "eval", None), "model", None),
                    "results": (log.results.model_dump() if getattr(log, "results", None) else None),
                }
            )
        except Exception as ex:  # noqa: BLE001
            rows.append({"name": info.name, "error": str(ex)})

    summary = {
        "log_dir": str(log_dir),
        "num_logs": len(rows),
        "logs": rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_path, summary)
    return summary
