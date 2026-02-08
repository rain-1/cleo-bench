"""Interactive manual-review helper for unresolved Cleo Bench items."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .constants import DATA_MANUAL
from .extract import html_to_text
from .io_utils import ensure_parent, read_jsonl, write_jsonl

REVIEWABLE_FIELDS = {
    "i": "integral_latex",
    "c": "cleo_reference_latex",
    "a": "accepted_reference_latex",
}


def _read_override_map(path: Path) -> dict[str, dict]:
    records = read_jsonl(path)
    out: dict[str, dict] = {}
    for record in records:
        item_id = record.get("item_id")
        if isinstance(item_id, str):
            out[item_id] = dict(record)
    return out


def _write_override_map(path: Path, override_map: dict[str, dict]) -> None:
    ensure_parent(path)
    rows = [override_map[k] for k in sorted(override_map.keys())]
    write_jsonl(path, rows)


def _effective_value(item: dict, override: dict, key: str):
    if key in override:
        return override[key]
    return item.get(key)


def _normalize_override(item: dict, override: dict) -> dict:
    normalized = {"item_id": override["item_id"]}
    for k, v in override.items():
        if k == "item_id":
            continue
        if item.get(k) != v:
            normalized[k] = v
    return normalized


def _set_override_value(item: dict, override: dict, key: str, value):
    override[key] = value
    normalized = _normalize_override(item, override)
    return normalized


def _format_short(value, width: int = 140) -> str:
    if value is None:
        return "<null>"
    text = str(value).replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _print_item(
    item: dict,
    override: dict,
    index: int,
    total: int,
    output_fn: Callable[[str], None],
) -> None:
    output_fn("")
    output_fn(f"[{index + 1}/{total}] {item.get('item_id')}  status={item.get('status')}")
    output_fn(f"title: {_format_short(item.get('title_raw'))}")
    output_fn(f"reason: {_format_short(item.get('manual_review_reason'))}")
    output_fn(f"question: {_format_short(item.get('question_url'))}")
    output_fn(f"answer: {_format_short(item.get('cleo_answer_url'))}")
    output_fn("")
    output_fn(f"integral_latex: {_format_short(_effective_value(item, override, 'integral_latex'))}")
    output_fn(f"cleo_reference_latex: {_format_short(_effective_value(item, override, 'cleo_reference_latex'))}")
    output_fn(f"accepted_reference_latex: {_format_short(_effective_value(item, override, 'accepted_reference_latex'))}")
    output_fn(
        f"is_integral_candidate: {_effective_value(item, override, 'is_integral_candidate')}"
    )
    output_fn("")
    output_fn(
        "commands: [n]ext [p]rev [i]ntegral [c]leo [a]ccepted [x]mark-non-integral [u]nmark [v]iew-text [r]eset [g <idx|item_id>] [s]ave [q]uit [h]elp"
    )


def _print_help(output_fn: Callable[[str], None]) -> None:
    output_fn("")
    output_fn("Manual review commands")
    output_fn("  n               move to next item")
    output_fn("  p               move to previous item")
    output_fn("  i               edit integral_latex")
    output_fn("  c               edit cleo_reference_latex")
    output_fn("  a               edit accepted_reference_latex")
    output_fn("  x               mark as non-integral (is_integral_candidate=false)")
    output_fn("  u               unmark non-integral (is_integral_candidate=true)")
    output_fn("  v               show expanded question/answer text")
    output_fn("  r               reset overrides for current item")
    output_fn("  g <idx|item_id> jump to 1-based index or item id")
    output_fn("  s               save overrides now")
    output_fn("  q               save and quit")
    output_fn("  h               show this help")
    output_fn("")


def _edit_field(
    key: str,
    item: dict,
    override: dict,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> dict:
    current = _effective_value(item, override, key)
    output_fn(f"current {key}: {_format_short(current, width=1000)}")
    output_fn("enter new value. use /null to clear, blank to keep")
    value = input_fn(f"{key}> ")
    if value == "":
        return override
    if value.strip() == "/null":
        value = None
    updated = _set_override_value(item, dict(override), key, value)
    return updated


def _show_expanded(item: dict, output_fn: Callable[[str], None]) -> None:
    output_fn("")
    output_fn("--- question text ---")
    output_fn(html_to_text(str(item.get("question_body_html_raw", ""))))
    output_fn("")
    output_fn("--- cleo answer text ---")
    output_fn(html_to_text(str(item.get("cleo_body_html_raw", ""))))
    accepted = item.get("accepted_body_html_raw")
    if accepted:
        output_fn("")
        output_fn("--- accepted answer text ---")
        output_fn(html_to_text(str(accepted)))
    output_fn("")


def _resolve_jump(arg: str, items: list[dict]) -> int | None:
    arg = arg.strip()
    if not arg:
        return None
    if arg.isdigit():
        idx = int(arg) - 1
        if 0 <= idx < len(items):
            return idx
        return None
    for idx, item in enumerate(items):
        if item.get("item_id") == arg:
            return idx
    return None


def run_review(
    unresolved_path: Path | None = None,
    overrides_path: Path | None = None,
    start_item: str | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    auto_save: bool = True,
) -> dict:
    unresolved_path = unresolved_path or (DATA_MANUAL / "unresolved.jsonl")
    overrides_path = overrides_path or (DATA_MANUAL / "overrides.jsonl")

    items = read_jsonl(unresolved_path)
    if not items:
        output_fn(f"No unresolved items found at {unresolved_path}")
        return {
            "reviewed": 0,
            "overrides_file": str(overrides_path),
            "message": "no_unresolved_items",
        }

    override_map = _read_override_map(overrides_path)

    idx = 0
    if start_item:
        jump = _resolve_jump(start_item, items)
        if jump is not None:
            idx = jump

    changed = 0
    visited: set[str] = set()

    while True:
        item = items[idx]
        item_id = str(item.get("item_id"))
        visited.add(item_id)

        current_override = override_map.get(item_id, {"item_id": item_id})
        _print_item(item, current_override, idx, len(items), output_fn)

        command = input_fn("review> ").strip()
        if not command:
            command = "n"

        if command == "h":
            _print_help(output_fn)
            continue

        if command == "q":
            _write_override_map(overrides_path, override_map)
            return {
                "reviewed": len(visited),
                "overrides_file": str(overrides_path),
                "num_overrides": len(override_map),
                "changed": changed,
            }

        if command == "s":
            _write_override_map(overrides_path, override_map)
            output_fn(f"saved overrides: {overrides_path}")
            continue

        if command == "n":
            idx = (idx + 1) % len(items)
            continue

        if command == "p":
            idx = (idx - 1) % len(items)
            continue

        if command.startswith("g "):
            jump = _resolve_jump(command[2:], items)
            if jump is None:
                output_fn("invalid jump target")
            else:
                idx = jump
            continue

        if command == "v":
            _show_expanded(item, output_fn)
            continue

        if command == "r":
            if item_id in override_map:
                del override_map[item_id]
                changed += 1
                if auto_save:
                    _write_override_map(overrides_path, override_map)
                output_fn("cleared overrides for item")
            else:
                output_fn("no override exists for item")
            continue

        if command == "x":
            updated = _set_override_value(
                item,
                dict(current_override),
                "is_integral_candidate",
                False,
            )
            if len(updated) == 1:
                override_map.pop(item_id, None)
            else:
                override_map[item_id] = updated
            changed += 1
            if auto_save:
                _write_override_map(overrides_path, override_map)
            continue

        if command == "u":
            updated = _set_override_value(
                item,
                dict(current_override),
                "is_integral_candidate",
                True,
            )
            if len(updated) == 1:
                override_map.pop(item_id, None)
            else:
                override_map[item_id] = updated
            changed += 1
            if auto_save:
                _write_override_map(overrides_path, override_map)
            continue

        if command in REVIEWABLE_FIELDS:
            key = REVIEWABLE_FIELDS[command]
            updated = _edit_field(key, item, dict(current_override), input_fn, output_fn)
            if len(updated) == 1:
                override_map.pop(item_id, None)
            else:
                override_map[item_id] = updated
            changed += 1
            if auto_save:
                _write_override_map(overrides_path, override_map)
            continue

        output_fn("unknown command. type 'h' for help")
