"""Command line interface for Cleo Bench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import (
    DEFAULT_ACCOUNT_ID,
    DEFAULT_DPS,
    DEFAULT_SITE,
    DEFAULT_SNAPSHOT_DATE,
    DEFAULT_TOLERANCE,
)
from .pipeline import (
    build_dataset,
    export_inspect,
    fetch_snapshot,
    summarize_eval_logs,
    validate_dataset,
)
from .review import run_review


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cleo-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="Download raw Stack Exchange snapshot")
    p_fetch.add_argument("--account-id", type=int, default=DEFAULT_ACCOUNT_ID)
    p_fetch.add_argument("--site", default=DEFAULT_SITE)
    p_fetch.add_argument("--snapshot-date", default=DEFAULT_SNAPSHOT_DATE)
    p_fetch.add_argument("--stackexchange-key", default=None)

    p_build = sub.add_parser("build", help="Build processed dataset from raw snapshot")
    p_build.add_argument("--snapshot-date", default=DEFAULT_SNAPSHOT_DATE)
    p_build.add_argument("--raw-bundle-path", type=Path, default=None)
    p_build.add_argument("--output-path", type=Path, default=None)

    p_validate = sub.add_parser("validate", help="Compute numeric values and flags")
    p_validate.add_argument("--input-path", type=Path, default=None)
    p_validate.add_argument("--output-path", type=Path, default=None)
    p_validate.add_argument("--scorable-path", type=Path, default=None)
    p_validate.add_argument("--unresolved-path", type=Path, default=None)
    p_validate.add_argument("--overrides-path", type=Path, default=None)
    p_validate.add_argument("--parser-repair-model", default=None)
    p_validate.add_argument("--dps", type=int, default=DEFAULT_DPS)
    p_validate.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)

    p_export = sub.add_parser("export-inspect", help="Export Inspect-compatible datasets")
    p_export.add_argument("--input-path", type=Path, default=None)
    p_export.add_argument("--output-dir", type=Path, default=None)

    p_eval = sub.add_parser("summarize-eval", help="Summarize Inspect eval logs")
    p_eval.add_argument("--log-dir", type=Path, default=None)
    p_eval.add_argument("--output-path", type=Path, default=None)

    p_review = sub.add_parser("review", help="Interactive manual review for unresolved items")
    p_review.add_argument("--unresolved-path", type=Path, default=None)
    p_review.add_argument("--overrides-path", type=Path, default=None)
    p_review.add_argument("--start-item", default=None)
    p_review.add_argument(
        "--no-auto-save",
        action="store_true",
        help="Disable auto-save after each edit (manual save with command 's').",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        out = fetch_snapshot(
            account_id=args.account_id,
            site=args.site,
            snapshot_date=args.snapshot_date,
            stackexchange_key=args.stackexchange_key,
        )
        _print(out)
        return

    if args.command == "build":
        out = build_dataset(
            snapshot_date=args.snapshot_date,
            raw_bundle_path=args.raw_bundle_path,
            output_path=args.output_path,
        )
        _print(out)
        return

    if args.command == "validate":
        out = validate_dataset(
            input_path=args.input_path,
            output_path=args.output_path,
            scorable_path=args.scorable_path,
            unresolved_path=args.unresolved_path,
            overrides_path=args.overrides_path,
            parser_repair_model=args.parser_repair_model,
            dps=args.dps,
            tolerance=args.tolerance,
        )
        _print(out)
        return

    if args.command == "export-inspect":
        out = export_inspect(
            input_path=args.input_path,
            output_dir=args.output_dir,
        )
        _print(out)
        return

    if args.command == "summarize-eval":
        out = summarize_eval_logs(log_dir=args.log_dir, output_path=args.output_path)
        _print(out)
        return

    if args.command == "review":
        out = run_review(
            unresolved_path=args.unresolved_path,
            overrides_path=args.overrides_path,
            start_item=args.start_item,
            auto_save=not args.no_auto_save,
        )
        _print(out)
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
