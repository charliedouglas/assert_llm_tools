"""
assert-review CLI — single note and batch CSV evaluation.

Usage:
    assert-review evaluate <note_file> --framework <id> [options]
    assert-review batch <csv_file> --framework <id> [options]

Examples:
    assert-review evaluate note.txt --framework fca_suitability_v1
    assert-review evaluate note.txt --framework fca_suitability_v1 --output json
    assert-review batch notes.csv --framework fca_suitability_v1 --note-column text
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

from assert_core.llm.config import LLMConfig

from .evaluate_note import evaluate_note
from .models import GapReport

# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_llm_config(args: argparse.Namespace) -> Optional[LLMConfig]:
    """Build LLMConfig from CLI arguments, or return None to use default."""
    if not hasattr(args, "provider") or args.provider is None:
        return None
    kwargs = {"provider": args.provider, "model_id": args.model}
    if args.provider == "bedrock":
        kwargs["region"] = args.region
    elif args.provider == "openai":
        kwargs["api_key"] = args.api_key
    return LLMConfig(**kwargs)


def _report_to_text(report: GapReport) -> str:
    """Render a GapReport as human-readable plain text."""
    lines = [
        f"Framework : {report.framework_id} v{report.framework_version}",
        f"Result    : {'PASS' if report.passed else 'FAIL'}  ({report.overall_rating})",
        f"Score     : {report.overall_score:.2%}",
        f"Elements  : {report.stats.present_count}/{report.stats.total_elements} present",
        "",
        "Summary:",
        f"  {report.summary}",
        "",
    ]

    gaps = [it for it in report.items if it.status != "present"]
    if gaps:
        lines.append("Gaps:")
        for item in gaps:
            req = "required" if item.required else "optional"
            lines.append(
                f"  [{item.severity.upper()}] {item.element_id} — {item.status} "
                f"(score {item.score:.2f}, {req})"
            )
            if item.evidence:
                lines.append(f"    Evidence : {item.evidence}")
            for s in item.suggestions:
                lines.append(f"    Suggest  : {s}")
    else:
        lines.append("No gaps identified.")

    return "\n".join(lines)


def _report_to_dict(report: GapReport) -> dict:
    """Convert a GapReport to a JSON-serialisable dict."""
    return {
        "framework_id": report.framework_id,
        "framework_version": report.framework_version,
        "passed": report.passed,
        "overall_score": report.overall_score,
        "overall_rating": report.overall_rating,
        "summary": report.summary,
        "stats": {
            "total_elements": report.stats.total_elements,
            "required_elements": report.stats.required_elements,
            "present_count": report.stats.present_count,
            "partial_count": report.stats.partial_count,
            "missing_count": report.stats.missing_count,
            "critical_gaps": report.stats.critical_gaps,
            "high_gaps": report.stats.high_gaps,
            "medium_gaps": report.stats.medium_gaps,
            "low_gaps": report.stats.low_gaps,
            "required_missing_count": report.stats.required_missing_count,
        },
        "items": [
            {
                "element_id": it.element_id,
                "status": it.status,
                "score": it.score,
                "evidence": it.evidence,
                "severity": it.severity,
                "required": it.required,
                "notes": it.notes,
                "suggestions": it.suggestions,
            }
            for it in report.items
        ],
        "metadata": report.metadata,
    }


# ── Subcommand: evaluate ───────────────────────────────────────────────────────

def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate a single note file."""
    note_path = Path(args.note_file)
    if not note_path.exists():
        print(f"Error: note file not found: {note_path}", file=sys.stderr)
        return 1

    note_text = note_path.read_text(encoding="utf-8")
    llm_config = _build_llm_config(args)

    try:
        report = evaluate_note(
            note_text=note_text,
            framework=args.framework,
            llm_config=llm_config,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output == "json":
        print(json.dumps(_report_to_dict(report), indent=2))
    else:
        print(_report_to_text(report))

    return 0 if report.passed else 2


# ── Subcommand: batch ──────────────────────────────────────────────────────────

def cmd_batch(args: argparse.Namespace) -> int:
    """Evaluate multiple notes from a CSV file."""
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    llm_config = _build_llm_config(args)
    results = []
    errors = 0

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if args.note_column not in (reader.fieldnames or []):
            print(
                f"Error: column '{args.note_column}' not found in CSV. "
                f"Available columns: {reader.fieldnames}",
                file=sys.stderr,
            )
            return 1

        for row_num, row in enumerate(reader, start=2):
            note_text = row[args.note_column]
            meta = {k: v for k, v in row.items() if k != args.note_column}

            try:
                report = evaluate_note(
                    note_text=note_text,
                    framework=args.framework,
                    llm_config=llm_config,
                    verbose=args.verbose,
                    metadata=meta,
                )
                results.append(_report_to_dict(report))
            except Exception as exc:
                errors += 1
                print(f"Warning: row {row_num} failed: {exc}", file=sys.stderr)
                results.append({"error": str(exc), "metadata": meta})

    if args.output == "json":
        print(json.dumps(results, indent=2))
    else:
        for i, r in enumerate(results, start=1):
            if "error" in r:
                print(f"\n--- Row {i} ERROR: {r['error']} ---")
            else:
                # Reconstruct a minimal text summary for batch mode
                print(f"\n--- Row {i} ---")
                print(f"Result : {'PASS' if r['passed'] else 'FAIL'}  ({r['overall_rating']})")
                print(f"Score  : {r['overall_score']:.2%}")

    if errors:
        print(f"\n{errors} row(s) failed.", file=sys.stderr)

    return 0


# ── Parser ─────────────────────────────────────────────────────────────────────

def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Add shared LLM configuration arguments to a subparser."""
    llm_group = parser.add_argument_group("LLM configuration")
    llm_group.add_argument(
        "--provider",
        choices=["bedrock", "openai"],
        default="bedrock",
        help="LLM provider (default: bedrock)",
    )
    llm_group.add_argument(
        "--model",
        default="anthropic.claude-v2",
        dest="model",
        help="Model ID (default: anthropic.claude-v2)",
    )
    llm_group.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)",
    )
    llm_group.add_argument(
        "--api-key",
        default=None,
        help="API key for OpenAI",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assert-review",
        description="LLM-based compliance note evaluation for financial services.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── evaluate ──────────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a single compliance note file.",
    )
    eval_parser.add_argument("note_file", help="Path to the compliance note text file.")
    eval_parser.add_argument(
        "--framework",
        required=True,
        help="Framework ID (e.g. fca_suitability_v1) or path to a YAML framework file.",
    )
    eval_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include LLM reasoning notes in output.",
    )
    _add_llm_args(eval_parser)
    eval_parser.set_defaults(func=cmd_evaluate)

    # ── batch ─────────────────────────────────────────────────────────────────
    batch_parser = subparsers.add_parser(
        "batch",
        help="Evaluate multiple notes from a CSV file.",
    )
    batch_parser.add_argument("csv_file", help="Path to the CSV file.")
    batch_parser.add_argument(
        "--framework",
        required=True,
        help="Framework ID or path to a YAML framework file.",
    )
    batch_parser.add_argument(
        "--note-column",
        default="note_text",
        help="CSV column containing the note text (default: note_text).",
    )
    batch_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    batch_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include LLM reasoning notes in output.",
    )
    _add_llm_args(batch_parser)
    batch_parser.set_defaults(func=cmd_batch)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
