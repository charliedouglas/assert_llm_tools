"""
assert_llm_tools.cli
~~~~~~~~~~~~~~~~~~~~

Command-line entry point for assert_llm_tools.

Usage examples:
    assert evaluate --framework fca_suitability_v1 --input note.txt
    assert evaluate --framework fca_suitability_v1 --input note.txt --output report.json
    assert evaluate --framework path/to/custom.yaml --input note.txt --output out.json
    assert evaluate --framework fca_suitability_v1 --input note.txt --verbose
    assert evaluate --framework fca_suitability_v1 --input note.txt --mask-pii
    assert evaluate --framework fca_suitability_v1 --input note.txt --summary-only
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Module-level import so tests can patch `assert_llm_tools.cli.evaluate_note`.
from .metrics.note.evaluate_note import evaluate_note  # noqa: E402


# ── ANSI colour helpers ────────────────────────────────────────────────────────

def _supports_colour() -> bool:
    """Return True if the terminal likely supports ANSI colour codes."""
    import os
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


_COLOUR = None  # lazily initialised on first use


def _use_colour() -> bool:
    global _COLOUR
    if _COLOUR is None:
        _COLOUR = _supports_colour()
    return _COLOUR


def _bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if _use_colour() else text


def _red(text: str) -> str:
    return f"\033[31m{text}\033[0m" if _use_colour() else text


def _green(text: str) -> str:
    return f"\033[32m{text}\033[0m" if _use_colour() else text


def _yellow(text: str) -> str:
    return f"\033[33m{text}\033[0m" if _use_colour() else text


def _cyan(text: str) -> str:
    return f"\033[36m{text}\033[0m" if _use_colour() else text


def _dim(text: str) -> str:
    return f"\033[2m{text}\033[0m" if _use_colour() else text


# ── Severity ordering / display ────────────────────────────────────────────────

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_SEVERITY_COLOUR = {
    "critical": _red,
    "high": _yellow,
    "medium": _cyan,
    "low": _dim,
}

_STATUS_BULLET = {
    "present": "✓",
    "partial": "◐",
    "missing": "●",
}
_STATUS_COLOUR = {
    "present": _green,
    "partial": _yellow,
    "missing": _red,
}


# ── Terminal formatter ─────────────────────────────────────────────────────────

def _format_report(report) -> str:
    """
    Render a GapReport as a human-readable string for terminal output.

    Layout::

        Framework: FCA Suitability Note Framework (v1.0.0)
        Overall:   Non-Compliant ❌
        Score:     0.42  (17/9 elements assessed)

        GAPS:
          ● [CRITICAL] recommendation_rationale — missing
            Suggestions: ...
          ◐ [HIGH] financial_situation — partial (0.6)

        PRESENT:
          ✓ client_objectives
          ✓ risk_attitude

        Summary:
          <LLM-generated summary text>
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    fw_label = f"{report.framework_id} (v{report.framework_version})"
    lines.append(_bold(f"Framework: {fw_label}"))

    if report.passed:
        overall_label = _green("Compliant ✅")
    else:
        overall_label = _red("Non-Compliant ❌")
    lines.append(f"Overall:   {overall_label}")

    score_pct = f"{report.overall_score:.2f}"
    stats = report.stats
    lines.append(
        f"Score:     {_bold(score_pct)}  "
        f"({stats.present_count} present / {stats.partial_count} partial / "
        f"{stats.missing_count} missing  |  {stats.total_elements} elements)"
    )

    if report.pii_masked:
        lines.append(_dim("  ⚑ PII masking was applied before evaluation"))

    # ── Gaps ──────────────────────────────────────────────────────────────────
    gap_items = sorted(
        [it for it in report.items if it.status != "present"],
        key=lambda it: (_SEVERITY_ORDER.get(it.severity, 99), it.element_id),
    )

    if gap_items:
        lines.append("")
        lines.append(_bold("GAPS:"))
        for item in gap_items:
            bullet = _STATUS_COLOUR[item.status](_STATUS_BULLET[item.status])
            sev_tag = _SEVERITY_COLOUR.get(item.severity, lambda x: x)(
                f"[{item.severity.upper()}]"
            )
            score_tag = f"  (score: {item.score:.2f})" if item.status == "partial" else ""
            lines.append(f"  {bullet} {sev_tag} {item.element_id} — {item.status}{score_tag}")

            if item.notes:
                # Wrap notes at ~76 chars for readability
                _append_wrapped(lines, f"Notes: {item.notes}", indent="      ")

            if not item.required:
                lines.append(_dim("      (optional element)"))
    else:
        lines.append("")
        lines.append(_green("GAPS: none — all elements present ✅"))

    # ── Present elements (compact) ────────────────────────────────────────────
    present_items = [it for it in report.items if it.status == "present"]
    if present_items:
        lines.append("")
        lines.append(_bold("PRESENT:"))
        for item in present_items:
            lines.append(f"  {_green('✓')} {item.element_id}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if report.summary:
        lines.append("")
        lines.append(_bold("Summary:"))
        _append_wrapped(lines, report.summary, indent="  ")

    lines.append("")
    return "\n".join(lines)


def _format_summary_report(report) -> str:
    """
    Render a GapReport as a compact summary-only string for terminal output.

    Only emits headline stats: framework, overall rating, score, and severity
    gap counts. No element list, no suggestions, no evidence.

    Layout::

        Framework: fca_suitability_v1 (v1.0.0)
        Overall:   Non-Compliant ❌
        Score:     0.42
        Gaps:      critical=1  high=0  medium=0  low=0
        Elements:  3 total  (1 present / 1 partial / 1 missing)
    """
    lines: list[str] = []

    fw_label = f"{report.framework_id} (v{report.framework_version})"
    lines.append(_bold(f"Framework: {fw_label}"))

    if report.passed:
        overall_label = _green("Compliant ✅")
    else:
        overall_label = _red("Non-Compliant ❌")
    lines.append(f"Overall:   {overall_label}")

    lines.append(f"Score:     {_bold(f'{report.overall_score:.2f}')}")

    stats = report.stats
    lines.append(
        f"Gaps:      "
        f"{_red('critical=' + str(stats.critical_gaps))}  "
        f"{_yellow('high=' + str(stats.high_gaps))}  "
        f"{_cyan('medium=' + str(stats.medium_gaps))}  "
        f"{_dim('low=' + str(stats.low_gaps))}"
    )
    lines.append(
        f"Elements:  {stats.total_elements} total  "
        f"({stats.present_count} present / {stats.partial_count} partial / "
        f"{stats.missing_count} missing)"
    )

    if report.pii_masked:
        lines.append(_dim("  ⚑ PII masking was applied before evaluation"))

    lines.append("")
    return "\n".join(lines)


def _append_wrapped(lines: list, text: str, indent: str = "  ", width: int = 76) -> None:
    """Word-wrap *text* at *width* chars and append lines with *indent*."""
    import textwrap
    available = width - len(indent)
    for wrapped_line in textwrap.wrap(text, width=max(available, 20)):
        lines.append(f"{indent}{wrapped_line}")


# ── JSON serialisation ────────────────────────────────────────────────────────

def _report_to_dict(report) -> dict:
    """Convert a GapReport to a JSON-serialisable dict."""
    return asdict(report)


def _report_to_summary_dict(report) -> dict:
    """
    Convert a GapReport to a JSON-serialisable summary dict.

    Omits the ``elements`` / ``items`` array; retains only headline stats
    suitable for quick batch overviews.
    """
    d = asdict(report)
    d.pop("items", None)
    return d


# ── Sub-command: evaluate ─────────────────────────────────────────────────────

def _cmd_evaluate(args: argparse.Namespace) -> int:
    """
    Entry point for `assert evaluate`.

    Returns an integer exit code (0 = pass, 1 = non-compliant, 2 = error).
    """
    # ── Read note text ────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 2

    try:
        note_text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error reading input file: {exc}", file=sys.stderr)
        return 2

    if not note_text.strip():
        print("Error: input file is empty.", file=sys.stderr)
        return 2

    # ── Build LLM config ─────────────────────────────────────────────────────
    llm_config = _build_llm_config(args)

    # ── Run evaluation ────────────────────────────────────────────────────────
    try:
        print(f"Evaluating note against framework '{args.framework}' …", file=sys.stderr)
        report = evaluate_note(
            note_text=note_text,
            framework=args.framework,
            llm_config=llm_config,
            mask_pii=args.mask_pii,
            verbose=args.verbose,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"Framework error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2

    summary_only = getattr(args, "summary_only", False)

    # ── Terminal output ───────────────────────────────────────────────────────
    if summary_only:
        print(_format_summary_report(report))
    else:
        print(_format_report(report))

    # ── JSON output ───────────────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_dict = (
                _report_to_summary_dict(report) if summary_only
                else _report_to_dict(report)
            )
            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(report_dict, fh, indent=2)
            print(f"Report written to: {output_path}", file=sys.stderr)
        except OSError as exc:
            print(f"Error writing output file: {exc}", file=sys.stderr)
            return 2

    # Exit 0 if passed, 1 if non-compliant (useful for CI pipelines)
    return 0 if report.passed else 1


def _build_llm_config(args: argparse.Namespace):
    """
    Construct an LLMConfig from CLI args / environment variables.

    Priority: explicit CLI flags > environment variables > defaults.
    If no provider flags are given, returns None (evaluate_note will use
    its own default, typically Bedrock/Claude).
    """
    provider = getattr(args, "provider", None)
    if not provider:
        return None  # use evaluate_note's built-in default

    from .llm.config import LLMConfig

    return LLMConfig(
        provider=provider,
        model_id=args.model_id,
        region=getattr(args, "region", None),
        api_key=getattr(args, "api_key", None),
    )


# ── Argument parser ───────────────────────────────────────────────────────────

def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    """Add shared LLM provider override arguments to a sub-parser."""
    llm_group = parser.add_argument_group(
        "LLM provider (optional)",
        description=(
            "Override the default LLM provider. If omitted, the library "
            "default (Bedrock/Claude) is used."
        ),
    )
    llm_group.add_argument(
        "--provider",
        choices=["bedrock", "openai"],
        default=None,
        help="LLM provider to use.",
    )
    llm_group.add_argument(
        "--model-id",
        dest="model_id",
        default=None,
        help="Model ID / name for the chosen provider.",
    )
    llm_group.add_argument(
        "--region",
        default=None,
        help="AWS region (Bedrock only).",
    )
    llm_group.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key (OpenAI only).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="assert",
        description="assert_llm_tools — LLM-based compliance and quality evaluation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=_get_version(),
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── evaluate ──────────────────────────────────────────────────────────────
    eval_p = sub.add_parser(
        "evaluate",
        help="Evaluate a compliance note against a regulatory framework.",
        description=(
            "Evaluate a compliance note against a regulatory framework and "
            "produce a gap report."
        ),
    )
    eval_p.add_argument(
        "--framework", "-f",
        required=True,
        metavar="FRAMEWORK",
        help=(
            "Built-in framework ID (e.g. 'fca_suitability_v1') or path to a "
            "custom YAML framework file."
        ),
    )
    eval_p.add_argument(
        "--input", "-i",
        required=True,
        metavar="FILE",
        help="Path to the plain-text compliance note to evaluate.",
    )
    eval_p.add_argument(
        "--output", "-o",
        metavar="FILE",
        default=None,
        help="Optional path for JSON report output (e.g. report.json).",
    )
    eval_p.add_argument(
        "--summary-only",
        action="store_true",
        default=False,
        dest="summary_only",
        help=(
            "Print headline stats only (framework, overall rating, score, "
            "severity gap counts). Omits element list, suggestions, and evidence. "
            "JSON output omits the 'items' array."
        ),
    )
    eval_p.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Include LLM reasoning notes in the report output.",
    )
    eval_p.add_argument(
        "--mask-pii",
        action="store_true",
        default=False,
        dest="mask_pii",
        help="Apply PII detection and masking before sending the note to the LLM.",
    )
    eval_p.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        dest="no_color",
        help="Disable ANSI colour in terminal output.",
    )
    _add_llm_args(eval_p)

    return parser


def _get_version() -> str:
    """Return the installed package version, falling back gracefully."""
    try:
        from importlib.metadata import version
        return f"assert_llm_tools {version('assert_llm_tools')}"
    except Exception:
        return "assert_llm_tools (version unknown)"


# ── Main entry point ──────────────────────────────────────────────────────────

def main(argv: Optional[list] = None) -> None:
    """
    Primary entry point wired up via pyproject.toml [project.scripts].

    Parses arguments, dispatches to the appropriate sub-command handler,
    and exits with a meaningful return code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Apply --no-color before any rendering
    if getattr(args, "no_color", False):
        global _COLOUR
        _COLOUR = False

    if args.command == "evaluate":
        exit_code = _cmd_evaluate(args)
    else:
        parser.print_help()
        exit_code = 2

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
