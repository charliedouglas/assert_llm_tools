"""
Tests for assert_llm_tools.cli

Covers:
- Argument parsing (happy path and error cases)
- Terminal formatter output structure
- JSON serialisation helper
- _cmd_evaluate error paths (no LLM calls made)
- --summary-only flag (END-54)
- assert batch subcommand (END-53): CSV parsing, output format, error handling
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal GapReport / GapItem stubs so tests don't need a live LLM
# ---------------------------------------------------------------------------
from assert_llm_tools.metrics.note.models import (
    GapItem,
    GapReport,
    GapReportStats,
)


def _make_stats(**overrides) -> GapReportStats:
    defaults = dict(
        total_elements=3,
        required_elements=2,
        present_count=1,
        partial_count=1,
        missing_count=1,
        critical_gaps=1,
        high_gaps=0,
        medium_gaps=0,
        low_gaps=0,
        required_missing_count=1,
    )
    defaults.update(overrides)
    return GapReportStats(**defaults)


def _make_report(passed: bool = False) -> GapReport:
    items = [
        GapItem(
            element_id="client_objectives",
            status="present",
            score=0.95,
            evidence="Client wants retirement income",
            severity="critical",
            required=True,
            notes=None,
        ),
        GapItem(
            element_id="risk_attitude",
            status="partial",
            score=0.55,
            evidence="Risk discussed briefly",
            severity="high",
            required=True,
            notes="Risk category not named explicitly",
        ),
        GapItem(
            element_id="recommendation_rationale",
            status="missing",
            score=0.0,
            evidence="",
            severity="critical",
            required=True,
            notes=None,
        ),
    ]
    return GapReport(
        framework_id="fca_suitability_v1",
        framework_version="1.0.0",
        passed=passed,
        overall_score=0.42,
        items=items,
        summary="Two elements are absent or partial.",
        stats=_make_stats(),
        pii_masked=False,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Arg-parsing tests
# ---------------------------------------------------------------------------

class TestArgParsing:
    """Test that the argument parser correctly handles inputs."""

    def _parse(self, argv):
        from assert_llm_tools.cli import _build_parser
        return _build_parser().parse_args(argv)

    def test_evaluate_required_args(self):
        args = self._parse(["evaluate", "--framework", "fca_suitability_v1", "--input", "note.txt"])
        assert args.command == "evaluate"
        assert args.framework == "fca_suitability_v1"
        assert args.input == "note.txt"
        assert args.output is None
        assert args.verbose is False
        assert args.mask_pii is False

    def test_evaluate_with_output(self):
        args = self._parse([
            "evaluate", "--framework", "fca_suitability_v1",
            "--input", "note.txt", "--output", "report.json",
        ])
        assert args.output == "report.json"

    def test_evaluate_verbose_flag(self):
        args = self._parse([
            "evaluate", "-f", "fca_suitability_v1", "-i", "note.txt", "-v",
        ])
        assert args.verbose is True

    def test_evaluate_mask_pii_flag(self):
        args = self._parse([
            "evaluate", "--framework", "fca_suitability_v1",
            "--input", "note.txt", "--mask-pii",
        ])
        assert args.mask_pii is True

    def test_evaluate_no_color_flag(self):
        args = self._parse([
            "evaluate", "--framework", "fca_suitability_v1",
            "--input", "note.txt", "--no-color",
        ])
        assert args.no_color is True

    def test_evaluate_provider_flags(self):
        args = self._parse([
            "evaluate", "--framework", "fca_suitability_v1",
            "--input", "note.txt",
            "--provider", "openai", "--model-id", "gpt-4o", "--api-key", "sk-test",
        ])
        assert args.provider == "openai"
        assert args.model_id == "gpt-4o"
        assert args.api_key == "sk-test"

    def test_missing_framework_raises(self):
        from assert_llm_tools.cli import _build_parser
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["evaluate", "--input", "note.txt"])

    def test_missing_input_raises(self):
        from assert_llm_tools.cli import _build_parser
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["evaluate", "--framework", "fca_suitability_v1"])

    def test_no_subcommand_raises(self):
        from assert_llm_tools.cli import _build_parser
        with pytest.raises(SystemExit):
            _build_parser().parse_args([])


# ---------------------------------------------------------------------------
# Terminal formatter tests
# ---------------------------------------------------------------------------

class TestFormatReport:
    """Test the human-readable terminal formatter."""

    def setup_method(self):
        # Force colour OFF for deterministic output
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

    def test_contains_framework_id(self):
        from assert_llm_tools.cli import _format_report
        report = _make_report()
        output = _format_report(report)
        assert "fca_suitability_v1" in output

    def test_contains_version(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "v1.0.0" in output

    def test_non_compliant_label(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report(passed=False))
        assert "Non-Compliant" in output

    def test_compliant_label(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report(passed=True))
        assert "Compliant" in output

    def test_score_shown(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "0.42" in output

    def test_gaps_section(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "GAPS:" in output
        assert "recommendation_rationale" in output

    def test_present_section(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "PRESENT:" in output
        assert "client_objectives" in output

    def test_summary_section(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "Summary:" in output
        assert "Two elements are absent or partial." in output

    def test_missing_bullet_symbol(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "●" in output  # missing bullet

    def test_partial_bullet_symbol(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "◐" in output  # partial bullet

    def test_present_check_symbol(self):
        from assert_llm_tools.cli import _format_report
        output = _format_report(_make_report())
        assert "✓" in output


# ---------------------------------------------------------------------------
# JSON serialisation tests
# ---------------------------------------------------------------------------

class TestReportToDict:
    def test_round_trip(self):
        from assert_llm_tools.cli import _report_to_dict
        report = _make_report()
        d = _report_to_dict(report)
        # Should be JSON-serialisable
        serialised = json.dumps(d)
        data = json.loads(serialised)
        assert data["framework_id"] == "fca_suitability_v1"
        assert data["overall_score"] == pytest.approx(0.42)
        assert len(data["items"]) == 3

    def test_items_have_expected_keys(self):
        from assert_llm_tools.cli import _report_to_dict
        report = _make_report()
        d = _report_to_dict(report)
        item = d["items"][0]
        assert "element_id" in item
        assert "status" in item
        assert "score" in item
        assert "severity" in item

    def test_stats_included(self):
        from assert_llm_tools.cli import _report_to_dict
        d = _report_to_dict(_make_report())
        assert "stats" in d
        assert d["stats"]["total_elements"] == 3


# ---------------------------------------------------------------------------
# _cmd_evaluate error-path tests (no LLM calls)
# ---------------------------------------------------------------------------

class TestCmdEvaluateErrors:
    """Test error handling in _cmd_evaluate without making real LLM calls."""

    def _args(self, **kwargs):
        defaults = dict(
            framework="fca_suitability_v1",
            input="note.txt",
            output=None,
            verbose=False,
            mask_pii=False,
            summary_only=False,
            provider=None,
            model_id=None,
            region=None,
            api_key=None,
        )
        defaults.update(kwargs)
        return MagicMock(**defaults)

    def test_missing_input_file_returns_2(self, tmp_path):
        from assert_llm_tools.cli import _cmd_evaluate
        args = self._args(input=str(tmp_path / "nonexistent.txt"))
        code = _cmd_evaluate(args)
        assert code == 2

    def test_empty_input_file_returns_2(self, tmp_path):
        from assert_llm_tools.cli import _cmd_evaluate
        note = tmp_path / "note.txt"
        note.write_text("   \n\n")
        args = self._args(input=str(note))
        code = _cmd_evaluate(args)
        assert code == 2

    def test_unknown_framework_returns_2(self, tmp_path):
        from assert_llm_tools.cli import _cmd_evaluate
        note = tmp_path / "note.txt"
        note.write_text("Some note text.")
        args = self._args(framework="nonexistent_framework_xyz", input=str(note))
        code = _cmd_evaluate(args)
        assert code == 2

    def test_happy_path_non_compliant(self, tmp_path, capsys):
        """Happy path: mock evaluate_note, assert exit code 1 (non-compliant)."""
        from assert_llm_tools.cli import _cmd_evaluate
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

        note = tmp_path / "note.txt"
        note.write_text("Client A wants retirement income. Balanced risk.")

        report = _make_report(passed=False)
        args = self._args(input=str(note))

        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            code = _cmd_evaluate(args)

        assert code == 1  # non-compliant → exit 1
        captured = capsys.readouterr()
        assert "fca_suitability_v1" in captured.out
        assert "Non-Compliant" in captured.out

    def test_happy_path_compliant(self, tmp_path, capsys):
        """Happy path: mock evaluate_note, assert exit code 0 (compliant)."""
        from assert_llm_tools.cli import _cmd_evaluate
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

        note = tmp_path / "note.txt"
        note.write_text("Excellent note with all elements present.")

        report = _make_report(passed=True)
        args = self._args(input=str(note))

        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            code = _cmd_evaluate(args)

        assert code == 0  # compliant → exit 0

    def test_json_output_written(self, tmp_path, capsys):
        """--output flag should write valid JSON to the specified path."""
        from assert_llm_tools.cli import _cmd_evaluate
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

        note = tmp_path / "note.txt"
        note.write_text("Some compliance note.")
        out_file = tmp_path / "report.json"

        report = _make_report(passed=False)
        args = self._args(input=str(note), output=str(out_file))

        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_evaluate(args)

        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["framework_id"] == "fca_suitability_v1"
        assert "items" in data
        assert "stats" in data


# ---------------------------------------------------------------------------
# main() integration smoke test
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_missing_input_exits(self, tmp_path):
        from assert_llm_tools.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main([
                "evaluate",
                "--framework", "fca_suitability_v1",
                "--input", str(tmp_path / "missing.txt"),
            ])
        assert exc_info.value.code == 2

    def test_main_happy_path_exits_correctly(self, tmp_path):
        from assert_llm_tools.cli import main
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

        note = tmp_path / "note.txt"
        note.write_text("A compliance note.")
        report = _make_report(passed=True)

        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            with pytest.raises(SystemExit) as exc_info:
                main([
                    "evaluate",
                    "--framework", "fca_suitability_v1",
                    "--input", str(note),
                    "--no-color",
                ])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# END-54: --summary-only flag tests
# ---------------------------------------------------------------------------

class TestSummaryOnlyFlag:
    """Tests for the --summary-only flag on the evaluate subcommand."""

    def setup_method(self):
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

    # ── Arg-parsing ──────────────────────────────────────────────────────────

    def test_summary_only_flag_parsed(self):
        from assert_llm_tools.cli import _build_parser
        args = _build_parser().parse_args([
            "evaluate", "--framework", "fca_suitability_v1",
            "--input", "note.txt", "--summary-only",
        ])
        assert args.summary_only is True

    def test_summary_only_default_false(self):
        from assert_llm_tools.cli import _build_parser
        args = _build_parser().parse_args([
            "evaluate", "--framework", "fca_suitability_v1", "--input", "note.txt",
        ])
        assert args.summary_only is False

    # ── _format_summary_report terminal output ────────────────────────────────

    def test_summary_report_contains_framework(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        assert "fca_suitability_v1" in output

    def test_summary_report_contains_score(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        assert "0.42" in output

    def test_summary_report_contains_overall_rating(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report(passed=False))
        assert "Non-Compliant" in output

    def test_summary_report_compliant_label(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report(passed=True))
        assert "Compliant" in output

    def test_summary_report_contains_gap_counts(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        # Stats: critical=1, high=0, medium=0, low=0
        assert "critical=1" in output
        assert "high=0" in output

    def test_summary_report_contains_element_counts(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        assert "3 total" in output
        assert "1 present" in output

    def test_summary_report_omits_element_ids(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        # Individual element IDs should NOT appear in summary output
        assert "recommendation_rationale" not in output
        assert "client_objectives" not in output
        assert "risk_attitude" not in output

    def test_summary_report_omits_gaps_section(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        assert "GAPS:" not in output

    def test_summary_report_omits_present_section(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        assert "PRESENT:" not in output

    def test_summary_report_omits_summary_text(self):
        from assert_llm_tools.cli import _format_summary_report
        output = _format_summary_report(_make_report())
        assert "Two elements are absent or partial." not in output

    # ── _report_to_summary_dict JSON output ───────────────────────────────────

    def test_summary_dict_omits_items(self):
        from assert_llm_tools.cli import _report_to_summary_dict
        d = _report_to_summary_dict(_make_report())
        assert "items" not in d

    def test_summary_dict_retains_stats(self):
        from assert_llm_tools.cli import _report_to_summary_dict
        d = _report_to_summary_dict(_make_report())
        assert "stats" in d
        assert d["stats"]["total_elements"] == 3

    def test_summary_dict_retains_score(self):
        from assert_llm_tools.cli import _report_to_summary_dict
        d = _report_to_summary_dict(_make_report())
        assert d["overall_score"] == pytest.approx(0.42)

    def test_summary_dict_retains_framework_id(self):
        from assert_llm_tools.cli import _report_to_summary_dict
        d = _report_to_summary_dict(_make_report())
        assert d["framework_id"] == "fca_suitability_v1"

    def test_summary_dict_is_json_serialisable(self):
        from assert_llm_tools.cli import _report_to_summary_dict
        d = _report_to_summary_dict(_make_report())
        serialised = json.dumps(d)
        data = json.loads(serialised)
        assert "framework_id" in data

    # ── _cmd_evaluate with --summary-only ─────────────────────────────────────

    def test_cmd_evaluate_summary_only_terminal(self, tmp_path, capsys):
        from assert_llm_tools.cli import _cmd_evaluate
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

        note = tmp_path / "note.txt"
        note.write_text("A compliance note.")

        report = _make_report(passed=False)
        args = MagicMock(
            framework="fca_suitability_v1",
            input=str(note),
            output=None,
            verbose=False,
            mask_pii=False,
            summary_only=True,
            provider=None,
            model_id=None,
            region=None,
            api_key=None,
        )

        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            code = _cmd_evaluate(args)

        assert code == 1
        captured = capsys.readouterr()
        # Summary output present
        assert "fca_suitability_v1" in captured.out
        assert "Non-Compliant" in captured.out
        assert "critical=1" in captured.out
        # Element-level detail absent
        assert "recommendation_rationale" not in captured.out
        assert "GAPS:" not in captured.out

    def test_cmd_evaluate_summary_only_json_omits_items(self, tmp_path):
        from assert_llm_tools.cli import _cmd_evaluate
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

        note = tmp_path / "note.txt"
        note.write_text("A compliance note.")
        out_file = tmp_path / "summary.json"

        report = _make_report(passed=False)
        args = MagicMock(
            framework="fca_suitability_v1",
            input=str(note),
            output=str(out_file),
            verbose=False,
            mask_pii=False,
            summary_only=True,
            provider=None,
            model_id=None,
            region=None,
            api_key=None,
        )

        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_evaluate(args)

        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert "items" not in data
        assert "stats" in data
        assert "overall_score" in data

    def test_cmd_evaluate_no_summary_only_json_includes_items(self, tmp_path):
        """Without --summary-only, JSON output should include items array."""
        from assert_llm_tools.cli import _cmd_evaluate
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

        note = tmp_path / "note.txt"
        note.write_text("A compliance note.")
        out_file = tmp_path / "full.json"

        report = _make_report(passed=False)
        args = MagicMock(
            framework="fca_suitability_v1",
            input=str(note),
            output=str(out_file),
            verbose=False,
            mask_pii=False,
            summary_only=False,
            provider=None,
            model_id=None,
            region=None,
            api_key=None,
        )

        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_evaluate(args)

        data = json.loads(out_file.read_text())
        assert "items" in data
        assert len(data["items"]) == 3


# ---------------------------------------------------------------------------
# END-53: batch subcommand tests
# ---------------------------------------------------------------------------

import csv as _csv_module  # noqa: E402 (module-level for helpers)


def _write_notes_csv(path, rows: list) -> None:
    """Helper: write a list of dicts as a CSV file."""
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = _csv_module.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class TestBatchArgParsing:
    """Test argument parsing for `assert batch`."""

    def _parse(self, argv):
        from assert_llm_tools.cli import _build_parser
        return _build_parser().parse_args(argv)

    def test_batch_required_args(self):
        args = self._parse([
            "batch", "--framework", "fca_wealth",
            "--input", "notes.csv",
        ])
        assert args.command == "batch"
        assert args.framework == "fca_wealth"
        assert args.input == "notes.csv"
        assert args.output is None
        assert args.summary_only is False
        assert args.mask_pii is False

    def test_batch_with_csv_output(self):
        args = self._parse([
            "batch", "--framework", "fca_wealth",
            "--input", "notes.csv", "--output", "results.csv",
        ])
        assert args.output == "results.csv"

    def test_batch_with_json_output(self):
        args = self._parse([
            "batch", "--framework", "fca_wealth",
            "--input", "notes.csv", "--output", "results.json",
        ])
        assert args.output == "results.json"

    def test_batch_summary_only_flag(self):
        args = self._parse([
            "batch", "--framework", "fca_wealth",
            "--input", "notes.csv", "--summary-only",
        ])
        assert args.summary_only is True

    def test_batch_missing_framework_raises(self):
        from assert_llm_tools.cli import _build_parser
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["batch", "--input", "notes.csv"])

    def test_batch_missing_input_raises(self):
        from assert_llm_tools.cli import _build_parser
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["batch", "--framework", "fca_wealth"])


class TestReadCsv:
    """Test CSV parsing helper."""

    def test_reads_text_column(self, tmp_path):
        from assert_llm_tools.cli import _read_csv
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"text": "Note A"}, {"text": "Note B"}])
        rows = _read_csv(csv_file)
        assert len(rows) == 2
        assert rows[0]["text"] == "Note A"

    def test_reads_note_column(self, tmp_path):
        from assert_llm_tools.cli import _read_csv
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"note": "Note C"}])
        rows = _read_csv(csv_file)
        assert rows[0]["note"] == "Note C"

    def test_reads_optional_metadata_columns(self, tmp_path):
        from assert_llm_tools.cli import _read_csv
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [
            {"note_id": "001", "adviser": "Jane", "text": "Note text"},
        ])
        rows = _read_csv(csv_file)
        assert rows[0]["note_id"] == "001"
        assert rows[0]["adviser"] == "Jane"

    def test_missing_text_column_raises(self, tmp_path):
        from assert_llm_tools.cli import _read_csv
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"id": "1", "content": "No text col"}])
        with pytest.raises(ValueError, match="'text' or 'note' column"):
            _read_csv(csv_file)

    def test_empty_file_returns_empty_list(self, tmp_path):
        from assert_llm_tools.cli import _read_csv
        csv_file = tmp_path / "notes.csv"
        csv_file.write_text("text\n")  # header only, no data
        rows = _read_csv(csv_file)
        assert rows == []

    def test_nonexistent_file_raises(self, tmp_path):
        from assert_llm_tools.cli import _read_csv
        with pytest.raises(OSError):
            _read_csv(tmp_path / "missing.csv")


class TestBuildBatchRecord:
    """Test _build_batch_record helper."""

    def test_record_contains_note_id(self):
        from assert_llm_tools.cli import _build_batch_record
        record = _build_batch_record("note-001", {}, _make_report(), summary_only=True)
        assert record["note_id"] == "note-001"

    def test_record_contains_score(self):
        from assert_llm_tools.cli import _build_batch_record
        record = _build_batch_record("001", {}, _make_report(), summary_only=True)
        assert record["overall_score"] == pytest.approx(0.42, abs=0.001)

    def test_record_contains_passed_flag(self):
        from assert_llm_tools.cli import _build_batch_record
        record = _build_batch_record("001", {}, _make_report(passed=True), summary_only=True)
        assert record["passed"] is True

    def test_record_contains_severity_counts(self):
        from assert_llm_tools.cli import _build_batch_record
        record = _build_batch_record("001", {}, _make_report(), summary_only=True)
        assert record["critical_gaps"] == 1
        assert record["high_gaps"] == 0

    def test_record_overall_rating_non_compliant(self):
        from assert_llm_tools.cli import _build_batch_record
        record = _build_batch_record("001", {}, _make_report(passed=False), summary_only=True)
        # score 0.42 → "Partially Compliant"
        assert record["overall_rating"] in {
            "Non-Compliant", "Partially Compliant", "Mostly Compliant"
        }

    def test_record_summary_only_omits_report(self):
        from assert_llm_tools.cli import _build_batch_record
        record = _build_batch_record("001", {}, _make_report(), summary_only=True)
        assert "report" not in record

    def test_record_non_summary_includes_report(self):
        from assert_llm_tools.cli import _build_batch_record
        record = _build_batch_record("001", {}, _make_report(), summary_only=False)
        assert "report" in record
        assert "items" in record["report"]

    def test_record_carries_adviser_metadata(self):
        from assert_llm_tools.cli import _build_batch_record
        row = {"text": "Note", "adviser": "Jane Smith"}
        record = _build_batch_record("001", row, _make_report(), summary_only=True)
        assert record.get("adviser") == "Jane Smith"


class TestWriteCsvOutput:
    """Test _write_csv_output."""

    def test_csv_has_expected_columns(self, tmp_path):
        from assert_llm_tools.cli import _write_csv_output, _build_batch_record
        records = [_build_batch_record("001", {}, _make_report(passed=True), summary_only=True)]
        out = tmp_path / "out.csv"
        _write_csv_output(out, records)

        with open(out, newline="") as fh:
            rows = list(_csv_module.DictReader(fh))

        assert len(rows) == 1
        assert rows[0]["note_id"] == "001"
        assert "overall_rating" in rows[0]
        assert "overall_score" in rows[0]
        assert "passed" in rows[0]
        assert "critical_gaps" in rows[0]

    def test_csv_multiple_rows(self, tmp_path):
        from assert_llm_tools.cli import _write_csv_output, _build_batch_record
        report = _make_report()
        records = [
            _build_batch_record("001", {}, report, summary_only=True),
            _build_batch_record("002", {}, report, summary_only=True),
        ]
        out = tmp_path / "out.csv"
        _write_csv_output(out, records)

        with open(out, newline="") as fh:
            rows = list(_csv_module.DictReader(fh))
        assert len(rows) == 2
        assert rows[1]["note_id"] == "002"

    def test_csv_optional_adviser_column_included(self, tmp_path):
        from assert_llm_tools.cli import _write_csv_output, _build_batch_record
        row = {"text": "Note", "adviser": "Bob"}
        records = [_build_batch_record("001", row, _make_report(), summary_only=True)]
        out = tmp_path / "out.csv"
        _write_csv_output(out, records)

        with open(out, newline="") as fh:
            rows = list(_csv_module.DictReader(fh))
        assert rows[0].get("adviser") == "Bob"


class TestWriteJsonOutput:
    """Test _write_json_output."""

    def test_json_output_is_array(self, tmp_path):
        from assert_llm_tools.cli import _write_json_output, _build_batch_record
        records = [_build_batch_record("001", {}, _make_report(), summary_only=False)]
        out = tmp_path / "out.json"
        _write_json_output(out, records)

        data = json.loads(out.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_json_record_has_full_report(self, tmp_path):
        from assert_llm_tools.cli import _write_json_output, _build_batch_record
        records = [_build_batch_record("001", {}, _make_report(), summary_only=False)]
        out = tmp_path / "out.json"
        _write_json_output(out, records)

        data = json.loads(out.read_text())
        assert "report" in data[0]
        assert "items" in data[0]["report"]


class TestCmdBatch:
    """Integration tests for _cmd_batch (LLM mocked)."""

    def setup_method(self):
        import assert_llm_tools.cli as cli_module
        cli_module._COLOUR = False

    def _args(self, **kwargs) -> MagicMock:
        defaults = dict(
            framework="fca_wealth",
            input="notes.csv",
            output=None,
            summary_only=False,
            mask_pii=False,
            verbose=False,
            provider=None,
            model_id=None,
            region=None,
            api_key=None,
        )
        defaults.update(kwargs)
        return MagicMock(**defaults)

    def test_missing_input_returns_2(self, tmp_path):
        from assert_llm_tools.cli import _cmd_batch
        args = self._args(input=str(tmp_path / "missing.csv"))
        assert _cmd_batch(args) == 2

    def test_empty_csv_returns_2(self, tmp_path):
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("text\n")  # header only
        args = self._args(input=str(csv_file))
        assert _cmd_batch(args) == 2

    def test_happy_path_csv_output(self, tmp_path):
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [
            {"note_id": "001", "text": "Client wants retirement income."},
            {"note_id": "002", "text": "Another suitability note."},
        ])
        out_file = tmp_path / "results.csv"
        args = self._args(input=str(csv_file), output=str(out_file))

        report = _make_report(passed=False)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            code = _cmd_batch(args)

        assert out_file.exists()
        with open(out_file, newline="") as fh:
            rows = list(_csv_module.DictReader(fh))
        assert len(rows) == 2
        assert rows[0]["note_id"] == "001"
        assert rows[1]["note_id"] == "002"
        assert code == 1  # all non-compliant → exit 1

    def test_happy_path_all_compliant_exit_0(self, tmp_path):
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"text": "Compliant note."}])
        args = self._args(input=str(csv_file))

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            code = _cmd_batch(args)

        assert code == 0

    def test_happy_path_json_output(self, tmp_path):
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"text": "A suitability note."}])
        out_file = tmp_path / "results.json"
        args = self._args(input=str(csv_file), output=str(out_file))

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_batch(args)

        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_summary_only_json_omits_report_key(self, tmp_path):
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"text": "A note."}])
        out_file = tmp_path / "results.json"
        args = self._args(input=str(csv_file), output=str(out_file), summary_only=True)

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_batch(args)

        data = json.loads(out_file.read_text())
        assert "report" not in data[0]

    def test_skips_empty_text_rows(self, tmp_path, capsys):
        """Rows with empty text are skipped with a warning to stderr."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [
            {"note_id": "001", "text": "Good note."},
            {"note_id": "002", "text": "   "},  # empty — should be skipped
        ])
        args = self._args(input=str(csv_file))

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_batch(args)

        captured = capsys.readouterr()
        assert "SKIP" in captured.err or "skip" in captured.err.lower()

    def test_skips_failed_rows_continues_processing(self, tmp_path):
        """If one note raises during evaluation, processing continues."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [
            {"note_id": "001", "text": "Good note."},
            {"note_id": "002", "text": "Another note."},
        ])
        out_file = tmp_path / "results.csv"
        args = self._args(input=str(csv_file), output=str(out_file))

        good_report = _make_report(passed=True)
        call_count = 0

        def mock_evaluate(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("LLM timeout")
            return good_report

        with patch("assert_llm_tools.cli.evaluate_note", side_effect=mock_evaluate):
            _cmd_batch(args)

        # Second note should still be written despite first failing
        with open(out_file, newline="") as fh:
            rows = list(_csv_module.DictReader(fh))
        assert len(rows) == 1
        assert rows[0]["note_id"] == "002"

    def test_progress_messages_printed_to_stderr(self, tmp_path, capsys):
        """Each note should emit a progress counter line to stderr."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [
            {"text": "Note 1"},
            {"text": "Note 2"},
            {"text": "Note 3"},
        ])
        args = self._args(input=str(csv_file))

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_batch(args)

        captured = capsys.readouterr()
        assert "1/3" in captured.err
        assert "2/3" in captured.err
        assert "3/3" in captured.err

    def test_note_column_accepted(self, tmp_path):
        """'note' column (instead of 'text') should be accepted."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"note": "Note content here."}])
        out_file = tmp_path / "results.csv"
        args = self._args(input=str(csv_file), output=str(out_file))

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            code = _cmd_batch(args)

        assert code == 0

    def test_no_output_prints_table_to_stdout(self, tmp_path, capsys):
        """Without --output, a summary table is printed to stdout."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"note_id": "001", "text": "Some note."}])
        args = self._args(input=str(csv_file), output=None)

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_batch(args)

        captured = capsys.readouterr()
        assert "001" in captured.out

    def test_auto_assigns_note_id_when_missing(self, tmp_path):
        """If CSV has no note_id column, the row index is used as note_id."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"text": "Note without ID."}])
        out_file = tmp_path / "results.csv"
        args = self._args(input=str(csv_file), output=str(out_file))

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_batch(args)

        with open(out_file, newline="") as fh:
            rows = list(_csv_module.DictReader(fh))
        assert rows[0]["note_id"] == "1"

    def test_csv_output_has_correct_column_headers(self, tmp_path):
        """CSV output should have all required header columns."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"text": "A note."}])
        out_file = tmp_path / "results.csv"
        args = self._args(input=str(csv_file), output=str(out_file))

        report = _make_report(passed=True)
        with patch("assert_llm_tools.cli.evaluate_note", return_value=report):
            _cmd_batch(args)

        with open(out_file, newline="") as fh:
            reader = _csv_module.DictReader(fh)
            headers = reader.fieldnames or []

        assert "note_id" in headers
        assert "overall_rating" in headers
        assert "overall_score" in headers
        assert "passed" in headers
        assert "critical_gaps" in headers
        assert "high_gaps" in headers
        assert "medium_gaps" in headers
        assert "low_gaps" in headers

    def test_all_rows_fail_returns_error(self, tmp_path):
        """If all rows error-out, _cmd_batch returns 2."""
        from assert_llm_tools.cli import _cmd_batch
        csv_file = tmp_path / "notes.csv"
        _write_notes_csv(csv_file, [{"text": "   "}, {"text": ""}])
        args = self._args(input=str(csv_file))

        with patch("assert_llm_tools.cli.evaluate_note", side_effect=RuntimeError("fail")):
            code = _cmd_batch(args)

        assert code == 2
