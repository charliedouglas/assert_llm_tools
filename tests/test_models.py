"""
tests/test_models.py
====================

Unit tests for assert_llm_tools.metrics.note.models:
  - GapItem    — field defaults, type checks, construction
  - GapReport  — field defaults, metadata default, pii_masked default
  - GapReportStats — all integer fields, construction
  - PassPolicy — default values, field types, custom values

No LLM calls are made; these are pure data-model tests.
"""
from __future__ import annotations

import dataclasses
from dataclasses import fields

import pytest

from assert_llm_tools.metrics.note.models import (
    GapItem,
    GapReport,
    GapReportStats,
    PassPolicy,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_gap_item(**overrides) -> GapItem:
    defaults = dict(
        element_id="elem_x",
        status="present",
        score=0.8,
        evidence="some evidence",
        severity="critical",
        required=True,
    )
    defaults.update(overrides)
    return GapItem(**defaults)


def _make_stats(**overrides) -> GapReportStats:
    defaults = dict(
        total_elements=9,
        required_elements=7,
        present_count=7,
        partial_count=1,
        missing_count=1,
        critical_gaps=0,
        high_gaps=1,
        medium_gaps=0,
        low_gaps=0,
        required_missing_count=2,
    )
    defaults.update(overrides)
    return GapReportStats(**defaults)


def _make_report(**overrides) -> GapReport:
    item = _make_gap_item()
    stats = _make_stats(
        total_elements=1,
        required_elements=1,
        present_count=1,
        partial_count=0,
        missing_count=0,
        critical_gaps=0,
        high_gaps=0,
        medium_gaps=0,
        low_gaps=0,
        required_missing_count=0,
    )
    defaults = dict(
        framework_id="test_fw",
        framework_version="1.0.0",
        passed=True,
        overall_score=0.8,
        items=[item],
        summary="All elements present.",
        stats=stats,
    )
    defaults.update(overrides)
    return GapReport(**defaults)


# ═══════════════════════════════════════════════════════════════════════════════
# GapItem
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapItem:

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(GapItem)

    def test_construction_all_required_fields(self):
        item = GapItem(
            element_id="risk_profile",
            status="present",
            score=0.9,
            evidence="Risk assessed via Dynamic Planner",
            severity="critical",
            required=True,
        )
        assert item.element_id == "risk_profile"
        assert item.status == "present"
        assert item.score == pytest.approx(0.9)
        assert item.evidence == "Risk assessed via Dynamic Planner"
        assert item.severity == "critical"
        assert item.required is True

    def test_notes_defaults_to_none(self):
        """Optional notes field defaults to None when not supplied."""
        item = _make_gap_item()
        assert item.notes is None

    def test_notes_can_be_set(self):
        item = _make_gap_item(notes="Some LLM reasoning here")
        assert item.notes == "Some LLM reasoning here"

    def test_required_false(self):
        item = _make_gap_item(required=False)
        assert item.required is False

    def test_status_partial(self):
        item = _make_gap_item(status="partial", score=0.45)
        assert item.status == "partial"
        assert item.score == pytest.approx(0.45)

    def test_status_missing_zero_score(self):
        item = _make_gap_item(status="missing", score=0.0, evidence="")
        assert item.status == "missing"
        assert item.score == pytest.approx(0.0)
        assert item.evidence == ""

    def test_all_severity_values(self):
        for sev in ("critical", "high", "medium", "low"):
            item = _make_gap_item(severity=sev)
            assert item.severity == sev

    def test_score_boundary_zero(self):
        item = _make_gap_item(score=0.0)
        assert item.score == pytest.approx(0.0)

    def test_score_boundary_one(self):
        item = _make_gap_item(score=1.0)
        assert item.score == pytest.approx(1.0)

    def test_field_names(self):
        """GapItem must have exactly these fields."""
        field_names = {f.name for f in fields(GapItem)}
        expected = {"element_id", "status", "score", "evidence", "severity", "required", "notes"}
        assert expected == field_names

    def test_equality_same_values(self):
        """Two GapItems with the same values should be equal (dataclass default)."""
        a = _make_gap_item(notes=None)
        b = _make_gap_item(notes=None)
        assert a == b

    def test_inequality_different_score(self):
        a = _make_gap_item(score=0.8)
        b = _make_gap_item(score=0.5)
        assert a != b


# ═══════════════════════════════════════════════════════════════════════════════
# GapReport
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapReport:

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(GapReport)

    def test_construction_required_fields(self):
        report = _make_report()
        assert report.framework_id == "test_fw"
        assert report.framework_version == "1.0.0"
        assert report.passed is True
        assert isinstance(report.overall_score, float)
        assert isinstance(report.items, list)
        assert isinstance(report.summary, str)

    def test_pii_masked_defaults_to_false(self):
        """pii_masked field default is False."""
        report = _make_report()
        assert report.pii_masked is False

    def test_pii_masked_can_be_true(self):
        report = _make_report(pii_masked=True)
        assert report.pii_masked is True

    def test_metadata_defaults_to_empty_dict(self):
        """metadata field default is {} (each instance gets a fresh dict)."""
        report = _make_report()
        assert report.metadata == {}
        assert isinstance(report.metadata, dict)

    def test_metadata_not_shared_between_instances(self):
        """Default metadata dict must not be shared between instances."""
        r1 = _make_report()
        r2 = _make_report()
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata

    def test_metadata_can_hold_arbitrary_values(self):
        meta = {"note_id": "N-001", "adviser": "Jane Doe", "batch": 42}
        report = _make_report(metadata=meta)
        assert report.metadata["note_id"] == "N-001"
        assert report.metadata["adviser"] == "Jane Doe"
        assert report.metadata["batch"] == 42

    def test_passed_false(self):
        report = _make_report(passed=False)
        assert report.passed is False

    def test_items_list_multiple_elements(self):
        items = [_make_gap_item(element_id=f"e{i}") for i in range(5)]
        report = _make_report(items=items)
        assert len(report.items) == 5

    def test_overall_score_is_float(self):
        report = _make_report(overall_score=0.73)
        assert isinstance(report.overall_score, float)

    def test_stats_is_gap_report_stats(self):
        report = _make_report()
        assert isinstance(report.stats, GapReportStats)

    def test_field_names(self):
        field_names = {f.name for f in fields(GapReport)}
        expected = {
            "framework_id", "framework_version", "passed", "overall_score",
            "items", "summary", "stats", "pii_masked", "metadata",
        }
        assert expected == field_names


# ═══════════════════════════════════════════════════════════════════════════════
# GapReportStats
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapReportStats:

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(GapReportStats)

    def test_construction(self):
        stats = _make_stats()
        assert stats.total_elements == 9
        assert stats.required_elements == 7
        assert stats.present_count == 7
        assert stats.partial_count == 1
        assert stats.missing_count == 1
        assert stats.critical_gaps == 0
        assert stats.high_gaps == 1
        assert stats.medium_gaps == 0
        assert stats.low_gaps == 0
        assert stats.required_missing_count == 2

    def test_all_fields_are_int(self):
        """All fields on GapReportStats must be integers."""
        stats = _make_stats()
        for f in fields(GapReportStats):
            value = getattr(stats, f.name)
            assert isinstance(value, int), f"{f.name} should be int, got {type(value)}"

    def test_zero_stats(self):
        """All-zero stats are valid (empty framework edge case)."""
        stats = _make_stats(
            total_elements=0,
            required_elements=0,
            present_count=0,
            partial_count=0,
            missing_count=0,
            critical_gaps=0,
            high_gaps=0,
            medium_gaps=0,
            low_gaps=0,
            required_missing_count=0,
        )
        assert stats.total_elements == 0

    def test_field_names(self):
        field_names = {f.name for f in fields(GapReportStats)}
        expected = {
            "total_elements", "required_elements", "present_count", "partial_count",
            "missing_count", "critical_gaps", "high_gaps", "medium_gaps", "low_gaps",
            "required_missing_count",
        }
        assert expected == field_names


# ═══════════════════════════════════════════════════════════════════════════════
# PassPolicy
# ═══════════════════════════════════════════════════════════════════════════════

class TestPassPolicyModel:

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(PassPolicy)

    def test_default_values(self):
        p = PassPolicy()
        assert p.block_on_critical_missing is True
        assert p.block_on_critical_partial is True
        assert p.block_on_high_missing is True
        assert p.critical_partial_threshold == pytest.approx(0.5)

    def test_all_bool_fields_are_bool(self):
        p = PassPolicy()
        assert isinstance(p.block_on_critical_missing, bool)
        assert isinstance(p.block_on_critical_partial, bool)
        assert isinstance(p.block_on_high_missing, bool)

    def test_threshold_is_float(self):
        p = PassPolicy()
        assert isinstance(p.critical_partial_threshold, float)

    def test_custom_lenient_policy(self):
        p = PassPolicy(
            block_on_critical_missing=False,
            block_on_critical_partial=False,
            block_on_high_missing=False,
            critical_partial_threshold=0.0,
        )
        assert p.block_on_critical_missing is False
        assert p.block_on_critical_partial is False
        assert p.block_on_high_missing is False
        assert p.critical_partial_threshold == pytest.approx(0.0)

    def test_custom_strict_threshold(self):
        p = PassPolicy(critical_partial_threshold=0.75)
        assert p.critical_partial_threshold == pytest.approx(0.75)

    def test_mixed_policy_custom_threshold(self):
        p = PassPolicy(
            block_on_critical_missing=True,
            block_on_critical_partial=True,
            block_on_high_missing=False,
            critical_partial_threshold=0.3,
        )
        assert p.block_on_critical_missing is True
        assert p.block_on_high_missing is False
        assert p.critical_partial_threshold == pytest.approx(0.3)

    def test_field_names(self):
        field_names = {f.name for f in fields(PassPolicy)}
        expected = {
            "block_on_critical_missing",
            "block_on_critical_partial",
            "block_on_high_missing",
            "critical_partial_threshold",
        }
        assert expected == field_names

    def test_equality_default_instances(self):
        """Two default PassPolicy instances should be equal."""
        assert PassPolicy() == PassPolicy()

    def test_inequality_different_threshold(self):
        assert PassPolicy(critical_partial_threshold=0.3) != PassPolicy(critical_partial_threshold=0.7)
