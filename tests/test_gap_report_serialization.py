"""
tests/test_gap_report_serialization.py — END-50
================================================

Tests for GapItem.to_dict(), GapReportStats.to_dict(), GapReport.to_dict(),
and GapReport.to_json() serialisation methods.

Covers:
  - All required top-level keys present in output
  - Correct types for every field
  - evidence None → JSON null (missing elements)
  - meeting_type nullable
  - metadata includes model, provider, evaluation_time, framework_version
  - to_json() produces valid, round-trippable JSON
  - evaluate_note() pipeline injects system metadata automatically
"""
from __future__ import annotations

# ── Stub native deps (same pattern as test_evaluate_note.py) ─────────────────
import sys
import types


class _AutoMock(types.ModuleType):
    def __getattr__(self, name: str) -> "_AutoMock":
        child = _AutoMock(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs) -> "_AutoMock":
        return _AutoMock("_call_result")


for _stub in ("boto3", "botocore", "botocore.config", "botocore.exceptions", "openai"):
    sys.modules[_stub] = _AutoMock(_stub)

sys.modules["botocore.config"].Config = type(
    "Config", (), {"__init__": lambda self, **kw: None}
)
sys.modules["openai"].OpenAI = type(
    "OpenAI", (), {"__init__": lambda self, **kw: None}
)

# ── Safe to import now ────────────────────────────────────────────────────────
import json
import pytest
from unittest.mock import MagicMock

import assert_llm_tools.metrics.base as _base_mod

_shared_mock_llm = MagicMock(name="shared_mock_llm")
_base_mod.BedrockLLM = lambda cfg: _shared_mock_llm  # type: ignore[assignment]

from assert_llm_tools.metrics.note.models import (
    GapItem,
    GapReport,
    GapReportStats,
    PassPolicy,
)
from assert_llm_tools.metrics.note.evaluate_note import NoteEvaluator


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_stats(**overrides) -> GapReportStats:
    defaults = dict(
        total_elements=5,
        required_elements=3,
        present_count=3,
        partial_count=1,
        missing_count=1,
        critical_gaps=0,
        high_gaps=1,
        medium_gaps=0,
        low_gaps=0,
        required_missing_count=1,
    )
    defaults.update(overrides)
    return GapReportStats(**defaults)


def _make_gap_item(
    element_id: str = "client_objectives",
    status: str = "present",
    score: float = 0.9,
    evidence=None,
    severity: str = "critical",
    required: bool = True,
    suggestions=None,
    name: str = "Client Objectives",
) -> GapItem:
    if suggestions is None:
        suggestions = []
    return GapItem(
        element_id=element_id,
        status=status,
        score=score,
        evidence=evidence if evidence is not None else (
            "Client confirmed retirement goal" if status != "missing" else None
        ),
        severity=severity,
        required=required,
        suggestions=suggestions,
        name=name,
    )


def _make_report(**overrides) -> GapReport:
    items = [
        _make_gap_item("client_objectives", "present", 0.9,
                       evidence="Client confirmed retirement goal",
                       name="Client Objectives"),
        _make_gap_item("risk_attitude", "partial", 0.5,
                       evidence="Risk mentioned briefly",
                       suggestions=["Document ATR score"],
                       name="Risk Attitude"),
        _make_gap_item("capacity_for_loss", "missing", 0.0,
                       evidence=None,
                       suggestions=["Document capacity for loss"],
                       name="Capacity For Loss"),
    ]
    defaults = dict(
        framework_id="fca_suitability_v1",
        framework_version="1.0.0",
        passed=False,
        overall_score=0.6333,
        overall_rating="Requires Attention",
        items=items,
        summary="Evaluation summary.",
        stats=_make_stats(),
        pii_masked=False,
        metadata={
            "model": "anthropic.claude-v2",
            "provider": "bedrock",
            "evaluation_time": "2026-02-19T09:20:00+00:00",
            "framework_version": "1.0.0",
        },
        meeting_type=None,
    )
    defaults.update(overrides)
    return GapReport(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# GapItem.to_dict()
# ─────────────────────────────────────────────────────────────────────────────

class TestGapItemToDict:

    def test_returns_dict(self):
        item = _make_gap_item()
        assert isinstance(item.to_dict(), dict)

    def test_required_keys_present(self):
        d = _make_gap_item().to_dict()
        for key in ("id", "name", "status", "severity", "score", "evidence", "suggestions"):
            assert key in d, f"Missing key: {key}"

    def test_id_maps_to_element_id(self):
        item = _make_gap_item(element_id="risk_attitude")
        assert item.to_dict()["id"] == "risk_attitude"

    def test_name_field_present(self):
        item = _make_gap_item(name="Risk Attitude")
        assert item.to_dict()["name"] == "Risk Attitude"

    def test_name_none_when_not_set(self):
        item = GapItem("x", "present", 0.9, "ev", "critical", required=True)
        # name defaults to None when not explicitly set
        assert item.to_dict()["name"] is None

    def test_status_correct(self):
        item = _make_gap_item(status="partial")
        assert item.to_dict()["status"] == "partial"

    def test_severity_correct(self):
        item = _make_gap_item(severity="high")
        assert item.to_dict()["severity"] == "high"

    def test_score_is_float(self):
        item = _make_gap_item(score=0.75)
        d = item.to_dict()
        assert isinstance(d["score"], float)
        assert d["score"] == pytest.approx(0.75)

    def test_evidence_none_for_missing_element(self):
        """Missing element → evidence is None (serialises as JSON null)."""
        item = _make_gap_item(status="missing", evidence=None)
        d = item.to_dict()
        assert d["evidence"] is None

    def test_evidence_string_for_present_element(self):
        item = _make_gap_item(status="present", evidence="Client confirmed goal")
        assert isinstance(item.to_dict()["evidence"], str)

    def test_suggestions_is_list(self):
        item = _make_gap_item(status="partial", suggestions=["Do X", "Do Y"])
        d = item.to_dict()
        assert isinstance(d["suggestions"], list)
        assert len(d["suggestions"]) == 2

    def test_suggestions_empty_for_present_element(self):
        item = _make_gap_item(status="present", suggestions=[])
        assert item.to_dict()["suggestions"] == []

    def test_suggestions_list_is_independent_copy(self):
        """Mutating the returned list must not affect the GapItem."""
        item = _make_gap_item(suggestions=["A"])
        d = item.to_dict()
        d["suggestions"].append("B")
        assert item.suggestions == ["A"]

    def test_no_extra_internal_fields_leaked(self):
        """notes and required are internal; must not appear in to_dict() output."""
        item = _make_gap_item()
        d = item.to_dict()
        assert "notes" not in d
        assert "required" not in d
        assert "element_id" not in d


# ─────────────────────────────────────────────────────────────────────────────
# GapReportStats.to_dict()
# ─────────────────────────────────────────────────────────────────────────────

class TestGapReportStatsToDict:

    def test_returns_dict(self):
        assert isinstance(_make_stats().to_dict(), dict)

    def test_all_required_keys_present(self):
        d = _make_stats().to_dict()
        expected_keys = {
            "total_elements", "required_elements",
            "present_count", "partial_count", "missing_count",
            "critical_gaps", "high_gaps", "medium_gaps", "low_gaps",
            "required_missing_count",
        }
        assert expected_keys <= set(d.keys()), (
            f"Missing keys: {expected_keys - set(d.keys())}"
        )

    def test_values_are_integers(self):
        d = _make_stats().to_dict()
        for key, value in d.items():
            assert isinstance(value, int), f"{key} should be int, got {type(value)}"

    def test_values_match_source(self):
        stats = _make_stats(total_elements=9, present_count=7, missing_count=2,
                             critical_gaps=1, high_gaps=0)
        d = stats.to_dict()
        assert d["total_elements"] == 9
        assert d["present_count"] == 7
        assert d["missing_count"] == 2
        assert d["critical_gaps"] == 1
        assert d["high_gaps"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# GapReport.to_dict()
# ─────────────────────────────────────────────────────────────────────────────

class TestGapReportToDict:

    def test_returns_dict(self):
        assert isinstance(_make_report().to_dict(), dict)

    def test_top_level_keys_present(self):
        d = _make_report().to_dict()
        for key in ("framework", "meeting_type", "overall_score", "overall_rating",
                    "severity_summary", "elements", "metadata"):
            assert key in d, f"Missing top-level key: {key}"

    def test_framework_is_dict_with_id_and_version(self):
        d = _make_report().to_dict()
        fw = d["framework"]
        assert isinstance(fw, dict)
        assert fw["id"] == "fca_suitability_v1"
        assert fw["version"] == "1.0.0"

    def test_meeting_type_none_by_default(self):
        d = _make_report().to_dict()
        assert d["meeting_type"] is None

    def test_meeting_type_string_when_set(self):
        report = _make_report(meeting_type="annual_review")
        assert report.to_dict()["meeting_type"] == "annual_review"

    def test_overall_score_is_float(self):
        d = _make_report().to_dict()
        assert isinstance(d["overall_score"], float)
        assert 0.0 <= d["overall_score"] <= 1.0

    def test_overall_rating_is_valid_string(self):
        d = _make_report().to_dict()
        valid = {"Compliant", "Minor Gaps", "Requires Attention", "Non-Compliant"}
        assert d["overall_rating"] in valid

    def test_severity_summary_is_dict(self):
        d = _make_report().to_dict()
        assert isinstance(d["severity_summary"], dict)

    def test_severity_summary_has_expected_keys(self):
        d = _make_report().to_dict()
        ss = d["severity_summary"]
        for key in ("total_elements", "present_count", "missing_count",
                    "critical_gaps", "high_gaps"):
            assert key in ss, f"severity_summary missing: {key}"

    def test_elements_is_list(self):
        d = _make_report().to_dict()
        assert isinstance(d["elements"], list)

    def test_elements_count_matches_items(self):
        report = _make_report()
        d = report.to_dict()
        assert len(d["elements"]) == len(report.items)

    def test_each_element_has_required_keys(self):
        d = _make_report().to_dict()
        for elem in d["elements"]:
            for key in ("id", "name", "status", "severity", "score",
                        "evidence", "suggestions"):
                assert key in elem, f"Element missing key: {key}"

    def test_missing_element_evidence_is_null(self):
        """Missing element must serialise evidence as null (None), not empty string."""
        d = _make_report().to_dict()
        missing_elems = [e for e in d["elements"] if e["status"] == "missing"]
        assert missing_elems, "Test report must have at least one missing element"
        for elem in missing_elems:
            assert elem["evidence"] is None, (
                f"Element {elem['id']}: evidence should be null for missing status"
            )

    def test_metadata_is_dict(self):
        assert isinstance(_make_report().to_dict()["metadata"], dict)

    def test_metadata_has_model_key(self):
        d = _make_report().to_dict()
        assert "model" in d["metadata"]

    def test_metadata_has_provider_key(self):
        d = _make_report().to_dict()
        assert "provider" in d["metadata"]

    def test_metadata_has_evaluation_time_key(self):
        d = _make_report().to_dict()
        assert "evaluation_time" in d["metadata"]

    def test_metadata_has_framework_version_key(self):
        d = _make_report().to_dict()
        assert "framework_version" in d["metadata"]
        assert d["metadata"]["framework_version"] == "1.0.0"

    def test_metadata_framework_version_always_set_even_without_caller_metadata(self):
        """framework_version in metadata is auto-populated from the dataclass field."""
        report = _make_report(metadata={})  # no caller metadata at all
        d = report.to_dict()
        assert d["metadata"]["framework_version"] == "1.0.0"

    def test_metadata_caller_values_preserved(self):
        report = _make_report(metadata={
            "model": "gpt-4o",
            "provider": "openai",
            "evaluation_time": "2026-02-19T09:00:00+00:00",
            "framework_version": "1.0.0",
            "note_id": "N-001",
            "adviser_ref": "A-999",
        })
        d = report.to_dict()
        assert d["metadata"]["note_id"] == "N-001"
        assert d["metadata"]["adviser_ref"] == "A-999"
        assert d["metadata"]["model"] == "gpt-4o"

    def test_caller_metadata_takes_precedence_over_framework_version(self):
        """If caller explicitly sets framework_version in metadata, it wins."""
        report = _make_report(
            framework_version="2.0.0",
            metadata={"framework_version": "OVERRIDE"},
        )
        d = report.to_dict()
        assert d["metadata"]["framework_version"] == "OVERRIDE"


# ─────────────────────────────────────────────────────────────────────────────
# GapReport.to_json()
# ─────────────────────────────────────────────────────────────────────────────

class TestGapReportToJson:

    def test_returns_string(self):
        assert isinstance(_make_report().to_json(), str)

    def test_valid_json(self):
        raw = _make_report().to_json()
        parsed = json.loads(raw)  # must not raise
        assert isinstance(parsed, dict)

    def test_default_indent_2(self):
        raw = _make_report().to_json()
        # With indent=2, each level adds 2 spaces
        assert "\n  " in raw

    def test_custom_indent(self):
        raw = _make_report().to_json(indent=4)
        assert "\n    " in raw

    def test_round_trip_top_level_keys(self):
        d_original = _make_report().to_dict()
        d_roundtrip = json.loads(_make_report().to_json())
        for key in ("framework", "meeting_type", "overall_score", "overall_rating",
                    "severity_summary", "elements", "metadata"):
            assert key in d_roundtrip, f"Round-trip missing key: {key}"

    def test_null_evidence_serialised_as_json_null(self):
        """None evidence → 'null' in JSON, not 'None' (Python string)."""
        raw = _make_report().to_json()
        parsed = json.loads(raw)
        missing = [e for e in parsed["elements"] if e["status"] == "missing"]
        assert missing
        for elem in missing:
            assert elem["evidence"] is None

    def test_meeting_type_null_serialised_as_json_null(self):
        raw = _make_report(meeting_type=None).to_json()
        assert '"meeting_type": null' in raw

    def test_meeting_type_string_round_trips(self):
        raw = _make_report(meeting_type="annual_review").to_json()
        parsed = json.loads(raw)
        assert parsed["meeting_type"] == "annual_review"

    def test_overall_rating_is_string_in_json(self):
        parsed = json.loads(_make_report().to_json())
        assert isinstance(parsed["overall_rating"], str)

    def test_score_is_number_in_json(self):
        parsed = json.loads(_make_report().to_json())
        assert isinstance(parsed["overall_score"], float)

    def test_suggestions_is_array_in_json(self):
        parsed = json.loads(_make_report().to_json())
        for elem in parsed["elements"]:
            assert isinstance(elem["suggestions"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: evaluate() pipeline injects system metadata
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateNoteMetadataInjection:
    """
    Verifies that NoteEvaluator.evaluate() populates the system metadata keys
    so that to_dict() output is always audit-complete.
    """

    _FAKE_NOTE = (
        "Client: Jane Smith. Objective: retirement in 15 years. "
        "Risk: balanced. Capacity for loss: can absorb 20% loss."
    )

    _ELEMENT_RESPONSES = [
        "STATUS: present\nSCORE: 0.9\nEVIDENCE: retirement in 15 years\nNOTES: OK",
        "STATUS: present\nSCORE: 0.85\nEVIDENCE: balanced\nNOTES: OK",
        "STATUS: present\nSCORE: 0.8\nEVIDENCE: 20% loss\nNOTES: OK",
        "STATUS: present\nSCORE: 0.75\nEVIDENCE: income £60k\nNOTES: OK",
        "STATUS: present\nSCORE: 0.7\nEVIDENCE: ISAs 8 years\nNOTES: OK",
        "STATUS: present\nSCORE: 0.9\nEVIDENCE: matches profile\nNOTES: OK",
        "STATUS: present\nSCORE: 0.8\nEVIDENCE: OCF 0.45%\nNOTES: OK",
        "STATUS: present\nSCORE: 0.6\nEVIDENCE: alternatives\nNOTES: OK",
        "STATUS: present\nSCORE: 0.7\nEVIDENCE: confirmed\nNOTES: OK",
    ]

    _SUMMARY = "All 9 FCA elements are present."

    def _run(self, extra_meta=None):
        ev = NoteEvaluator()
        ev.llm = MagicMock()
        ev.llm.generate.side_effect = self._ELEMENT_RESPONSES + [self._SUMMARY]
        return ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1",
                           metadata=extra_meta or {})

    def test_evaluation_time_is_iso_string(self):
        """evaluation_time in metadata must be an ISO-8601 string."""
        report = self._run()
        et = report.metadata.get("evaluation_time")
        assert isinstance(et, str), "evaluation_time must be a string"
        # Basic ISO-8601 sanity: should contain 'T' and end with timezone info
        assert "T" in et, f"evaluation_time not ISO: {et}"

    def test_framework_version_in_metadata(self):
        report = self._run()
        assert report.metadata.get("framework_version") == "1.0.0"

    def test_caller_metadata_preserved_alongside_system_keys(self):
        report = self._run(extra_meta={"note_id": "NOTE-42"})
        assert report.metadata.get("note_id") == "NOTE-42"
        assert "evaluation_time" in report.metadata

    def test_caller_can_override_evaluation_time(self):
        fixed_time = "2099-01-01T00:00:00+00:00"
        report = self._run(extra_meta={"evaluation_time": fixed_time})
        assert report.metadata["evaluation_time"] == fixed_time

    def test_to_dict_output_has_all_metadata_keys(self):
        report = self._run()
        d = report.to_dict()
        meta = d["metadata"]
        for key in ("model", "provider", "evaluation_time", "framework_version"):
            assert key in meta, f"metadata missing key: {key}"

    def test_elements_have_name_field_populated(self):
        """All GapItems created by evaluate() have a non-None name."""
        report = self._run()
        for item in report.items:
            assert item.name is not None, (
                f"GapItem {item.element_id} has no name"
            )

    def test_element_name_derived_from_id(self):
        """'client_objectives' → 'Client Objectives' in element name."""
        report = self._run()
        objectives = next(
            (it for it in report.items if it.element_id == "client_objectives"), None
        )
        assert objectives is not None
        assert objectives.name == "Client Objectives"

    def test_element_name_appears_in_to_dict(self):
        report = self._run()
        d = report.to_dict()
        elem_dict = next(
            (e for e in d["elements"] if e["id"] == "client_objectives"), None
        )
        assert elem_dict is not None
        assert elem_dict["name"] == "Client Objectives"

    def test_full_json_round_trip_from_evaluate(self):
        """to_json() output from a real evaluate() call round-trips cleanly."""
        report = self._run()
        raw = report.to_json()
        parsed = json.loads(raw)

        assert parsed["framework"]["id"] == "fca_suitability_v1"
        assert parsed["framework"]["version"] == "1.0.0"
        assert parsed["meeting_type"] is None
        assert isinstance(parsed["overall_score"], float)
        assert parsed["overall_rating"] in {
            "Compliant", "Minor Gaps", "Requires Attention", "Non-Compliant"
        }
        assert isinstance(parsed["elements"], list)
        assert len(parsed["elements"]) == 9
        assert "evaluation_time" in parsed["metadata"]
        assert "framework_version" in parsed["metadata"]
