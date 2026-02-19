"""
test_meeting_type.py — Tests for END-46: meeting_type parameter with framework overrides
=========================================================================================

Covers:
  - meeting_type=None uses full framework (no overrides)
  - known meeting_type applies overrides correctly (required/severity promoted/demoted)
  - unknown meeting_type is a no-op (falls back to full framework, no error)
  - meeting_type is recorded in GapReport output
  - evaluate_note() public entry point accepts meeting_type
"""
from __future__ import annotations

import sys
import types


# ── 1. Stub native deps before any library imports ────────────────────────────
class _AutoMock(types.ModuleType):
    def __getattr__(self, name: str) -> "_AutoMock":
        child = _AutoMock(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs) -> "_AutoMock":
        return _AutoMock("_call_result")


for _stub_name in ("boto3", "botocore", "botocore.config", "botocore.exceptions", "openai"):
    sys.modules[_stub_name] = _AutoMock(_stub_name)

sys.modules["botocore.config"].Config = type("Config", (), {"__init__": lambda self, **kw: None})
sys.modules["openai"].OpenAI = type("OpenAI", (), {"__init__": lambda self, **kw: None})

# ── 2. Patch BedrockLLM before importing library ──────────────────────────────
import assert_llm_tools.metrics.base as _base_mod
from unittest.mock import MagicMock, patch

_shared_mock_llm = MagicMock(name="shared_mock_llm")
_base_mod.BedrockLLM = lambda cfg: _shared_mock_llm

import pytest
from assert_llm_tools.metrics.note.models import GapItem, GapReport, PassPolicy
from assert_llm_tools.metrics.note.evaluate_note import NoteEvaluator, evaluate_note

# ── 3. Framework fixtures ─────────────────────────────────────────────────────

# Minimal framework with meeting_type_overrides — mirrors fca_wealth.yaml structure
_FW_WITH_OVERRIDES: dict = {
    "framework_id": "test_fw_overrides",
    "name": "Test Framework With Overrides",
    "version": "1.0.0",
    "regulator": "TEST",
    "elements": [
        {
            "id": "client_id",
            "description": "Client identity verification",
            "required": True,
            "severity": "critical",
        },
        {
            "id": "meeting_context",
            "description": "Meeting context",
            "required": True,
            "severity": "high",
        },
        {
            "id": "objectives",
            "description": "Client objectives",
            "required": True,
            "severity": "critical",
        },
        {
            "id": "esg",
            "description": "ESG preference assessment",
            "required": True,
            "severity": "medium",
        },
    ],
    "meeting_type_overrides": {
        "annual_review": {
            "description": "Annual review overrides",
            "elements": {
                "client_id": {
                    "required": False,
                    "severity": "low",
                },
                "objectives": {
                    "required": True,
                    "severity": "critical",
                    # no change — stays as-is
                },
                "esg": {
                    "required": True,
                    "severity": "high",   # promoted from medium → high
                },
            },
        },
        "ad_hoc_call": {
            "description": "Ad hoc call overrides",
            "elements": {
                "client_id": {
                    "required": False,
                    "severity": "low",
                },
                "objectives": {
                    "required": False,
                    "severity": "medium",
                },
                "esg": {
                    "required": False,
                    "severity": "low",
                },
            },
        },
    },
}

# Framework with no meeting_type_overrides section at all
_FW_WITHOUT_OVERRIDES: dict = {
    "framework_id": "test_fw_no_overrides",
    "name": "Test Framework No Overrides",
    "version": "1.0.0",
    "regulator": "TEST",
    "elements": [
        {
            "id": "client_id",
            "description": "Client identity verification",
            "required": True,
            "severity": "critical",
        },
    ],
}

_FAKE_NOTE = "Client meeting held. Client ID verified. Objectives discussed. ESG preferences assessed."


# ── 4. Helpers ────────────────────────────────────────────────────────────────

def _make_evaluator(pass_policy: PassPolicy | None = None) -> NoteEvaluator:
    ev = NoteEvaluator(pass_policy=pass_policy)
    ev.llm = MagicMock(name="test_mock_llm")
    return ev


def _element_response(status: str = "present", score: float = 0.9) -> str:
    return f"STATUS: {status}\nSCORE: {score}\nEVIDENCE: Some evidence\nNOTES: OK"


def _mock_responses(n_elements: int, summary: str = "All good.") -> list:
    """Return n_elements element responses + one summary response."""
    return [_element_response() for _ in range(n_elements)] + [summary]


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: meeting_type=None (no overrides)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeetingTypeNone:

    def test_meeting_type_none_uses_full_framework(self):
        """
        meeting_type=None → elements evaluated with their original required/severity.
        client_id must remain required=True, severity=critical (no override).
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type=None)

        client_id_item = next(it for it in report.items if it.element_id == "client_id")
        assert client_id_item.required is True
        assert client_id_item.severity == "critical"

    def test_meeting_type_none_recorded_as_none_in_report(self):
        """meeting_type=None → GapReport.meeting_type is None."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type=None)

        assert report.meeting_type is None

    def test_meeting_type_defaults_to_none(self):
        """Omitting meeting_type entirely → same as passing None."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES)

        assert report.meeting_type is None

    def test_meeting_type_none_does_not_mutate_framework(self):
        """evaluate() with meeting_type=None must not modify the framework dict."""
        import copy
        fw_copy = copy.deepcopy(_FW_WITH_OVERRIDES)

        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)
        ev.evaluate(_FAKE_NOTE, fw_copy, meeting_type=None)

        assert fw_copy["elements"][0]["required"] is True
        assert fw_copy["elements"][0]["severity"] == "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: annual_review overrides
# ═══════════════════════════════════════════════════════════════════════════════

class TestAnnualReviewOverrides:

    def test_client_id_demoted_to_not_required_on_annual_review(self):
        """
        annual_review override: client_id required=False, severity=low.
        The GapItem for client_id must reflect the override.
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="annual_review")

        client_id_item = next(it for it in report.items if it.element_id == "client_id")
        assert client_id_item.required is False, (
            "client_id should be not-required under annual_review override"
        )
        assert client_id_item.severity == "low", (
            "client_id severity should be 'low' under annual_review override"
        )

    def test_esg_promoted_to_high_severity_on_annual_review(self):
        """
        annual_review override: esg severity promoted from medium → high.
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="annual_review")

        esg_item = next(it for it in report.items if it.element_id == "esg")
        assert esg_item.severity == "high", (
            "esg severity should be promoted to 'high' under annual_review override"
        )
        assert esg_item.required is True

    def test_objectives_unchanged_on_annual_review(self):
        """
        annual_review override specifies objectives with same values as base —
        element properties should remain required=True, severity=critical.
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="annual_review")

        objectives_item = next(it for it in report.items if it.element_id == "objectives")
        assert objectives_item.required is True
        assert objectives_item.severity == "critical"

    def test_meeting_context_unaffected_by_annual_review_override(self):
        """
        meeting_context has no annual_review override → stays required=True, severity=high.
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="annual_review")

        context_item = next(it for it in report.items if it.element_id == "meeting_context")
        assert context_item.required is True
        assert context_item.severity == "high"

    def test_annual_review_meeting_type_recorded_in_report(self):
        """GapReport.meeting_type == 'annual_review' when that override is applied."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="annual_review")

        assert report.meeting_type == "annual_review"

    def test_annual_review_does_not_mutate_original_framework(self):
        """
        Applying annual_review overrides must NOT modify the original framework dict
        (deep-copy must be used internally).
        """
        import copy
        fw_copy = copy.deepcopy(_FW_WITH_OVERRIDES)

        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)
        ev.evaluate(_FAKE_NOTE, fw_copy, meeting_type="annual_review")

        # Original framework must be untouched
        client_id_elem = next(e for e in fw_copy["elements"] if e["id"] == "client_id")
        assert client_id_elem["required"] is True, (
            "evaluate() must not mutate the source framework dict"
        )
        assert client_id_elem["severity"] == "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: ad_hoc_call overrides
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdHocCallOverrides:

    def test_objectives_demoted_to_optional_on_ad_hoc_call(self):
        """
        ad_hoc_call override: objectives required=False.
        An optional missing element should not cause the report to fail.
        """
        ev = _make_evaluator()
        # Make objectives missing
        responses = [
            _element_response("present", 0.9),   # client_id
            _element_response("present", 0.9),   # meeting_context
            "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent",  # objectives
            _element_response("present", 0.9),   # esg
            "All elements assessed.",             # summary
        ]
        ev.llm.generate.side_effect = responses

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="ad_hoc_call")

        objectives_item = next(it for it in report.items if it.element_id == "objectives")
        assert objectives_item.required is False
        # A non-required missing element should not block pass
        assert report.passed is True, (
            "Non-required missing element should not fail the report"
        )

    def test_esg_demoted_to_not_required_on_ad_hoc_call(self):
        """ad_hoc_call override: esg required=False, severity=low."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="ad_hoc_call")

        esg_item = next(it for it in report.items if it.element_id == "esg")
        assert esg_item.required is False
        assert esg_item.severity == "low"

    def test_ad_hoc_call_meeting_type_recorded_in_report(self):
        """GapReport.meeting_type == 'ad_hoc_call'."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="ad_hoc_call")

        assert report.meeting_type == "ad_hoc_call"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: unknown meeting_type → no-op fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnknownMeetingTypeNoOp:

    def test_unknown_meeting_type_uses_full_framework(self):
        """
        An unrecognised meeting_type falls back to the full framework — no error raised.
        Elements keep their original required/severity values.
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        # Must not raise
        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="quarterly_update")

        client_id_item = next(it for it in report.items if it.element_id == "client_id")
        assert client_id_item.required is True
        assert client_id_item.severity == "critical"

    def test_unknown_meeting_type_not_recorded_in_report(self):
        """
        Unknown meeting_type → GapReport.meeting_type is None
        (effective_meeting_type stays None when override not found).
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        report = ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="quarterly_update")

        assert report.meeting_type is None, (
            "Unknown meeting_type should not be recorded — report.meeting_type must be None"
        )

    def test_meeting_type_on_framework_without_overrides_section(self):
        """
        Framework with no meeting_type_overrides section → unknown type is a no-op.
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=1)

        # Must not raise
        report = ev.evaluate(_FAKE_NOTE, _FW_WITHOUT_OVERRIDES, meeting_type="annual_review")

        assert report.meeting_type is None
        assert len(report.items) == 1

    def test_unknown_meeting_type_does_not_raise_exception(self):
        """Unknown meeting_type must never raise — it is always a silent no-op."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = _mock_responses(n_elements=4)

        try:
            ev.evaluate(_FAKE_NOTE, _FW_WITH_OVERRIDES, meeting_type="completely_unknown_xyz")
        except Exception as exc:
            pytest.fail(f"evaluate() raised an exception for unknown meeting_type: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: GapReport.meeting_type field
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapReportMeetingTypeField:

    def test_gap_report_has_meeting_type_field(self):
        """GapReport dataclass must have a meeting_type attribute."""
        from assert_llm_tools.metrics.note.models import GapReport
        import dataclasses
        fields = {f.name for f in dataclasses.fields(GapReport)}
        assert "meeting_type" in fields, "GapReport must have a 'meeting_type' field"

    def test_gap_report_meeting_type_defaults_to_none(self):
        """GapReport.meeting_type defaults to None."""
        from assert_llm_tools.metrics.note.models import GapReport, GapReportStats
        report = GapReport(
            framework_id="x",
            framework_version="1.0",
            passed=True,
            overall_score=1.0,
            items=[],
            summary="ok",
            stats=GapReportStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        assert report.meeting_type is None

    def test_gap_report_meeting_type_can_be_set(self):
        """GapReport.meeting_type can hold a string value."""
        from assert_llm_tools.metrics.note.models import GapReport, GapReportStats
        report = GapReport(
            framework_id="x",
            framework_version="1.0",
            passed=True,
            overall_score=1.0,
            items=[],
            summary="ok",
            stats=GapReportStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            meeting_type="annual_review",
        )
        assert report.meeting_type == "annual_review"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: evaluate_note() public entry point accepts meeting_type
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateNoteFunctionMeetingType:

    def test_evaluate_note_accepts_meeting_type_kwarg(self):
        """evaluate_note() top-level function accepts meeting_type keyword argument."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = _mock_responses(n_elements=4)

        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=_FAKE_NOTE,
                framework=_FW_WITH_OVERRIDES,
                meeting_type="annual_review",
            )

        assert isinstance(report, GapReport)
        assert report.meeting_type == "annual_review"

    def test_evaluate_note_meeting_type_none_accepted(self):
        """evaluate_note() with meeting_type=None works correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = _mock_responses(n_elements=4)

        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=_FAKE_NOTE,
                framework=_FW_WITH_OVERRIDES,
                meeting_type=None,
            )

        assert report.meeting_type is None

    def test_evaluate_note_overrides_applied_via_top_level_function(self):
        """
        When called via evaluate_note(), the annual_review override is applied:
        client_id becomes not-required.
        """
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = _mock_responses(n_elements=4)

        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=_FAKE_NOTE,
                framework=_FW_WITH_OVERRIDES,
                meeting_type="annual_review",
            )

        client_id_item = next(it for it in report.items if it.element_id == "client_id")
        assert client_id_item.required is False


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: fca_wealth framework integration (real YAML, mocked LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestFcaWealthFrameworkIntegration:
    """
    Smoke-tests against the real fca_wealth.yaml to ensure the override
    structure from END-47 is correctly applied by the END-46 implementation.
    """

    _NOTE = (
        "Client meeting. ID verified. Meeting context noted. "
        "Objectives discussed. Risk profile assessed. Suitability rationale provided. "
        "Vulnerability assessed. Existing arrangements reviewed. Charges disclosed. "
        "Tax considered. ESG preferences assessed. Actions agreed. Client agreement obtained."
    )

    def _responses_for_n(self, n: int) -> list:
        return [_element_response() for _ in range(n)] + ["Summary complete."]

    def test_fca_wealth_annual_review_client_id_not_required(self):
        """
        fca_wealth.yaml annual_review: client_id should be required=False after override.
        """
        ev = _make_evaluator()
        fw = __import__(
            "assert_llm_tools.metrics.note.loader",
            fromlist=["load_framework"],
        ).load_framework("fca_wealth")
        n_elements = len(fw["elements"])
        ev.llm.generate.side_effect = self._responses_for_n(n_elements)

        report = ev.evaluate(self._NOTE, "fca_wealth", meeting_type="annual_review")

        client_id_item = next(it for it in report.items if it.element_id == "client_id")
        assert client_id_item.required is False
        assert client_id_item.severity == "low"
        assert report.meeting_type == "annual_review"

    def test_fca_wealth_ad_hoc_call_objectives_optional(self):
        """
        fca_wealth.yaml ad_hoc_call: objectives should be required=False after override.
        """
        ev = _make_evaluator()
        fw = __import__(
            "assert_llm_tools.metrics.note.loader",
            fromlist=["load_framework"],
        ).load_framework("fca_wealth")
        n_elements = len(fw["elements"])
        ev.llm.generate.side_effect = self._responses_for_n(n_elements)

        report = ev.evaluate(self._NOTE, "fca_wealth", meeting_type="ad_hoc_call")

        objectives_item = next(it for it in report.items if it.element_id == "objectives")
        assert objectives_item.required is False
        assert report.meeting_type == "ad_hoc_call"

    def test_fca_wealth_new_client_client_id_critical(self):
        """
        fca_wealth.yaml new_client: client_id stays required=True, severity=critical.
        """
        ev = _make_evaluator()
        fw = __import__(
            "assert_llm_tools.metrics.note.loader",
            fromlist=["load_framework"],
        ).load_framework("fca_wealth")
        n_elements = len(fw["elements"])
        ev.llm.generate.side_effect = self._responses_for_n(n_elements)

        report = ev.evaluate(self._NOTE, "fca_wealth", meeting_type="new_client")

        client_id_item = next(it for it in report.items if it.element_id == "client_id")
        assert client_id_item.required is True
        assert client_id_item.severity == "critical"
        assert report.meeting_type == "new_client"

    def test_fca_wealth_unknown_meeting_type_fallback(self):
        """
        fca_wealth.yaml with unknown meeting_type → no-op, full framework applied.
        """
        ev = _make_evaluator()
        fw = __import__(
            "assert_llm_tools.metrics.note.loader",
            fromlist=["load_framework"],
        ).load_framework("fca_wealth")
        n_elements = len(fw["elements"])
        ev.llm.generate.side_effect = self._responses_for_n(n_elements)

        report = ev.evaluate(self._NOTE, "fca_wealth", meeting_type="board_meeting")

        assert report.meeting_type is None
        client_id_item = next(it for it in report.items if it.element_id == "client_id")
        # Base framework: client_id required=True, severity=critical
        assert client_id_item.required is True
        assert client_id_item.severity == "critical"
