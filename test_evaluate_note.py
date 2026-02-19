"""
test_evaluate_note.py — QA test suite for evaluate_note() (END-42 / P1-05)
===========================================================================

Covers:
  Unit tests  : loader, validator, parser, scorer, pass-policy, stats  (no real LLM)
  Integration : full evaluate_note() pipeline with mocked LLM
"""
from __future__ import annotations

# ── 1. Stub out native deps that aren't installed in the test env ─────────────
#    Must happen BEFORE any assert_llm_tools imports are triggered.
import sys
import types


class _AutoMock(types.ModuleType):
    """Thin module stub: attribute access returns child stubs, calls return stubs."""

    def __getattr__(self, name: str) -> "_AutoMock":
        child = _AutoMock(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs) -> "_AutoMock":  # noqa: D105
        return _AutoMock("_call_result")


for _stub_name in ("boto3", "botocore", "botocore.config", "botocore.exceptions", "openai"):
    sys.modules[_stub_name] = _AutoMock(_stub_name)

# botocore.config.Config must be a real class (instantiated in bedrock.py)
sys.modules["botocore.config"].Config = type("Config", (), {"__init__": lambda self, **kw: None})
# openai.OpenAI must be a real class (instantiated in openai.py)
sys.modules["openai"].OpenAI = type("OpenAI", (), {"__init__": lambda self, **kw: None})

# ── 2. Now safe to import the library ─────────────────────────────────────────
import pytest
from unittest.mock import MagicMock, patch

import assert_llm_tools.metrics.base as _base_mod

# Patch BedrockLLM on the base module so NoteEvaluator() never calls boto3
_shared_mock_llm = MagicMock(name="shared_mock_llm")
_base_mod.BedrockLLM = lambda cfg: _shared_mock_llm  # type: ignore[assignment]

from assert_llm_tools.metrics.note.loader import load_framework, _validate_framework
from assert_llm_tools.metrics.note.models import GapItem, GapReport, GapReportStats, PassPolicy
from assert_llm_tools.metrics.note.evaluate_note import NoteEvaluator, evaluate_note

# ── 3. Helpers ─────────────────────────────────────────────────────────────────

_MINIMAL_VALID_FRAMEWORK: dict = {
    "framework_id": "test_fw",
    "name": "Test Framework",
    "version": "1.0.0",
    "regulator": "TEST",
    "elements": [
        {
            "id": "elem_a",
            "description": "Element A description",
            "required": True,
            "severity": "critical",
        }
    ],
}

_ELEMENT_CRITICAL = {
    "id": "elem_a",
    "description": "Element A",
    "required": True,
    "severity": "critical",
    "guidance": "Look for X.",
}

_ELEMENT_HIGH = {
    "id": "elem_b",
    "description": "Element B",
    "required": True,
    "severity": "high",
}

_ELEMENT_OPTIONAL_CRITICAL = {
    "id": "elem_c",
    "description": "Element C — optional but critical severity",
    "required": False,
    "severity": "critical",
}

_ELEMENT_OPTIONAL_HIGH = {
    "id": "elem_d",
    "description": "Element D — optional high",
    "required": False,
    "severity": "high",
}


def _make_evaluator(verbose: bool = False, pass_policy: PassPolicy | None = None) -> NoteEvaluator:
    """Create a NoteEvaluator with a fresh per-test MagicMock as its LLM."""
    ev = NoteEvaluator(verbose=verbose, pass_policy=pass_policy)
    ev.llm = MagicMock(name="test_mock_llm")
    return ev


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — loader
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFramework:

    def test_load_framework_with_valid_dict_returns_unchanged(self):
        """load_framework(dict) → same dict returned after validation."""
        fw = load_framework(_MINIMAL_VALID_FRAMEWORK)
        assert fw is _MINIMAL_VALID_FRAMEWORK
        assert fw["framework_id"] == "test_fw"

    def test_load_framework_builtin_id_fca_suitability_v1(self):
        """load_framework("fca_suitability_v1") loads the bundled YAML."""
        fw = load_framework("fca_suitability_v1")
        assert fw["framework_id"] == "fca_suitability_v1"
        assert fw["regulator"] == "FCA"
        assert isinstance(fw["elements"], list)
        assert len(fw["elements"]) == 9, "FCA framework must have exactly 9 elements"
        # Spot-check one element
        ids = {e["id"] for e in fw["elements"]}
        assert "client_objectives" in ids
        assert "risk_attitude" in ids

    def test_load_framework_invalid_id_raises_file_not_found(self):
        """load_framework("nonexistent_framework") → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("nonexistent_framework_abc123")

    def test_load_framework_invalid_file_path_raises_file_not_found(self):
        """load_framework('/no/such/file.yaml') → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("/no/such/file.yaml")


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _validate_framework
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateFramework:

    def test_validate_missing_elements_field_raises_value_error(self):
        """Framework missing 'elements' → ValueError."""
        bad_fw = {
            "framework_id": "x",
            "name": "X",
            "version": "1.0",
            "regulator": "X",
            # 'elements' deliberately absent
        }
        with pytest.raises(ValueError, match="elements"):
            _validate_framework(bad_fw)

    def test_validate_missing_framework_id_raises_value_error(self):
        """Framework missing 'framework_id' → ValueError."""
        bad_fw = {
            "name": "X",
            "version": "1.0",
            "regulator": "X",
            "elements": [
                {"id": "a", "description": "A", "required": True, "severity": "high"}
            ],
        }
        with pytest.raises(ValueError, match="framework_id"):
            _validate_framework(bad_fw)

    def test_validate_empty_elements_list_raises_value_error(self):
        """Framework with empty elements list → ValueError."""
        bad_fw = {**_MINIMAL_VALID_FRAMEWORK, "elements": []}
        with pytest.raises(ValueError):
            _validate_framework(bad_fw)

    def test_validate_element_missing_severity_raises_value_error(self):
        """Element missing 'severity' field → ValueError."""
        bad_fw = {
            **_MINIMAL_VALID_FRAMEWORK,
            "elements": [
                {"id": "a", "description": "A", "required": True}  # no severity
            ],
        }
        with pytest.raises(ValueError, match="severity"):
            _validate_framework(bad_fw)

    def test_validate_element_invalid_severity_raises_value_error(self):
        """Element with invalid severity string → ValueError."""
        bad_fw = {
            **_MINIMAL_VALID_FRAMEWORK,
            "elements": [
                {"id": "a", "description": "A", "required": True, "severity": "extreme"}
            ],
        }
        with pytest.raises(ValueError, match="extreme"):
            _validate_framework(bad_fw)

    def test_validate_valid_framework_does_not_raise(self):
        """Well-formed framework → no exception."""
        _validate_framework(_MINIMAL_VALID_FRAMEWORK)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _parse_element_response
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseElementResponse:

    def _ev(self, verbose: bool = False) -> NoteEvaluator:
        return _make_evaluator(verbose=verbose)

    def test_well_formed_response_returns_correct_gap_item(self):
        """Well-formed LLM response → correctly populated GapItem."""
        ev = self._ev()
        response = (
            "STATUS: present\n"
            "SCORE: 0.85\n"
            "EVIDENCE: Client clearly states retirement goal in 10 years\n"
            "NOTES: Fully documented"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)

        assert item.element_id == "elem_a"
        assert item.status == "present"
        assert item.score == pytest.approx(0.85, abs=1e-3)
        assert "retirement goal" in item.evidence
        assert item.severity == "critical"
        assert item.required is True
        assert item.notes is None  # verbose=False → notes suppressed

    def test_well_formed_verbose_response_includes_notes(self):
        """When verbose=True, notes field is populated from LLM NOTES line."""
        ev = self._ev(verbose=True)
        response = (
            "STATUS: partial\n"
            "SCORE: 0.4\n"
            "EVIDENCE: Some mention of goals\n"
            "NOTES: Insufficient detail provided"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)

        assert item.notes == "Insufficient detail provided"

    def test_missing_score_field_no_crash_returns_sensible_default(self):
        """Missing SCORE field → no exception; score falls back to a valid float."""
        ev = self._ev()
        response = (
            "STATUS: missing\n"
            "EVIDENCE: None found\n"
            "NOTES: Element absent"
        )
        # Must not raise
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)

        assert item.status == "missing"
        assert isinstance(item.score, float)
        assert 0.0 <= item.score <= 1.0

    def test_status_missing_but_score_high_score_corrected_to_zero(self):
        """STATUS=missing with SCORE=0.9 → score corrected to 0.0 (consistency fix)."""
        ev = self._ev()
        response = (
            "STATUS: missing\n"
            "SCORE: 0.9\n"
            "EVIDENCE: None found\n"
            "NOTES: LLM was inconsistent"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)

        assert item.status == "missing"
        assert item.score == pytest.approx(0.0), (
            "Score must be corrected to 0.0 when STATUS=missing and raw score > 0.2"
        )

    def test_status_partial_preserves_score(self):
        """STATUS=partial with mid-range score → score unchanged."""
        ev = self._ev()
        response = (
            "STATUS: partial\n"
            "SCORE: 0.45\n"
            "EVIDENCE: Brief mention only\n"
            "NOTES: Needs more detail"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)

        assert item.status == "partial"
        assert item.score == pytest.approx(0.45, abs=1e-3)

    def test_evidence_none_found_normalized_to_empty_string(self):
        """'None found' evidence → stored as empty string."""
        ev = self._ev()
        response = (
            "STATUS: missing\n"
            "SCORE: 0.0\n"
            "EVIDENCE: None found\n"
            "NOTES: Absent"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.evidence == ""

    def test_garbled_response_no_crash(self):
        """Completely garbled LLM response → GapItem with safe defaults, no exception."""
        ev = self._ev()
        response = "I cannot assess this. The note is too vague."
        # Must not raise
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)

        assert item.element_id == "elem_a"
        assert item.status == "missing"  # default when STATUS not found
        assert isinstance(item.score, float)

    def test_empty_response_no_crash(self):
        """Empty string response → safe GapItem defaults, no exception."""
        ev = self._ev()
        item = ev._parse_element_response("", _ELEMENT_CRITICAL)

        assert item.element_id == "elem_a"
        assert item.status == "missing"
        assert isinstance(item.score, float)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _compute_overall_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeOverallScore:

    def _ev(self) -> NoteEvaluator:
        return _make_evaluator()

    def test_required_elements_weighted_twice(self):
        """
        Required elements are weighted 2×, optional 1×.
        Two items: required score=0.8 (w=2), optional score=0.4 (w=1).
        Expected = (0.8×2 + 0.4×1) / (2+1) = 2.0/3.0 ≈ 0.6667
        """
        ev = self._ev()
        items = [
            GapItem("a", "present", 0.8, "evidence", "critical", required=True),
            GapItem("b", "partial", 0.4, "evidence", "medium", required=False),
        ]
        score = ev._compute_overall_score(items)
        assert score == pytest.approx(round(2.0 / 3.0, 4), abs=1e-4)

    def test_all_required_equal_weighting(self):
        """All required → weighted mean equals simple mean."""
        ev = self._ev()
        items = [
            GapItem("a", "present", 1.0, "", "critical", required=True),
            GapItem("b", "present", 0.0, "", "high", required=True),
        ]
        score = ev._compute_overall_score(items)
        assert score == pytest.approx(0.5, abs=1e-4)

    def test_empty_items_returns_zero(self):
        """Empty item list → 0.0."""
        ev = self._ev()
        assert ev._compute_overall_score([]) == 0.0

    def test_single_required_item_score_returned(self):
        """Single required item → its score returned."""
        ev = self._ev()
        items = [GapItem("a", "present", 0.75, "", "high", required=True)]
        assert ev._compute_overall_score(items) == pytest.approx(0.75, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _determine_pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminePass:

    def _ev(self, **policy_kwargs) -> NoteEvaluator:
        policy = PassPolicy(**policy_kwargs) if policy_kwargs else PassPolicy()
        return _make_evaluator(pass_policy=policy)

    def test_critical_required_missing_returns_false(self):
        """Required critical element missing → fail (block_on_critical_missing=True)."""
        ev = self._ev()
        items = [GapItem("a", "missing", 0.0, "", "critical", required=True)]
        assert ev._determine_pass(items) is False

    def test_high_required_missing_returns_false(self):
        """Required high element missing → fail (block_on_high_missing=True)."""
        ev = self._ev()
        items = [GapItem("b", "missing", 0.0, "", "high", required=True)]
        assert ev._determine_pass(items) is False

    def test_only_optional_critical_missing_returns_true(self):
        """Optional critical element missing → PASS (optional never blocks)."""
        ev = self._ev()
        items = [GapItem("c", "missing", 0.0, "", "critical", required=False)]
        assert ev._determine_pass(items) is True

    def test_only_optional_high_missing_returns_true(self):
        """Optional high element missing → PASS."""
        ev = self._ev()
        items = [GapItem("d", "missing", 0.0, "", "high", required=False)]
        assert ev._determine_pass(items) is True

    def test_critical_partial_below_threshold_returns_false(self):
        """Required critical element partial with score < 0.5 → fail."""
        ev = self._ev()
        items = [GapItem("a", "partial", 0.3, "", "critical", required=True)]
        assert ev._determine_pass(items) is False

    def test_critical_partial_at_threshold_returns_true(self):
        """Required critical element partial with score = 0.5 → pass (not below threshold)."""
        ev = self._ev()
        items = [GapItem("a", "partial", 0.5, "", "critical", required=True)]
        assert ev._determine_pass(items) is True

    def test_all_present_returns_true(self):
        """All required elements present → pass."""
        ev = self._ev()
        items = [
            GapItem("a", "present", 0.9, "", "critical", required=True),
            GapItem("b", "present", 0.8, "", "high", required=True),
        ]
        assert ev._determine_pass(items) is True

    def test_critical_disabled_policy_passes_despite_missing(self):
        """block_on_critical_missing=False → critical missing does not block."""
        ev = self._ev(block_on_critical_missing=False, block_on_high_missing=False)
        items = [GapItem("a", "missing", 0.0, "", "critical", required=True)]
        assert ev._determine_pass(items) is True

    def test_medium_required_missing_does_not_block(self):
        """Medium-severity required element missing → does NOT block pass."""
        ev = self._ev()
        items = [GapItem("m", "missing", 0.0, "", "medium", required=True)]
        assert ev._determine_pass(items) is True


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — GapReportStats
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapReportStats:

    def _ev(self) -> NoteEvaluator:
        return _make_evaluator()

    def _make_items(self) -> list[GapItem]:
        return [
            GapItem("a", "present", 0.9, "ev", "critical", required=True),   # critical, present
            GapItem("b", "partial", 0.4, "ev", "high",     required=True),   # high, partial
            GapItem("c", "missing", 0.0, "",   "medium",   required=False),  # medium, missing, optional
            GapItem("d", "missing", 0.0, "",   "low",      required=True),   # low, missing, required
            GapItem("e", "present", 0.8, "ev", "critical", required=False),  # critical, present, optional
        ]

    def test_total_elements(self):
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        assert stats.total_elements == 5

    def test_required_elements_count(self):
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # a, b, d are required; c, e are optional
        assert stats.required_elements == 3

    def test_present_count(self):
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # a, e are present
        assert stats.present_count == 2

    def test_partial_count(self):
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # b is partial
        assert stats.partial_count == 1

    def test_missing_count(self):
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # c, d are missing
        assert stats.missing_count == 2

    def test_critical_gaps(self):
        """Critical gaps = critical elements that are NOT present."""
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # a=critical+present (not a gap), e=critical+present (not a gap) → 0 critical gaps
        assert stats.critical_gaps == 0

    def test_high_gaps(self):
        """High gaps = high elements that are NOT present."""
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # b=high+partial → 1 high gap
        assert stats.high_gaps == 1

    def test_medium_gaps(self):
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # c=medium+missing → 1 medium gap
        assert stats.medium_gaps == 1

    def test_low_gaps(self):
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # d=low+missing → 1 low gap
        assert stats.low_gaps == 1

    def test_required_missing_count(self):
        """required_missing_count = required elements that are missing OR partial."""
        ev = self._ev()
        stats = ev._compute_stats(self._make_items())
        # b=required+partial, d=required+missing → 2
        assert stats.required_missing_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — PassPolicy defaults
# ═══════════════════════════════════════════════════════════════════════════════

class TestPassPolicyDefaults:

    def test_default_block_on_critical_missing_is_true(self):
        p = PassPolicy()
        assert p.block_on_critical_missing is True

    def test_default_block_on_critical_partial_is_true(self):
        p = PassPolicy()
        assert p.block_on_critical_partial is True

    def test_default_block_on_high_missing_is_true(self):
        p = PassPolicy()
        assert p.block_on_high_missing is True

    def test_default_critical_partial_threshold_is_0_5(self):
        p = PassPolicy()
        assert p.critical_partial_threshold == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — full evaluate_note() pipeline with mock LLM
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateNoteIntegration:
    """
    Exercises the full evaluate_note() code path using the built-in FCA framework
    (9 elements). The LLM is mocked so no real API calls are made.
    """

    _FAKE_NOTE = (
        "Client: Jane Smith. Objective: retirement in 15 years. "
        "Risk profile: balanced (score 5/10). "
        "Capacity for loss: client can absorb losses up to 20% without affecting living standards. "
        "Financial situation: income £60k, savings £50k ISA, mortgage outstanding £200k. "
        "Knowledge: client has held ISAs for 8 years, familiar with equity funds. "
        "Recommendation: Global Equity Fund selected because it matches the client's balanced "
        "risk profile and 15-year horizon. Charges: OCF 0.45%, adviser fee 0.5% p.a. "
        "Alternatives considered: bond fund and cash ISA rejected as insufficient growth potential. "
        "Client confirmed receipt of the suitability report."
    )

    # One LLM response per FCA element (9 total) — all healthy
    _ELEMENT_RESPONSES = [
        "STATUS: present\nSCORE: 0.9\nEVIDENCE: retirement in 15 years\nNOTES: Clear",
        "STATUS: present\nSCORE: 0.85\nEVIDENCE: balanced (score 5/10)\nNOTES: Clear",
        "STATUS: present\nSCORE: 0.8\nEVIDENCE: absorb losses up to 20%\nNOTES: Good",
        "STATUS: present\nSCORE: 0.75\nEVIDENCE: income £60k, savings £50k\nNOTES: Present",
        "STATUS: present\nSCORE: 0.7\nEVIDENCE: held ISAs for 8 years\nNOTES: Adequate",
        "STATUS: present\nSCORE: 0.9\nEVIDENCE: matches balanced profile\nNOTES: Linked",
        "STATUS: present\nSCORE: 0.8\nEVIDENCE: OCF 0.45% adviser 0.5%\nNOTES: Disclosed",
        "STATUS: present\nSCORE: 0.6\nEVIDENCE: bond fund rejected\nNOTES: Mentioned",
        "STATUS: present\nSCORE: 0.7\nEVIDENCE: client confirmed receipt\nNOTES: OK",
    ]

    _SUMMARY_RESPONSE = (
        "The note meets all 9 FCA suitability requirements. "
        "Client objectives, risk, capacity for loss, financials, knowledge, "
        "rationale, charges, alternatives, and confirmation are all documented."
    )

    def _mock_llm_side_effects(self):
        """Return list: one response per element + one summary response."""
        return self._ELEMENT_RESPONSES + [self._SUMMARY_RESPONSE]

    def test_evaluate_note_returns_gap_report(self):
        """evaluate_note() returns a GapReport instance."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert isinstance(report, GapReport)

    def test_evaluate_note_correct_framework_id(self):
        """GapReport.framework_id matches the loaded framework."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert report.framework_id == "fca_suitability_v1"

    def test_evaluate_note_items_count(self):
        """GapReport has one GapItem per framework element (9 for FCA)."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert len(report.items) == 9

    def test_evaluate_note_all_gap_items_are_gap_item_instances(self):
        """Every item in GapReport.items is a GapItem dataclass."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        for item in report.items:
            assert isinstance(item, GapItem), f"Expected GapItem, got {type(item)}"

    def test_evaluate_note_passed_is_true_when_all_present(self):
        """All elements present → passed=True."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert report.passed is True

    def test_evaluate_note_overall_score_is_float_between_0_and_1(self):
        """overall_score is a float in [0, 1]."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert isinstance(report.overall_score, float)
        assert 0.0 <= report.overall_score <= 1.0

    def test_evaluate_note_stats_are_gap_report_stats(self):
        """GapReport.stats is a GapReportStats dataclass."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert isinstance(report.stats, GapReportStats)

    def test_evaluate_note_stats_total_equals_items(self):
        """stats.total_elements == len(items)."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert report.stats.total_elements == len(report.items)

    def test_evaluate_note_summary_is_non_empty_string(self):
        """GapReport.summary is a non-empty string."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert isinstance(report.summary, str)
        assert len(report.summary.strip()) > 0

    def test_evaluate_note_pii_masked_false_by_default(self):
        """pii_masked=False when mask_pii not requested."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1", mask_pii=False)
        assert report.pii_masked is False

    def test_evaluate_note_with_critical_missing_fails(self):
        """If a required critical element is missing → passed=False."""
        # Override first element response (client_objectives) to missing
        ev = _make_evaluator()
        responses = list(self._ELEMENT_RESPONSES)
        responses[0] = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent"
        ev.llm.generate.side_effect = responses + [self._SUMMARY_RESPONSE]

        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        # client_objectives is critical+required → must fail
        assert report.passed is False

    def test_evaluate_note_metadata_passed_through(self):
        """metadata dict is preserved in GapReport."""
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._mock_llm_side_effects()
        meta = {"note_id": "N-001", "adviser": "Jane Doe"}
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1", metadata=meta)
        assert report.metadata["note_id"] == "N-001"
        assert report.metadata["adviser"] == "Jane Doe"

    def test_evaluate_note_via_top_level_function(self):
        """evaluate_note() public entry point works end-to-end."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = self._mock_llm_side_effects()

        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=self._FAKE_NOTE,
                framework="fca_suitability_v1",
            )

        assert isinstance(report, GapReport)
        assert report.framework_id == "fca_suitability_v1"
        assert len(report.items) == 9


# ═══════════════════════════════════════════════════════════════════════════════
# CODE REVIEW ASSERTIONS (import-level checks)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeReviewChecks:
    """
    Lightweight checks that verify structural / design constraints
    without needing to run full LLM calls.
    """

    def test_note_evaluator_extends_base_calculator(self):
        """NoteEvaluator must subclass BaseCalculator."""
        from assert_llm_tools.metrics.base import BaseCalculator
        assert issubclass(NoteEvaluator, BaseCalculator)

    def test_evaluate_note_importable_from_top_level(self):
        """evaluate_note is importable from assert_llm_tools package root."""
        import assert_llm_tools
        assert callable(assert_llm_tools.evaluate_note)

    def test_gap_report_importable_from_top_level(self):
        import assert_llm_tools
        assert assert_llm_tools.GapReport is GapReport

    def test_gap_item_importable_from_top_level(self):
        import assert_llm_tools
        assert assert_llm_tools.GapItem is GapItem

    def test_gap_report_stats_importable_from_top_level(self):
        import assert_llm_tools
        assert assert_llm_tools.GapReportStats is GapReportStats

    def test_pass_policy_exported_from_top_level(self):
        """
        PassPolicy is exported from the top-level package so callers can customise
        pass thresholds without reaching into internals.

        """
        import assert_llm_tools
        from assert_llm_tools import PassPolicy as TopLevelPassPolicy

        assert TopLevelPassPolicy is PassPolicy, (
            "PassPolicy exported from top-level should be the same class as metrics.note.models.PassPolicy"
        )

    def test_detect_and_mask_pii_not_reimplemented(self):
        """
        evaluate_note.py must import detect_and_mask_pii from utils, not redefine it.

        NOTE: assert_llm_tools.metrics.note.__init__ re-exports the `evaluate_note`
        *function*, which shadows the submodule name when doing a dotted import.
        We therefore fetch the real module object via sys.modules.
        """
        # metrics.note.__init__ re-exports evaluate_note (function), shadowing
        # the submodule — we must go to sys.modules for the real module object.
        ev_mod = sys.modules.get("assert_llm_tools.metrics.note.evaluate_note")
        assert ev_mod is not None and isinstance(ev_mod, types.ModuleType), (
            "evaluate_note module not found in sys.modules — was it imported?"
        )

        import assert_llm_tools.utils as utils_mod

        ev_fn = getattr(ev_mod, "detect_and_mask_pii", None)
        utils_fn = getattr(utils_mod, "detect_and_mask_pii", None)

        assert ev_fn is not None, "detect_and_mask_pii not imported into evaluate_note module"
        assert utils_fn is not None, "detect_and_mask_pii not found in utils module"
        assert ev_fn is utils_fn, (
            "detect_and_mask_pii in evaluate_note.py is NOT the same object as utils.py — "
            "it may have been reimplemented rather than imported!"
        )

    def test_no_direct_bedrock_or_openai_instantiation_in_note_evaluator(self):
        """
        NoteEvaluator source must not directly instantiate BedrockLLM or OpenAILLM.
        All LLM access should go through self.llm (inherited from BaseCalculator).
        """
        import inspect
        import assert_llm_tools.metrics.note.evaluate_note as ev_mod

        source = inspect.getsource(ev_mod)
        # These constructor calls must not appear in note/evaluate_note.py
        assert "BedrockLLM(" not in source, "NoteEvaluator directly instantiates BedrockLLM"
        assert "OpenAILLM(" not in source, "NoteEvaluator directly instantiates OpenAILLM"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — PassPolicy: configurable thresholds (END-51)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPassPolicyConfigurableThresholds:
    """
    Verify that all configurable threshold fields exist on PassPolicy with
    the correct defaults, and that custom values are accepted.
    """

    def test_required_pass_threshold_default(self):
        """required_pass_threshold defaults to 0.6."""
        p = PassPolicy()
        assert p.required_pass_threshold == pytest.approx(0.6)

    def test_score_correction_missing_cutoff_default(self):
        """score_correction_missing_cutoff defaults to 0.2."""
        p = PassPolicy()
        assert p.score_correction_missing_cutoff == pytest.approx(0.2)

    def test_score_correction_present_min_default(self):
        """score_correction_present_min defaults to 0.5."""
        p = PassPolicy()
        assert p.score_correction_present_min == pytest.approx(0.5)

    def test_score_correction_present_floor_default(self):
        """score_correction_present_floor defaults to 0.7."""
        p = PassPolicy()
        assert p.score_correction_present_floor == pytest.approx(0.7)

    def test_custom_required_pass_threshold_accepted(self):
        """PassPolicy accepts a custom required_pass_threshold."""
        p = PassPolicy(required_pass_threshold=0.75)
        assert p.required_pass_threshold == pytest.approx(0.75)

    def test_custom_score_correction_missing_cutoff_accepted(self):
        """PassPolicy accepts a custom score_correction_missing_cutoff."""
        p = PassPolicy(score_correction_missing_cutoff=0.1)
        assert p.score_correction_missing_cutoff == pytest.approx(0.1)

    def test_custom_score_correction_present_min_accepted(self):
        p = PassPolicy(score_correction_present_min=0.4)
        assert p.score_correction_present_min == pytest.approx(0.4)

    def test_custom_score_correction_present_floor_accepted(self):
        p = PassPolicy(score_correction_present_floor=0.8)
        assert p.score_correction_present_floor == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — _determine_pass with required_pass_threshold (END-51)
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminePassRequiredThreshold:
    """
    Verify that required_pass_threshold is applied to HIGH, MEDIUM, and LOW
    required elements when they are partially evidenced, and that changing the
    threshold alters outcomes as expected.
    """

    def _ev(self, **policy_kwargs) -> NoteEvaluator:
        policy = PassPolicy(**policy_kwargs) if policy_kwargs else PassPolicy()
        return _make_evaluator(pass_policy=policy)

    # ── HIGH partial ───────────────────────────────────────────────────────────

    def test_high_partial_below_default_threshold_returns_false(self):
        """
        Required HIGH element, status=partial, score=0.5 < default threshold 0.6
        → fail.
        """
        ev = self._ev()  # required_pass_threshold=0.6 by default
        items = [GapItem("b", "partial", 0.5, "some evidence", "high", required=True)]
        assert ev._determine_pass(items) is False

    def test_high_partial_at_default_threshold_returns_true(self):
        """
        Required HIGH element, status=partial, score=0.6 == threshold 0.6
        → pass (not strictly below threshold).
        """
        ev = self._ev()
        items = [GapItem("b", "partial", 0.6, "some evidence", "high", required=True)]
        assert ev._determine_pass(items) is True

    def test_high_partial_above_default_threshold_returns_true(self):
        """
        Required HIGH element, status=partial, score=0.8 > threshold 0.6
        → pass.
        """
        ev = self._ev()
        items = [GapItem("b", "partial", 0.8, "good evidence", "high", required=True)]
        assert ev._determine_pass(items) is True

    def test_custom_required_pass_threshold_high_partial_pass(self):
        """
        Custom threshold=0.3; HIGH+partial+score=0.4 → pass (0.4 >= 0.3).
        """
        ev = self._ev(required_pass_threshold=0.3)
        items = [GapItem("b", "partial", 0.4, "some evidence", "high", required=True)]
        assert ev._determine_pass(items) is True

    def test_custom_required_pass_threshold_high_partial_fail(self):
        """
        Custom threshold=0.9; HIGH+partial+score=0.7 → fail (0.7 < 0.9).
        """
        ev = self._ev(required_pass_threshold=0.9)
        items = [GapItem("b", "partial", 0.7, "some evidence", "high", required=True)]
        assert ev._determine_pass(items) is False

    def test_high_missing_still_fails_regardless_of_threshold(self):
        """
        HIGH+missing is blocked by block_on_high_missing, independent of
        required_pass_threshold.
        """
        ev = self._ev(required_pass_threshold=0.0)  # threshold effectively disabled
        items = [GapItem("b", "missing", 0.0, "", "high", required=True)]
        assert ev._determine_pass(items) is False

    def test_optional_high_partial_below_threshold_still_passes(self):
        """Optional elements are never blocked, even with low score."""
        ev = self._ev()
        items = [GapItem("b", "partial", 0.1, "tiny evidence", "high", required=False)]
        assert ev._determine_pass(items) is True

    # ── MEDIUM partial ─────────────────────────────────────────────────────────

    def test_medium_partial_below_threshold_returns_false(self):
        """
        Required MEDIUM element, status=partial, score=0.4 < threshold 0.6
        → fail.
        """
        ev = self._ev()
        items = [GapItem("m", "partial", 0.4, "vague mention", "medium", required=True)]
        assert ev._determine_pass(items) is False

    def test_medium_partial_above_threshold_returns_true(self):
        """Required MEDIUM element, partial with score above threshold → pass."""
        ev = self._ev()
        items = [GapItem("m", "partial", 0.7, "decent mention", "medium", required=True)]
        assert ev._determine_pass(items) is True

    def test_medium_missing_still_does_not_block(self):
        """
        Medium+missing never blocks (existing behaviour preserved).
        required_pass_threshold does NOT apply to missing status.
        """
        ev = self._ev()
        items = [GapItem("m", "missing", 0.0, "", "medium", required=True)]
        assert ev._determine_pass(items) is True

    # ── LOW partial ────────────────────────────────────────────────────────────

    def test_low_partial_below_threshold_returns_false(self):
        """Required LOW element, partial with score below threshold → fail."""
        ev = self._ev()
        items = [GapItem("l", "partial", 0.3, "hint", "low", required=True)]
        assert ev._determine_pass(items) is False

    def test_low_partial_above_threshold_returns_true(self):
        """Required LOW element, partial with score above threshold → pass."""
        ev = self._ev()
        items = [GapItem("l", "partial", 0.65, "decent", "low", required=True)]
        assert ev._determine_pass(items) is True

    # ── Critical is unaffected by required_pass_threshold ─────────────────────

    def test_critical_partial_uses_critical_partial_threshold_not_required(self):
        """
        CRITICAL elements use critical_partial_threshold, NOT required_pass_threshold.
        score=0.55 is below required_pass_threshold=0.6 but above
        critical_partial_threshold=0.5 → should PASS.
        """
        ev = self._ev(
            critical_partial_threshold=0.5,
            required_pass_threshold=0.6,
        )
        items = [GapItem("a", "partial", 0.55, "evidence", "critical", required=True)]
        assert ev._determine_pass(items) is True

    # ── Defaults reproduce all existing _determine_pass behaviour ──────────────

    def test_defaults_reproduce_critical_missing_fails(self):
        ev = self._ev()
        items = [GapItem("a", "missing", 0.0, "", "critical", required=True)]
        assert ev._determine_pass(items) is False

    def test_defaults_reproduce_high_missing_fails(self):
        ev = self._ev()
        items = [GapItem("b", "missing", 0.0, "", "high", required=True)]
        assert ev._determine_pass(items) is False

    def test_defaults_reproduce_medium_missing_passes(self):
        ev = self._ev()
        items = [GapItem("m", "missing", 0.0, "", "medium", required=True)]
        assert ev._determine_pass(items) is True

    def test_defaults_reproduce_optional_critical_missing_passes(self):
        ev = self._ev()
        items = [GapItem("c", "missing", 0.0, "", "critical", required=False)]
        assert ev._determine_pass(items) is True

    def test_defaults_reproduce_critical_partial_below_05_fails(self):
        ev = self._ev()
        items = [GapItem("a", "partial", 0.3, "", "critical", required=True)]
        assert ev._determine_pass(items) is False

    def test_defaults_reproduce_critical_partial_at_05_passes(self):
        ev = self._ev()
        items = [GapItem("a", "partial", 0.5, "", "critical", required=True)]
        assert ev._determine_pass(items) is True

    def test_defaults_reproduce_all_present_passes(self):
        ev = self._ev()
        items = [
            GapItem("a", "present", 0.9, "", "critical", required=True),
            GapItem("b", "present", 0.8, "", "high", required=True),
        ]
        assert ev._determine_pass(items) is True


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — score correction thresholds configurable (END-51)
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreCorrectionThresholds:
    """
    Verify that score correction in _parse_element_response() uses the
    configurable PassPolicy thresholds, and that defaults reproduce the
    original hard-coded behaviour.
    """

    def _ev(self, **policy_kwargs) -> NoteEvaluator:
        policy = PassPolicy(**policy_kwargs) if policy_kwargs else PassPolicy()
        return _make_evaluator(pass_policy=policy)

    def test_default_missing_score_corrected_above_0_2(self):
        """STATUS=missing, score=0.9 → corrected to 0.0 (0.9 > default cutoff 0.2)."""
        ev = self._ev()
        response = "STATUS: missing\nSCORE: 0.9\nEVIDENCE: None found\nNOTES: Absent"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.status == "missing"
        assert item.score == pytest.approx(0.0)

    def test_default_missing_score_at_0_2_not_corrected(self):
        """STATUS=missing, score=0.2 → NOT corrected (not strictly above cutoff 0.2)."""
        ev = self._ev()
        response = "STATUS: missing\nSCORE: 0.2\nEVIDENCE: None found\nNOTES: Absent"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.status == "missing"
        assert item.score == pytest.approx(0.2)

    def test_custom_missing_cutoff_raises_correction_threshold(self):
        """
        score_correction_missing_cutoff=0.05; score=0.1 > 0.05 → corrected to 0.0.
        With default (0.2), score=0.1 would NOT be corrected (0.1 <= 0.2).
        """
        ev_default = self._ev()
        ev_custom = self._ev(score_correction_missing_cutoff=0.05)
        response = "STATUS: missing\nSCORE: 0.1\nEVIDENCE: None found\nNOTES: Absent"
        item_default = ev_default._parse_element_response(response, _ELEMENT_CRITICAL)
        item_custom = ev_custom._parse_element_response(response, _ELEMENT_CRITICAL)
        # With default cutoff 0.2: 0.1 <= 0.2 → score preserved as 0.1
        assert item_default.score == pytest.approx(0.1)
        # With custom cutoff 0.05: 0.1 > 0.05 → score corrected to 0.0
        assert item_custom.score == pytest.approx(0.0)

    def test_default_present_score_corrected_when_below_0_5(self):
        """STATUS=present, score=0.3 < default present_min 0.5 → corrected to ≥ 0.7."""
        ev = self._ev()
        response = "STATUS: present\nSCORE: 0.3\nEVIDENCE: Something\nNOTES: Present"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.status == "present"
        assert item.score >= 0.7  # plain float comparison avoids pytest.approx >= issue

    def test_default_present_score_not_corrected_at_0_5(self):
        """STATUS=present, score=0.5 == present_min 0.5 → NOT corrected (not below)."""
        ev = self._ev()
        response = "STATUS: present\nSCORE: 0.5\nEVIDENCE: Something\nNOTES: Present"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.status == "present"
        assert item.score == pytest.approx(0.5)

    def test_custom_present_floor_raises_minimum_corrected_score(self):
        """
        Custom score_correction_present_floor=0.85; score=0.3 (below min 0.5)
        → corrected to ≥ 0.85.
        """
        ev = self._ev(score_correction_present_floor=0.85)
        response = "STATUS: present\nSCORE: 0.3\nEVIDENCE: Something\nNOTES: Present"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.score >= 0.85  # plain float comparison

    def test_custom_present_min_widens_correction_window(self):
        """
        Custom score_correction_present_min=0.8; score=0.7 is now below min
        → corrected upward. With default min=0.5, score=0.7 would be kept.
        """
        ev_default = self._ev()
        response = "STATUS: present\nSCORE: 0.7\nEVIDENCE: Something\nNOTES: Present"
        item_default = ev_default._parse_element_response(response, _ELEMENT_CRITICAL)
        # Default: 0.7 >= 0.5 → not corrected → score stays 0.7
        assert item_default.score == pytest.approx(0.7)
        # Custom: 0.7 < 0.8 → corrected to max(0.7, floor=0.85)
        ev_custom = self._ev(score_correction_present_min=0.8, score_correction_present_floor=0.85)
        item_custom = ev_custom._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item_custom.score >= 0.85  # plain float comparison


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST — custom thresholds via evaluate_note() entry point (END-51)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigurableThresholdsIntegration:
    """
    Verify that custom PassPolicy thresholds can be passed through the public
    evaluate_note() / NoteEvaluator.evaluate() entry points and produce
    different outcomes from the default policy.
    """

    _FAKE_NOTE = (
        "Client: Jane Smith. Risk profile: balanced. "
        "Charges: OCF 0.45%. Alternatives: briefly mentioned."
    )

    # FCA suitability v1 element order & severities (for reference):
    #   0: client_objectives        critical required
    #   1: risk_attitude            critical required
    #   2: capacity_for_loss        critical required
    #   3: financial_situation      HIGH     required  ← we vary this one
    #   4: knowledge_and_experience HIGH     required
    #   5: recommendation_rationale critical required
    #   6: charges_and_costs        HIGH     required
    #   7: alternatives_considered  medium   optional
    #   8: client_confirmation      low      optional
    def _make_side_effects(self, fin_status: str, fin_score: float) -> list:
        """9 element responses + 1 summary; financial_situation (HIGH) is variable."""
        return [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: retirement\nNOTES: OK",           # [0] critical
            "STATUS: present\nSCORE: 0.8\nEVIDENCE: balanced\nNOTES: OK",              # [1] critical
            "STATUS: present\nSCORE: 0.8\nEVIDENCE: 20% loss ok\nNOTES: OK",          # [2] critical
            f"STATUS: {fin_status}\nSCORE: {fin_score}\nEVIDENCE: income 60k\nNOTES: Variable",  # [3] HIGH
            "STATUS: present\nSCORE: 0.7\nEVIDENCE: held ISAs\nNOTES: OK",            # [4] HIGH
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: matches profile\nNOTES: OK",      # [5] critical
            "STATUS: present\nSCORE: 0.8\nEVIDENCE: OCF 0.45%\nNOTES: OK",           # [6] HIGH
            "STATUS: present\nSCORE: 0.6\nEVIDENCE: bond rejected\nNOTES: OK",        # [7] medium opt
            "STATUS: present\nSCORE: 0.7\nEVIDENCE: client confirmed\nNOTES: OK",     # [8] low opt
            "Summary: mostly compliant with one gap.",                                  # summary
        ]

    def test_default_policy_high_partial_below_threshold_fails(self):
        """
        HIGH required element (financial_situation) partial with score=0.55 < 0.6
        → report.passed=False with default policy.
        """
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._make_side_effects("partial", 0.55)
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert report.passed is False

    def test_relaxed_threshold_high_partial_passes(self):
        """
        Same scenario with required_pass_threshold=0.4 → score=0.55 >= 0.4 → passed=True.
        """
        ev = _make_evaluator(pass_policy=PassPolicy(required_pass_threshold=0.4))
        ev.llm.generate.side_effect = self._make_side_effects("partial", 0.55)
        report = ev.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert report.passed is True

    def test_strict_threshold_high_partial_score_above_default_still_fails(self):
        """
        score=0.75 >= default 0.6 → passes by default.
        But with required_pass_threshold=0.8 → 0.75 < 0.8 → fails.
        """
        ev_default = _make_evaluator()
        ev_default.llm.generate.side_effect = self._make_side_effects("partial", 0.75)
        report_default = ev_default.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert report_default.passed is True  # 0.75 >= 0.6 → passes with default

        ev_strict = _make_evaluator(pass_policy=PassPolicy(required_pass_threshold=0.8))
        ev_strict.llm.generate.side_effect = self._make_side_effects("partial", 0.75)
        report_strict = ev_strict.evaluate(self._FAKE_NOTE, "fca_suitability_v1")
        assert report_strict.passed is False  # 0.75 < 0.8 → fails with strict

    def test_pass_policy_accessible_via_evaluate_note_function(self):
        """
        Custom PassPolicy can be passed via the top-level evaluate_note() function.
        """
        import assert_llm_tools.metrics.base as _base_mod

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = self._make_side_effects("partial", 0.55)

        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=self._FAKE_NOTE,
                framework="fca_suitability_v1",
                pass_policy=PassPolicy(required_pass_threshold=0.4),  # relaxed
            )

        assert report.passed is True  # 0.55 >= 0.4 → passes
