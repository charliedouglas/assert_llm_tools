"""
test_smoke.py — assert-review package smoke tests (END-90)
===========================================================

Covers:
  Unit tests  : loader, validator, parser, scorer, pass-policy, stats  (no real LLM)
  Integration : full evaluate_note() pipeline with mocked LLM
  API surface : public imports, re-exports
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from assert_review import GapItem, GapReport, GapReportStats, LLMConfig, PassPolicy, evaluate_note
from assert_review.evaluate_note import NoteEvaluator
from assert_review.loader import _validate_framework, load_framework

# ── Fixtures ───────────────────────────────────────────────────────────────────

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


def _make_evaluator(verbose: bool = False, pass_policy: PassPolicy | None = None) -> NoteEvaluator:
    """Create a NoteEvaluator with a fresh MagicMock as its LLM (no real API calls)."""
    with patch("assert_core.llm.bedrock.BedrockLLM._initialize"):
        ev = NoteEvaluator(verbose=verbose, pass_policy=pass_policy)
    ev.llm = MagicMock(name="test_mock_llm")
    return ev


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API SURFACE
# ═══════════════════════════════════════════════════════════════════════════════

class TestPublicAPI:

    def test_evaluate_note_importable(self):
        assert callable(evaluate_note)

    def test_models_importable_from_top_level(self):
        from assert_review import GapReport, GapItem, GapReportStats, PassPolicy
        assert GapReport is not None
        assert GapItem is not None
        assert GapReportStats is not None
        assert PassPolicy is not None

    def test_llm_config_re_exported(self):
        from assert_review import LLMConfig
        assert LLMConfig is not None

    def test_note_evaluator_importable(self):
        from assert_review.evaluate_note import NoteEvaluator
        assert NoteEvaluator is not None

    def test_note_evaluator_extends_base_calculator(self):
        from assert_core.metrics.base import BaseCalculator
        assert issubclass(NoteEvaluator, BaseCalculator)


# ═══════════════════════════════════════════════════════════════════════════════
# LOADER
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFramework:

    def test_load_framework_with_valid_dict_returns_unchanged(self):
        fw = load_framework(_MINIMAL_VALID_FRAMEWORK)
        assert fw is _MINIMAL_VALID_FRAMEWORK
        assert fw["framework_id"] == "test_fw"

    def test_load_builtin_fca_suitability_v1(self):
        fw = load_framework("fca_suitability_v1")
        assert fw["framework_id"] == "fca_suitability_v1"
        assert fw["regulator"] == "FCA"
        assert len(fw["elements"]) == 9
        ids = {e["id"] for e in fw["elements"]}
        assert "client_objectives" in ids
        assert "risk_attitude" in ids

    def test_invalid_id_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_framework("nonexistent_framework_xyz")

    def test_invalid_file_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_framework("/no/such/file.yaml")


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMEWORK VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateFramework:

    def test_missing_elements_field_raises(self):
        bad = {"framework_id": "x", "name": "X", "version": "1.0", "regulator": "X"}
        with pytest.raises(ValueError, match="elements"):
            _validate_framework(bad)

    def test_missing_framework_id_raises(self):
        bad = {
            "name": "X", "version": "1.0", "regulator": "X",
            "elements": [{"id": "a", "description": "A", "required": True, "severity": "high"}],
        }
        with pytest.raises(ValueError, match="framework_id"):
            _validate_framework(bad)

    def test_empty_elements_list_raises(self):
        with pytest.raises(ValueError):
            _validate_framework({**_MINIMAL_VALID_FRAMEWORK, "elements": []})

    def test_element_missing_severity_raises(self):
        bad = {
            **_MINIMAL_VALID_FRAMEWORK,
            "elements": [{"id": "a", "description": "A", "required": True}],
        }
        with pytest.raises(ValueError, match="severity"):
            _validate_framework(bad)

    def test_invalid_severity_value_raises(self):
        bad = {
            **_MINIMAL_VALID_FRAMEWORK,
            "elements": [{"id": "a", "description": "A", "required": True, "severity": "extreme"}],
        }
        with pytest.raises(ValueError, match="extreme"):
            _validate_framework(bad)

    def test_valid_framework_does_not_raise(self):
        _validate_framework(_MINIMAL_VALID_FRAMEWORK)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseElementResponse:

    def test_well_formed_present_response(self):
        ev = _make_evaluator()
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
        assert item.notes is None  # verbose=False

    def test_verbose_includes_notes(self):
        ev = _make_evaluator(verbose=True)
        response = (
            "STATUS: partial\nSCORE: 0.4\n"
            "EVIDENCE: Some mention\nNOTES: Insufficient detail"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.notes == "Insufficient detail"

    def test_missing_score_no_crash(self):
        ev = _make_evaluator()
        response = "STATUS: missing\nEVIDENCE: None found\nNOTES: Absent"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.status == "missing"
        assert isinstance(item.score, float)

    def test_missing_status_high_score_corrected(self):
        ev = _make_evaluator()
        response = "STATUS: missing\nSCORE: 0.9\nEVIDENCE: None found\nNOTES: Inconsistent"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.status == "missing"
        assert item.score == pytest.approx(0.0)

    def test_evidence_none_on_missing_is_none(self):
        ev = _make_evaluator()
        response = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.evidence is None

    def test_evidence_none_found_on_partial_is_empty_string(self):
        ev = _make_evaluator()
        response = "STATUS: partial\nSCORE: 0.4\nEVIDENCE: None found\nNOTES: Incomplete"
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.status == "partial"
        assert item.evidence == ""
        assert item.evidence is not None

    def test_garbled_response_no_crash(self):
        ev = _make_evaluator()
        item = ev._parse_element_response("I cannot assess this.", _ELEMENT_CRITICAL)
        assert item.status == "missing"
        assert isinstance(item.score, float)

    def test_empty_response_no_crash(self):
        ev = _make_evaluator()
        item = ev._parse_element_response("", _ELEMENT_CRITICAL)
        assert item.status == "missing"


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeOverallScore:

    def test_required_elements_weighted_twice(self):
        ev = _make_evaluator()
        items = [
            GapItem("a", "present", 0.8, "evidence", "critical", required=True),
            GapItem("b", "partial", 0.4, "evidence", "medium", required=False),
        ]
        score = ev._compute_overall_score(items)
        assert score == pytest.approx(round(2.0 / 3.0, 4), abs=1e-4)

    def test_empty_items_returns_zero(self):
        ev = _make_evaluator()
        assert ev._compute_overall_score([]) == 0.0

    def test_single_required_item(self):
        ev = _make_evaluator()
        items = [GapItem("a", "present", 0.75, "", "high", required=True)]
        assert ev._compute_overall_score(items) == pytest.approx(0.75, abs=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# PASS POLICY
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeterminePass:

    def test_critical_required_missing_fails(self):
        ev = _make_evaluator()
        items = [GapItem("a", "missing", 0.0, None, "critical", required=True)]
        assert ev._determine_pass(items) is False

    def test_high_required_missing_fails(self):
        ev = _make_evaluator()
        items = [GapItem("b", "missing", 0.0, None, "high", required=True)]
        assert ev._determine_pass(items) is False

    def test_optional_critical_missing_passes(self):
        ev = _make_evaluator()
        items = [GapItem("c", "missing", 0.0, None, "critical", required=False)]
        assert ev._determine_pass(items) is True

    def test_critical_partial_below_threshold_fails(self):
        ev = _make_evaluator()
        items = [GapItem("a", "partial", 0.3, "", "critical", required=True)]
        assert ev._determine_pass(items) is False

    def test_critical_partial_at_threshold_passes(self):
        ev = _make_evaluator()
        items = [GapItem("a", "partial", 0.5, "", "critical", required=True)]
        assert ev._determine_pass(items) is True

    def test_all_present_passes(self):
        ev = _make_evaluator()
        items = [
            GapItem("a", "present", 0.9, "", "critical", required=True),
            GapItem("b", "present", 0.8, "", "high", required=True),
        ]
        assert ev._determine_pass(items) is True

    def test_medium_required_missing_does_not_block(self):
        ev = _make_evaluator()
        items = [GapItem("m", "missing", 0.0, None, "medium", required=True)]
        assert ev._determine_pass(items) is True


# ═══════════════════════════════════════════════════════════════════════════════
# PASS POLICY DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPassPolicyDefaults:

    def test_defaults(self):
        p = PassPolicy()
        assert p.block_on_critical_missing is True
        assert p.block_on_critical_partial is True
        assert p.block_on_high_missing is True
        assert p.critical_partial_threshold == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapReportStats:

    def _items(self):
        return [
            GapItem("a", "present", 0.9, "ev", "critical", required=True),
            GapItem("b", "partial", 0.4, "ev", "high",     required=True),
            GapItem("c", "missing", 0.0, None, "medium",   required=False),
            GapItem("d", "missing", 0.0, None, "low",      required=True),
            GapItem("e", "present", 0.8, "ev", "critical", required=False),
        ]

    def test_counts(self):
        ev = _make_evaluator()
        stats = ev._compute_stats(self._items())
        assert stats.total_elements == 5
        assert stats.required_elements == 3
        assert stats.present_count == 2
        assert stats.partial_count == 1
        assert stats.missing_count == 2

    def test_gap_counts_by_severity(self):
        ev = _make_evaluator()
        stats = ev._compute_stats(self._items())
        assert stats.critical_gaps == 0   # both critical elements present
        assert stats.high_gaps == 1       # b is partial
        assert stats.medium_gaps == 1     # c is missing
        assert stats.low_gaps == 1        # d is missing

    def test_required_missing_count(self):
        ev = _make_evaluator()
        stats = ev._compute_stats(self._items())
        assert stats.required_missing_count == 2   # b (partial) + d (missing)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERALL RATING
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverallRating:

    def _item(self, status, severity, required=True, score=None):
        s = 0.9 if status == "present" else (0.4 if status == "partial" else 0.0)
        if score is not None:
            s = score
        ev_text = "evidence" if status != "missing" else None
        return GapItem("x", status, s, ev_text, severity, required)

    def test_compliant_all_present(self):
        ev = _make_evaluator()
        items = [self._item("present", "critical"), self._item("present", "high")]
        assert ev._determine_overall_rating(items, passed=True) == "Compliant"

    def test_minor_gaps_passed_with_partial(self):
        ev = _make_evaluator()
        items = [self._item("present", "critical"), self._item("partial", "medium")]
        assert ev._determine_overall_rating(items, passed=True) == "Minor Gaps"

    def test_non_compliant_critical_required_missing(self):
        ev = _make_evaluator()
        items = [self._item("missing", "critical", required=True)]
        assert ev._determine_overall_rating(items, passed=False) == "Non-Compliant"

    def test_requires_attention_failed_high_only(self):
        ev = _make_evaluator(pass_policy=PassPolicy(block_on_critical_missing=True, block_on_high_missing=True, block_on_critical_partial=False))
        items = [
            self._item("present", "critical", required=True),
            self._item("missing", "high", required=True),
        ]
        assert ev._determine_overall_rating(items, passed=False) == "Requires Attention"


# ═══════════════════════════════════════════════════════════════════════════════
# SUGGESTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSuggestions:

    def test_suggestions_for_missing(self):
        ev = _make_evaluator()
        response = (
            "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent\n"
            "SUGGESTIONS: Document the client's retirement objective | Include target date | Note income requirements"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert len(item.suggestions) == 3

    def test_suggestions_empty_for_present(self):
        ev = _make_evaluator()
        response = (
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Documented\nNOTES: Good\nSUGGESTIONS: None"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert item.suggestions == []

    def test_suggestions_capped_at_three(self):
        ev = _make_evaluator()
        response = (
            "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent\n"
            "SUGGESTIONS: S1 | S2 | S3 | S4"
        )
        item = ev._parse_element_response(response, _ELEMENT_CRITICAL)
        assert len(item.suggestions) == 3

    def test_prompt_contains_suggestions_field(self):
        ev = _make_evaluator()
        prompt = ev._build_element_prompt("Some note text", _ELEMENT_CRITICAL)
        assert "SUGGESTIONS" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — full pipeline with mocked LLM
# ═══════════════════════════════════════════════════════════════════════════════

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


class TestEvaluateNoteIntegration:

    def _all_responses(self):
        return _ELEMENT_RESPONSES + [_SUMMARY_RESPONSE]

    def test_returns_gap_report(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert isinstance(report, GapReport)

    def test_correct_framework_id(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert report.framework_id == "fca_suitability_v1"

    def test_items_count_matches_elements(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert len(report.items) == 9

    def test_all_present_passes(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert report.passed is True

    def test_overall_score_in_range(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert 0.0 <= report.overall_score <= 1.0

    def test_stats_total_matches_items(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert report.stats.total_elements == len(report.items)

    def test_summary_is_non_empty_string(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert isinstance(report.summary, str) and report.summary.strip()

    def test_overall_rating_compliant(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert report.overall_rating == "Compliant"

    def test_critical_missing_fails_and_non_compliant(self):
        ev = _make_evaluator()
        responses = list(_ELEMENT_RESPONSES)
        responses[0] = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent"
        ev.llm.generate.side_effect = responses + [_SUMMARY_RESPONSE]
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        assert report.passed is False
        assert report.overall_rating == "Non-Compliant"

    def test_missing_element_has_null_evidence(self):
        ev = _make_evaluator()
        responses = list(_ELEMENT_RESPONSES)
        responses[0] = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent"
        ev.llm.generate.side_effect = responses + [_SUMMARY_RESPONSE]
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1")
        missing = next(it for it in report.items if it.status == "missing")
        assert missing.evidence is None

    def test_metadata_preserved(self):
        ev = _make_evaluator()
        ev.llm.generate.side_effect = self._all_responses()
        meta = {"note_id": "N-001", "adviser": "Jane Doe"}
        report = ev.evaluate(_FAKE_NOTE, "fca_suitability_v1", metadata=meta)
        assert report.metadata["note_id"] == "N-001"

    def test_top_level_evaluate_note_function(self):
        """evaluate_note() top-level entry point works end-to-end."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = self._all_responses()
        with patch("assert_core.llm.bedrock.BedrockLLM._initialize"), \
             patch("assert_core.llm.bedrock.BedrockLLM.generate", side_effect=self._all_responses()):
            with patch("assert_core.metrics.base.BedrockLLM") as mock_cls:
                mock_cls.return_value = mock_llm
                report = evaluate_note(note_text=_FAKE_NOTE, framework="fca_suitability_v1")

        assert isinstance(report, GapReport)
        assert report.framework_id == "fca_suitability_v1"
        assert len(report.items) == 9
