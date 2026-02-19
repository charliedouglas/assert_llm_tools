"""
tests/test_evaluation_engine.py
================================

Comprehensive unit and integration tests for the evaluation engine:
  - NoteEvaluator.__init__   — defaults, custom args
  - _parse_element_response  — parsing, edge cases, score correction
  - _compute_overall_score   — weighting logic
  - _compute_stats           — aggregate counts
  - _determine_pass          — (core logic covered more deeply in test_pass_policy.py)
  - evaluate() / evaluate_note() — full pipeline integration with mocked LLM
  - custom_instruction        — propagation to prompts
  - Framework dict input      — pre-loaded dict accepted by evaluate()
  - Verbose mode              — notes field populated
  - metadata passthrough
  - Empty note text
  - Error paths
"""
from __future__ import annotations

import sys
import types

import pytest
from unittest.mock import MagicMock, call, patch

import assert_llm_tools.metrics.base as _base_mod

from assert_llm_tools.metrics.note.loader import load_framework
from assert_llm_tools.metrics.note.models import GapItem, GapReport, GapReportStats, PassPolicy
from assert_llm_tools.metrics.note.evaluate_note import NoteEvaluator, evaluate_note


# ── Shared framework / note fixtures ──────────────────────────────────────────

_FRAMEWORK_1_ELEMENT: dict = {
    "framework_id": "test_single",
    "name": "Single-Element Test Framework",
    "version": "0.1",
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

_FRAMEWORK_2_ELEMENTS: dict = {
    "framework_id": "test_two",
    "name": "Two-Element Test Framework",
    "version": "0.1",
    "regulator": "TEST",
    "elements": [
        {
            "id": "elem_required",
            "description": "Required critical element",
            "required": True,
            "severity": "critical",
        },
        {
            "id": "elem_optional",
            "description": "Optional high element",
            "required": False,
            "severity": "high",
        },
    ],
}

_FRAMEWORK_WITH_GUIDANCE: dict = {
    "framework_id": "test_guidance",
    "name": "Framework With Guidance",
    "version": "0.1",
    "regulator": "TEST",
    "elements": [
        {
            "id": "risk_profile",
            "description": "Client risk profile documented",
            "required": True,
            "severity": "critical",
            "guidance": "Look for a named questionnaire, score, and assigned risk category.",
        }
    ],
}

_NOTE_TEXT = "Client objectives: retire at 65. Risk: balanced (score 5/10). CfL: low-medium."

_ELEM_CRITICAL: dict = {
    "id": "elem_a",
    "description": "Element A",
    "required": True,
    "severity": "critical",
}

_ELEM_HIGH_OPTIONAL: dict = {
    "id": "elem_b",
    "description": "Element B (optional)",
    "required": False,
    "severity": "high",
}


def _make_ev(
    verbose: bool = False,
    pass_policy: PassPolicy | None = None,
    custom_instruction: str | None = None,
) -> NoteEvaluator:
    """Create a NoteEvaluator with a fresh per-test MagicMock LLM."""
    ev = NoteEvaluator(
        verbose=verbose,
        pass_policy=pass_policy,
        custom_instruction=custom_instruction,
    )
    ev.llm = MagicMock(name="test_mock_llm")
    return ev


# ── FCA 9-element mock responses ──────────────────────────────────────────────

_FCA_ELEMENT_RESPONSES = [
    "STATUS: present\nSCORE: 0.9\nEVIDENCE: retire at 65\nNOTES: Clear objectives",
    "STATUS: present\nSCORE: 0.85\nEVIDENCE: balanced score 5/10\nNOTES: Named instrument",
    "STATUS: present\nSCORE: 0.8\nEVIDENCE: low-medium assessed\nNOTES: CfL distinct",
    "STATUS: present\nSCORE: 0.75\nEVIDENCE: income and savings\nNOTES: Present",
    "STATUS: present\nSCORE: 0.7\nEVIDENCE: ISAs for 8 years\nNOTES: Adequate",
    "STATUS: present\nSCORE: 0.9\nEVIDENCE: matches balanced profile\nNOTES: Linked",
    "STATUS: present\nSCORE: 0.8\nEVIDENCE: OCF 0.45% adviser 0.5%\nNOTES: Disclosed",
    "STATUS: present\nSCORE: 0.6\nEVIDENCE: bond fund rejected\nNOTES: Mentioned",
    "STATUS: present\nSCORE: 0.7\nEVIDENCE: client confirmed receipt\nNOTES: OK",
]
_FCA_SUMMARY = "All 9 FCA elements are present and well-documented."

_FCA_FAKE_NOTE = (
    "Client: Jane Smith. Objective: retirement at 65. "
    "Risk: balanced (Dynamic Planner 5/10). CfL: absorb losses up to 20%. "
    "Financial situation: income £60k, savings £50k ISA. "
    "Knowledge: 8 years ISA experience. "
    "Recommendation: Global Equity Fund matches balanced profile and 15yr horizon. "
    "Charges: OCF 0.45%, adviser 0.5% p.a. "
    "Alternatives: bond fund rejected — insufficient growth. "
    "Client confirmed receipt of suitability report."
)


def _fca_side_effects() -> list[str]:
    return list(_FCA_ELEMENT_RESPONSES) + [_FCA_SUMMARY]


# ═══════════════════════════════════════════════════════════════════════════════
# NoteEvaluator.__init__ defaults
# ═══════════════════════════════════════════════════════════════════════════════

class TestNoteEvaluatorInit:

    def test_verbose_defaults_to_false(self):
        ev = _make_ev()
        assert ev.verbose is False

    def test_verbose_true_set(self):
        ev = _make_ev(verbose=True)
        assert ev.verbose is True

    def test_pass_policy_defaults_to_default_pass_policy(self):
        ev = _make_ev()
        assert isinstance(ev.pass_policy, PassPolicy)
        assert ev.pass_policy == PassPolicy()

    def test_custom_pass_policy_stored(self):
        policy = PassPolicy(block_on_high_missing=False, critical_partial_threshold=0.6)
        ev = _make_ev(pass_policy=policy)
        assert ev.pass_policy is policy

    def test_custom_instruction_none_by_default(self):
        ev = _make_ev()
        assert ev.custom_instruction is None

    def test_custom_instruction_stored(self):
        ev = _make_ev(custom_instruction="Use FCA terminology.")
        assert ev.custom_instruction == "Use FCA terminology."

    def test_is_note_evaluator_subclass(self):
        from assert_llm_tools.metrics.base import BaseCalculator
        ev = _make_ev()
        assert isinstance(ev, BaseCalculator)


# ═══════════════════════════════════════════════════════════════════════════════
# _parse_element_response — parsing & score corrections
# ═══════════════════════════════════════════════════════════════════════════════

class TestParseElementResponse:

    def test_well_formed_present_response(self):
        ev = _make_ev()
        resp = "STATUS: present\nSCORE: 0.85\nEVIDENCE: Client states retirement goal\nNOTES: Clear"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)

        assert item.element_id == "elem_a"
        assert item.status == "present"
        assert item.score == pytest.approx(0.85, abs=1e-3)
        assert "retirement" in item.evidence
        assert item.severity == "critical"
        assert item.required is True

    def test_verbose_false_notes_suppressed(self):
        ev = _make_ev(verbose=False)
        resp = "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Strong"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.notes is None

    def test_verbose_true_notes_populated(self):
        ev = _make_ev(verbose=True)
        resp = "STATUS: partial\nSCORE: 0.4\nEVIDENCE: Brief mention\nNOTES: Needs more detail"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.notes == "Needs more detail"

    def test_partial_status_parsed(self):
        ev = _make_ev()
        resp = "STATUS: partial\nSCORE: 0.4\nEVIDENCE: Some mention\nNOTES: Incomplete"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.status == "partial"
        assert item.score == pytest.approx(0.4, abs=1e-3)

    def test_missing_status_parsed(self):
        ev = _make_ev()
        resp = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.status == "missing"
        assert item.score == pytest.approx(0.0)

    def test_missing_status_high_score_corrected_to_zero(self):
        """STATUS=missing with inflated score → score corrected to 0.0."""
        ev = _make_ev()
        resp = "STATUS: missing\nSCORE: 0.9\nEVIDENCE: None\nNOTES: LLM inconsistent"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.status == "missing"
        assert item.score == pytest.approx(0.0, abs=1e-3)

    def test_evidence_none_found_normalised_to_empty(self):
        ev = _make_ev()
        resp = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.evidence == ""

    def test_evidence_none_normalised_to_empty(self):
        ev = _make_ev()
        resp = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None\nNOTES: Absent"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.evidence == ""

    def test_real_evidence_preserved(self):
        ev = _make_ev()
        resp = "STATUS: present\nSCORE: 0.9\nEVIDENCE: Client targets £28k p.a. at 67\nNOTES: Good"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert "£28k" in item.evidence

    def test_missing_score_field_returns_valid_float(self):
        ev = _make_ev()
        resp = "STATUS: missing\nEVIDENCE: Nothing\nNOTES: Absent"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert isinstance(item.score, float)
        assert 0.0 <= item.score <= 1.0

    def test_garbled_response_safe_defaults(self):
        ev = _make_ev()
        item = ev._parse_element_response("This response is completely garbled.", _ELEM_CRITICAL)
        assert item.element_id == "elem_a"
        assert item.status == "missing"
        assert isinstance(item.score, float)

    def test_empty_response_safe_defaults(self):
        ev = _make_ev()
        item = ev._parse_element_response("", _ELEM_CRITICAL)
        assert item.element_id == "elem_a"
        assert item.status == "missing"

    def test_optional_element_properties_propagated(self):
        ev = _make_ev()
        resp = "STATUS: present\nSCORE: 0.7\nEVIDENCE: mentioned briefly\nNOTES: OK"
        item = ev._parse_element_response(resp, _ELEM_HIGH_OPTIONAL)
        assert item.element_id == "elem_b"
        assert item.severity == "high"
        assert item.required is False

    def test_case_insensitive_status_present(self):
        """STATUS: PRESENT (uppercase) is parsed correctly."""
        ev = _make_ev()
        resp = "STATUS: PRESENT\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.status == "present"

    def test_status_partial_score_preserved(self):
        """STATUS=partial with mid-range score is not corrected."""
        ev = _make_ev()
        resp = "STATUS: partial\nSCORE: 0.45\nEVIDENCE: Brief mention\nNOTES: Incomplete"
        item = ev._parse_element_response(resp, _ELEM_CRITICAL)
        assert item.status == "partial"
        assert item.score == pytest.approx(0.45, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_overall_score — weighting
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeOverallScore:

    def test_empty_items_returns_zero(self):
        ev = _make_ev()
        assert ev._compute_overall_score([]) == pytest.approx(0.0)

    def test_single_required_item_score_returned(self):
        ev = _make_ev()
        items = [GapItem("a", "present", 0.75, "", "high", required=True)]
        assert ev._compute_overall_score(items) == pytest.approx(0.75, abs=1e-4)

    def test_single_optional_item_score_returned(self):
        ev = _make_ev()
        items = [GapItem("a", "present", 0.6, "", "low", required=False)]
        assert ev._compute_overall_score(items) == pytest.approx(0.6, abs=1e-4)

    def test_required_weighted_twice_optional_once(self):
        """Required 2× weight, optional 1× weight: (0.8×2 + 0.4×1) / 3 = 0.6667."""
        ev = _make_ev()
        items = [
            GapItem("a", "present", 0.8, "", "critical", required=True),
            GapItem("b", "partial", 0.4, "", "medium", required=False),
        ]
        expected = round((0.8 * 2 + 0.4 * 1) / 3.0, 4)
        assert ev._compute_overall_score(items) == pytest.approx(expected, abs=1e-4)

    def test_all_required_equal_weighting(self):
        """All required → weighted mean equals simple mean."""
        ev = _make_ev()
        items = [
            GapItem("a", "present", 1.0, "", "critical", required=True),
            GapItem("b", "missing", 0.0, "", "high", required=True),
        ]
        assert ev._compute_overall_score(items) == pytest.approx(0.5, abs=1e-4)

    def test_all_optional_equal_weighting(self):
        """All optional → same 1× weight each; mean of scores."""
        ev = _make_ev()
        items = [
            GapItem("a", "present", 0.8, "", "medium", required=False),
            GapItem("b", "present", 0.4, "", "low", required=False),
        ]
        assert ev._compute_overall_score(items) == pytest.approx(0.6, abs=1e-4)

    def test_score_rounded_to_4_decimal_places(self):
        """_compute_overall_score rounds to 4 decimal places."""
        ev = _make_ev()
        items = [
            GapItem("a", "present", 1.0, "", "critical", required=True),
            GapItem("b", "partial", 0.5, "", "high", required=True),
            GapItem("c", "missing", 0.0, "", "medium", required=True),
        ]
        score = ev._compute_overall_score(items)
        # Result should be rounded to 4 decimal places
        assert score == round(score, 4)

    def test_all_present_perfect_score(self):
        ev = _make_ev()
        items = [GapItem(f"e{i}", "present", 1.0, "", "critical", required=True) for i in range(5)]
        assert ev._compute_overall_score(items) == pytest.approx(1.0)

    def test_all_missing_zero_score(self):
        ev = _make_ev()
        items = [GapItem(f"e{i}", "missing", 0.0, "", "critical", required=True) for i in range(3)]
        assert ev._compute_overall_score(items) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# _compute_stats — aggregate counts
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeStats:

    def _items(self) -> list[GapItem]:
        return [
            GapItem("a", "present", 0.9, "ev", "critical", required=True),
            GapItem("b", "partial", 0.4, "ev", "high",     required=True),
            GapItem("c", "missing", 0.0, "",   "medium",   required=False),
            GapItem("d", "missing", 0.0, "",   "low",      required=True),
            GapItem("e", "present", 0.8, "ev", "critical", required=False),
        ]

    def test_total_elements(self):
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        assert stats.total_elements == 5

    def test_required_elements_count(self):
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        assert stats.required_elements == 3  # a, b, d

    def test_present_count(self):
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        assert stats.present_count == 2  # a, e

    def test_partial_count(self):
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        assert stats.partial_count == 1  # b

    def test_missing_count(self):
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        assert stats.missing_count == 2  # c, d

    def test_critical_gaps_count(self):
        """Critical gaps = critical elements that are NOT present."""
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        # a=critical+present (no gap), e=critical+present (no gap) → 0
        assert stats.critical_gaps == 0

    def test_high_gaps_count(self):
        """High gaps = high elements that are NOT present."""
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        # b=high+partial → 1 gap
        assert stats.high_gaps == 1

    def test_medium_gaps_count(self):
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        # c=medium+missing → 1 gap
        assert stats.medium_gaps == 1

    def test_low_gaps_count(self):
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        # d=low+missing → 1 gap
        assert stats.low_gaps == 1

    def test_required_missing_count(self):
        """required_missing_count = required elements that are missing OR partial."""
        ev = _make_ev()
        stats = ev._compute_stats(self._items())
        # b=required+partial, d=required+missing → 2
        assert stats.required_missing_count == 2

    def test_empty_items_all_zero(self):
        ev = _make_ev()
        stats = ev._compute_stats([])
        assert stats.total_elements == 0
        assert stats.required_elements == 0
        assert stats.present_count == 0
        assert stats.missing_count == 0
        assert stats.critical_gaps == 0

    def test_all_present_all_required_zero_gaps(self):
        ev = _make_ev()
        items = [GapItem(f"e{i}", "present", 1.0, "ev", "critical", required=True) for i in range(3)]
        stats = ev._compute_stats(items)
        assert stats.present_count == 3
        assert stats.missing_count == 0
        assert stats.critical_gaps == 0
        assert stats.required_missing_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate() — full pipeline (mocked LLM, 1-element framework)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluatePipeline:

    def _single_element_responses(self) -> list[str]:
        return [
            "STATUS: present\nSCORE: 0.85\nEVIDENCE: Element well documented\nNOTES: Good",
            "This note covers the element adequately.",
        ]

    def test_returns_gap_report(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert isinstance(report, GapReport)

    def test_framework_id_correct(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.framework_id == "test_single"

    def test_framework_version_correct(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.framework_version == "0.1"

    def test_items_count_matches_framework(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert len(report.items) == 1

    def test_items_are_gap_item_instances(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        for item in report.items:
            assert isinstance(item, GapItem)

    def test_stats_is_gap_report_stats(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert isinstance(report.stats, GapReportStats)

    def test_summary_is_non_empty_string(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert isinstance(report.summary, str)
        assert len(report.summary.strip()) > 0

    def test_passed_true_when_element_present(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.passed is True

    def test_passed_false_when_critical_missing(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = [
            "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Absent",
            "Summary: critical element missing.",
        ]
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.passed is False

    def test_pii_masked_false_by_default(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT, mask_pii=False)
        assert report.pii_masked is False

    def test_metadata_empty_dict_by_default(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.metadata == {}

    def test_metadata_preserved(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        meta = {"note_id": "N-42", "adviser": "P. Okonkwo"}
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT, metadata=meta)
        assert report.metadata["note_id"] == "N-42"
        assert report.metadata["adviser"] == "P. Okonkwo"

    def test_overall_score_in_range(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert 0.0 <= report.overall_score <= 1.0

    def test_stats_total_matches_items_len(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.stats.total_elements == len(report.items)

    def test_two_element_framework_two_llm_calls_plus_summary(self):
        """evaluate() makes one LLM call per element + one for summary."""
        ev = _make_ev()
        ev.llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good",
            "STATUS: partial\nSCORE: 0.5\nEVIDENCE: Brief\nNOTES: Partial",
            "Summary of two elements.",
        ]
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_2_ELEMENTS)
        assert len(report.items) == 2
        assert ev.llm.generate.call_count == 3  # 2 elements + 1 summary

    def test_framework_as_preloaded_dict(self):
        """evaluate() accepts a pre-loaded framework dict directly."""
        ev = _make_ev()
        ev.llm.generate.side_effect = self._single_element_responses()
        # Pass the dict directly (not a string ID)
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.framework_id == "test_single"

    def test_invalid_framework_string_raises_file_not_found(self):
        """Unknown string framework ID → FileNotFoundError from loader."""
        ev = _make_ev()
        with pytest.raises(FileNotFoundError):
            ev.evaluate(_NOTE_TEXT, "absolutely_nonexistent_framework_abc")

    def test_empty_note_text_no_exception(self):
        """Empty note_text must not raise; results in all-missing items."""
        ev = _make_ev()
        ev.llm.generate.side_effect = [
            "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None found\nNOTES: Empty note",
            "Empty note: all elements missing.",
        ]
        report = ev.evaluate("", _FRAMEWORK_1_ELEMENT)
        assert isinstance(report, GapReport)
        assert report.items[0].status == "missing"


# ═══════════════════════════════════════════════════════════════════════════════
# custom_instruction propagation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCustomInstructionPropagation:

    def test_custom_instruction_appears_in_llm_prompt(self):
        """custom_instruction text must appear in the LLM prompt for each element."""
        custom = "This note follows the FCA Finalised Guidance FG11/5 template."
        ev = _make_ev(custom_instruction=custom)
        ev.llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good",
            "Summary here.",
        ]
        ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)

        # Check the prompt for the element LLM call contains the custom instruction
        element_call_args = ev.llm.generate.call_args_list[0]
        prompt = element_call_args[0][0]  # first positional arg
        assert custom in prompt, f"custom_instruction not found in prompt: {prompt[:200]}"

    def test_no_custom_instruction_prompt_excludes_additional_instructions(self):
        """Without custom_instruction, 'Additional instructions' block absent from prompt."""
        ev = _make_ev(custom_instruction=None)
        ev.llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good",
            "Summary.",
        ]
        ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)

        element_call_args = ev.llm.generate.call_args_list[0]
        prompt = element_call_args[0][0]
        assert "Additional instructions" not in prompt

    def test_guidance_appears_in_prompt_when_element_has_guidance(self):
        """Element guidance text must appear in the LLM prompt."""
        ev = _make_ev()
        ev.llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Named questionnaire\nNOTES: Good",
            "Summary.",
        ]
        ev.evaluate(_NOTE_TEXT, _FRAMEWORK_WITH_GUIDANCE)

        element_call_args = ev.llm.generate.call_args_list[0]
        prompt = element_call_args[0][0]
        assert "named questionnaire" in prompt.lower() or "questionnaire" in prompt


# ═══════════════════════════════════════════════════════════════════════════════
# verbose mode
# ═══════════════════════════════════════════════════════════════════════════════

class TestVerboseMode:

    def test_verbose_false_all_notes_none(self):
        ev = _make_ev(verbose=False)
        ev.llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Detailed reasoning",
            "Summary.",
        ]
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.items[0].notes is None

    def test_verbose_true_notes_populated(self):
        ev = _make_ev(verbose=True)
        ev.llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Detailed reasoning",
            "Summary.",
        ]
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)
        assert report.items[0].notes == "Detailed reasoning"


# ═══════════════════════════════════════════════════════════════════════════════
# FCA 9-element framework full integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestFCAIntegration:

    def test_fca_returns_9_items(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert len(report.items) == 9

    def test_fca_all_items_are_gap_items(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        for item in report.items:
            assert isinstance(item, GapItem)

    def test_fca_all_present_passes(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert report.passed is True

    def test_fca_overall_score_float_in_range(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert isinstance(report.overall_score, float)
        assert 0.0 <= report.overall_score <= 1.0

    def test_fca_framework_id_correct(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert report.framework_id == "fca_suitability_v1"

    def test_fca_stats_total_equals_9(self):
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert report.stats.total_elements == 9

    def test_fca_first_element_is_client_objectives(self):
        """Element ordering in output must match framework definition order."""
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert report.items[0].element_id == "client_objectives"

    def test_fca_critical_missing_fails(self):
        """If client_objectives (critical) is missing → passed=False."""
        responses = list(_FCA_ELEMENT_RESPONSES)
        responses[0] = "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None\nNOTES: Absent"
        ev = _make_ev()
        ev.llm.generate.side_effect = responses + [_FCA_SUMMARY]
        report = ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert report.passed is False

    def test_fca_llm_called_10_times(self):
        """9 element calls + 1 summary call = 10 LLM calls for FCA framework."""
        ev = _make_ev()
        ev.llm.generate.side_effect = _fca_side_effects()
        ev.evaluate(_FCA_FAKE_NOTE, "fca_suitability_v1")
        assert ev.llm.generate.call_count == 10


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate_note() top-level function
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateNoteTopLevel:

    def test_returns_gap_report(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good",
            "Summary.",
        ]
        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=_NOTE_TEXT,
                framework=_FRAMEWORK_1_ELEMENT,
            )
        assert isinstance(report, GapReport)

    def test_correct_framework_id(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good",
            "Summary.",
        ]
        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(note_text=_NOTE_TEXT, framework=_FRAMEWORK_1_ELEMENT)
        assert report.framework_id == "test_single"

    def test_top_level_passes_metadata(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good",
            "Summary.",
        ]
        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=_NOTE_TEXT,
                framework=_FRAMEWORK_1_ELEMENT,
                metadata={"batch": "2026-02"},
            )
        assert report.metadata["batch"] == "2026-02"

    def test_top_level_passes_pass_policy(self):
        """evaluate_note() with a custom PassPolicy → policy applied."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            "STATUS: missing\nSCORE: 0.0\nEVIDENCE: None\nNOTES: Absent",
            "Summary: critical missing.",
        ]
        lenient = PassPolicy(block_on_critical_missing=False)
        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(
                note_text=_NOTE_TEXT,
                framework=_FRAMEWORK_1_ELEMENT,
                pass_policy=lenient,
            )
        # Lenient policy: critical missing should not block
        assert report.passed is True

    def test_top_level_fca_9_items(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = _fca_side_effects()
        with patch.object(_base_mod, "BedrockLLM", return_value=mock_llm):
            report = evaluate_note(note_text=_FCA_FAKE_NOTE, framework="fca_suitability_v1")
        assert len(report.items) == 9


# ═══════════════════════════════════════════════════════════════════════════════
# Summary generation failure fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestSummaryFallback:

    def test_summary_llm_failure_returns_fallback_string(self):
        """If summary LLM call raises, a fallback string is used; no exception propagates."""
        ev = _make_ev()
        element_resp = "STATUS: present\nSCORE: 0.9\nEVIDENCE: Found\nNOTES: Good"

        def side_effect(prompt, **kwargs):
            if ev.llm.generate.call_count == 1:
                return element_resp
            raise RuntimeError("LLM timeout")

        ev.llm.generate.side_effect = side_effect
        report = ev.evaluate(_NOTE_TEXT, _FRAMEWORK_1_ELEMENT)

        assert isinstance(report.summary, str)
        assert len(report.summary.strip()) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Structural / import-level checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuralChecks:

    def test_evaluate_note_importable_from_top_level(self):
        import assert_llm_tools
        assert callable(assert_llm_tools.evaluate_note)

    def test_gap_report_importable_from_top_level(self):
        import assert_llm_tools
        assert assert_llm_tools.GapReport is GapReport

    def test_pass_policy_importable_from_top_level(self):
        from assert_llm_tools import PassPolicy as TopPolicy
        assert TopPolicy is PassPolicy

    def test_gap_item_importable_from_top_level(self):
        from assert_llm_tools import GapItem as TopGapItem
        assert TopGapItem is GapItem

    def test_note_evaluator_is_base_calculator(self):
        from assert_llm_tools.metrics.base import BaseCalculator
        assert issubclass(NoteEvaluator, BaseCalculator)

    def test_detect_and_mask_pii_imported_from_utils(self):
        """evaluate_note module must use utils.detect_and_mask_pii, not reimplement it."""
        ev_mod = sys.modules.get("assert_llm_tools.metrics.note.evaluate_note")
        import assert_llm_tools.utils as utils_mod

        if ev_mod is None or not isinstance(ev_mod, types.ModuleType):
            pytest.skip("evaluate_note module not yet in sys.modules")

        ev_fn = getattr(ev_mod, "detect_and_mask_pii", None)
        utils_fn = getattr(utils_mod, "detect_and_mask_pii", None)
        assert ev_fn is not None, "detect_and_mask_pii not imported into evaluate_note"
        assert utils_fn is not None, "detect_and_mask_pii not in utils"
        assert ev_fn is utils_fn, "detect_and_mask_pii reimplemented rather than imported"

    def test_no_direct_llm_instantiation_in_evaluate_note(self):
        """NoteEvaluator must not directly instantiate BedrockLLM or OpenAILLM."""
        import inspect
        import assert_llm_tools.metrics.note.evaluate_note as ev_mod
        source = inspect.getsource(ev_mod)
        assert "BedrockLLM(" not in source
        assert "OpenAILLM(" not in source
