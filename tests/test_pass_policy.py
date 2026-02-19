"""
tests/test_pass_policy.py
==========================

Comprehensive tests for PassPolicy threshold interactions:
  - Default strict policy
  - Custom threshold values at and around boundaries
  - All four configurable fields in isolation and combination
  - block_on_critical_partial with custom critical_partial_threshold
  - block_on_high_missing toggle
  - Optional elements never block regardless of policy

These tests operate purely on _determine_pass() (no LLM calls).
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from assert_llm_tools.metrics.note.models import GapItem, PassPolicy
from assert_llm_tools.metrics.note.evaluate_note import NoteEvaluator


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ev(policy: PassPolicy | None = None) -> NoteEvaluator:
    """Return a NoteEvaluator with the given policy and a mocked LLM."""
    ev = NoteEvaluator(pass_policy=policy)
    ev.llm = MagicMock(name="mock_llm")
    return ev


def _item(
    element_id: str = "x",
    status: str = "present",
    score: float = 1.0,
    severity: str = "critical",
    required: bool = True,
) -> GapItem:
    return GapItem(
        element_id=element_id,
        status=status,
        score=score,
        evidence="",
        severity=severity,
        required=required,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Default policy — strict
# ═══════════════════════════════════════════════════════════════════════════════

class TestDefaultPolicyStrict:

    def test_all_present_passes(self):
        ev = _ev()
        items = [
            _item("a", "present", 1.0, "critical", True),
            _item("b", "present", 0.9, "high", True),
            _item("c", "present", 0.8, "medium", False),
        ]
        assert ev._determine_pass(items) is True

    def test_critical_required_missing_fails(self):
        ev = _ev()
        items = [_item("a", "missing", 0.0, "critical", True)]
        assert ev._determine_pass(items) is False

    def test_high_required_missing_fails(self):
        ev = _ev()
        items = [_item("a", "missing", 0.0, "high", True)]
        assert ev._determine_pass(items) is False

    def test_critical_partial_below_half_fails(self):
        ev = _ev()
        items = [_item("a", "partial", 0.49, "critical", True)]
        assert ev._determine_pass(items) is False

    def test_critical_partial_at_half_passes(self):
        """Score == 0.5 (the default threshold) should NOT block (strict less-than)."""
        ev = _ev()
        items = [_item("a", "partial", 0.5, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_critical_partial_above_half_passes(self):
        ev = _ev()
        items = [_item("a", "partial", 0.8, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_medium_required_missing_passes(self):
        """Medium-severity required element missing does NOT block under default policy."""
        ev = _ev()
        items = [_item("a", "missing", 0.0, "medium", True)]
        assert ev._determine_pass(items) is True

    def test_low_required_missing_passes(self):
        ev = _ev()
        items = [_item("a", "missing", 0.0, "low", True)]
        assert ev._determine_pass(items) is True

    def test_high_partial_does_not_block(self):
        """High partial with low score does NOT block — only high MISSING blocks."""
        ev = _ev()
        items = [_item("a", "partial", 0.1, "high", True)]
        assert ev._determine_pass(items) is True


# ═══════════════════════════════════════════════════════════════════════════════
# block_on_critical_missing toggle
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockOnCriticalMissing:

    def test_enabled_critical_missing_fails(self):
        ev = _ev(PassPolicy(block_on_critical_missing=True))
        items = [_item("a", "missing", 0.0, "critical", True)]
        assert ev._determine_pass(items) is False

    def test_disabled_critical_missing_passes(self):
        ev = _ev(PassPolicy(
            block_on_critical_missing=False,
            block_on_critical_partial=False,
            block_on_high_missing=False,
        ))
        items = [_item("a", "missing", 0.0, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_disabled_critical_missing_but_high_missing_still_blocks(self):
        ev = _ev(PassPolicy(
            block_on_critical_missing=False,
            block_on_high_missing=True,
        ))
        items = [
            _item("a", "missing", 0.0, "critical", True),
            _item("b", "missing", 0.0, "high", True),
        ]
        assert ev._determine_pass(items) is False


# ═══════════════════════════════════════════════════════════════════════════════
# block_on_high_missing toggle
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockOnHighMissing:

    def test_enabled_high_missing_fails(self):
        ev = _ev(PassPolicy(block_on_high_missing=True))
        items = [_item("a", "missing", 0.0, "high", True)]
        assert ev._determine_pass(items) is False

    def test_disabled_high_missing_passes(self):
        ev = _ev(PassPolicy(block_on_high_missing=False))
        items = [_item("a", "missing", 0.0, "high", True)]
        assert ev._determine_pass(items) is True

    def test_disabled_high_missing_critical_still_blocks(self):
        ev = _ev(PassPolicy(block_on_high_missing=False, block_on_critical_missing=True))
        items = [
            _item("a", "missing", 0.0, "high", True),    # high missing — disabled, should not block
            _item("b", "missing", 0.0, "critical", True), # critical missing — should block
        ]
        assert ev._determine_pass(items) is False

    def test_high_present_does_not_affect_pass(self):
        ev = _ev(PassPolicy(block_on_high_missing=True))
        items = [_item("a", "present", 1.0, "high", True)]
        assert ev._determine_pass(items) is True


# ═══════════════════════════════════════════════════════════════════════════════
# block_on_critical_partial + critical_partial_threshold combinations
# ═══════════════════════════════════════════════════════════════════════════════

class TestCriticalPartialThreshold:

    def test_default_threshold_0_5_boundary_exact(self):
        """Exact default threshold: score==0.5 should NOT block (< operator)."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.5))
        items = [_item("a", "partial", 0.5, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_default_threshold_0_5_just_below(self):
        """Just below default threshold: 0.499 should block."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.5))
        items = [_item("a", "partial", 0.499, "critical", True)]
        assert ev._determine_pass(items) is False

    def test_threshold_0_7_score_0_65_fails(self):
        """Custom threshold 0.7: score 0.65 is below → block."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.7))
        items = [_item("a", "partial", 0.65, "critical", True)]
        assert ev._determine_pass(items) is False

    def test_threshold_0_7_score_0_7_passes(self):
        """Custom threshold 0.7: score at exactly 0.7 → NOT blocked (< operator)."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.7))
        items = [_item("a", "partial", 0.7, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_threshold_0_7_score_0_75_passes(self):
        """Custom threshold 0.7: score 0.75 → pass."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.7))
        items = [_item("a", "partial", 0.75, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_threshold_0_3_score_0_25_fails(self):
        """Lenient threshold 0.3: score 0.25 → block."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.3))
        items = [_item("a", "partial", 0.25, "critical", True)]
        assert ev._determine_pass(items) is False

    def test_threshold_0_3_score_0_3_passes(self):
        """Lenient threshold 0.3: score exactly 0.3 → pass (< operator)."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.3))
        items = [_item("a", "partial", 0.3, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_block_on_critical_partial_disabled_ignores_threshold(self):
        """block_on_critical_partial=False → even score 0.0 partial does not block."""
        ev = _ev(PassPolicy(
            block_on_critical_partial=False,
            critical_partial_threshold=0.99,  # would block anything below 0.99 if enabled
        ))
        items = [_item("a", "partial", 0.01, "critical", True)]
        assert ev._determine_pass(items) is True

    def test_threshold_does_not_affect_high_partial(self):
        """critical_partial_threshold has no effect on high-severity partial elements."""
        ev = _ev(PassPolicy(
            block_on_critical_partial=True,
            block_on_high_missing=True,
            critical_partial_threshold=0.9,
        ))
        items = [
            _item("a", "partial", 0.1, "high", True),   # high partial — threshold not applicable
        ]
        # High partial (not missing) should NOT block regardless of threshold
        assert ev._determine_pass(items) is True

    def test_threshold_zero_always_passes_partial(self):
        """Threshold of 0.0 means any partial score (even 0.001) passes."""
        ev = _ev(PassPolicy(block_on_critical_partial=True, critical_partial_threshold=0.0))
        items = [_item("a", "partial", 0.001, "critical", True)]
        assert ev._determine_pass(items) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Optional elements never block
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptionalElementsNeverBlock:

    def test_optional_critical_missing_does_not_block(self):
        ev = _ev()
        items = [_item("a", "missing", 0.0, "critical", required=False)]
        assert ev._determine_pass(items) is True

    def test_optional_high_missing_does_not_block(self):
        ev = _ev()
        items = [_item("a", "missing", 0.0, "high", required=False)]
        assert ev._determine_pass(items) is True

    def test_optional_critical_partial_below_threshold_does_not_block(self):
        ev = _ev()
        items = [_item("a", "partial", 0.1, "critical", required=False)]
        assert ev._determine_pass(items) is True

    def test_mixed_optional_blocking_required_blocks(self):
        """If required element blocks, optional elements must not override."""
        ev = _ev()
        items = [
            _item("a", "missing", 0.0, "critical", required=True),   # → blocks
            _item("b", "present", 1.0, "high", required=False),       # → OK
        ]
        assert ev._determine_pass(items) is False

    def test_only_optional_elements_passes(self):
        """Report with only optional elements always passes, regardless of status."""
        ev = _ev()
        items = [
            _item("a", "missing", 0.0, "critical", required=False),
            _item("b", "partial", 0.1, "high", required=False),
            _item("c", "missing", 0.0, "low", required=False),
        ]
        assert ev._determine_pass(items) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Full lenient policy
# ═══════════════════════════════════════════════════════════════════════════════

class TestFullyLenientPolicy:

    def test_all_disabled_everything_passes(self):
        """With all block flags disabled, even worst-case items should pass."""
        ev = _ev(PassPolicy(
            block_on_critical_missing=False,
            block_on_critical_partial=False,
            block_on_high_missing=False,
        ))
        items = [
            _item("a", "missing", 0.0, "critical", True),
            _item("b", "missing", 0.0, "high", True),
            _item("c", "partial", 0.0, "critical", True),
        ]
        assert ev._determine_pass(items) is True

    def test_all_disabled_empty_list_passes(self):
        ev = _ev(PassPolicy(
            block_on_critical_missing=False,
            block_on_critical_partial=False,
            block_on_high_missing=False,
        ))
        assert ev._determine_pass([]) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Multiple blocking conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultipleBlockingConditions:

    def test_multiple_required_critical_missing_all_block(self):
        """Any of multiple missing critical elements should block."""
        ev = _ev()
        items = [
            _item("a", "missing", 0.0, "critical", True),
            _item("b", "missing", 0.0, "critical", True),
        ]
        assert ev._determine_pass(items) is False

    def test_one_present_one_missing_blocks(self):
        """One present, one critical missing → blocked by the missing one."""
        ev = _ev()
        items = [
            _item("a", "present", 1.0, "critical", True),
            _item("b", "missing", 0.0, "critical", True),
        ]
        assert ev._determine_pass(items) is False

    def test_critical_partial_and_high_missing_both_would_block(self):
        """With default policy, both blocking conditions active → fails."""
        ev = _ev()
        items = [
            _item("a", "partial", 0.3, "critical", True),  # blocks (partial < 0.5)
            _item("b", "missing", 0.0, "high", True),       # also blocks
        ]
        assert ev._determine_pass(items) is False
