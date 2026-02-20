"""
Data models for compliance note evaluation.

GapItem, GapReport, GapReportStats, and PassPolicy are importable directly
from assert_review.models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# ── Type aliases ───────────────────────────────────────────────────────────────

ElementStatus = Literal["present", "partial", "missing"]
ElementSeverity = Literal["critical", "high", "medium", "low", "warning", "info"]
OverallRating = Literal["Compliant", "Minor Gaps", "Requires Attention", "Non-Compliant"]


# ── Element-level result ───────────────────────────────────────────────────────

@dataclass
class GapItem:
    """
    Evaluation result for a single framework element.

    Attributes:
        element_id:   The element's id as defined in the framework YAML.
        status:       Whether the element is present, partially present, or missing.
        score:        Confidence/quality score for the element, 0.0–1.0.
                      1.0 = fully present and well-documented.
                      0.5 = partially addressed.
                      0.0 = absent.
        evidence:     Verbatim or paraphrased excerpt from the note that supports the
                      status assessment. None when the element is missing entirely;
                      for partial elements, captures what was found and what is absent.
        severity:     Severity copied from the framework element definition —
                      reflects the compliance impact if this element is absent/partial.
        required:     Whether the element is required per the framework.
        notes:        (optional) Free-text LLM commentary on the gap or quality.
                      Populated only when verbose=True.
        suggestions:  Actionable remediation suggestions (1–3 items) for gaps.
                      Empty list when status is "present" — no action needed.
    """

    element_id: str
    status: ElementStatus
    score: float          # 0.0–1.0
    evidence: Optional[str]   # None when element is missing
    severity: ElementSeverity
    required: bool
    notes: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)


# ── Top-level report ──────────────────────────────────────────────────────────

@dataclass
class GapReport:
    """
    Full evaluation report for a compliance note against a framework.

    Attributes:
        framework_id:       ID of the framework used for evaluation.
        framework_version:  Version of the framework.
        passed:             Overall pass/fail. True only if no required critical
                            or high elements are missing/partial below threshold.
        overall_score:      Weighted mean of element scores (required elements
                            weighted 2×, optional elements 1×), 0.0–1.0.
        overall_rating:     Human-readable compliance rating derived from pass/fail
                            status and gap severity profile. One of:
                            "Compliant"          — passed, no gaps at all.
                            "Minor Gaps"         — passed, but some elements partial
                                                   or optional elements missing.
                            "Requires Attention" — failed due to high/medium gaps;
                                                   no critical blockers.
                            "Non-Compliant"      — failed due to critical required
                                                   element gaps.
        items:              List of GapItem, one per framework element.
        summary:            Human-readable summary of the evaluation produced by the LLM.
        stats:              Breakdown counts — see GapReportStats.
        metadata:           Arbitrary key/value pairs (e.g. note_id, adviser_ref).
    """

    framework_id: str
    framework_version: str
    passed: bool
    overall_score: float          # 0.0–1.0
    overall_rating: OverallRating
    items: List[GapItem]
    summary: str
    stats: "GapReportStats"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Summary statistics ─────────────────────────────────────────────────────────

@dataclass
class GapReportStats:
    """
    Summary statistics for a GapReport.

    Attributes:
        total_elements:         Total number of elements in the framework.
        required_elements:      Number of required elements.
        present_count:          Elements with status == "present".
        partial_count:          Elements with status == "partial".
        missing_count:          Elements with status == "missing".
        critical_gaps:          Missing/partial elements with severity == "critical".
        high_gaps:              Missing/partial elements with severity == "high".
        medium_gaps:            Missing/partial elements with severity == "medium".
        low_gaps:               Missing/partial elements with severity == "low".
        warning_gaps:           Missing/partial elements with severity == "warning".
        info_gaps:              Missing/partial elements with severity == "info".
        required_missing_count: Required elements that are missing or partial.
    """

    total_elements: int
    required_elements: int
    present_count: int
    partial_count: int
    missing_count: int
    critical_gaps: int
    high_gaps: int
    medium_gaps: int
    low_gaps: int
    warning_gaps: int
    info_gaps: int
    required_missing_count: int


# ── Pass policy ────────────────────────────────────────────────────────────────

@dataclass
class PassPolicy:
    """
    Configures the pass/fail threshold for GapReport.passed.

    Attributes:
        block_on_critical_missing:  Fail if any critical required element is missing.
        block_on_critical_partial:  Fail if any critical required element is partial
                                    with score below critical_partial_threshold.
        block_on_high_missing:      Fail if any high required element is missing.
        block_on_warning_missing:   Fail if any warning required element is missing.
        critical_partial_threshold: Minimum score for a critical element to not
                                    block on partial status. Default 0.5.
    """

    block_on_critical_missing: bool = True
    block_on_critical_partial: bool = True
    block_on_high_missing: bool = True
    block_on_warning_missing: bool = True
    critical_partial_threshold: float = 0.5
