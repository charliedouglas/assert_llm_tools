"""
Data models for compliance note evaluation.

GapItem, GapReport, GapReportStats, and PassPolicy are importable directly
from assert_llm_tools.metrics.note.models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


# ── Type aliases ───────────────────────────────────────────────────────────────

ElementStatus = Literal["present", "partial", "missing"]
ElementSeverity = Literal["critical", "high", "medium", "low"]
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
        name:         Human-readable element name (derived from element_id when not
                      explicitly set, e.g. "client_objectives" → "Client Objectives").
    """

    element_id: str
    status: ElementStatus
    score: float          # 0.0–1.0
    evidence: Optional[str]   # None when element is missing
    severity: ElementSeverity
    required: bool
    notes: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    name: Optional[str] = None  # Human-readable label; derived from element_id if absent

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise this element result to a plain dict matching the JSON schema.

        Returns:
            {
                "id":          element_id string,
                "name":        human-readable name (or None),
                "status":      "present" | "partial" | "missing",
                "severity":    "critical" | "high" | "medium" | "low",
                "score":       float 0.0–1.0,
                "evidence":    string or null (null for missing elements),
                "suggestions": list of strings (empty for present elements),
            }
        """
        return {
            "id": self.element_id,
            "name": self.name,
            "status": self.status,
            "severity": self.severity,
            "score": self.score,
            "evidence": self.evidence,
            "suggestions": list(self.suggestions),
        }


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
        pii_masked:         True if PII masking was applied before evaluation.
        metadata:           Key/value pairs for audit trail — automatically populated
                            by the evaluator with: model, provider, evaluation_time
                            (ISO-8601 UTC), and framework_version. Callers may inject
                            additional fields (e.g. note_id, adviser_ref) which are
                            preserved alongside the system keys.
        meeting_type:       Type of meeting that generated the note (nullable).
                            Reserved for future use — pass None for now.
    """

    framework_id: str
    framework_version: str
    passed: bool
    overall_score: float          # 0.0–1.0
    overall_rating: OverallRating
    items: List[GapItem]
    summary: str
    stats: "GapReportStats"
    pii_masked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    meeting_type: Optional[str] = None

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the full report to a plain dict matching the structured
        JSON gap report schema (END-50).

        Schema
        ──────
        {
            "framework":        { "id": str, "version": str },
            "meeting_type":     str | null,
            "overall_score":    float (0.0–1.0),
            "overall_rating":   "Compliant"|"Minor Gaps"|"Requires Attention"|"Non-Compliant",
            "severity_summary": { ...counts by severity and status... },
            "elements":         [ { id, name, status, severity, score, evidence,
                                    suggestions }, ... ],
            "metadata":         { model, provider, evaluation_time, framework_version,
                                  ...any additional caller-supplied keys... },
        }

        The ``metadata`` section merges the system keys (framework_version) with
        whatever the evaluator injected at evaluation time (model, provider,
        evaluation_time) and any caller-supplied key/value pairs.  Caller-supplied
        values take precedence if they collide with system keys.

        Returns:
            dict: JSON-serialisable representation of this report.
        """
        # Build metadata: system key framework_version + everything stored in
        # self.metadata (which already contains model/provider/evaluation_time
        # when the report was created via NoteEvaluator.evaluate()).
        metadata: Dict[str, Any] = {
            "framework_version": self.framework_version,
        }
        metadata.update(self.metadata)  # caller keys take precedence

        return {
            "framework": {
                "id": self.framework_id,
                "version": self.framework_version,
            },
            "meeting_type": self.meeting_type,
            "overall_score": self.overall_score,
            "overall_rating": self.overall_rating,
            "severity_summary": self.stats.to_dict(),
            "elements": [item.to_dict() for item in self.items],
            "metadata": metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Return the report as a pretty-printed JSON string.

        Args:
            indent: JSON indentation level (default 2).

        Returns:
            str: JSON-encoded gap report.

        Example::

            report = evaluate_note(note, "fca_suitability_v1")
            print(report.to_json())
        """
        return json.dumps(self.to_dict(), indent=indent)


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
    required_missing_count: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise statistics to a plain dict for JSON output.

        Returns:
            {
                "total_elements":         int,
                "required_elements":      int,
                "present_count":          int,
                "partial_count":          int,
                "missing_count":          int,
                "critical_gaps":          int,
                "high_gaps":              int,
                "medium_gaps":            int,
                "low_gaps":               int,
                "required_missing_count": int,
            }
        """
        return {
            "total_elements": self.total_elements,
            "required_elements": self.required_elements,
            "present_count": self.present_count,
            "partial_count": self.partial_count,
            "missing_count": self.missing_count,
            "critical_gaps": self.critical_gaps,
            "high_gaps": self.high_gaps,
            "medium_gaps": self.medium_gaps,
            "low_gaps": self.low_gaps,
            "required_missing_count": self.required_missing_count,
        }


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
        critical_partial_threshold: Minimum score for a critical element to not
                                    block on partial status. Default 0.5.
    """

    block_on_critical_missing: bool = True
    block_on_critical_partial: bool = True
    block_on_high_missing: bool = True
    critical_partial_threshold: float = 0.5
