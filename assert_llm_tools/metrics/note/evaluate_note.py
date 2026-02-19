"""
NoteEvaluator and evaluate_note() — LLM-based compliance note evaluator.

Each framework element is evaluated in a focused, independent LLM call so
that prompts stay short and scores remain reliable. Results are assembled
into a structured GapReport.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Union

from ...llm.config import LLMConfig
from ...utils import detect_and_mask_pii
from ..base import BaseCalculator
from .loader import load_framework
from .models import GapItem, GapReport, GapReportStats, PassPolicy

logger = logging.getLogger(__name__)


# ── Public entry point ─────────────────────────────────────────────────────────

def evaluate_note(
    note_text: str,
    framework: Union[str, dict],
    llm_config: Optional[LLMConfig] = None,
    *,
    mask_pii: bool = False,
    verbose: bool = False,
    custom_instruction: Optional[str] = None,
    pass_policy: Optional[PassPolicy] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GapReport:
    """
    Evaluate a compliance note against a regulatory framework definition.

    Uses an LLM to assess each framework element for presence, quality,
    and supporting evidence within the note text, returning a structured
    GapReport.

    Args:
        note_text (str):
            The full text of the compliance note to evaluate.

        framework (str | dict):
            Either a path to a YAML framework file, a built-in framework_id
            string (e.g. "fca_suitability_v1"), or a pre-loaded framework dict.

        llm_config (LLMConfig, optional):
            LLM provider configuration. If None, a default Bedrock/Claude
            config is used (consistent with BaseCalculator default).

        mask_pii (bool):
            If True, apply PII detection and masking to note_text before
            passing it to the LLM. pii_masked=True is recorded in the report.
            Default: False.

        verbose (bool):
            If True, GapItem.notes will contain the raw LLM reasoning for
            each element assessment. Default: False.

        custom_instruction (str, optional):
            Additional instruction appended to every element prompt, e.g. to
            handle firm-specific note formats or terminology.

        pass_policy (PassPolicy, optional):
            Override the default pass/fail thresholds. If None, the default
            PassPolicy is used.

        metadata (dict, optional):
            Arbitrary key/value pairs attached to GapReport.metadata.

    Returns:
        GapReport: Structured evaluation result.

    Raises:
        FileNotFoundError: If framework is a string path/id that cannot be resolved.
        ValueError: If the framework YAML is missing required fields.
    """
    calculator = NoteEvaluator(
        llm_config=llm_config,
        custom_instruction=custom_instruction,
        verbose=verbose,
        pass_policy=pass_policy,
    )
    return calculator.evaluate(
        note_text=note_text,
        framework=framework,
        mask_pii=mask_pii,
        metadata=metadata or {},
    )


# ── NoteEvaluator ─────────────────────────────────────────────────────────────

class NoteEvaluator(BaseCalculator):
    """
    LLM-based evaluator for compliance notes against a regulatory framework.

    Extends BaseCalculator, reusing its LLM initialisation and helper methods.
    Each framework element is evaluated in a separate LLM call to keep prompts
    focused and scores reliable.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        custom_instruction: Optional[str] = None,
        verbose: bool = False,
        pass_policy: Optional[PassPolicy] = None,
    ) -> None:
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose
        self.pass_policy = pass_policy or PassPolicy()

    # ── Main evaluation pipeline ───────────────────────────────────────────────

    def evaluate(
        self,
        note_text: str,
        framework: Union[str, dict],
        mask_pii: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GapReport:
        """Run the full evaluation pipeline and return a GapReport."""
        # 1. Load & validate framework
        fw = load_framework(framework)

        # 2. Optionally mask PII
        pii_masked = False
        if mask_pii:
            try:
                note_text, _ = detect_and_mask_pii(note_text)
                pii_masked = True
                logger.info("PII masking applied to note text.")
            except Exception as exc:
                logger.warning("PII masking failed (%s); continuing with original text.", exc)

        # 3. Evaluate each element independently
        items: List[GapItem] = []
        for element in fw["elements"]:
            logger.debug("Evaluating element: %s", element["id"])
            item = self._evaluate_element(note_text, element)
            items.append(item)

        # 4. Generate overall summary via a single additional LLM call
        summary = self._generate_summary(note_text, fw, items)

        # 5. Compute derived statistics
        stats = self._compute_stats(items)
        overall_score = self._compute_overall_score(items)
        passed = self._determine_pass(items)

        return GapReport(
            framework_id=fw["framework_id"],
            framework_version=fw["version"],
            passed=passed,
            overall_score=overall_score,
            items=items,
            summary=summary,
            stats=stats,
            pii_masked=pii_masked,
            metadata=metadata or {},
        )

    # ── Per-element evaluation ─────────────────────────────────────────────────

    def _evaluate_element(
        self,
        note_text: str,
        element: dict,
    ) -> GapItem:
        """
        Evaluate a single framework element against the note text.

        Constructs a focused LLM prompt and parses the structured response
        into a GapItem.
        """
        prompt = self._build_element_prompt(note_text, element)
        response = self.llm.generate(prompt, max_tokens=400)
        return self._parse_element_response(response, element)

    def _build_element_prompt(
        self,
        note_text: str,
        element: dict,
    ) -> str:
        """Construct the LLM prompt for a single element assessment."""
        guidance_block = ""
        if element.get("guidance"):
            guidance_block = (
                f"\nEvaluation guidance:\n{element['guidance'].strip()}\n"
            )

        examples_block = ""
        if element.get("examples"):
            examples_lines = "\n".join(
                f'- "{ex}"' for ex in element["examples"]
            )
            examples_block = (
                f"\nEXAMPLES (phrases that would count as evidence):\n"
                f"{examples_lines}\n"
            )

        anti_patterns_block = ""
        if element.get("anti_patterns"):
            anti_patterns_lines = "\n".join(
                f'- "{ap}"' for ap in element["anti_patterns"]
            )
            anti_patterns_block = (
                f"\nANTI-PATTERNS (these do NOT constitute compliant evidence):\n"
                f"{anti_patterns_lines}\n"
            )

        custom_block = ""
        if self.custom_instruction:
            custom_block = (
                f"\nAdditional instructions:\n{self.custom_instruction.strip()}\n"
            )

        severity_label = element.get("severity", "medium")
        required_label = "REQUIRED" if element.get("required", True) else "RECOMMENDED"

        prompt = (
            f"System: You are a regulatory compliance reviewer assessing whether a "
            f"financial advice note meets a specific requirement under {severity_label.upper()} "
            f"severity. Be precise and conservative — only mark an element as 'present' "
            f"if it is clearly and adequately documented. Partial credit ('partial') "
            f"applies when the topic is raised but insufficiently documented; 'missing' "
            f"means there is no meaningful mention whatsoever.\n\n"
            f"Requirement ({required_label}): {element['description'].strip()}"
            f"{guidance_block}"
            f"{examples_block}"
            f"{anti_patterns_block}"
            f"{custom_block}\n"
            f"Note text:\n"
            f"---\n"
            f"{note_text}\n"
            f"---\n\n"
            f"Assess this requirement and respond using ONLY this exact format "
            f"(no other text):\n"
            f"STATUS: present|partial|missing\n"
            f"SCORE: <float 0.0-1.0>\n"
            f"EVIDENCE: <direct quote or paraphrase from the note, or \"None found\">\n"
            f"NOTES: <brief explanation of your assessment>"
        )
        return prompt

    def _parse_element_response(
        self,
        response: str,
        element: dict,
    ) -> GapItem:
        """
        Parse structured LLM response into a GapItem.

        Expected format:
            STATUS: present|partial|missing
            SCORE: 0.0–1.0
            EVIDENCE: <verbatim excerpt or "None found">
            NOTES: <optional reasoning>

        Defensive parsing: handles missing fields, unexpected whitespace,
        and LLM format drift without raising exceptions.
        """
        # Parse labelled fields — multi-line values are captured greedily
        # until the next label or end of string.
        label_pattern = re.compile(
            r"^(STATUS|SCORE|EVIDENCE|NOTES)\s*:\s*(.+?)(?=\n(?:STATUS|SCORE|EVIDENCE|NOTES)\s*:|$)",
            re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        parsed: Dict[str, str] = {}
        for match in label_pattern.finditer(response):
            key = match.group(1).upper()
            value = match.group(2).strip()
            parsed[key] = value

        # ── STATUS ────────────────────────────────────────────────────────────
        raw_status = parsed.get("STATUS", "").lower()
        if "present" in raw_status and "partial" not in raw_status:
            status = "present"
        elif "partial" in raw_status:
            status = "partial"
        else:
            status = "missing"

        # ── SCORE ─────────────────────────────────────────────────────────────
        raw_score = parsed.get("SCORE", "")
        score = self._extract_float_from_response(raw_score, default=0.0)
        # Align score with status when LLM is inconsistent
        if status == "missing" and score > 0.2:
            score = 0.0
        elif status == "present" and score < 0.5:
            score = max(score, 0.7)

        # ── EVIDENCE ──────────────────────────────────────────────────────────
        evidence_raw = parsed.get("EVIDENCE", "None found")
        evidence = evidence_raw if evidence_raw.lower() not in ("", "none", "none found") else ""

        # ── NOTES ─────────────────────────────────────────────────────────────
        notes: Optional[str] = None
        if self.verbose:
            notes = parsed.get("NOTES", "").strip() or None

        return GapItem(
            element_id=element["id"],
            status=status,
            score=score,
            evidence=evidence,
            severity=element["severity"],
            required=element.get("required", True),
            notes=notes,
        )

    # ── Overall summary ────────────────────────────────────────────────────────

    def _generate_summary(
        self,
        note_text: str,
        framework: dict,
        items: List[GapItem],
    ) -> str:
        """
        Generate a concise human-readable summary of the evaluation results
        via a single LLM call.
        """
        gaps = [
            f"  - [{item.severity.upper()}] {item.element_id}: {item.status}"
            for item in items
            if item.status != "present"
        ]
        gaps_text = "\n".join(gaps) if gaps else "  (none)"

        present_count = sum(1 for it in items if it.status == "present")
        total = len(items)

        prompt = (
            f"System: You are a regulatory compliance analyst. Write a concise (3–5 sentence) "
            f"plain-English summary of the following compliance note evaluation.\n\n"
            f"Framework: {framework['name']} (v{framework['version']})\n"
            f"Elements assessed: {total}\n"
            f"Elements present: {present_count}/{total}\n"
            f"Gaps identified:\n{gaps_text}\n\n"
            f"Write a professional, factual summary suitable for an audit trail. "
            f"Do not invent information beyond what is provided above."
        )
        try:
            return self.llm.generate(prompt, max_tokens=300).strip()
        except Exception as exc:
            logger.warning("Summary generation failed (%s); using fallback.", exc)
            return (
                f"Evaluated {total} framework elements; "
                f"{present_count} present, {len(gaps)} gap(s) identified."
            )

    # ── Statistics & scoring ───────────────────────────────────────────────────

    def _compute_stats(self, items: List[GapItem]) -> GapReportStats:
        """Compute summary statistics from a list of GapItems."""
        total = len(items)
        required = sum(1 for it in items if it.required)
        present = sum(1 for it in items if it.status == "present")
        partial = sum(1 for it in items if it.status == "partial")
        missing = sum(1 for it in items if it.status == "missing")

        def _gap_count(severity: str) -> int:
            return sum(
                1 for it in items
                if it.severity == severity and it.status != "present"
            )

        required_missing = sum(
            1 for it in items
            if it.required and it.status in ("missing", "partial")
        )

        return GapReportStats(
            total_elements=total,
            required_elements=required,
            present_count=present,
            partial_count=partial,
            missing_count=missing,
            critical_gaps=_gap_count("critical"),
            high_gaps=_gap_count("high"),
            medium_gaps=_gap_count("medium"),
            low_gaps=_gap_count("low"),
            required_missing_count=required_missing,
        )

    def _compute_overall_score(self, items: List[GapItem]) -> float:
        """
        Weighted mean of element scores.
        Required elements are weighted 2×, optional elements 1×.
        """
        if not items:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for item in items:
            weight = 2.0 if item.required else 1.0
            weighted_sum += item.score * weight
            total_weight += weight

        return round(weighted_sum / total_weight, 4) if total_weight > 0 else 0.0

    def _determine_pass(self, items: List[GapItem]) -> bool:
        """Apply the PassPolicy to determine overall pass/fail."""
        policy = self.pass_policy

        for item in items:
            if not item.required:
                continue  # Optional elements never block a pass

            if item.severity == "critical":
                if policy.block_on_critical_missing and item.status == "missing":
                    return False
                if (
                    policy.block_on_critical_partial
                    and item.status == "partial"
                    and item.score < policy.critical_partial_threshold
                ):
                    return False

            elif item.severity == "high":
                if policy.block_on_high_missing and item.status == "missing":
                    return False

        return True
