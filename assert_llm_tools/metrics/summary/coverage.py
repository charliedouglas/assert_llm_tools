from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class CoverageCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating coverage/completeness of summaries.

    Measures how well a summary covers the claims from the reference text
    by extracting claims from the reference and checking if they appear in the summary.
    This provides a measure of completeness/recall - what percentage of the source
    information is captured in the summary.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None, verbose: bool = False):
        """
        Initialize coverage calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            verbose: Whether to include detailed claim-level analysis in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose

    def _check_claims_in_summary_batch(self, claims: List[str], summary: str) -> List[bool]:
        """
        Check if claims from the reference text are present in the summary.

        Args:
            claims: List of claims from reference text to check
            summary: Summary text to check against

        Returns:
            List of boolean values indicating if each claim is present in the summary
        """
        if not claims:
            return []

        claims_text = "\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )
        prompt = f"""You are a coverage verification assistant that determines if claims from a source document are present in a summary.

For each claim below, determine if the information from that claim appears in the summary (even if worded differently or paraphrased).

Summary:
```
{summary}
```

Claims from source document to check:
{claims_text}

Respond with EXACTLY one line per claim, containing ONLY the word 'supported' or 'unsupported'.
- 'supported' = the claim's information appears in the summary
- 'unsupported' = the claim's information is missing from the summary

Do not include any explanation, reasoning, or numbering in your response.

For example, if there are 3 claims, your response should look exactly like:
supported
unsupported
supported"""

        if self.custom_instruction:
            prompt += f"\n\nAdditional Instructions:\n{self.custom_instruction}"

        response = self.llm.generate(prompt, max_tokens=300)

        # Clean up response and split into lines
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

        # Filter out any lines that don't contain our expected response formats
        # Accept both new (supported/unsupported) and old (present/missing) keywords
        valid_lines = []
        for line in lines:
            line_lower = line.lower()
            if ("supported" in line_lower or "unsupported" in line_lower or
                "present" in line_lower or "missing" in line_lower):
                valid_lines.append(line_lower)

        # Make sure we have a result for each claim
        results = valid_lines[:len(claims)]  # Truncate if too many
        if len(results) < len(claims):
            # Pad with "unsupported" if too few (being conservative - assume not covered)
            results.extend(["unsupported"] * (len(claims) - len(results)))

        # Determine if each result indicates presence
        # Accept both new and old keyword formats for backward compatibility
        present = []
        for result in results:
            # Check for positive indicators (supported/present) without negation
            is_present = (
                ("supported" in result and "unsupported" not in result) or
                ("present" in result and "not present" not in result)
            )
            present.append(is_present)

        return present

    def calculate_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate coverage score for a summary based on coverage of source claims.

        This metric measures how many claims from the reference text are present in the summary,
        providing a measure of completeness/recall. Higher scores indicate better coverage
        of the source material.

        Args:
            reference: Original reference text
            candidate: Summary to evaluate

        Returns:
            Dictionary with coverage score and claim statistics
        """
        # Extract claims from the reference (source material)
        reference_claims = self._extract_claims(reference, context="source")

        if not reference_claims:  # avoid division by zero
            return {
                "coverage": 1.0,  # No claims in reference means perfect coverage
                "reference_claims_count": 0,
                "claims_in_summary_count": 0,
            }

        # Check which reference claims appear in the summary
        claims_present_results = self._check_claims_in_summary_batch(reference_claims, candidate)
        claims_in_summary_count = sum(claims_present_results)

        # Calculate coverage score as recall
        coverage_score = claims_in_summary_count / len(reference_claims)

        result = {
            "coverage": coverage_score,
            "reference_claims_count": len(reference_claims),
            "claims_in_summary_count": claims_in_summary_count,
        }

        # Include detailed claim-level analysis when verbose is enabled
        if self.verbose:
            result["claims_analysis"] = [
                {"claim": claim, "is_covered": is_present}
                for claim, is_present in zip(reference_claims, claims_present_results)
            ]

        return result


def calculate_coverage(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None, verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate coverage score by measuring how many claims from the reference appear in the summary.

    This metric measures completeness/recall: it extracts all claims from the reference text
    and checks how many of them are present in the summary, providing a score between 0-1.
    A score of 1.0 means all source claims are covered, while 0.0 means none are covered.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed claim-level analysis showing each extracted
            claim and whether it was found in the summary

    Returns:
        Dict[str, float]: Dictionary containing:
            - coverage: Score from 0-1 (claims_in_summary / total_reference_claims)
            - reference_claims_count: Total claims extracted from reference
            - claims_in_summary_count: Number of reference claims present in summary
            - claims_analysis (only if verbose=True): List of dicts with claim text and coverage status
    """
    calculator = CoverageCalculator(llm_config, custom_instruction=custom_instruction, verbose=verbose)
    return calculator.calculate_score(reference, candidate)
