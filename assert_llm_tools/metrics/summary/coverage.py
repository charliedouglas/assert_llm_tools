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

    def __init__(self, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None):
        """
        Initialize coverage calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction

    def _check_claims_in_summary_batch(self, claims: List[str], summary: str) -> List[bool]:
        """
        Check if claims from the reference text are present in the summary.

        Args:
            claims: List of claims from reference text to check
            summary: Summary text to check against

        Returns:
            List of boolean values indicating if each claim is present in the summary
        """
        claims_text = "\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )
        prompt = f"""
        System: You are a helpful assistant that determines if claims from a source document are present in a summary.
        For each claim, determine if the information from that claim appears in the summary (even if worded differently).
        Answer with only 'true' if the claim's information is present in the summary, or 'false' if it is missing.

        Summary: {summary}

        Claims from source document to check:
        {claims_text}

        For each claim, answer with only 'true' or 'false', one per line."""

        if self.custom_instruction:
            prompt += f"\n\nAdditional Instructions:\n{self.custom_instruction}"

        prompt += "\n\nAssistant:"

        response = self.llm.generate(prompt, max_tokens=300)
        results = response.strip().split("\n")
        return [result.strip().lower() == "true" for result in results]

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
        reference_claims = self._extract_claims(reference)

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

        return {
            "coverage": coverage_score,
            "reference_claims_count": len(reference_claims),
            "claims_in_summary_count": claims_in_summary_count,
        }


def calculate_coverage(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None
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

    Returns:
        Dict[str, float]: Dictionary containing:
            - coverage: Score from 0-1 (claims_in_summary / total_reference_claims)
            - reference_claims_count: Total claims extracted from reference
            - claims_in_summary_count: Number of reference claims present in summary
    """
    calculator = CoverageCalculator(llm_config, custom_instruction=custom_instruction)
    return calculator.calculate_score(reference, candidate)
