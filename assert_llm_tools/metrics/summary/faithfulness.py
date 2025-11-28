from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class FaithfulnessCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating faithfulness of summaries.

    Measures how well a summary covers the claims from the reference text
    by extracting claims from the reference and checking if they appear in the summary.
    This provides a measure of completeness/recall.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None, verbose: bool = False):
        """
        Initialize faithfulness calculator.

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
        Calculate faithfulness score for a summary based on coverage of source claims.

        This metric measures how many claims from the reference text are present in the summary,
        providing a measure of completeness/recall rather than accuracy.

        Args:
            reference: Original reference text
            candidate: Summary to evaluate

        Returns:
            Dictionary with faithfulness score and claim statistics
        """
        # Extract claims from the reference (source material)
        reference_claims = self._extract_claims(reference)

        if not reference_claims:  # avoid division by zero
            return {
                "faithfulness": 1.0,  # No claims in reference means perfect coverage
                "reference_claims_count": 0,
                "claims_in_summary_count": 0,
            }

        # Check which reference claims appear in the summary
        claims_present_results = self._check_claims_in_summary_batch(reference_claims, candidate)
        claims_in_summary_count = sum(claims_present_results)

        # Calculate faithfulness score as coverage/recall
        faithfulness_score = claims_in_summary_count / len(reference_claims)

        result = {
            "faithfulness": faithfulness_score,
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


def calculate_faithfulness(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None, verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate faithfulness score by measuring how many claims from the reference appear in the summary.

    This metric measures completeness/recall: it extracts all claims from the reference text
    and checks how many of them are present in the summary, providing a score between 0-1.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed claim-level analysis

    Returns:
        Dict[str, float]: Dictionary containing:
            - faithfulness: Score from 0-1 (claims_in_summary / total_reference_claims)
            - reference_claims_count: Total claims extracted from reference
            - claims_in_summary_count: Number of reference claims present in summary
            - claims_analysis (only if verbose=True): List of dicts with claim text and coverage status
    """
    calculator = FaithfulnessCalculator(llm_config, custom_instruction=custom_instruction, verbose=verbose)
    return calculator.calculate_score(reference, candidate)
