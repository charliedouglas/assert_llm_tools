from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class FaithfulnessCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating faithfulness of summaries.

    Measures how factually consistent a summary is with the reference text
    by extracting claims from the summary and verifying them against the reference.
    """

    def _verify_claims_batch(self, claims: List[str], context: str) -> List[bool]:
        """
        Verify if claims can be inferred from the reference text.

        Args:
            claims: List of claims to verify
            context: Reference text to check against

        Returns:
            List of boolean values indicating if each claim is supported
        """
        claims_text = "\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )
        prompt = f"""
        System: You are a helpful assistant that verifies if claims can be directly inferred from given context. 
        For each claim, answer with only 'true' or 'false'.

        Context: {context}

        Claims to verify:
        {claims_text}

        For each claim, answer with only 'true' or 'false', one per line.

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=300)
        results = response.strip().split("\n")
        return [result.strip().lower() == "true" for result in results]

    def calculate_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate faithfulness score for a summary.

        Args:
            reference: Original reference text
            candidate: Summary to evaluate

        Returns:
            Dictionary with faithfulness score and claim statistics
        """
        # Extract claims from both texts
        reference_claims = self._extract_claims(reference)
        summary_claims = self._extract_claims(candidate)

        if not summary_claims:  # avoid division by zero
            return {
                "faithfulness": 1.0,  # No claims means no unfaithful claims
                "reference_claims_count": len(reference_claims),
                "summary_claims_count": 0,
                "verified_claims_count": 0,
            }

        # Verify all claims in a single batch
        verification_results = self._verify_claims_batch(summary_claims, reference)
        verified_claims_count = sum(verification_results)

        # Calculate faithfulness score
        faithfulness_score = (
            verified_claims_count / len(summary_claims) if summary_claims else 1.0
        )

        return {
            "faithfulness": faithfulness_score,
            "reference_claims_count": len(reference_claims),
            "summary_claims_count": len(summary_claims),
            "verified_claims_count": verified_claims_count,
        }


def calculate_faithfulness(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, float]:
    """
    Calculate faithfulness score by comparing claims in the summary against the reference text.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use

    Returns:
        Dict[str, float]: Dictionary containing faithfulness score and claim counts
    """
    calculator = FaithfulnessCalculator(llm_config)
    return calculator.calculate_score(reference, candidate)
