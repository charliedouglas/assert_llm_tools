from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class HallucinationCalculator(SummaryMetricCalculator):
    """
    Calculator for detecting hallucinations in summaries.
    
    Identifies claims in the summary that cannot be supported by the reference text,
    specifically targeting content that appears to be fabricated or hallucinated.
    """

    def _detect_hallucinations_batch(self, claims: List[str], context: str) -> List[bool]:
        """
        Detect if claims are hallucinations not supported by the reference text.

        Args:
            claims: List of claims to verify
            context: Reference text to check against

        Returns:
            List of boolean values indicating if each claim is a hallucination
        """
        if not claims:
            return []
            
        claims_text = "\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )
        prompt = f"""
        You are a hallucination detection assistant that verifies if claims in a summary are supported by the original text.
        
        For each claim below, determine if it contains ANY information NOT present in or directly inferable from the original text.
        
        Original text:
        ```
        {context}
        ```
        
        Claims to verify:
        {claims_text}
        
        Respond with EXACTLY one line per claim, containing ONLY the word 'hallucination' or 'supported'.
        Do not include any explanation, reasoning, or numbering in your response.
        
        For example, if there are 3 claims, your response should look exactly like:
        supported
        hallucination
        supported
        """

        response = self.llm.generate(prompt, max_tokens=300)
        results = response.strip().split("\n")
        
        # Make sure we have a result for each claim
        if len(results) != len(claims):
            # Try to salvage by taking only the first len(claims) results
            results = results[:len(claims)] if len(results) > len(claims) else results + ['supported'] * (len(claims) - len(results))
            
        return [result.strip().lower() == "hallucination" for result in results]

    def calculate_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate hallucination score for a summary.

        Args:
            reference: Original reference text
            candidate: Summary to evaluate

        Returns:
            Dictionary with hallucination score and claim statistics
        """
        # Extract claims from summary
        summary_claims = self._extract_claims(candidate)

        if not summary_claims:  # avoid division by zero
            return {
                "hallucination_free": 1.0,  # No claims means no hallucinations
                "summary_claims_count": 0,
                "hallucinated_claims_count": 0,
            }

        # Detect hallucinations in all claims
        hallucination_results = self._detect_hallucinations_batch(summary_claims, reference)
        hallucinated_claims_count = sum(hallucination_results)

        # Calculate hallucination-free score (higher is better)
        hallucination_free_score = 1.0 - (
            hallucinated_claims_count / len(summary_claims) if summary_claims else 0.0
        )

        return {
            "hallucination_free": hallucination_free_score,
            "summary_claims_count": len(summary_claims),
            "hallucinated_claims_count": hallucinated_claims_count,
        }


def calculate_hallucination(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, float]:
    """
    Calculate hallucination score by identifying claims in the summary not supported by the reference text.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use

    Returns:
        Dict[str, float]: Dictionary containing hallucination score and claim counts
    """
    calculator = HallucinationCalculator(llm_config)
    return calculator.calculate_score(reference, candidate)