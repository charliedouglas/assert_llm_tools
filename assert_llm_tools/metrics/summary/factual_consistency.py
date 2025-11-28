from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class FactualConsistencyCalculator(SummaryMetricCalculator):
    """
    Calculator for evaluating factual consistency of summaries.

    Identifies claims in the summary that are supported or unsupported by the reference text.
    This provides a measure of precision/accuracy - what percentage of summary claims
    are factually grounded in the source material.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None, verbose: bool = False):
        """
        Initialize factual consistency calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            verbose: Whether to include detailed claim-level analysis in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose

    def _verify_claims_batch(self, claims: List[str], context: str) -> List[bool]:
        """
        Verify if claims are supported by the reference text.

        Args:
            claims: List of claims to verify
            context: Reference text to check against

        Returns:
            List of boolean values indicating if each claim is supported (True) or unsupported (False)
        """
        if not claims:
            return []

        claims_text = "\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )
        prompt = f"""
        You are a factual consistency verification assistant that determines if claims in a summary are supported by the original text.

        For each claim below, determine if it is supported by or can be directly inferred from the original text.

        Original text:
        ```
        {context}
        ```

        Claims to verify:
        {claims_text}

        Respond with EXACTLY one line per claim, containing ONLY the word 'supported' or 'unsupported'.
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
        valid_lines = []
        for line in lines:
            line_lower = line.lower()
            # Check for supported or unsupported anywhere in the line
            if "supported" in line_lower or "unsupported" in line_lower:
                valid_lines.append(line_lower)

        # Make sure we have a result for each claim
        results = valid_lines[:len(claims)]  # Truncate if too many
        if len(results) < len(claims):
            # Pad with "unsupported" if too few (being conservative - assume not supported)
            results.extend(["unsupported"] * (len(claims) - len(results)))

        # Determine if each result indicates support
        supported = []
        for result in results:
            # If the result contains "unsupported", count it as unsupported
            # This handles cases like "this is unsupported" or just "unsupported"
            is_supported = "unsupported" not in result or "not unsupported" in result
            supported.append(is_supported)

        return supported

    def calculate_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate factual consistency score for a summary.

        Args:
            reference: Original reference text
            candidate: Summary to evaluate

        Returns:
            Dictionary with factual consistency score and claim statistics
        """
        # Extract claims from summary using context-aware extraction
        summary_claims = self._extract_claims(candidate, context="summary")

        if not summary_claims:  # avoid division by zero
            return {
                "factual_consistency": 1.0,  # No claims means perfect consistency
                "summary_claims_count": 0,
                "unsupported_claims_count": 0,
            }

        # Verify each summary claim against the reference
        supported_results = self._verify_claims_batch(summary_claims, reference)
        supported_claims_count = sum(supported_results)
        unsupported_claims_count = len(summary_claims) - supported_claims_count

        # Calculate factual consistency score (higher is better)
        factual_consistency_score = supported_claims_count / len(summary_claims)

        result = {
            "factual_consistency": factual_consistency_score,
            "summary_claims_count": len(summary_claims),
            "supported_claims_count": supported_claims_count,
            "unsupported_claims_count": unsupported_claims_count,
        }

        # Include detailed claim-level analysis when verbose is enabled
        if self.verbose:
            result["claims_analysis"] = [
                {"claim": claim, "is_supported": is_supported}
                for claim, is_supported in zip(summary_claims, supported_results)
            ]

        return result


def calculate_factual_consistency(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None, verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate factual consistency score by verifying if claims in the summary are supported by the reference text.

    This metric measures precision/accuracy: it extracts all claims from the summary text
    and checks how many of them are supported by the reference, providing a score between 0-1.
    A score of 1.0 means all summary claims are supported, while 0.0 means none are supported.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed claim-level analysis showing each extracted
            claim and whether it is supported by the reference

    Returns:
        Dict[str, float]: Dictionary containing:
            - factual_consistency: Score from 0-1 (supported_claims / total_summary_claims)
            - summary_claims_count: Total claims extracted from summary
            - supported_claims_count: Number of summary claims supported by reference
            - unsupported_claims_count: Number of summary claims not supported by reference
            - claims_analysis (only if verbose=True): List of dicts with claim text and support status
    """
    calculator = FactualConsistencyCalculator(llm_config, custom_instruction=custom_instruction, verbose=verbose)
    return calculator.calculate_score(reference, candidate)
