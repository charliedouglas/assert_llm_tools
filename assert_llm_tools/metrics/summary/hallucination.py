from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ..base import SummaryMetricCalculator


class HallucinationCalculator(SummaryMetricCalculator):
    """
    Calculator for detecting hallucinations in summaries.

    Identifies claims in the summary that cannot be supported by the reference text,
    specifically targeting content that appears to be fabricated or hallucinated.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, custom_instruction: Optional[str] = None, verbose: bool = False):
        """
        Initialize hallucination calculator.

        Args:
            llm_config: Configuration for LLM
            custom_instruction: Optional custom instruction to add to the LLM prompt
            verbose: Whether to include detailed claim-level analysis in the output
        """
        super().__init__(llm_config)
        self.custom_instruction = custom_instruction
        self.verbose = verbose

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
            # Check for hallucination or supported anywhere in the line
            if "hallucination" in line_lower or "supported" in line_lower:
                valid_lines.append(line_lower)
        
        # Make sure we have a result for each claim
        results = valid_lines[:len(claims)]  # Truncate if too many
        if len(results) < len(claims):
            # Pad with "supported" if too few (being conservative)
            results.extend(["supported"] * (len(claims) - len(results)))
            
        # Determine if each result indicates a hallucination
        hallucinations = []
        for result in results:
            # If the result contains "hallucination", count it as a hallucination
            # This handles cases like "this is a hallucination" or just "hallucination"
            is_hallucination = "hallucination" in result and "not hallucination" not in result
            hallucinations.append(is_hallucination)
            
        return hallucinations

    def _extract_summary_claims(self, text: str) -> List[str]:
        """
        Extract factual claims specifically from a summary text using LLM.

        Args:
            text: Summary text to extract claims from

        Returns:
            List of extracted claims
        """
        prompt = f"""
        You are a claim extraction assistant. Your task is to break down the following summary into separate factual claims.
        
        Guidelines:
        - Extract all verifiable statements of fact
        - Each claim should be a single, atomic piece of information
        - Split compound claims into separate individual claims
        - Keep each claim concise but complete enough to be verified
        - Include specific details, numbers, dates, names when present
        - Exclude opinions, judgments, or subjective statements
        
        Summary to analyze:
        ```
        {text}
        ```
        
        Return only the list of claims, one per line. Do not include any other text.
        """

        response = self.llm.generate(prompt, max_tokens=500)
        claims = response.strip().split("\n")
        return [claim.strip() for claim in claims if claim.strip()]

    def calculate_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate hallucination score for a summary.

        Args:
            reference: Original reference text
            candidate: Summary to evaluate

        Returns:
            Dictionary with hallucination score and claim statistics
        """
        # Extract claims from summary using specialized method for summaries
        summary_claims = self._extract_summary_claims(candidate)

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

        result = {
            "hallucination_score": hallucination_free_score,  # More consistent with other metric names
            "summary_claims_count": len(summary_claims),
            "hallucinated_claims_count": hallucinated_claims_count,
        }

        # Include detailed claim-level analysis when verbose is enabled
        if self.verbose:
            result["claims_analysis"] = [
                {"claim": claim, "is_hallucination": is_hallucination}
                for claim, is_hallucination in zip(summary_claims, hallucination_results)
            ]

        return result


def calculate_hallucination(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None,
    custom_instruction: Optional[str] = None, verbose: bool = False
) -> Dict[str, float]:
    """
    Calculate hallucination score by identifying claims in the summary not supported by the reference text.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use
        custom_instruction (Optional[str]): Custom instruction to add to the LLM prompt for evaluation
        verbose (bool): If True, include detailed claim-level analysis

    Returns:
        Dict[str, float]: Dictionary containing:
            - hallucination_score: Score from 0-1 (1 = no hallucinations)
            - summary_claims_count: Total claims extracted from summary
            - hallucinated_claims_count: Number of hallucinated claims
            - claims_analysis (only if verbose=True): List of dicts with claim text and hallucination status
    """
    calculator = HallucinationCalculator(llm_config, custom_instruction=custom_instruction, verbose=verbose)
    return calculator.calculate_score(reference, candidate)