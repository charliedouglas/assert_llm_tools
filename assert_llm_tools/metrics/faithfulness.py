from typing import Dict, List, Optional
from assert_llm_tools.llm.config import LLMConfig
from assert_llm_tools.llm.bedrock import BedrockLLM
from assert_llm_tools.llm.openai import OpenAILLM


class FaithfulnessCalculator:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            # Default to Bedrock with Claude
            llm_config = LLMConfig(
                provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1"
            )

        if llm_config.provider == "bedrock":
            self.llm = BedrockLLM(llm_config)
        elif llm_config.provider == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _extract_claims(self, text: str) -> List[str]:
        prompt = f"""
        System: You are a helpful assistant that extracts factual claims from text. Extract all factual claims from the given text. Output each claim on a new line. Only include objective, verifiable claims. Do not include opinions or subjective statements.

        Human: Here is the text to analyze:
        {text}

        Please list all factual claims, one per line.

        Assistant: Here are the factual claims:"""

        response = self.llm.generate(prompt, max_tokens=500)
        claims = response.strip().split("\n")
        return [claim.strip() for claim in claims if claim.strip()]

    def _verify_claims_batch(self, claims: List[str], context: str) -> List[bool]:
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

    # Extract claims from both texts
    reference_claims = calculator._extract_claims(reference)
    summary_claims = calculator._extract_claims(candidate)

    if not reference_claims:  # avoid division by zero
        return {
            "faithfulness": 0.0,
            "reference_claims_count": 0,
            "summary_claims_count": len(summary_claims),
            "verified_claims_count": 0,
        }

    # Verify all claims in a single batch
    verification_results = calculator._verify_claims_batch(summary_claims, reference)
    verified_claims_count = sum(verification_results)

    # Calculate faithfulness score based on reference claims
    faithfulness_score = verified_claims_count / len(reference_claims)

    return {
        "faithfulness": faithfulness_score,
        "reference_claims_count": len(reference_claims),
        "summary_claims_count": len(summary_claims),
        "verified_claims_count": verified_claims_count,
    }
