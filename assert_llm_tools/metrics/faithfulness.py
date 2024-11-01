from typing import Dict, List, Optional
from ..llm.config import LLMConfig
from ..llm.bedrock import BedrockLLM
from ..llm.openai import OpenAILLM


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

    def _verify_claim(self, claim: str, context: str) -> bool:
        prompt = f"""
        System: You are a helpful assistant that verifies if claims can be directly inferred from given context. Answer only with 'true' or 'false'.

        Human: Can the following claim be directly inferred from the given context?

        Context: {context}
        Claim: {claim}

        Answer with only 'true' or 'false'.

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=10)
        return response.strip().lower() == "true"


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
        Dict[str, float]: Dictionary containing faithfulness score
    """
    calculator = FaithfulnessCalculator(llm_config)

    # Extract claims from both texts
    summary_claims = calculator._extract_claims(candidate)

    # Verify each claim from summary against the reference
    verified_claims = sum(
        calculator._verify_claim(claim, reference) for claim in summary_claims
    )

    # Calculate faithfulness score
    faithfulness_score = (
        verified_claims / len(summary_claims) if summary_claims else 0.0
    )

    return {"faithfulness": faithfulness_score}
