from typing import Dict, Optional, Union, List
from ...llm.config import LLMConfig
from ...llm.bedrock import BedrockLLM
from ...llm.openai import OpenAILLM


class ContextRelevanceCalculator:
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

    def calculate_score(self, question: str, context: Union[str, List[str]]) -> float:
        # If context is a list, join with newlines
        if isinstance(context, list):
            context_text = "\n\n".join(context)
        else:
            context_text = context

        prompt = f"""You are an expert evaluator. Assess how relevant the retrieved context is to the given question.

Question: {question}
Retrieved Context: {context_text}

Rate the relevance on a scale of 0 to 1, where:
0.0: Completely irrelevant - The context has no connection to the question
0.5: Partially relevant - The context contains some relevant information but includes unnecessary content or misses key aspects
1.0: Highly relevant - The context contains precisely the information needed to answer the question

Important: Your response must start with just the numerical score between 0.00 to 1.00. 

Score:"""

        # Get response from LLM
        response = self.llm.generate(prompt).strip()

        # Extract the first line and convert to float
        try:
            score = float(response.split("\n")[0].strip())
        except (ValueError, IndexError):
            # If parsing fails, default to 0
            score = 0.0

        # Ensure score is within bounds
        return max(0.0, min(1.0, score))


def calculate_context_relevance(
    question: str,
    context: Union[str, List[str]],
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate how relevant the retrieved context is to the given question.

    Args:
        question: The input question
        context: Retrieved context(s). Can be a single string or list of strings.
        llm_config: Configuration for LLM-based evaluation

    Returns:
        Dictionary containing the context_relevance score
    """
    calculator = ContextRelevanceCalculator(llm_config)
    score = calculator.calculate_score(question, context)

    return {"context_relevance": score}
