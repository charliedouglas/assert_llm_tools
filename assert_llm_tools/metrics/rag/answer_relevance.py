from typing import Dict, Optional
from ...llm.config import LLMConfig
from ...llm.bedrock import BedrockLLM
from ...llm.openai import OpenAILLM


class AnswerRelevanceCalculator:
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

    def calculate_score(self, question: str, answer: str) -> float:
        prompt = f"""You are an expert evaluator. Assess how relevant the following answer is to the given question.
    
Question: {question}
Answer: {answer}

Rate the relevance on a scale of 0 to 1, where:
0.0: Completely irrelevant - The answer has no connection to the question
0.5: Partially relevant - The answer addresses some aspects but misses key points or includes irrelevant information
1.0: Highly relevant - The answer directly addresses the question

Important: Your response must start with just the numerical score (0.0 to 1.0). 
You may provide explanation after the score on a new line.

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


def calculate_answer_relevance(
    question: str,
    answer: str,
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate how relevant the answer is to the given question.

    Args:
        question: The input question
        answer: The generated answer to evaluate
        llm_config: Configuration for LLM-based evaluation

    Returns:
        Dictionary containing the answer_relevance score
    """
    calculator = AnswerRelevanceCalculator(llm_config)
    score = calculator.calculate_score(question, answer)

    return {"answer_relevance": score}
