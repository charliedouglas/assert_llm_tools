from typing import Dict, List, Optional
from ..llm.config import LLMConfig
from ..llm.bedrock import BedrockLLM
from ..llm.openai import OpenAILLM


class TopicPreservationCalculator:
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

    def _extract_topics(self, text: str) -> List[str]:
        prompt = f"""
        System: You are a helpful assistant that extracts main topics from text. Extract all distinct topics from the given text. Output each topic on a new line. Focus on key subjects and themes discussed in the text.

        Human: Here is the text to analyze:
        {text}

        Please list all main topics, one per line.

        Assistant: Here are the main topics:"""

        response = self.llm.generate(prompt, max_tokens=500)
        topics = response.strip().split("\n")
        return [topic.strip() for topic in topics if topic.strip()]


def calculate_topic_preservation(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, float]:
    """
    Calculate topic preservation score by comparing topics in the summary against the reference text.

    Args:
        reference (str): The original full text
        candidate (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for the LLM to use

    Returns:
        Dict[str, float]: Dictionary containing topic preservation score
    """
    calculator = TopicPreservationCalculator(llm_config)

    # Extract topics from both texts
    reference_topics = calculator._extract_topics(reference)
    summary_topics = calculator._extract_topics(candidate)

    # Calculate topic preservation score
    # If summary has more topics than reference, cap at 1.0
    topic_preservation_score = min(
        len(summary_topics) / len(reference_topics) if reference_topics else 0.0, 1.0
    )

    return {"topic_preservation": topic_preservation_score}
