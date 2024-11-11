from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ...llm.bedrock import BedrockLLM
from ...llm.openai import OpenAILLM


class TopicPreservationCalculator:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            # Default to Bedrock with Claude
            llm_config = LLMConfig(
                provider="bedrock",
                model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
                region="us-east-1",
            )

        if llm_config.provider == "bedrock":
            self.llm = BedrockLLM(llm_config)
        elif llm_config.provider == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _extract_topics(self, text: str) -> List[str]:
        prompt = f"""
        System: You are a helpful assistant that extracts main topics from text. Your task is to identify only the high-level, key topics from the given text. Focus on broad themes and main subjects rather than specific details.

        Guidelines:
        - Extract only major, overarching topics
        - Avoid specific details or subtopics
        - Aim for 3-5 main topics
        - Use broad, general terms
        - Each topic should be a few words at most
        - Similar topics should be merged

        Human: Here is the text to analyze:
        {text}

        Please list only the main, high-level topics, one per line.

        Assistant: Here are the main topics:"""

        response = self.llm.generate(prompt, max_tokens=500)
        topics = response.strip().split("\n")
        return [topic.strip() for topic in topics if topic.strip()]

    def _check_topics_in_summary(self, topics: List[str], summary: str) -> List[bool]:
        topics_str = "\n".join([f"- {topic}" for topic in topics])
        prompt = f"""
        System: You are a helpful assistant that analyzes whether specific topics are covered in a given text. Your task is to determine if each topic is present in the summary, even if not explicitly mentioned.

        Human: Here is a summary text:
        {summary}

        Please analyze if each of the following topics from the original text is covered in the summary. 
        Respond with only "yes" or "no" for each topic, one per line:

        {topics_str}

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=500)
        results = response.strip().split("\n")
        return [r.strip().lower() == "yes" for r in results]

    def compute_score(self, reference: str, candidate: str) -> Dict[str, any]:
        # First, extract topics from reference text
        reference_topics = self._extract_topics(reference)

        # Then check which topics are present in the summary
        topic_present = self._check_topics_in_summary(reference_topics, candidate)

        # Separate preserved and missing topics
        preserved_topics = [
            topic for topic, present in zip(reference_topics, topic_present) if present
        ]
        missing_topics = [
            topic
            for topic, present in zip(reference_topics, topic_present)
            if not present
        ]

        # Calculate preservation score
        topic_preservation_score = (
            len(preserved_topics) / len(reference_topics) if reference_topics else 0.0
        )

        return {
            "topic_preservation": topic_preservation_score,
            "reference_topics": reference_topics,
            "preserved_topics": preserved_topics,
            "missing_topics": missing_topics,
        }


# Add this wrapper function at the module level (outside the class)
def calculate_topic_preservation(
    reference: str, candidate: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, any]:
    """
    Wrapper function to calculate topic preservation score.
    """
    calculator = TopicPreservationCalculator(llm_config)
    return calculator.compute_score(reference, candidate)
