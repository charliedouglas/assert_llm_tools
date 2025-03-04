from typing import Optional, Dict, Union, List, Any
from ..llm.config import LLMConfig
from ..llm.bedrock import BedrockLLM
from ..llm.openai import OpenAILLM


class BaseCalculator:
    """
    Base class for all metric calculators.

    Handles common initialization logic for LLM-based metrics, including
    default configuration and LLM client initialization.
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        default_provider: str = "bedrock",
        default_model: str = "anthropic.claude-v2",
        default_region: str = "us-east-1",
    ):
        """
        Initialize base calculator with LLM configuration.

        Args:
            llm_config: Configuration for LLM. If None, a default config is created.
            default_provider: Default LLM provider if no config provided.
            default_model: Default model ID if no config provided.
            default_region: Default region (for Bedrock) if no config provided.
        """
        # Use provided config or create default
        if llm_config is None:
            llm_config = LLMConfig(
                provider=default_provider, model_id=default_model, region=default_region
            )

        # Initialize appropriate LLM client
        if llm_config.provider == "bedrock":
            self.llm = BedrockLLM(llm_config)
        elif llm_config.provider == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _extract_float_from_response(
        self, response: str, default: float = 0.5
    ) -> float:
        """
        Extract a float value from the first line of an LLM response.

        Args:
            response: Raw LLM response text
            default: Default value if parsing fails

        Returns:
            Extracted float value, bounded between 0.0 and 1.0
        """
        try:
            score = float(response.split("\n")[0].strip())
            return max(0.0, min(1.0, score))
        except (ValueError, IndexError):
            return default


class SummaryMetricCalculator(BaseCalculator):
    """
    Base class for summary evaluation metrics.

    Extends BaseCalculator with methods specific to summary evaluation.
    """

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from text using LLM.

        Args:
            text: Text to extract topics from

        Returns:
            List of extracted topics
        """
        prompt = f"""
        System: You are a topic extraction assistant. Your task is to identify the main topics from the text.

        Guidelines:
        - Extract 3-5 primary topics
        - Topics should be at the same level of abstraction
        - Merge related concepts into single topics
        - Exclude action items, recommendations, and time-specific references
        - Keep topics to 2-3 words maximum

        Human: Here is the text to analyze:
        {text}

        Please list only the main, high-level topics, one per line.

        Assistant: Here are the main topics:"""

        response = self.llm.generate(prompt, max_tokens=500)
        topics = response.strip().split("\n")
        return [topic.strip() for topic in topics if topic.strip()]

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text using LLM.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claims
        """
        prompt = f"""
        System: You are a helpful assistant that extracts factual claims from text. Extract all factual claims from the given text. Output each claim on a new line. Only include objective, verifiable claims. Do not include opinions or subjective statements.

        Human: Here is the text to analyze:
        {text}

        Please list all factual claims, one per line.

        Assistant: Here are the factual claims:"""

        response = self.llm.generate(prompt, max_tokens=500)
        claims = response.strip().split("\n")
        return [claim.strip() for claim in claims if claim.strip()]


class RAGMetricCalculator(BaseCalculator):
    """
    Base class for RAG evaluation metrics.

    Extends BaseCalculator with methods specific to RAG evaluation.
    """

    def _normalize_context(self, context: Union[str, List[str]]) -> str:
        """
        Convert context to a single string if it's a list.

        Args:
            context: Context as string or list of strings

        Returns:
            Normalized context as a single string
        """
        if isinstance(context, list):
            return "\n\n".join(context)
        return context

    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text using LLM.

        Args:
            text: Text to extract claims from

        Returns:
            List of extracted claims
        """
        prompt = f"""
        System: You are a helpful assistant that extracts factual claims from text. Extract all factual claims from the given text. Output each claim on a new line. Only include objective, verifiable claims. Do not include opinions or subjective statements.

        Human: Here is the text to analyze:
        {text}

        Please list all factual claims, one per line.

        Assistant: Here are the factual claims:"""

        response = self.llm.generate(prompt, max_tokens=500)
        claims = response.strip().split("\n")
        return [claim.strip() for claim in claims if claim.strip()]

    def _extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics from text using LLM.

        Args:
            text: Text to extract topics from

        Returns:
            List of extracted topics
        """
        prompt = f"""
        System: You are a helpful assistant that extracts main topics from text. Extract all key topics or subjects mentioned. Output each topic on a new line. Be specific but concise.

        Human: Here is the text to analyze:
        {text}

        Please list all key topics, one per line.

        Assistant: Here are the key topics:"""

        response = self.llm.generate(prompt, max_tokens=500)
        topics = response.strip().split("\n")
        return [topic.strip() for topic in topics if topic.strip()]
