import re
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

    def _parse_claim_list(self, response: str) -> List[str]:
        """
        Parse claims from LLM response, handling various formats.

        Handles:
        - Numbered lists: "1. claim", "1) claim", "(1) claim"
        - Bulleted lists: "- claim", "* claim", "• claim"
        - Plain newline-separated claims
        - Meta-commentary filtering ("Here are the claims:")

        Args:
            response: Raw LLM response text

        Returns:
            List of cleaned claim strings
        """
        lines = response.strip().split("\n")
        claims = []

        # Patterns to skip (meta-commentary)
        skip_patterns = [
            r"^here are",
            r"^the (factual )?claims",
            r"^claims:",
            r"^$",
        ]

        # Pattern to strip prefixes (numbered lists, bullets)
        prefix_pattern = r"^\s*(?:[\d]+[.\)]\s*|\(?[\d]+\)\s*|[-*•]\s*)"

        for line in lines:
            line = line.strip()

            # Skip meta-commentary
            if any(re.match(p, line.lower()) for p in skip_patterns):
                continue

            # Strip numbering/bullet prefixes
            cleaned = re.sub(prefix_pattern, "", line).strip()

            if cleaned:
                claims.append(cleaned)

        return claims

    def _extract_claims(self, text: str, context: str = "general") -> List[str]:
        """
        Extract factual claims from text using LLM.

        Args:
            text: Text to extract claims from
            context: Type of text - affects extraction granularity
                     "source": Extract comprehensive claims from source document
                     "summary": Extract atomic, verifiable claims from summary
                     "general": Default balanced extraction

        Returns:
            List of extracted claims
        """
        context_guidelines = {
            "source": "For this source document, extract all significant factual claims to ensure comprehensive coverage analysis. Be thorough but avoid extracting trivial or redundant information.",
            "summary": "For this summary, extract the core claims being asserted. Each claim should be independently verifiable against a source document.",
            "general": "Extract claims at a balanced level of granularity.",
        }

        context_instruction = context_guidelines.get(context, context_guidelines["general"])

        prompt = f"""System: You are a claim extraction assistant that identifies distinct factual claims in text.

Guidelines:
- Extract all verifiable statements of fact
- Each claim should be atomic (one fact per claim)
- Split compound sentences into separate claims when they contain multiple facts
- Keep each claim concise but complete enough to be independently verified
- Include specific details: numbers, dates, names, measurements when present
- Exclude opinions, judgments, subjective statements, and hedged language
- Exclude procedural statements (e.g., "This document describes...")

{context_instruction}

Example:
Input: "The company reported $5.2 billion in revenue for Q3 2024, representing a 15% increase year-over-year."
Output:
The company reported $5.2 billion in revenue for Q3 2024
Q3 2024 revenue represented a 15% increase year-over-year

Text to analyze:
{text}

Extract the factual claims, one per line:"""

        response = self.llm.generate(prompt, max_tokens=1500)
        return self._parse_claim_list(response)


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
