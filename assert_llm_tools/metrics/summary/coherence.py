from typing import Dict, List, Optional
from ...llm.config import LLMConfig
from ...llm.bedrock import BedrockLLM
from ...llm.openai import OpenAILLM
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine


class CoherenceCalculator:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        # Initialize embedding model for semantic analysis
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        if llm_config is None:
            # Default to Bedrock with Claude 3
            llm_config = LLMConfig(
                provider="bedrock",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                region="us-east-1",
            )

        if llm_config.provider == "bedrock":
            self.llm = BedrockLLM(llm_config)
        elif llm_config.provider == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    def _calculate_sentence_similarity(self, sentences: List[str]) -> float:
        """
        Calculate average cosine similarity between consecutive sentences
        using sentence embeddings.
        """
        if len(sentences) <= 1:
            return 1.0  # If only one sentence, it's coherent by default

        # Get embeddings for all sentences
        embeddings = self.embedding_model.encode(sentences)

        # Calculate similarity between consecutive sentences
        similarities = []
        for i in range(len(sentences) - 1):
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)

        # Return average similarity
        return float(np.mean(similarities))

    def _evaluate_discourse_coherence(self, text: str) -> float:
        """
        Use LLM to evaluate discourse-level coherence including:
        - Proper use of transition words
        - Logical flow of ideas
        - Consistent referencing (pronouns, etc.)
        - Topic progression
        """
        prompt = f"""Evaluate the coherence of the following text. Focus on:
1. Logical flow between sentences and paragraphs
2. Appropriate use of transition words and phrases
3. Consistent referencing (pronouns, definite articles)
4. Natural topic progression
5. Absence of abrupt topic shifts

Text to evaluate:
"{text}"

Rate the coherence on a scale of 0 to 1, where:
0.0: Completely incoherent - sentences appear random and disconnected
0.5: Partially coherent - some logical connections but with gaps or inconsistencies
1.0: Highly coherent - smooth and logical progression throughout

Important: Your response must be only a numerical score between 0.0 and 1.0.
"""

        # Get response from LLM
        response = self.llm.generate(prompt).strip()

        # Extract float from response
        try:
            score = float(response)
            # Ensure score is within bounds
            return max(0.0, min(1.0, score))
        except ValueError:
            # Default to middle score if parsing fails
            return 0.5

    def calculate_coherence(self, text: str) -> float:
        """
        Calculate overall coherence score using both embedding-based
        and LLM-based approaches.
        """
        # Split text into sentences
        sentences = sent_tokenize(text)

        if len(sentences) <= 1:
            return 1.0  # Single sentence is coherent by default

        # Get similarity-based coherence
        similarity_score = self._calculate_sentence_similarity(sentences)

        # Get discourse-based coherence
        discourse_score = self._evaluate_discourse_coherence(text)

        # Combine scores (weighted more toward discourse evaluation)
        final_score = 0.3 * similarity_score + 0.7 * discourse_score

        return final_score


def calculate_coherence(
    summary: str, llm_config: Optional[LLMConfig] = None
) -> Dict[str, float]:
    """
    Evaluate coherence of a summary.

    Args:
        summary (str): The summary to evaluate
        llm_config (Optional[LLMConfig]): Configuration for LLM-based evaluation

    Returns:
        Dict[str, float]: Dictionary containing the coherence score
    """
    calculator = CoherenceCalculator(llm_config)
    score = calculator.calculate_coherence(summary)

    return {"coherence": score}
