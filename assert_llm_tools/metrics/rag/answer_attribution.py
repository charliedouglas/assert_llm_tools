from typing import Dict, Optional, Union, List
from ...llm.config import LLMConfig
from ...llm.bedrock import BedrockLLM
from ...llm.openai import OpenAILLM
from sentence_transformers import SentenceTransformer
import numpy as np


class AnswerAttributionCalculator:
    def __init__(self, llm_config: Optional[LLMConfig] = None):
        if llm_config is None:
            llm_config = LLMConfig(
                provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1"
            )

        if llm_config.provider == "bedrock":
            self.llm = BedrockLLM(llm_config)
        elif llm_config.provider == "openai":
            self.llm = OpenAILLM(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _calculate_embedding_similarity(self, answer: str, context: str) -> float:
        # Get embeddings
        answer_embedding = self.embedding_model.encode(answer)
        context_embedding = self.embedding_model.encode(context)

        # Calculate cosine similarity
        similarity = np.dot(answer_embedding, context_embedding) / (
            np.linalg.norm(answer_embedding) * np.linalg.norm(context_embedding)
        )

        return float(similarity)

    def _calculate_ngram_overlap(self, answer: str, context: str, n: int = 3) -> float:
        answer_words = answer.lower().split()
        context_words = context.lower().split()

        if len(answer_words) < n:
            return 0.0

        answer_ngrams = set(
            " ".join(answer_words[i : i + n]) for i in range(len(answer_words) - n + 1)
        )
        context_ngrams = set(
            " ".join(context_words[i : i + n])
            for i in range(len(context_words) - n + 1)
        )

        if not answer_ngrams:
            return 0.0

        overlap = len(answer_ngrams.intersection(context_ngrams)) / len(answer_ngrams)
        return overlap

    def _calculate_llm_score(self, answer: str, context: str) -> float:
        prompt = f"""You are an expert evaluator. Assess whether the given answer appears to be derived from the provided context.

Context: {context}
Answer: {answer}

Rate on a scale of 0 to 1, where:
0.0: Answer shows no evidence of using the context
0.5: Answer partially uses the context but includes external information
1.0: Answer is completely derived from the context

Important: Your response must start with just the numerical score (0.00 to 1.00).
You may provide explanation after the score on a new line.

Score:"""

        response = self.llm.generate(prompt).strip()

        try:
            score = float(response.split("\n")[0].strip())
        except (ValueError, IndexError):
            score = 0.0

        return max(0.0, min(1.0, score))

    def calculate_score(self, answer: str, context: Union[str, List[str]]) -> float:
        # If context is a list, join with newlines
        if isinstance(context, list):
            context_text = "\n\n".join(context)
        else:
            context_text = context

        # Calculate individual scores
        embedding_score = self._calculate_embedding_similarity(answer, context_text)
        ngram_score = self._calculate_ngram_overlap(answer, context_text)
        llm_score = self._calculate_llm_score(answer, context_text)

        # Combine scores with weights
        weights = {"embedding": 0.3, "ngram": 0.3, "llm": 0.4}

        final_score = (
            weights["embedding"] * embedding_score
            + weights["ngram"] * ngram_score
            + weights["llm"] * llm_score
        )

        return max(0.0, min(1.0, final_score))


def calculate_answer_attribution(
    answer: str,
    context: Union[str, List[str]],
    llm_config: Optional[LLMConfig] = None,
) -> Dict[str, float]:
    """
    Evaluate how much of the answer appears to be derived from the provided context.

    Args:
        answer: The generated answer to evaluate
        context: Retrieved context(s). Can be a single string or list of strings.
        llm_config: Configuration for LLM-based evaluation

    Returns:
        Dictionary containing the answer_attribution score
    """
    calculator = AnswerAttributionCalculator(llm_config)
    score = calculator.calculate_score(answer, context)

    return {"answer_attribution": score}
