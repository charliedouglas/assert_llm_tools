from typing import Dict, List, Optional, Union
from ...llm.config import LLMConfig
from ..base import RAGMetricCalculator


class RAGFaithfulnessCalculator(RAGMetricCalculator):
    """
    Calculator for evaluating faithfulness of RAG answers.

    Measures how factually consistent an answer is with the retrieved context
    by analyzing both claims and topics.
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, verbose: bool = False):
        """
        Initialize RAG faithfulness calculator.

        Args:
            llm_config: Configuration for LLM
            verbose: Whether to include detailed claim/topic-level analysis in the output
        """
        super().__init__(llm_config)
        self.verbose = verbose

    def _verify_claims_batch(self, claims: List[str], context: str) -> List[bool]:
        """
        Verify if claims can be inferred from the provided context.

        Args:
            claims: List of claims to verify
            context: Context to check against

        Returns:
            List of boolean values indicating if each claim is supported
        """
        if not claims:
            return []

        claims_text = "\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )
        prompt = f"""You are a factual verification assistant that determines if claims can be supported by the given context.

For each claim below, determine if it is supported by or can be directly inferred from the context.

Context:
```
{context}
```

Claims to verify:
{claims_text}

Respond with EXACTLY one line per claim, containing ONLY the word 'supported' or 'unsupported'.
Do not include any explanation, reasoning, or numbering in your response.

For example, if there are 3 claims, your response should look exactly like:
supported
unsupported
supported"""

        response = self.llm.generate(prompt, max_tokens=300)

        # Clean up response and split into lines
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

        # Filter for valid responses (accept both new and legacy keywords)
        valid_lines = []
        for line in lines:
            line_lower = line.lower()
            if ("supported" in line_lower or "unsupported" in line_lower or
                "true" in line_lower or "false" in line_lower):
                valid_lines.append(line_lower)

        # Make sure we have a result for each claim
        results = valid_lines[:len(claims)]
        if len(results) < len(claims):
            # Pad with "unsupported" if too few (being conservative)
            results.extend(["unsupported"] * (len(claims) - len(results)))

        # Determine if each result indicates support
        supported = []
        for result in results:
            is_supported = (
                ("supported" in result and "unsupported" not in result) or
                (result == "true")
            )
            supported.append(is_supported)

        return supported

    def _verify_topics_batch(self, topics: List[str], context: str) -> List[bool]:
        """
        Verify if topics are substantively discussed in the context.

        Args:
            topics: List of topics to verify
            context: Context to check against

        Returns:
            List of boolean values indicating if each topic is present
        """
        topics_text = "\n".join(
            f"Topic {i+1}: {topic}" for i, topic in enumerate(topics)
        )
        prompt = f"""
        System: You are a helpful assistant that verifies if topics are substantively discussed in the given context. 
        For each topic, carefully check if the context contains meaningful information about it.
        Answer ONLY 'true' if the topic is clearly discussed in the context, or 'false' if it is not mentioned or only briefly referenced.

        Context:
        {context}

        Topics to verify:
        {topics_text}

        For each topic listed above, respond with ONLY 'true' or 'false' on a new line, indicating whether the topic is substantively discussed in the context.

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=300)
        results = response.strip().split("\n")

        # Ensure we have a result for each topic
        if len(results) != len(topics):
            # If response length doesn't match, assume false for missing results
            results.extend(["false"] * (len(topics) - len(results)))

        return [result.strip().lower() == "true" for result in results[: len(topics)]]

    def calculate_score(
        self, answer: str, context: Union[str, List[str]]
    ) -> Dict[str, Union[float, List[str], int]]:
        """
        Calculate faithfulness score for a RAG answer.

        Args:
            answer: The generated answer to evaluate
            context: The context(s) used to generate the answer

        Returns:
            Dictionary with faithfulness scores and analysis
        """
        # Normalize context if it's a list
        context_text = self._normalize_context(context)

        # Extract and verify claims from the answer (using "summary" context since
        # answers are generated content similar to summaries)
        answer_claims = self._extract_claims(answer, context="summary")
        claims_verification = (
            self._verify_claims_batch(answer_claims, context_text)
            if answer_claims
            else []
        )
        verified_claims_count = sum(claims_verification)

        # Extract and verify topics
        answer_topics = self._extract_topics(answer)
        topics_verification = (
            self._verify_topics_batch(answer_topics, context_text)
            if answer_topics
            else []
        )

        # Identify missing topics
        topics_not_found = [
            topic
            for topic, is_present in zip(answer_topics, topics_verification)
            if not is_present
        ]

        # Calculate scores
        claims_score = (
            verified_claims_count / len(answer_claims) if answer_claims else 1.0
        )
        topics_score = (
            sum(topics_verification) / len(answer_topics) if answer_topics else 1.0
        )

        # Combined faithfulness score (average of claims and topics scores)
        faithfulness_score = (claims_score + topics_score) / 2

        result = {
            "faithfulness": faithfulness_score,
            "claims_score": claims_score,
            "topics_score": topics_score,
            "claims_count": len(answer_claims),
            "verified_claims_count": verified_claims_count,
            "topics_count": len(answer_topics),
            "verified_topics_count": sum(topics_verification),
        }

        # Include detailed claim/topic-level analysis when verbose is enabled
        if self.verbose:
            result["claims_analysis"] = [
                {"claim": claim, "is_verified": is_verified}
                for claim, is_verified in zip(answer_claims, claims_verification)
            ]
            result["topics_analysis"] = [
                {"topic": topic, "is_in_context": is_present}
                for topic, is_present in zip(answer_topics, topics_verification)
            ]
            result["topics_not_found_in_context"] = topics_not_found

        return result


def calculate_rag_faithfulness(
    answer: str, context: Union[str, List[str]], llm_config: LLMConfig,
    verbose: bool = False
) -> Dict[str, Union[float, List[str], int]]:
    """
    Calculate faithfulness score by comparing claims and topics in the answer against the provided context.

    Args:
        answer (str): The generated answer to evaluate
        context (Union[str, List[str]]): The context(s) used to generate the answer
        llm_config (LLMConfig): Configuration for the LLM to use
        verbose (bool): If True, include detailed claim and topic-level analysis

    Returns:
        Dict containing:
            - faithfulness: Combined faithfulness score (0-1)
            - claims_score: Score based on claim verification
            - topics_score: Score based on topic verification
            - claims_count: Number of claims extracted
            - verified_claims_count: Number of claims verified
            - topics_count: Number of topics extracted
            - verified_topics_count: Number of topics found in context
            - claims_analysis (only if verbose=True): List of claims with verification status
            - topics_analysis (only if verbose=True): List of topics with context status
            - topics_not_found_in_context (only if verbose=True): List of missing topics
    """
    calculator = RAGFaithfulnessCalculator(llm_config, verbose=verbose)
    return calculator.calculate_score(answer, context)
