from typing import Dict, List, Optional, Union
from assert_llm_tools.llm.config import LLMConfig
from assert_llm_tools.llm.bedrock import BedrockLLM
from assert_llm_tools.llm.openai import OpenAILLM


class RAGFaithfulnessCalculator:
    def __init__(self, llm_config: LLMConfig):
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

    def _verify_claims_batch(
        self, claims: List[str], context: Union[str, List[str]]
    ) -> List[bool]:
        # If context is a list, join with newlines
        if isinstance(context, list):
            context = "\n\n".join(context)

        claims_text = "\n".join(
            f"Claim {i+1}: {claim}" for i, claim in enumerate(claims)
        )
        prompt = f"""
        System: You are a helpful assistant that verifies if claims can be directly supported by the given context. 
        For each claim, answer with only 'true' or 'false'.

        Context: {context}

        Claims to verify:
        {claims_text}

        For each claim, answer with only 'true' or 'false', one per line.

        Assistant:"""

        response = self.llm.generate(prompt, max_tokens=300)
        results = response.strip().split("\n")
        return [result.strip().lower() == "true" for result in results]

    def _extract_topics(self, text: str) -> List[str]:
        prompt = f"""
        System: You are a helpful assistant that extracts main topics from text. Extract all key topics or subjects mentioned. Output each topic on a new line. Be specific but concise.

        Human: Here is the text to analyze:
        {text}

        Please list all key topics, one per line.

        Assistant: Here are the key topics:"""

        response = self.llm.generate(prompt, max_tokens=500)
        topics = response.strip().split("\n")
        return [topic.strip() for topic in topics if topic.strip()]

    def _verify_topics_batch(
        self, topics: List[str], context: Union[str, List[str]]
    ) -> List[bool]:
        if isinstance(context, list):
            context = "\n\n".join(context)

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


def calculate_faithfulness(
    answer: str, context: Union[str, List[str]], llm_config: LLMConfig
) -> Dict[str, Union[float, List[str], int]]:
    """
    Calculate faithfulness score by comparing claims and topics in the answer against the provided context.

    Args:
        answer (str): The generated answer to evaluate
        context (Union[str, List[str]]): The context(s) used to generate the answer
        llm_config (LLMConfig): Configuration for the LLM to use

    Returns:
        Dict containing faithfulness scores, claim counts, and topic analysis
    """
    calculator = RAGFaithfulnessCalculator(llm_config)

    # Extract and verify claims
    answer_claims = calculator._extract_claims(answer)
    claims_verification = (
        calculator._verify_claims_batch(answer_claims, context) if answer_claims else []
    )
    verified_claims_count = sum(claims_verification)

    # Extract and verify topics
    answer_topics = calculator._extract_topics(answer)
    topics_verification = (
        calculator._verify_topics_batch(answer_topics, context) if answer_topics else []
    )

    # Identify missing topics
    topics_not_found = [
        topic
        for topic, is_present in zip(answer_topics, topics_verification)
        if not is_present
    ]

    # Calculate scores
    claims_score = (
        (verified_claims_count / len(answer_claims)) if answer_claims else 1.0
    )
    topics_score = (
        (sum(topics_verification) / len(answer_topics)) if answer_topics else 1.0
    )

    # Combined faithfulness score (average of claims and topics scores)
    faithfulness_score = (claims_score + topics_score) / 2

    return {
        "faithfulness": faithfulness_score,
        "claims_score": claims_score,
        "topics_score": topics_score,
        "claims_count": len(answer_claims),
        "verified_claims_count": verified_claims_count,
        "topics_count": len(answer_topics),
        "verified_topics_count": sum(topics_verification),
        "topics_found": answer_topics,
        "topics_not_found_in_context": topics_not_found,
    }
