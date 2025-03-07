import pytest
from assert_llm_tools.llm.config import LLMConfig

@pytest.fixture
def sample_text():
    return """
    The James Webb Space Telescope (JWST) has revolutionized our understanding of the cosmos since its launch in 2021. 
    As the largest and most powerful space telescope ever built, it has provided unprecedented views of distant galaxies, 
    exoplanets, and cosmic phenomena. The telescope's infrared capabilities allow it to peer through cosmic dust and gas, 
    revealing previously hidden details about star formation and galaxy evolution. Scientists have already used JWST data 
    to make groundbreaking discoveries, including observations of some of the earliest galaxies formed after the Big Bang 
    and detailed atmospheric analysis of potentially habitable exoplanets.
    """

@pytest.fixture
def sample_summary():
    return """
    The James Webb Space Telescope, launched in 2021, is revolutionizing space observation with its powerful infrared 
    capabilities, enabling scientists to study early galaxies and exoplanets in unprecedented detail.
    """

@pytest.fixture
def rag_question():
    return "What is the Eiffel Tower?"

@pytest.fixture
def rag_context():
    return "The Eiffel Tower was completed in 1889. It stands 324 meters tall and is located in Paris, France."

@pytest.fixture
def rag_answer():
    return "The Eiffel Tower, located in Paris, was completed in 1889 and reaches a height of 324 meters."

@pytest.fixture
def mock_llm_config():
    """Mock LLM config that doesn't make real API calls for testing"""
    return LLMConfig(
        provider="openai",
        model_id="gpt-3.5-turbo",
        api_key="test_key"
    )

@pytest.fixture
def mocked_responses():
    """Dictionary of mocked responses for different prompt types"""
    return {
        "faithfulness": {
            "score": 0.85,
            "reasoning": "The summary accurately captures the key points..."
        },
        "topic_preservation": {
            "score": 0.9,
            "reasoning": "The summary maintains the most important topics..."
        },
        "redundancy": {
            "score": 0.95,
            "reasoning": "The summary contains no repeated information..."
        },
        "conciseness": {
            "score": 0.8,
            "reasoning": "The summary is concise and to the point..."
        },
        "coherence": {
            "score": 0.88,
            "reasoning": "The summary flows logically..."
        },
        "answer_relevance": {
            "score": 0.87,
            "reasoning": "The answer directly addresses the question..."
        },
        "context_relevance": {
            "score": 0.82,
            "reasoning": "The provided context is highly relevant..."
        },
        "answer_attribution": {
            "score": 0.93,
            "reasoning": "All information in the answer is supported by the context..."
        },
        "rag_faithfulness": {
            "score": 0.91,
            "reasoning": "The answer contains information present in the context..."
        },
        "completeness": {
            "score": 0.89,
            "reasoning": "The answer fully addresses all aspects of the question..."
        }
    }