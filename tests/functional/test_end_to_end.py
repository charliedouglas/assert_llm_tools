import pytest
from unittest.mock import patch, MagicMock
from assert_llm_tools.core import evaluate_summary, evaluate_rag
from assert_llm_tools.llm.config import LLMConfig


@pytest.mark.functional
def test_end_to_end_summary_evaluation(sample_texts):
    """
    Test end-to-end summary evaluation with basic metrics (no LLM required).
    This is a functional test that verifies the basic pipeline works.
    """
    # Use only non-LLM metrics to avoid requiring actual API access
    metrics = ["rouge", "bleu", "bert_score"]

    # Get sample texts
    full_text = sample_texts["source_text"]
    summary = sample_texts["summary"]

    # Run evaluation
    results = evaluate_summary(
        full_text=full_text,
        summary=summary,
        metrics=metrics,
        show_progress=False,  # Disable progress bar in tests
    )

    # Verify we got results for all expected metrics
    assert "rouge1_fmeasure" in results
    assert "rouge2_fmeasure" in results
    assert "rougeL_fmeasure" in results
    assert "bleu" in results
    assert "bert_score_f1" in results

    # Check that scores are within expected ranges
    for metric, score in results.items():
        assert 0 <= score <= 1, f"Score for {metric} out of range: {score}"


@pytest.mark.functional
def test_end_to_end_rag_evaluation(sample_texts):
    """
    Test end-to-end RAG evaluation with mocked LLM.
    This test verifies the RAG evaluation pipeline works.
    """
    # Mock the LLM to avoid actual API calls
    with patch("assert_llm_tools.llm.openai.OpenAI") as mock_openai:
        # Configure the mock
        instance = mock_openai.return_value
        mock_message = MagicMock()
        mock_message.content = "0.8\nThe answer is highly relevant to the question."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        instance.chat.completions.create.return_value.choices = [mock_choice]

        # Create LLM config
        llm_config = LLMConfig(
            provider="openai", model_id="gpt-3.5-turbo", api_key="test_key"
        )

        # Get sample data
        question = sample_texts["question"]
        answer = sample_texts["answer"]
        context = sample_texts["context"]

        # Test with just one metric to simplify
        metrics = ["answer_relevance"]

        # Run RAG evaluation
        results = evaluate_rag(
            question=question,
            answer=answer,
            context=context,
            llm_config=llm_config,
            metrics=metrics,
            show_progress=False,
        )

        # Verify results
        assert "answer_relevance" in results
        assert results["answer_relevance"] == 0.8  # This should match our mock response
