import pytest
from unittest.mock import patch, MagicMock
from assert_llm_tools.core import evaluate_rag


@patch("assert_llm_tools.metrics.rag.answer_relevance.evaluate_with_llm")
def test_evaluate_rag_answer_relevance(mock_evaluate, rag_question, rag_answer, rag_context, mock_llm_config, mocked_responses):
    """Test evaluate_rag with answer_relevance metric"""
    # Configure mock
    mock_evaluate.return_value = mocked_responses["answer_relevance"]
    
    # Call with answer_relevance metric
    metrics = ["answer_relevance"]
    results = evaluate_rag(
        rag_question,
        rag_answer,
        rag_context,
        llm_config=mock_llm_config,
        metrics=metrics,
        show_progress=False
    )
    
    # Verify the mock was called
    mock_evaluate.assert_called_once()
    
    # Check that the answer_relevance score is in the results
    assert "answer_relevance" in results
    assert results["answer_relevance"] == mocked_responses["answer_relevance"]["score"]


@patch("assert_llm_tools.metrics.rag.answer_attribution.evaluate_with_llm")
def test_evaluate_rag_answer_attribution(mock_evaluate, rag_question, rag_answer, rag_context, mock_llm_config, mocked_responses):
    """Test evaluate_rag with answer_attribution metric"""
    # Configure mock
    mock_evaluate.return_value = mocked_responses["answer_attribution"]
    
    # Call with answer_attribution metric
    metrics = ["answer_attribution"]
    results = evaluate_rag(
        rag_question,
        rag_answer,
        rag_context,
        llm_config=mock_llm_config,
        metrics=metrics,
        show_progress=False
    )
    
    # Verify the mock was called
    mock_evaluate.assert_called_once()
    
    # Check that the answer_attribution score is in the results
    assert "answer_attribution" in results
    assert results["answer_attribution"] == mocked_responses["answer_attribution"]["score"]


def test_evaluate_rag_missing_llm_config(rag_question, rag_answer, rag_context):
    """Test evaluate_rag requires LLM config for all metrics"""
    with pytest.raises(ValueError, match="LLM configuration required for metrics"):
        evaluate_rag(
            rag_question,
            rag_answer,
            rag_context,
            llm_config=None,  # Missing LLM config
            metrics=["answer_relevance"],
            show_progress=False
        )


def test_evaluate_rag_invalid_metrics(rag_question, rag_answer, rag_context, mock_llm_config):
    """Test evaluate_rag raises error with invalid metrics"""
    with pytest.raises(ValueError, match="Invalid metrics"):
        evaluate_rag(
            rag_question,
            rag_answer,
            rag_context,
            llm_config=mock_llm_config,
            metrics=["invalid_metric"],
            show_progress=False
        )


@patch("assert_llm_tools.metrics.rag.answer_relevance.evaluate_with_llm")
@patch("assert_llm_tools.metrics.rag.context_relevance.evaluate_with_llm")
@patch("assert_llm_tools.metrics.rag.faithfulness.evaluate_with_llm")
def test_evaluate_rag_multiple_metrics(
    mock_faith, mock_context, mock_answer,
    rag_question, rag_answer, rag_context, mock_llm_config, mocked_responses
):
    """Test evaluate_rag with multiple metrics"""
    # Configure mocks
    mock_answer.return_value = mocked_responses["answer_relevance"]
    mock_context.return_value = mocked_responses["context_relevance"]
    mock_faith.return_value = mocked_responses["rag_faithfulness"]
    
    # Call with multiple metrics
    metrics = ["answer_relevance", "context_relevance", "faithfulness"]
    results = evaluate_rag(
        rag_question,
        rag_answer,
        rag_context,
        llm_config=mock_llm_config,
        metrics=metrics,
        show_progress=False
    )
    
    # Verify all mocks were called
    mock_answer.assert_called_once()
    mock_context.assert_called_once()
    mock_faith.assert_called_once()
    
    # Check that all metrics are in the results
    assert "answer_relevance" in results
    assert "context_relevance" in results
    assert "faithfulness" in results
    
    # Check values match our mocked responses
    assert results["answer_relevance"] == mocked_responses["answer_relevance"]["score"]
    assert results["context_relevance"] == mocked_responses["context_relevance"]["score"]
    assert results["faithfulness"] == mocked_responses["rag_faithfulness"]["score"]


def test_evaluate_rag_with_list_context(rag_question, rag_answer, mock_llm_config):
    """Test evaluate_rag with context as a list of strings"""
    with patch("assert_llm_tools.metrics.rag.context_relevance.evaluate_with_llm") as mock_context:
        # Configure mock
        mock_context.return_value = {"score": 0.75, "reasoning": "Mock reasoning"}
        
        # Create a list context
        context_list = [
            "The Eiffel Tower was completed in 1889.",
            "It stands 324 meters tall.",
            "It is located in Paris, France."
        ]
        
        # Call with context as a list
        results = evaluate_rag(
            rag_question,
            rag_answer,
            context_list,
            llm_config=mock_llm_config,
            metrics=["context_relevance"],
            show_progress=False
        )
        
        # Verify the mock was called (implying the function handled the list properly)
        mock_context.assert_called_once()
        assert "context_relevance" in results