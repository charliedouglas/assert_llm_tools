import pytest
from unittest.mock import patch, MagicMock
from assert_llm_tools.core import evaluate_summary


def test_evaluate_summary_no_llm_metrics(sample_text, sample_summary):
    """Test evaluate_summary with metrics that don't require LLM"""
    metrics = ["rouge", "bleu", "bert_score"]
    
    results = evaluate_summary(
        sample_text,
        sample_summary,
        metrics=metrics,
        show_progress=False
    )
    
    # Check that each of the expected metrics is in the results
    assert "rouge1_precision" in results
    assert "rouge1_recall" in results
    assert "rouge1_fmeasure" in results
    assert "rouge2_precision" in results
    assert "bleu" in results
    assert "bert_score_precision" in results
    assert "bert_score_recall" in results
    assert "bert_score_f1" in results
    
    # Check that scores are in the expected range
    for key, value in results.items():
        assert 0 <= value <= 1


@patch("assert_llm_tools.metrics.summary.faithfulness.evaluate_with_llm")
def test_evaluate_summary_with_llm_metrics(mock_evaluate, sample_text, sample_summary, mock_llm_config, mocked_responses):
    """Test evaluate_summary with metrics that require LLM"""
    # Configure mock to return expected response
    mock_evaluate.return_value = mocked_responses["faithfulness"]
    
    # Call with faithfulness metric which requires LLM
    metrics = ["faithfulness"]
    results = evaluate_summary(
        sample_text,
        sample_summary,
        metrics=metrics,
        llm_config=mock_llm_config,
        show_progress=False
    )
    
    # Verify the mock was called
    mock_evaluate.assert_called_once()
    
    # Check that the faithfulness score is in the results
    assert "faithfulness" in results
    assert results["faithfulness"] == mocked_responses["faithfulness"]["score"]


def test_evaluate_summary_missing_llm_config():
    """Test evaluate_summary raises error when LLM metrics requested but no config provided"""
    with pytest.raises(ValueError, match="LLM configuration required for metrics"):
        evaluate_summary(
            "Some text",
            "A summary",
            metrics=["faithfulness"],  # This requires LLM
            show_progress=False
        )


def test_evaluate_summary_invalid_metrics():
    """Test evaluate_summary raises error with invalid metrics"""
    with pytest.raises(ValueError, match="Invalid metrics"):
        evaluate_summary(
            "Some text",
            "A summary",
            metrics=["invalid_metric"],
            show_progress=False
        )


@patch("assert_llm_tools.metrics.summary.faithfulness.evaluate_with_llm")
@patch("assert_llm_tools.metrics.summary.topic_preservation.evaluate_with_llm")
def test_evaluate_summary_multiple_metrics(mock_topic, mock_faith, sample_text, sample_summary, mock_llm_config, mocked_responses):
    """Test evaluate_summary with multiple metrics including LLM and non-LLM metrics"""
    # Configure mocks
    mock_faith.return_value = mocked_responses["faithfulness"]
    mock_topic.return_value = mocked_responses["topic_preservation"]
    
    # Call with multiple metrics
    metrics = ["rouge", "faithfulness", "topic_preservation"]
    results = evaluate_summary(
        sample_text,
        sample_summary,
        metrics=metrics,
        llm_config=mock_llm_config,
        show_progress=False
    )
    
    # Verify both LLM-based metrics were called
    mock_faith.assert_called_once()
    mock_topic.assert_called_once()
    
    # Check that all expected metrics are in the results
    assert "rouge1_fmeasure" in results
    assert "faithfulness" in results
    assert "topic_preservation" in results
    
    # Check LLM metric values match our mocked responses
    assert results["faithfulness"] == mocked_responses["faithfulness"]["score"]
    assert results["topic_preservation"] == mocked_responses["topic_preservation"]["score"]