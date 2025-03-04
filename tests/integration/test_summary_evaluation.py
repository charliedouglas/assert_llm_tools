import pytest
from unittest.mock import patch, MagicMock
from assert_llm_tools.core import evaluate_summary, AVAILABLE_SUMMARY_METRICS


# Mock the individual metric calculation functions
@pytest.fixture
def mock_metric_functions():
    with patch("assert_llm_tools.core.calculate_rouge") as mock_rouge, patch(
        "assert_llm_tools.core.calculate_bleu"
    ) as mock_bleu, patch(
        "assert_llm_tools.core.calculate_bert_score"
    ) as mock_bert, patch(
        "assert_llm_tools.core.calculate_faithfulness"
    ) as mock_faith, patch(
        "assert_llm_tools.core.calculate_topic_preservation"
    ) as mock_topic, patch(
        "assert_llm_tools.core.calculate_redundancy"
    ) as mock_redundancy, patch(
        "assert_llm_tools.core.calculate_conciseness_score"
    ) as mock_conciseness, patch(
        "assert_llm_tools.core.calculate_bart_score"
    ) as mock_bart, patch(
        "assert_llm_tools.core.calculate_coherence"
    ) as mock_coherence, patch(
        "assert_llm_tools.core.calculate_comet_score"
    ) as mock_comet, patch(
        "assert_llm_tools.core.calculate_comet_qe_score"
    ) as mock_comet_qe:

        # Configure mock return values
        mock_rouge.return_value = {
            "rouge1_fmeasure": 0.75,
            "rouge2_fmeasure": 0.6,
            "rougeL_fmeasure": 0.7,
        }
        mock_bleu.return_value = 0.65
        mock_bert.return_value = {
            "bert_score_precision": 0.8,
            "bert_score_recall": 0.7,
            "bert_score_f1": 0.75,
        }
        mock_faith.return_value = {"faithfulness": 0.9}
        mock_topic.return_value = {"topic_preservation": 0.8}
        mock_redundancy.return_value = {"redundancy_score": 0.95}
        mock_conciseness.return_value = 0.85
        mock_bart.return_value = {"bart_score": 0.7}
        mock_coherence.return_value = {"coherence": 0.8}
        mock_comet.return_value = {"comet_score": 0.75}
        mock_comet_qe.return_value = {"comet_qe_score": 0.7}

        yield {
            "rouge": mock_rouge,
            "bleu": mock_bleu,
            "bert_score": mock_bert,
            "faithfulness": mock_faith,
            "topic_preservation": mock_topic,
            "redundancy": mock_redundancy,
            "conciseness": mock_conciseness,
            "bart_score": mock_bart,
            "coherence": mock_coherence,
            "comet_score": mock_comet,
            "comet_qe_score": mock_comet_qe,
        }


def test_evaluate_summary_all_metrics(sample_texts, mock_metric_functions, llm_config):
    """Test that evaluate_summary calls all specified metrics and combines results correctly."""
    # Get sample texts
    full_text = sample_texts["source_text"]
    summary = sample_texts["summary"]

    # Run evaluation with all metrics
    results = evaluate_summary(
        full_text=full_text,
        summary=summary,
        metrics=AVAILABLE_SUMMARY_METRICS,
        llm_config=llm_config,
    )

    # Verify all metrics were called
    for metric_name, mock_func in mock_metric_functions.items():
        mock_func.assert_called_once()

    # Check that all expected metrics are in the results
    expected_metrics = [
        "rouge1_fmeasure",
        "rouge2_fmeasure",
        "rougeL_fmeasure",
        "bleu",
        "bert_score_precision",
        "bert_score_recall",
        "bert_score_f1",
        "faithfulness",
        "topic_preservation",
        "redundancy_score",
        "conciseness",
        "bart_score",
        "coherence",
        "comet_score",
        "comet_qe_score",
    ]

    for metric in expected_metrics:
        assert metric in results

    # Check specific values
    assert results["bleu"] == 0.65
    assert results["faithfulness"] == 0.9
    assert results["conciseness"] == 0.85


def test_evaluate_summary_subset_metrics(sample_texts, mock_metric_functions):
    """Test evaluate_summary with a subset of metrics."""
    full_text = sample_texts["source_text"]
    summary = sample_texts["summary"]

    # Select only non-LLM metrics
    selected_metrics = ["rouge", "bleu", "bert_score", "bart_score"]

    # Run evaluation with selected metrics
    results = evaluate_summary(
        full_text=full_text, summary=summary, metrics=selected_metrics
    )

    # Verify only selected metrics were called
    for metric_name, mock_func in mock_metric_functions.items():
        if metric_name in selected_metrics:
            mock_func.assert_called_once()
        else:
            mock_func.assert_not_called()

    # Check that only expected metrics are in the results
    unexpected_metrics = [
        "faithfulness",
        "topic_preservation",
        "redundancy_score",
        "conciseness",
        "coherence",
        "comet_score",
        "comet_qe_score",
    ]

    for metric in unexpected_metrics:
        assert metric not in results


def test_evaluate_summary_invalid_metrics(sample_texts):
    """Test that evaluate_summary raises an error for invalid metrics."""
    full_text = sample_texts["source_text"]
    summary = sample_texts["summary"]

    # Try with an invalid metric
    with pytest.raises(ValueError, match="Invalid metrics"):
        evaluate_summary(
            full_text=full_text, summary=summary, metrics=["rouge", "invalid_metric"]
        )


def test_evaluate_summary_llm_required(sample_texts, llm_config):
    """Test that evaluate_summary requires LLM config for LLM-based metrics."""
    full_text = sample_texts["source_text"]
    summary = sample_texts["summary"]

    # Try with LLM-based metrics but no LLM config
    with pytest.raises(ValueError, match="LLM configuration required"):
        evaluate_summary(
            full_text=full_text,
            summary=summary,
            metrics=["faithfulness", "topic_preservation"],
        )

    # Should work with LLM config provided
    with patch("assert_llm_tools.core.calculate_faithfulness") as mock_faith, patch(
        "assert_llm_tools.core.calculate_topic_preservation"
    ) as mock_topic:

        mock_faith.return_value = {"faithfulness": 0.9}
        mock_topic.return_value = {"topic_preservation": 0.8}

        results = evaluate_summary(
            full_text=full_text,
            summary=summary,
            metrics=["faithfulness", "topic_preservation"],
            llm_config=llm_config,
        )

        assert "faithfulness" in results
        assert "topic_preservation" in results
