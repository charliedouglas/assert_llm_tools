import pytest
from assert_llm_tools.metrics.summary.rouge import calculate_rouge


def test_calculate_rouge(sample_texts):
    """Test ROUGE score calculation."""
    # Get the sample texts from fixture
    reference = sample_texts["source_text"]
    candidate = sample_texts["summary"]

    # Calculate ROUGE scores
    rouge_scores = calculate_rouge(reference, candidate)

    # Check that all expected metrics are present
    expected_metrics = [
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_fmeasure",
        "rouge2_precision",
        "rouge2_recall",
        "rouge2_fmeasure",
        "rougeL_precision",
        "rougeL_recall",
        "rougeL_fmeasure",
    ]
    for metric in expected_metrics:
        assert metric in rouge_scores

    # Check that all scores are within the expected range (0-1)
    for metric, score in rouge_scores.items():
        assert 0 <= score <= 1

    # Check specific behavior: identical text should have perfect scores
    perfect_scores = calculate_rouge(reference, reference)
    assert perfect_scores["rouge1_fmeasure"] == 1.0
    assert perfect_scores["rouge2_fmeasure"] == 1.0

    # Check that completely different texts have low scores
    different_text = "This is a completely different text with no overlap."
    low_scores = calculate_rouge(reference, different_text)
    assert low_scores["rouge1_fmeasure"] < 0.3  # Threshold may need adjustment
    assert low_scores["rouge2_fmeasure"] < 0.1  # Bi-grams should have very low overlap
