import pytest
from assert_llm_tools.metrics.summary.rouge import calculate_rouge


def test_rouge_calculation():
    """Test ROUGE score calculation"""
    reference = "The cat sat on the mat."
    candidate = "The cat sat on the mat."
    
    # Perfect match should give scores of 1.0
    scores = calculate_rouge(reference, candidate)
    
    # Check that we have all expected metrics
    expected_metrics = [
        "rouge1_precision", "rouge1_recall", "rouge1_fmeasure",
        "rouge2_precision", "rouge2_recall", "rouge2_fmeasure",
        "rougeL_precision", "rougeL_recall", "rougeL_fmeasure",
    ]
    
    for metric in expected_metrics:
        assert metric in scores
        assert scores[metric] == 1.0


def test_rouge_partial_match():
    """Test ROUGE scores with partial matches"""
    reference = "The quick brown fox jumps over the lazy dog."
    candidate = "The brown fox jumps over the dog."
    
    scores = calculate_rouge(reference, candidate)
    
    # Partial match should have scores between 0 and 1
    for key, value in scores.items():
        assert 0 <= value <= 1
    
    # Rouge1 should be higher than Rouge2 for this partial match
    assert scores["rouge1_fmeasure"] > scores["rouge2_fmeasure"]


def test_rouge_no_match():
    """Test ROUGE scores with completely different texts"""
    reference = "The quick brown fox jumps over the lazy dog."
    candidate = "A completely different sentence with no overlap."
    
    scores = calculate_rouge(reference, candidate)
    
    # Rouge2 should be 0 (no bigram matches)
    assert scores["rouge2_precision"] == 0
    assert scores["rouge2_recall"] == 0
    assert scores["rouge2_fmeasure"] == 0
    
    # Rouge1 might have some matches due to common words like "the"
    # but should be very low
    assert scores["rouge1_fmeasure"] < 0.2