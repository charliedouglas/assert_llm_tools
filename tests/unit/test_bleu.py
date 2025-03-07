import pytest
from assert_llm_tools.metrics.summary.bleu import calculate_bleu


def test_bleu_exact_match():
    """Test BLEU score calculation with exact match"""
    reference = "The cat sat on the mat."
    candidate = "The cat sat on the mat."
    
    score = calculate_bleu(reference, candidate)
    assert score == 1.0


def test_bleu_partial_match():
    """Test BLEU score calculation with partial match"""
    reference = "The quick brown fox jumps over the lazy dog."
    candidate = "The brown fox jumps over the dog."
    
    score = calculate_bleu(reference, candidate)
    assert 0 < score < 1


def test_bleu_no_match():
    """Test BLEU score calculation with no match"""
    reference = "The quick brown fox jumps over the lazy dog."
    candidate = "A completely different sentence with no overlap."
    
    score = calculate_bleu(reference, candidate)
    assert score < 0.1  # Should be very low but might not be exactly 0 due to smoothing


def test_bleu_handles_case_differences():
    """Test that BLEU score ignores case differences"""
    reference = "The cat sat on the mat."
    candidate = "THE CAT SAT ON THE MAT."
    
    score = calculate_bleu(reference, candidate)
    assert score == 1.0


def test_bleu_weighting():
    """Test that BLEU weights affect the score"""
    reference = "The quick brown fox jumps over the lazy dog."
    
    # This matches all unigrams but not the bigram order
    scrambled = "dog lazy the over jumps fox brown quick The."
    
    # Since our implementation weighs unigrams more heavily (0.7, 0.3, 0, 0)
    # we should still get a decent score despite word order differences
    score = calculate_bleu(reference, scrambled)
    assert score > 0.5