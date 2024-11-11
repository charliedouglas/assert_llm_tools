from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from ... import initialize_nltk


def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score for a candidate summary against a reference text.
    Args:
        reference: The ground truth text
        candidate: The text to evaluate
    Returns:
        float: BLEU score between 0 and 1
    """
    # Ensure NLTK is initialized before calculating scores
    initialize_nltk()

    # Initialize smoothing function
    smoothie = SmoothingFunction().method1

    # Tokenize both texts
    reference_tokens = [word_tokenize(reference.lower())]
    candidate_tokens = word_tokenize(candidate.lower())

    # Use weights that emphasize unigrams and bigrams more
    weights = (0.7, 0.3, 0, 0)  # Focus on 1-grams and 2-grams only

    return sentence_bleu(
        reference_tokens, candidate_tokens, weights=weights, smoothing_function=smoothie
    )
