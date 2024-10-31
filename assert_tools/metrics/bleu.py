from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from .. import initialize_nltk


def calculate_bleu(reference: str, candidate: str) -> float:
    """
    Calculate BLEU score for a candidate summary against a reference text.
    """
    # Ensure NLTK is initialized before calculating scores
    initialize_nltk()

    reference_tokens = [word_tokenize(reference)]
    candidate_tokens = word_tokenize(candidate)

    return sentence_bleu(reference_tokens, candidate_tokens)
