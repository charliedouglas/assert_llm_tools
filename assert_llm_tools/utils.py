from nltk.corpus import stopwords
from typing import Set, Optional, List

_custom_stopwords: Set[str] = set()


def add_custom_stopwords(words: List[str]) -> None:
    """
    Add custom words to the stopwords list.

    Args:
        words (List[str]): List of words to add to stopwords
    """
    global _custom_stopwords
    _custom_stopwords.update(set(word.lower() for word in words))


def get_all_stopwords() -> Set[str]:
    """
    Get combined set of NLTK and custom stopwords.

    Returns:
        Set[str]: Combined set of stopwords
    """
    return set(stopwords.words("english")).union(_custom_stopwords)


def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning and normalizing it.
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Convert to lowercase
    text = text.lower()
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove stopwords from text using NLTK's stopwords list and custom stopwords.

    Args:
        text (str): Input text to process

    Returns:
        str: Text with stopwords removed
    """
    stop_words = get_all_stopwords()
    return " ".join([word for word in text.split() if word not in stop_words])
