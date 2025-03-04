import pytest
from assert_llm_tools.utils import (
    preprocess_text,
    remove_stopwords,
    add_custom_stopwords,
    get_all_stopwords,
)
from unittest.mock import patch


def test_preprocess_text():
    """Test text preprocessing functionality."""
    # Test whitespace normalization
    assert preprocess_text("  hello   world  ") == "hello world"

    # Test lowercase conversion
    assert preprocess_text("Hello World") == "hello world"

    # Test combination of both
    assert preprocess_text("  Hello   WORLD  ") == "hello world"


def test_remove_stopwords():
    """Test stopword removal functionality."""
    # Setup - ensure stopwords are initialized
    with patch("assert_llm_tools.utils.get_all_stopwords") as mock_get_stopwords:
        mock_get_stopwords.return_value = {"the", "and", "is", "in", "of"}

        # Test basic stopword removal
        text = "the cat and dog is in the house of dreams"
        result = remove_stopwords(text)
        assert result == "cat dog house dreams"

        # Test with no stopwords in text
        text = "cat dog house dreams"
        result = remove_stopwords(text)
        assert result == "cat dog house dreams"


def test_add_custom_stopwords():
    """Test adding custom stopwords."""
    # Add custom stopwords
    add_custom_stopwords(["cat", "dog"])

    # Get all stopwords and check if our custom words are included
    all_stopwords = get_all_stopwords()
    assert "cat" in all_stopwords
    assert "dog" in all_stopwords

    # Test case insensitivity
    add_custom_stopwords(["HOUSE"])
    all_stopwords = get_all_stopwords()
    assert "house" in all_stopwords
