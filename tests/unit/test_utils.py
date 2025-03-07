import pytest
from assert_llm_tools.utils import (
    add_custom_stopwords,
    get_all_stopwords,
    preprocess_text,
    remove_stopwords,
    initialize_nltk,
)


def test_initialize_nltk():
    """Test that NLTK initialization doesn't raise errors"""
    initialize_nltk()  # Should not raise exceptions


def test_add_custom_stopwords():
    """Test adding custom stopwords"""
    # Clear any existing custom stopwords from previous tests
    from assert_llm_tools.utils import _custom_stopwords
    _custom_stopwords.clear()
    
    # Add some custom stopwords
    custom_words = ["test", "custom", "UPPER"]
    add_custom_stopwords(custom_words)
    
    # Get all stopwords
    all_stopwords = get_all_stopwords()
    
    # Check that our custom words are included (case insensitive)
    assert "test" in all_stopwords
    assert "custom" in all_stopwords
    assert "upper" in all_stopwords  # Should be lowercase


def test_preprocess_text():
    """Test text preprocessing functionality"""
    # Test whitespace normalization
    text = "This   has    extra   spaces"
    processed = preprocess_text(text)
    assert processed == "this has extra spaces"
    
    # Test case normalization
    text = "This Has UPPER and lower CASE"
    processed = preprocess_text(text)
    assert processed == "this has upper and lower case"


def test_remove_stopwords():
    """Test removing stopwords from text"""
    # Clear any existing custom stopwords from previous tests
    from assert_llm_tools.utils import _custom_stopwords
    _custom_stopwords.clear()
    
    # Add a custom stopword
    add_custom_stopwords(["custom"])
    
    # Test with a mix of standard and custom stopwords
    text = "this is a test with custom stopwords"
    # 'this', 'is', 'a', 'with', and 'custom' should be removed
    processed = remove_stopwords(text)
    
    # Check result
    assert "this" not in processed.split()
    assert "is" not in processed.split()
    assert "a" not in processed.split()
    assert "with" not in processed.split()
    assert "custom" not in processed.split()
    assert "test" in processed.split()
    assert "stopwords" in processed.split()