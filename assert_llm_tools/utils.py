def preprocess_text(text: str) -> str:
    """
    Preprocess text by cleaning and normalizing it.
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    # Convert to lowercase
    text = text.lower()
    return text
