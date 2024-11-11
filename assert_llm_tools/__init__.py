# First, initialize NLTK
from .utils import initialize_nltk

initialize_nltk()

# Then import the rest
from .core import evaluate_summary, AVAILABLE_SUMMARY_METRICS
from .llm.config import LLMConfig

__all__ = ["evaluate_summary", "AVAILABLE_SUMMARY_METRICS", "LLMConfig"]
