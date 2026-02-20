import warnings

warnings.warn(
    "assert_llm_tools is deprecated and will no longer receive updates. "
    "Please migrate to the successor packages:\n"
    "  - assert-eval  (pip install assert-eval)  for summary evaluation\n"
    "  - assert-review (pip install assert-review) for compliance note evaluation\n"
    "See https://github.com/charliedouglas/assert for migration details.",
    DeprecationWarning,
    stacklevel=2,
)

# Import the core functionality
from .core import evaluate_summary, AVAILABLE_SUMMARY_METRICS, evaluate_note
from .llm.config import LLMConfig
from .utils import initialize_nltk
from .metrics.note.models import GapReport, GapItem, GapReportStats, PassPolicy

__all__ = [
    "evaluate_summary",
    "AVAILABLE_SUMMARY_METRICS",
    "LLMConfig",
    "initialize_nltk",
    # Note evaluation
    "evaluate_note",
    "GapReport",
    "GapItem",
    "GapReportStats",
    "PassPolicy",
]
