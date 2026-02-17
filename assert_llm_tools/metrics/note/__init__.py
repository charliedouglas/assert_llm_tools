"""
assert_llm_tools.metrics.note
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compliance note evaluation against regulatory framework definitions.

Public exports:
    evaluate_note   — top-level function; returns a GapReport
    NoteEvaluator   — class; extend or instantiate directly for advanced use
"""
from .evaluate_note import NoteEvaluator, evaluate_note

__all__ = ["evaluate_note", "NoteEvaluator"]
