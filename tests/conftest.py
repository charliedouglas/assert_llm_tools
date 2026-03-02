"""
tests/conftest.py
=================

Shared pytest configuration for the tests/ package.

Native dependencies that are not present in the test environment (boto3,
botocore, openai) are stubbed out here at the module level *before* any
assert_llm_tools import can trigger them.  conftest.py is the first thing
pytest loads, which guarantees the stubs are in sys.modules when any test
file is collected and imported.
"""
from __future__ import annotations

import sys
import types


# ── Minimal auto-mock ──────────────────────────────────────────────────────────

class _AutoMock(types.ModuleType):
    """
    Module stub where attribute access returns child stubs and calls return
    stubs.  Just enough to satisfy import-time attribute lookups.
    """

    def __getattr__(self, name: str) -> "_AutoMock":
        child = _AutoMock(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs) -> "_AutoMock":  # noqa: D105
        return _AutoMock("_call_result")


def _stub_modules() -> None:
    """Inject stubs for all native deps the library would otherwise import."""
    for name in (
        "boto3",
        "botocore",
        "botocore.config",
        "botocore.exceptions",
        "openai",
    ):
        if name not in sys.modules:
            sys.modules[name] = _AutoMock(name)

    # botocore.config.Config is instantiated directly → must be a real class
    sys.modules["botocore.config"].Config = type(
        "Config", (), {"__init__": lambda self, **kw: None}
    )

    # openai.OpenAI is instantiated directly → must be a real class
    sys.modules["openai"].OpenAI = type(
        "OpenAI", (), {"__init__": lambda self, **kw: None}
    )


_stub_modules()

# ── Patch BedrockLLM so NoteEvaluator() never calls boto3 ─────────────────────
# This must happen after _stub_modules() but before any test module imports
# NoteEvaluator.

from unittest.mock import MagicMock  # noqa: E402 (after stubs)

import assert_llm_tools.metrics.base as _base_mod  # noqa: E402

_shared_mock_llm = MagicMock(name="shared_conftest_mock_llm")
_base_mod.BedrockLLM = lambda cfg: _shared_mock_llm  # type: ignore[assignment]
