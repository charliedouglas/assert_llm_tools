"""
test_golden_calibration.py — golden dataset calibration tests.

Requires a live LLM. Skipped by default.

Run with:
    RUN_GOLDEN=1 pytest packages/assert-review/tests/golden/ -v -s
    pytest packages/assert-review/tests/golden/ -m golden -v -s

Configure the LLM via environment variables:
    GOLDEN_LLM_PROVIDER   — "bedrock" (default) or "openai"
    GOLDEN_MODEL_ID       — model ID (default: "us.amazon.nova-pro-v1:0")
    GOLDEN_AWS_REGION     — AWS region (default: "us-east-1")
    GOLDEN_OPENAI_API_KEY — required when GOLDEN_LLM_PROVIDER=openai
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import yaml

from assert_review import LLMConfig, evaluate_note

# ── Pytest marker ──────────────────────────────────────────────────────────────

pytestmark = pytest.mark.golden

# ── Skip guard ─────────────────────────────────────────────────────────────────

_SHOULD_RUN = os.environ.get("RUN_GOLDEN", "0").strip() == "1"


def _require_golden():
    if not _SHOULD_RUN:
        pytest.skip("Golden calibration skipped — set RUN_GOLDEN=1 to enable")


# ── Dataset and LLM config ─────────────────────────────────────────────────────

_DATASET_PATH = Path(__file__).parent / "dataset.yaml"

_MAX_CRITICAL_FP_RATE = 0.25  # fail if any critical element exceeds this false-positive rate


def _load_dataset() -> dict:
    with open(_DATASET_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_llm_config() -> LLMConfig:
    provider = os.environ.get("GOLDEN_LLM_PROVIDER", "bedrock")
    return LLMConfig(
        provider=provider,
        model_id=os.environ.get("GOLDEN_MODEL_ID", "us.amazon.nova-pro-v1:0"),
        region=os.environ.get("GOLDEN_AWS_REGION", "us-east-1"),
        api_key=os.environ.get("GOLDEN_OPENAI_API_KEY"),
    )


# ── Status helpers ─────────────────────────────────────────────────────────────

_STATUS_ORDER = {"present": 2, "partial": 1, "missing": 0}


def _is_false_positive(actual: str, expected: str) -> bool:
    """Model said 'present' when expected 'missing' — the dangerous error."""
    return actual == "present" and expected == "missing"


def _is_false_negative(actual: str, expected: str) -> bool:
    """Model said 'missing' when expected 'present' — less dangerous, human catches it."""
    return actual == "missing" and expected == "present"


def _run_note(note_entry: dict, llm_config: LLMConfig) -> dict:
    """Run a single dataset note through evaluate_note and return a result dict."""
    dataset = _load_dataset()
    report = evaluate_note(
        note_text=note_entry["note"],
        framework=dataset["framework_id"],
        llm_config=llm_config,
        verbose=False,
    )
    return {"id": note_entry["id"], "expected": note_entry["expected"], "report": report}


def _assert_note_result(result: dict) -> None:
    """
    Assert the evaluation result against expected outcomes.

    Only hard-asserts on:
    - overall_pass (if specified)
    - critical false positives (model says 'present' when expected 'missing' on a critical element)

    Does not hard-assert on non-critical element statuses — those are measured
    in the aggregate calibration report.
    """
    note_id = result["id"]
    expected = result["expected"]
    report = result["report"]

    if "overall_pass" in expected:
        assert report.passed == expected["overall_pass"], (
            f"[{note_id}] overall_pass: expected={expected['overall_pass']}, "
            f"got={report.passed} (rating={report.overall_rating}, "
            f"score={report.overall_score:.2f})"
        )

    item_map = {it.element_id: it for it in report.items}
    for elem_id, expected_status in expected.get("elements", {}).items():
        assert elem_id in item_map, (
            f"[{note_id}] Element '{elem_id}' not found in report items"
        )
        item = item_map[elem_id]
        actual_status = item.status
        if item.severity == "critical" and _is_false_positive(actual_status, expected_status):
            pytest.fail(
                f"[{note_id}] CRITICAL FALSE POSITIVE on '{elem_id}': "
                f"expected={expected_status}, got={actual_status} "
                f"(score={item.score:.2f}, evidence={str(item.evidence)[:80]!r})"
            )


# ── Per-note tests ─────────────────────────────────────────────────────────────

class TestGoldenNotes:
    """
    One test method per golden note scenario.

    Each test runs the note through evaluate_note() with a live LLM and checks:
    - overall_pass matches expected
    - no critical false positives (present when expected missing)
    """

    @pytest.fixture(autouse=True)
    def skip_unless_golden(self):
        _require_golden()

    @pytest.fixture(scope="class")
    def llm(self):
        return _get_llm_config()

    @pytest.fixture(scope="class")
    def notes_by_id(self):
        return {n["id"]: n for n in _load_dataset()["notes"]}

    def test_gold_standard(self, llm, notes_by_id):
        result = _run_note(notes_by_id["gold_standard"], llm)
        _assert_note_result(result)

    def test_thin_passing(self, llm, notes_by_id):
        result = _run_note(notes_by_id["thin_passing"], llm)
        _assert_note_result(result)

    def test_generic_rationale(self, llm, notes_by_id):
        result = _run_note(notes_by_id["generic_rationale"], llm)
        _assert_note_result(result)

    def test_conflated_atr_cfl(self, llm, notes_by_id):
        result = _run_note(notes_by_id["conflated_atr_cfl"], llm)
        _assert_note_result(result)

    def test_minimal_understanding(self, llm, notes_by_id):
        result = _run_note(notes_by_id["minimal_understanding"], llm)
        _assert_note_result(result)

    def test_missing_cfl(self, llm, notes_by_id):
        result = _run_note(notes_by_id["missing_cfl"], llm)
        _assert_note_result(result)

    def test_replacement_business(self, llm, notes_by_id):
        result = _run_note(notes_by_id["replacement_business"], llm)
        _assert_note_result(result)

    def test_bare_review(self, llm, notes_by_id):
        result = _run_note(notes_by_id["bare_review"], llm)
        _assert_note_result(result)


# ── Aggregate calibration report ──────────────────────────────────────────────

class TestCalibrationReport:
    """
    Runs all golden notes and prints a per-element precision/recall table.

    Fails if any critical element has a false-positive rate exceeding
    _MAX_CRITICAL_FP_RATE (25%). This is the primary calibration gate.
    """

    @pytest.fixture(autouse=True)
    def skip_unless_golden(self):
        _require_golden()

    def test_calibration_summary(self, capsys):
        dataset = _load_dataset()
        llm_config = _get_llm_config()

        # Collect (element_id, severity, expected_status, actual_status) for all notes
        records: List[Tuple[str, str, str, str]] = []

        for note_entry in dataset["notes"]:
            report = evaluate_note(
                note_text=note_entry["note"],
                framework=dataset["framework_id"],
                llm_config=llm_config,
                verbose=False,
            )
            item_map = {it.element_id: it for it in report.items}
            for elem_id, expected_status in note_entry["expected"].get("elements", {}).items():
                if elem_id in item_map:
                    item = item_map[elem_id]
                    records.append((elem_id, item.severity, expected_status, item.status))

        # Aggregate per element
        element_stats: Dict[str, Dict] = defaultdict(
            lambda: {"fp": 0, "fn": 0, "correct": 0, "total": 0, "severity": "unknown"}
        )
        for elem_id, severity, expected, actual in records:
            s = element_stats[elem_id]
            s["total"] += 1
            s["severity"] = severity
            if _is_false_positive(actual, expected):
                s["fp"] += 1
            elif _is_false_negative(actual, expected):
                s["fn"] += 1
            else:
                s["correct"] += 1

        # Print calibration table
        with capsys.disabled():
            print("\n" + "=" * 72)
            print("GOLDEN DATASET CALIBRATION REPORT")
            print(f"Framework: {dataset['framework_id']}  |  Notes: {len(dataset['notes'])}")
            print("=" * 72)
            print(f"{'Element':<38} {'Sev':<10} {'FP Rate':<10} {'FN Rate':<10} {'N'}")
            print("-" * 72)
            for elem_id in sorted(element_stats):
                s = element_stats[elem_id]
                n = s["total"]
                fp_rate = s["fp"] / n if n > 0 else 0.0
                fn_rate = s["fn"] / n if n > 0 else 0.0
                flag = (
                    "  *** EXCEEDS THRESHOLD"
                    if fp_rate > _MAX_CRITICAL_FP_RATE and s["severity"] == "critical"
                    else ""
                )
                print(
                    f"{elem_id:<38} {s['severity']:<10} {fp_rate:<10.1%} {fn_rate:<10.1%} {n}{flag}"
                )
            print("=" * 72)

        # Assert: no critical element exceeds the false-positive rate threshold
        violations = [
            (eid, s["fp"] / s["total"])
            for eid, s in element_stats.items()
            if s["severity"] == "critical"
            and s["total"] > 0
            and (s["fp"] / s["total"]) > _MAX_CRITICAL_FP_RATE
        ]
        assert not violations, (
            f"Critical element false-positive rate exceeds {_MAX_CRITICAL_FP_RATE:.0%}:\n"
            + "\n".join(f"  {eid}: {rate:.1%}" for eid, rate in violations)
        )
