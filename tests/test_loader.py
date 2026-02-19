"""
tests/test_loader.py
====================

Unit tests for assert_llm_tools.metrics.note.loader:
  - load_framework()       — dict passthrough, built-in ID, custom file path,
                             error handling
  - _validate_framework()  — required top-level fields, required element fields,
                             severity validation, edge cases
"""
from __future__ import annotations

import os
import tempfile

import pytest
import yaml

from assert_llm_tools.metrics.note.loader import _validate_framework, load_framework


# ── Shared fixtures ────────────────────────────────────────────────────────────

_MINIMAL_VALID: dict = {
    "framework_id": "test_fw",
    "name": "Test Framework",
    "version": "1.0.0",
    "regulator": "TEST",
    "elements": [
        {
            "id": "elem_a",
            "description": "Element A",
            "required": True,
            "severity": "critical",
        }
    ],
}

_ALL_SEVERITY_ELEMENTS: list[dict] = [
    {"id": "c", "description": "C", "required": True, "severity": "critical"},
    {"id": "h", "description": "H", "required": True, "severity": "high"},
    {"id": "m", "description": "M", "required": False, "severity": "medium"},
    {"id": "l", "description": "L", "required": False, "severity": "low"},
]


def _write_temp_yaml(data: dict) -> str:
    """Write a framework dict to a temporary YAML file; returns the file path."""
    fh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(data, fh)
    fh.close()
    return fh.name


# ═══════════════════════════════════════════════════════════════════════════════
# load_framework — dict passthrough
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFrameworkDict:

    def test_valid_dict_returned_unchanged(self):
        """load_framework(dict) → same object returned after validation."""
        fw = load_framework(_MINIMAL_VALID)
        assert fw is _MINIMAL_VALID

    def test_dict_framework_id_preserved(self):
        fw = load_framework(_MINIMAL_VALID)
        assert fw["framework_id"] == "test_fw"

    def test_dict_with_extra_fields_accepted(self):
        """Extra top-level keys (e.g. description, metadata) must not cause errors."""
        enriched = {
            **_MINIMAL_VALID,
            "description": "Extra field",
            "effective_date": "2024-01-01",
            "meeting_type_overrides": {},
        }
        fw = load_framework(enriched)
        assert fw["framework_id"] == "test_fw"

    def test_dict_with_all_severity_levels(self):
        """Framework with elements of all four severities is valid."""
        fw_data = {
            **_MINIMAL_VALID,
            "elements": _ALL_SEVERITY_ELEMENTS,
        }
        fw = load_framework(fw_data)
        assert len(fw["elements"]) == 4

    def test_dict_with_optional_element_fields_accepted(self):
        """Elements may have optional fields (guidance, examples, anti_patterns)."""
        fw_data = {
            **_MINIMAL_VALID,
            "elements": [
                {
                    "id": "rich",
                    "description": "Rich element",
                    "required": True,
                    "severity": "high",
                    "guidance": "Look for X",
                    "examples": ["Example one", "Example two"],
                    "anti_patterns": ["Bad pattern"],
                }
            ],
        }
        fw = load_framework(fw_data)
        assert fw["elements"][0]["id"] == "rich"


# ═══════════════════════════════════════════════════════════════════════════════
# load_framework — built-in IDs
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFrameworkBuiltin:

    def test_fca_suitability_v1_loads(self):
        """Built-in 'fca_suitability_v1' resolves and loads without error."""
        fw = load_framework("fca_suitability_v1")
        assert fw["framework_id"] == "fca_suitability_v1"

    def test_fca_suitability_v1_has_nine_elements(self):
        fw = load_framework("fca_suitability_v1")
        assert len(fw["elements"]) == 9

    def test_fca_suitability_v1_required_fields_present(self):
        """All elements in built-in framework have required fields."""
        fw = load_framework("fca_suitability_v1")
        required_fields = {"id", "description", "required", "severity"}
        for elem in fw["elements"]:
            missing = required_fields - set(elem.keys())
            assert not missing, f"Element {elem.get('id')} missing fields: {missing}"

    def test_fca_suitability_v1_known_element_ids(self):
        """Spot-check known element IDs exist in the FCA framework."""
        fw = load_framework("fca_suitability_v1")
        ids = {e["id"] for e in fw["elements"]}
        for expected_id in (
            "client_objectives",
            "risk_attitude",
            "capacity_for_loss",
            "financial_situation",
            "knowledge_and_experience",
            "recommendation_rationale",
            "charges_and_costs",
        ):
            assert expected_id in ids, f"Expected element '{expected_id}' not found"

    def test_fca_suitability_v1_regulator_is_fca(self):
        fw = load_framework("fca_suitability_v1")
        assert "FCA" in fw["regulator"]

    def test_unknown_builtin_id_raises_file_not_found(self):
        """String that is neither a valid path nor a built-in ID → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("totally_nonexistent_framework_xyz789")

    def test_unknown_builtin_error_message_mentions_name(self):
        """FileNotFoundError message should mention the framework ID."""
        with pytest.raises(FileNotFoundError, match="no_such_fw"):
            load_framework("no_such_fw")


# ═══════════════════════════════════════════════════════════════════════════════
# load_framework — custom YAML file paths
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFrameworkCustomPath:

    def test_absolute_yaml_path_loads(self):
        """Absolute path to a valid YAML file resolves and loads correctly."""
        path = _write_temp_yaml(_MINIMAL_VALID)
        try:
            fw = load_framework(path)
            assert fw["framework_id"] == "test_fw"
        finally:
            os.unlink(path)

    def test_custom_yaml_elements_returned(self):
        """Elements from a custom YAML are returned intact."""
        custom = {
            "framework_id": "custom_fw",
            "name": "Custom",
            "version": "2.0",
            "regulator": "Custom Regulator",
            "elements": [
                {"id": "e1", "description": "E1", "required": True, "severity": "high"},
                {"id": "e2", "description": "E2", "required": False, "severity": "low"},
            ],
        }
        path = _write_temp_yaml(custom)
        try:
            fw = load_framework(path)
            assert fw["framework_id"] == "custom_fw"
            assert len(fw["elements"]) == 2
            assert fw["elements"][0]["id"] == "e1"
        finally:
            os.unlink(path)

    def test_nonexistent_path_raises_file_not_found(self):
        """String that looks like a path but doesn't exist → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("/absolutely/does/not/exist/framework.yaml")

    def test_custom_yaml_with_all_optional_element_fields(self):
        """YAML with guidance/examples/anti_patterns loads without error."""
        custom = {
            "framework_id": "rich_fw",
            "name": "Rich Framework",
            "version": "1.0",
            "regulator": "TEST",
            "elements": [
                {
                    "id": "el",
                    "description": "Element",
                    "required": True,
                    "severity": "critical",
                    "guidance": "Detailed guidance here",
                    "examples": ["Good example"],
                    "anti_patterns": ["Bad pattern"],
                }
            ],
        }
        path = _write_temp_yaml(custom)
        try:
            fw = load_framework(path)
            assert fw["elements"][0]["guidance"] == "Detailed guidance here"
        finally:
            os.unlink(path)

    def test_invalid_yaml_content_raises_value_error(self):
        """YAML file with invalid framework structure → ValueError."""
        bad_data = {
            # Missing required top-level fields
            "name": "Bad Framework",
            "elements": [],
        }
        path = _write_temp_yaml(bad_data)
        try:
            with pytest.raises((ValueError, FileNotFoundError)):
                load_framework(path)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# _validate_framework — top-level field validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateFrameworkTopLevel:

    def test_valid_framework_no_exception(self):
        """Fully valid framework passes validation silently."""
        _validate_framework(_MINIMAL_VALID)  # must not raise

    def test_missing_framework_id_raises_value_error(self):
        bad = {k: v for k, v in _MINIMAL_VALID.items() if k != "framework_id"}
        with pytest.raises(ValueError, match="framework_id"):
            _validate_framework(bad)

    def test_missing_name_raises_value_error(self):
        bad = {k: v for k, v in _MINIMAL_VALID.items() if k != "name"}
        with pytest.raises(ValueError, match="name"):
            _validate_framework(bad)

    def test_missing_version_raises_value_error(self):
        bad = {k: v for k, v in _MINIMAL_VALID.items() if k != "version"}
        with pytest.raises(ValueError, match="version"):
            _validate_framework(bad)

    def test_missing_regulator_raises_value_error(self):
        bad = {k: v for k, v in _MINIMAL_VALID.items() if k != "regulator"}
        with pytest.raises(ValueError, match="regulator"):
            _validate_framework(bad)

    def test_missing_elements_raises_value_error(self):
        bad = {k: v for k, v in _MINIMAL_VALID.items() if k != "elements"}
        with pytest.raises(ValueError, match="elements"):
            _validate_framework(bad)

    def test_empty_elements_list_raises_value_error(self):
        bad = {**_MINIMAL_VALID, "elements": []}
        with pytest.raises(ValueError):
            _validate_framework(bad)

    def test_elements_not_a_list_raises_value_error(self):
        bad = {**_MINIMAL_VALID, "elements": "not a list"}
        with pytest.raises(ValueError):
            _validate_framework(bad)

    def test_empty_framework_dict_raises_value_error(self):
        with pytest.raises(ValueError):
            _validate_framework({})


# ═══════════════════════════════════════════════════════════════════════════════
# _validate_framework — per-element field validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateFrameworkElements:

    def _fw_with_elements(self, elements: list) -> dict:
        return {**_MINIMAL_VALID, "elements": elements}

    def test_element_missing_id_raises_value_error(self):
        bad_elem = {"description": "No id", "required": True, "severity": "high"}
        with pytest.raises(ValueError, match="id"):
            _validate_framework(self._fw_with_elements([bad_elem]))

    def test_element_missing_description_raises_value_error(self):
        bad_elem = {"id": "x", "required": True, "severity": "high"}
        with pytest.raises(ValueError, match="description"):
            _validate_framework(self._fw_with_elements([bad_elem]))

    def test_element_missing_required_flag_raises_value_error(self):
        bad_elem = {"id": "x", "description": "X", "severity": "high"}
        with pytest.raises(ValueError, match="required"):
            _validate_framework(self._fw_with_elements([bad_elem]))

    def test_element_missing_severity_raises_value_error(self):
        bad_elem = {"id": "x", "description": "X", "required": True}
        with pytest.raises(ValueError, match="severity"):
            _validate_framework(self._fw_with_elements([bad_elem]))

    def test_element_invalid_severity_raises_value_error(self):
        bad_elem = {"id": "x", "description": "X", "required": True, "severity": "urgent"}
        with pytest.raises(ValueError, match="urgent"):
            _validate_framework(self._fw_with_elements([bad_elem]))

    def test_all_valid_severities_accepted(self):
        """All four valid severity values must pass validation."""
        for sev in ("critical", "high", "medium", "low"):
            elem = {"id": "x", "description": "X", "required": True, "severity": sev}
            _validate_framework(self._fw_with_elements([elem]))  # must not raise

    def test_second_invalid_element_raises_value_error(self):
        """Validation error in second element is detected."""
        good_elem = {"id": "a", "description": "A", "required": True, "severity": "critical"}
        bad_elem = {"id": "b", "description": "B", "required": True, "severity": "extreme"}
        with pytest.raises(ValueError, match="extreme"):
            _validate_framework(self._fw_with_elements([good_elem, bad_elem]))

    def test_multiple_valid_elements(self):
        """Framework with 4 elements of different severities passes validation."""
        _validate_framework(
            self._fw_with_elements(_ALL_SEVERITY_ELEMENTS)
        )  # must not raise

    def test_element_with_optional_extra_fields_valid(self):
        """Extra element fields (guidance, examples) must not cause validation failure."""
        elem = {
            "id": "x",
            "description": "X",
            "required": False,
            "severity": "low",
            "guidance": "Extra guidance",
            "examples": ["ex1"],
            "anti_patterns": ["bad1"],
        }
        _validate_framework(self._fw_with_elements([elem]))  # must not raise
