"""
test_custom_framework_loading.py — Tests for END-48: custom framework loading.

Covers:
  - File-path detection (_is_file_path helper)
  - FileNotFoundError when a custom file path doesn't exist
  - yaml.YAMLError (with file path in message) for invalid YAML
  - ValueError for missing required top-level keys
  - ValueError for invalid element structure (id included in message)
  - Valid custom YAML file loads and validates correctly
  - Built-in framework IDs still resolve via the bundled directory
"""
from __future__ import annotations

# ── Stub out native deps before importing the library ─────────────────────────
import sys
import types


class _AutoMock(types.ModuleType):
    def __getattr__(self, name: str) -> "_AutoMock":
        child = _AutoMock(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs) -> "_AutoMock":
        return _AutoMock("_call_result")


for _stub in (
    "boto3",
    "botocore",
    "botocore.config",
    "botocore.exceptions",
    "openai",
    "nltk",
    "nltk.corpus",
    "nltk.tokenize",
    "nltk.data",
):
    sys.modules[_stub] = _AutoMock(_stub)

sys.modules["botocore.config"].Config = type(
    "Config", (), {"__init__": lambda self, **kw: None}
)
sys.modules["openai"].OpenAI = type(
    "OpenAI", (), {"__init__": lambda self, **kw: None}
)

# ── Library imports ────────────────────────────────────────────────────────────
import os
import textwrap
import tempfile
from pathlib import Path

import pytest
import yaml

from assert_llm_tools.metrics.note.loader import load_framework, _validate_framework, _is_file_path


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _write_temp_yaml(content: str, suffix: str = ".yaml") -> str:
    """Write *content* to a temp file and return its path string."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(content)
        return fh.name


_VALID_YAML = textwrap.dedent("""\
    framework_id: custom_test_fw
    name: Custom Test Framework
    version: 1.0.0
    regulator: TEST
    elements:
      - id: elem_one
        description: The note must contain element one.
        required: true
        severity: critical
        guidance: Look for element one.
      - id: elem_two
        description: The note should contain element two.
        required: false
        severity: medium
""")


# ═══════════════════════════════════════════════════════════════════════════════
# _is_file_path detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsFilePath:
    """Unit tests for the _is_file_path() detection helper."""

    def test_yaml_suffix_detected_as_file_path(self):
        assert _is_file_path("my_framework.yaml") is True

    def test_yml_suffix_detected_as_file_path(self):
        assert _is_file_path("my_framework.yml") is True

    def test_yaml_suffix_case_insensitive(self):
        assert _is_file_path("My_Framework.YAML") is True

    def test_absolute_path_detected_as_file_path(self):
        assert _is_file_path("/path/to/framework.yaml") is True

    def test_relative_path_with_directory_detected_as_file_path(self):
        assert _is_file_path("./frameworks/custom.yaml") is True

    def test_path_with_backslash_detected_as_file_path(self):
        assert _is_file_path("frameworks\\custom.yaml") is True

    def test_path_separator_without_extension_detected_as_file_path(self):
        # e.g. "some/path/framework" — has separator, no .yaml suffix
        assert _is_file_path("some/path/framework") is True

    def test_plain_id_not_detected_as_file_path(self):
        assert _is_file_path("fca_suitability_v1") is False

    def test_plain_id_with_hyphens_not_detected_as_file_path(self):
        assert _is_file_path("my-custom-framework") is False

    def test_plain_id_with_dots_but_no_yaml_not_detected_as_file_path(self):
        # e.g. "v1.0" — not a yaml path
        assert _is_file_path("v1.0") is False


# ═══════════════════════════════════════════════════════════════════════════════
# FileNotFoundError
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileNotFound:

    def test_nonexistent_yaml_path_raises_file_not_found(self):
        """A .yaml path that doesn't exist → FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="/no/such/path/framework.yaml"):
            load_framework("/no/such/path/framework.yaml")

    def test_nonexistent_yml_path_raises_file_not_found(self):
        """A .yml path that doesn't exist → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("/tmp/does_not_exist_abc123.yml")

    def test_nonexistent_path_with_separator_raises_file_not_found(self):
        """A path with directory separator (no .yaml) that doesn't exist → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("/no/such/directory/framework")

    def test_error_message_includes_path(self):
        """FileNotFoundError message should reference the supplied path."""
        bad_path = "/definitely/missing/custom_fw.yaml"
        with pytest.raises(FileNotFoundError, match="custom_fw.yaml"):
            load_framework(bad_path)

    def test_nonexistent_builtin_id_raises_file_not_found(self):
        """A built-in ID that doesn't exist → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("nonexistent_builtin_id_xyz")

    def test_builtin_error_message_is_helpful(self):
        """FileNotFoundError for a bad built-in ID mentions the frameworks directory."""
        with pytest.raises(FileNotFoundError, match="frameworks"):
            load_framework("nonexistent_builtin_id_xyz")


# ═══════════════════════════════════════════════════════════════════════════════
# Invalid YAML
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvalidYaml:

    def test_invalid_yaml_raises_yaml_error(self):
        """A file containing invalid YAML → yaml.YAMLError."""
        path = _write_temp_yaml("{ this is: [not valid yaml")
        try:
            with pytest.raises(yaml.YAMLError):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_invalid_yaml_error_includes_file_path(self):
        """yaml.YAMLError message must include the file path."""
        path = _write_temp_yaml(": bad: yaml: content: [[[")
        try:
            with pytest.raises(yaml.YAMLError, match=path):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_yaml_error_raised_for_yml_extension(self):
        """Invalid YAML in a .yml file also raises yaml.YAMLError."""
        path = _write_temp_yaml("key: [unclosed", suffix=".yml")
        try:
            with pytest.raises(yaml.YAMLError):
                load_framework(path)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Missing required top-level keys
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingTopLevelKeys:

    def test_missing_framework_id_raises_value_error(self):
        """YAML missing 'framework_id' → ValueError."""
        yaml_content = textwrap.dedent("""\
            name: No ID Framework
            version: 1.0.0
            regulator: TEST
            elements:
              - id: e1
                description: Desc
                required: true
                severity: high
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="framework_id"):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_missing_elements_raises_value_error(self):
        """YAML missing 'elements' → ValueError."""
        yaml_content = textwrap.dedent("""\
            framework_id: fw_no_elements
            name: No Elements
            version: 1.0.0
            regulator: TEST
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="elements"):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_missing_multiple_top_level_keys_raises_value_error(self):
        """YAML missing several top-level keys → single ValueError mentioning them."""
        yaml_content = textwrap.dedent("""\
            framework_id: partial_fw
            elements:
              - id: e1
                description: Desc
                required: true
                severity: low
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_empty_elements_list_raises_value_error(self):
        """YAML with 'elements: []' → ValueError."""
        yaml_content = textwrap.dedent("""\
            framework_id: empty_fw
            name: Empty Elements
            version: 1.0.0
            regulator: TEST
            elements: []
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError):
                load_framework(path)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Invalid element structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvalidElementStructure:

    def test_element_missing_severity_raises_value_error_with_id(self):
        """Element missing 'severity' → ValueError mentioning element id."""
        yaml_content = textwrap.dedent("""\
            framework_id: fw_bad_elem
            name: Bad Element Framework
            version: 1.0.0
            regulator: TEST
            elements:
              - id: bad_element_id
                description: Missing severity field.
                required: true
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="bad_element_id"):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_element_invalid_severity_raises_value_error_with_id(self):
        """Element with invalid severity → ValueError mentioning element id."""
        yaml_content = textwrap.dedent("""\
            framework_id: fw_bad_sev
            name: Bad Severity Framework
            version: 1.0.0
            regulator: TEST
            elements:
              - id: severity_violator
                description: Bad severity value.
                required: true
                severity: extreme
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="severity_violator"):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_element_missing_description_raises_value_error_with_id(self):
        """Element missing 'description' → ValueError mentioning element id."""
        yaml_content = textwrap.dedent("""\
            framework_id: fw_no_desc
            name: No Desc Framework
            version: 1.0.0
            regulator: TEST
            elements:
              - id: elem_without_desc
                required: true
                severity: high
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="elem_without_desc"):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_element_missing_required_field_raises_value_error_with_id(self):
        """Element missing 'required' → ValueError mentioning element id."""
        yaml_content = textwrap.dedent("""\
            framework_id: fw_no_req
            name: No Required Framework
            version: 1.0.0
            regulator: TEST
            elements:
              - id: elem_no_required
                description: Missing required field.
                severity: medium
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="elem_no_required"):
                load_framework(path)
        finally:
            os.unlink(path)

    def test_second_element_invalid_raises_value_error_with_its_id(self):
        """Only the second element is broken → ValueError mentions second element's id."""
        yaml_content = textwrap.dedent("""\
            framework_id: fw_second_bad
            name: Second Bad Framework
            version: 1.0.0
            regulator: TEST
            elements:
              - id: good_elem
                description: Fine.
                required: true
                severity: high
              - id: broken_elem
                description: Missing severity.
                required: false
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            with pytest.raises(ValueError, match="broken_elem"):
                load_framework(path)
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Valid custom YAML loads correctly
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidCustomYaml:

    def test_valid_yaml_file_loads_correctly(self):
        """A well-formed custom YAML file loads and returns correct data."""
        path = _write_temp_yaml(_VALID_YAML)
        try:
            fw = load_framework(path)
            assert fw["framework_id"] == "custom_test_fw"
            assert fw["name"] == "Custom Test Framework"
            assert fw["version"] == "1.0.0"
            assert fw["regulator"] == "TEST"
            assert isinstance(fw["elements"], list)
            assert len(fw["elements"]) == 2
        finally:
            os.unlink(path)

    def test_valid_yaml_elements_have_correct_ids(self):
        """Loaded elements have the expected ids."""
        path = _write_temp_yaml(_VALID_YAML)
        try:
            fw = load_framework(path)
            ids = {e["id"] for e in fw["elements"]}
            assert "elem_one" in ids
            assert "elem_two" in ids
        finally:
            os.unlink(path)

    def test_valid_yml_extension_also_loads(self):
        """A file with .yml extension loads without issues."""
        path = _write_temp_yaml(_VALID_YAML, suffix=".yml")
        try:
            fw = load_framework(path)
            assert fw["framework_id"] == "custom_test_fw"
        finally:
            os.unlink(path)

    def test_valid_yaml_with_optional_fields_loads(self):
        """Optional fields (guidance, examples, description, etc.) are preserved."""
        yaml_content = textwrap.dedent("""\
            framework_id: fw_with_extras
            name: Framework With Extras
            version: 2.0.0
            regulator: FCA
            description: An extended framework for testing.
            effective_date: "2026-01-01"
            reference: "COBS 9.2"
            elements:
              - id: elem_extras
                description: Has optional fields.
                required: true
                severity: critical
                guidance: Look for this and that.
                examples:
                  - "Example text showing compliance."
                anti_patterns:
                  - "Vague statement."
        """)
        path = _write_temp_yaml(yaml_content)
        try:
            fw = load_framework(path)
            assert fw["framework_id"] == "fw_with_extras"
            elem = fw["elements"][0]
            assert elem["guidance"].strip() == "Look for this and that."
            assert elem["examples"] == ["Example text showing compliance."]
        finally:
            os.unlink(path)

    def test_valid_dict_still_loads_unchanged(self):
        """Pre-loaded dict is returned as-is (existing behaviour preserved)."""
        framework_dict = {
            "framework_id": "dict_fw",
            "name": "Dict Framework",
            "version": "1.0.0",
            "regulator": "TEST",
            "elements": [
                {
                    "id": "dict_elem",
                    "description": "A dict element.",
                    "required": True,
                    "severity": "high",
                }
            ],
        }
        result = load_framework(framework_dict)
        assert result is framework_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in IDs still work
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuiltinIdResolution:

    def test_fca_suitability_v1_loads(self):
        """Built-in 'fca_suitability_v1' still loads correctly."""
        fw = load_framework("fca_suitability_v1")
        assert fw["framework_id"] == "fca_suitability_v1"
        assert fw["regulator"] == "FCA"
        assert isinstance(fw["elements"], list)
        assert len(fw["elements"]) > 0

    def test_builtin_elements_have_required_fields(self):
        """Every element in the built-in framework has required fields."""
        fw = load_framework("fca_suitability_v1")
        for elem in fw["elements"]:
            assert "id" in elem
            assert "description" in elem
            assert "required" in elem
            assert "severity" in elem

    def test_unknown_builtin_id_raises_file_not_found(self):
        """Unknown built-in ID (no path indicators) → FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_framework("totally_unknown_framework_id_xyz")

    def test_builtin_error_mentions_frameworks_directory(self):
        """FileNotFoundError for unknown built-in ID hints at the frameworks dir."""
        with pytest.raises(FileNotFoundError, match="frameworks"):
            load_framework("totally_unknown_framework_id_xyz")

    def test_builtin_error_suggests_yaml_extension_hint(self):
        """FileNotFoundError for unknown built-in ID hints that .yaml suffix denotes a file path."""
        with pytest.raises(FileNotFoundError, match=r"\.yaml"):
            load_framework("totally_unknown_framework_id_xyz")
