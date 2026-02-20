"""
Framework loader for compliance note evaluation.

load_framework() and _validate_framework() are intentionally kept separate
from evaluate_note.py so they can be unit-tested and used independently
(e.g. in a CLI validate-framework tool).
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Union

# Built-in frameworks shipped with the package.
# Resolved relative to this file so it works regardless of install location.
_BUILTIN_FRAMEWORKS_DIR = Path(__file__).parent / "frameworks"


def load_framework(framework: Union[str, dict]) -> Dict[str, Any]:
    """
    Load and validate a regulatory framework definition.

    Args:
        framework: One of:
            - A pre-loaded dict (returned as-is after validation).
            - A path string to a YAML file (absolute or relative).
            - A built-in framework_id string (e.g. "fca_suitability_v1"),
              resolved against the library's bundled frameworks directory.

    Returns:
        Validated framework dict.

    Raises:
        FileNotFoundError: If no matching YAML file can be found.
        ValueError: If the YAML is missing required top-level or element fields.
    """
    if isinstance(framework, dict):
        _validate_framework(framework)
        return framework

    # Try as a literal file path first
    path = Path(framework)
    if not path.exists():
        # Fall back to the built-in frameworks directory
        path = _BUILTIN_FRAMEWORKS_DIR / f"{framework}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Framework '{framework}' not found as a file path or as a built-in "
                f"framework ID. Built-in frameworks live in: {_BUILTIN_FRAMEWORKS_DIR}"
            )

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    _validate_framework(data)
    return data


def _validate_framework(framework: dict) -> None:
    """
    Raise ValueError if required framework fields are missing or invalid.

    Validates:
    - Top-level required keys: framework_id, name, version, regulator, elements
    - Per-element required keys: id, description, required, severity
    - Per-element severity values: critical | high | medium | low

    Args:
        framework: Framework dict to validate.

    Raises:
        ValueError: On any validation failure.
    """
    required_top_level = {"framework_id", "name", "version", "regulator", "elements"}
    missing_top = required_top_level - set(framework.keys())
    if missing_top:
        raise ValueError(
            f"Framework definition is missing required top-level fields: {missing_top}"
        )

    if not isinstance(framework["elements"], list) or len(framework["elements"]) == 0:
        raise ValueError("Framework 'elements' must be a non-empty list.")

    required_element_fields = {"id", "description", "required", "severity"}
    valid_severities = {"critical", "high", "medium", "low"}

    for i, element in enumerate(framework["elements"]):
        missing_fields = required_element_fields - set(element.keys())
        if missing_fields:
            raise ValueError(
                f"Framework element[{i}] (id={element.get('id', '<unknown>')}) "
                f"is missing required fields: {missing_fields}"
            )
        if element["severity"] not in valid_severities:
            raise ValueError(
                f"Framework element[{i}] (id={element.get('id', '<unknown>')}) "
                f"has invalid severity '{element['severity']}'. "
                f"Must be one of: {valid_severities}"
            )
