"""
Framework loader for compliance note evaluation.

load_framework() and _validate_framework() are intentionally kept separate
from evaluate_note.py so they can be unit-tested and used independently
(e.g. in a CLI validate-framework tool).
"""
import os

import yaml
from pathlib import Path
from typing import Any, Dict, Union

# Built-in frameworks shipped with the library.
# Resolved relative to this file so it works regardless of install location.
_BUILTIN_FRAMEWORKS_DIR = Path(__file__).parent.parent.parent / "frameworks"

# Suffixes that mark a string as a file-path rather than a built-in ID.
_FILE_SUFFIXES = (".yaml", ".yml")


def _is_file_path(value: str) -> bool:
    """
    Return True if *value* looks like a file path rather than a built-in ID.

    Detection rules (any one is sufficient):
    - Ends with ``.yaml`` or ``.yml`` (case-insensitive).
    - Contains a path separator (``/`` on POSIX, ``\\`` on Windows,
      or the platform ``os.sep``).
    """
    lower = value.lower()
    if any(lower.endswith(suffix) for suffix in _FILE_SUFFIXES):
        return True
    if "/" in value or "\\" in value or os.sep in value:
        return True
    return False


def load_framework(framework: Union[str, dict]) -> Dict[str, Any]:
    """
    Load and validate a regulatory framework definition.

    Detection logic
    ---------------
    When *framework* is a string the loader decides how to resolve it:

    * **File path** — if the string ends in ``.yaml`` / ``.yml`` *or* contains a
      path separator (``/`` or ``\\``), it is treated as a file-system path and
      loaded directly.  This means you can use absolute paths, relative paths,
      or any path containing a directory component.

    * **Built-in ID** — anything else (no extension, no separator) is treated as
      a built-in framework identifier and resolved against the library's bundled
      ``frameworks/`` directory (e.g. ``"fca_suitability_v1"``).

    YAML schema
    -----------
    Custom YAML files must follow the same schema as built-in frameworks.
    Required top-level keys:

    .. code-block:: yaml

        framework_id: my_custom_fw       # unique identifier string
        name: My Custom Framework        # human-readable name
        version: 1.0.0                   # semver string
        regulator: ACME                  # regulator / standard body
        elements:                        # list of one or more elements
          - id: elem_one                 # unique element identifier
            description: >              # plain-English requirement description
              The note must contain …
            required: true              # true = required, false = recommended
            severity: critical          # critical | high | medium | low
            guidance: >                 # (optional) evaluator guidance
              Look for …

    Optional top-level keys (not validated but preserved):
        ``description``, ``effective_date``, ``reference``,
        ``meeting_type_overrides``, and any other keys you add.

    Optional per-element keys:
        ``guidance``, ``examples``, ``anti_patterns``.

    Example
    -------
    .. code-block:: python

        from assert_llm_tools.metrics.note.evaluate_note import evaluate_note

        report = evaluate_note(
            note_text="Client meeting notes …",
            framework="/path/to/my_firm_framework.yaml",
        )
        print(report.passed, report.overall_score)

    Args:
        framework: One of:
            - A pre-loaded dict (returned as-is after validation).
            - A file-path string to a YAML file (absolute or relative).
              Detected by ``.yaml``/``.yml`` suffix or the presence of a
              path separator.
            - A built-in framework_id string (e.g. ``"fca_suitability_v1"``),
              resolved against the library's bundled frameworks directory.

    Returns:
        Validated framework dict.

    Raises:
        FileNotFoundError: If a file-path string points to a non-existent file,
            or if a built-in ID string cannot be resolved to a bundled file.
        yaml.YAMLError: If a YAML file exists but cannot be parsed.  The
            exception message includes the file path for easier debugging.
        ValueError: If the loaded data is missing required top-level keys
            (``framework_id``, ``name``, ``version``, ``regulator``,
            ``elements``), or if any element is missing required fields or has
            an invalid ``severity`` value.
    """
    if isinstance(framework, dict):
        _validate_framework(framework)
        return framework

    if _is_file_path(framework):
        # ── Custom file-path branch ──────────────────────────────────────────
        path = Path(framework)
        if not path.exists():
            raise FileNotFoundError(
                f"Custom framework file not found: '{framework}'. "
                f"Please check that the path is correct and the file exists."
            )
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Failed to parse YAML from framework file '{framework}': {exc}"
            ) from exc
    else:
        # ── Built-in ID branch ───────────────────────────────────────────────
        path = _BUILTIN_FRAMEWORKS_DIR / f"{framework}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"Built-in framework '{framework}' not found. "
                f"Built-in frameworks live in: {_BUILTIN_FRAMEWORKS_DIR}. "
                f"If you meant to load a custom file, make sure the path ends "
                f"in '.yaml' or '.yml', or contains a path separator."
            )
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

    _validate_framework(data, source=str(framework))
    return data


def _validate_framework(framework: dict, *, source: str = "<dict>") -> None:
    """
    Raise ValueError if required framework fields are missing or invalid.

    Validates:
    - Top-level required keys: framework_id, name, version, regulator, elements
    - Per-element required keys: id, description, required, severity
    - Per-element severity values: critical | high | medium | low

    Args:
        framework: Framework dict to validate.
        source: Human-readable description of where the framework came from
            (file path or ``"<dict>"``), included in error messages to aid
            debugging.

    Raises:
        ValueError: On any validation failure.
    """
    required_top_level = {"framework_id", "name", "version", "regulator", "elements"}
    missing_top = required_top_level - set(framework.keys())
    if missing_top:
        raise ValueError(
            f"Framework definition (source: {source}) is missing required "
            f"top-level fields: {missing_top}"
        )

    if not isinstance(framework["elements"], list) or len(framework["elements"]) == 0:
        raise ValueError(
            f"Framework (source: {source}) 'elements' must be a non-empty list."
        )

    required_element_fields = {"id", "description", "required", "severity"}
    valid_severities = {"critical", "high", "medium", "low"}

    for i, element in enumerate(framework["elements"]):
        elem_id = element.get("id", "<unknown>")
        missing_fields = required_element_fields - set(element.keys())
        if missing_fields:
            raise ValueError(
                f"Framework (source: {source}) element[{i}] "
                f"(id={elem_id}) is missing required fields: {missing_fields}"
            )
        if element["severity"] not in valid_severities:
            raise ValueError(
                f"Framework (source: {source}) element[{i}] "
                f"(id={elem_id}) has invalid severity '{element['severity']}'. "
                f"Must be one of: {valid_severities}"
            )
