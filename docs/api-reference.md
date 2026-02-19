# API Reference

`assert_llm_tools` provides an LLM-based evaluation pipeline for compliance notes.
The primary API is `evaluate_note()` / `NoteEvaluator`, backed by structured data
models (`GapReport`, `GapItem`, `GapReportStats`, `PassPolicy`).

All public symbols are importable from the top-level package:

```python
from assert_llm_tools import (
    evaluate_note,
    NoteEvaluator,
    GapReport,
    GapItem,
    GapReportStats,
    PassPolicy,
    LLMConfig,
)
```

---

## `evaluate_note()`

```python
def evaluate_note(
    note_text: str,
    framework: Union[str, dict],
    llm_config: Optional[LLMConfig] = None,
    *,
    mask_pii: bool = False,
    verbose: bool = False,
    custom_instruction: Optional[str] = None,
    pass_policy: Optional[PassPolicy] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GapReport:
```

Top-level convenience function. Creates a `NoteEvaluator` internally and calls
`evaluate()`. Use this for single-shot evaluations; use `NoteEvaluator` directly
when you need to reuse the same evaluator across multiple calls.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `note_text` | `str` | — | Full text of the compliance note to evaluate. |
| `framework` | `str \| dict` | — | Framework to evaluate against. Accepts: a built-in framework ID (e.g. `"fca_suitability_v1"`), a file path to a YAML framework definition, or a pre-loaded framework `dict`. |
| `llm_config` | `LLMConfig \| None` | `None` | LLM provider configuration. If `None`, defaults to the Bedrock/Claude configuration inherited from `BaseCalculator`. |
| `mask_pii` | `bool` | `False` | If `True`, runs PII detection and masking on `note_text` before sending it to the LLM. Sets `GapReport.pii_masked = True`. |
| `verbose` | `bool` | `False` | If `True`, `GapItem.notes` is populated with the raw LLM reasoning for each element assessment. |
| `custom_instruction` | `str \| None` | `None` | Additional instruction appended to every element evaluation prompt. Useful for firm-specific note formats or terminology. |
| `pass_policy` | `PassPolicy \| None` | `None` | Override the default pass/fail thresholds. If `None`, `PassPolicy()` defaults are used. |
| `metadata` | `dict \| None` | `None` | Arbitrary key/value pairs attached to `GapReport.metadata` (e.g. `note_id`, `adviser_ref`). |

### Returns

`GapReport` — structured evaluation result. See [`GapReport`](#gapreport) below.

### Raises

| Exception | When |
|-----------|------|
| `FileNotFoundError` | `framework` is a string that cannot be resolved to a file or built-in ID. |
| `ValueError` | Framework YAML is missing required fields or contains invalid values. |

---

## `NoteEvaluator`

```python
class NoteEvaluator(BaseCalculator):
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        custom_instruction: Optional[str] = None,
        verbose: bool = False,
        pass_policy: Optional[PassPolicy] = None,
    ) -> None: ...
```

LLM-based evaluator for compliance notes. Extends `BaseCalculator` and reuses its
LLM initialisation. Each framework element is evaluated in a separate LLM call to
keep prompts focused.

Prefer `NoteEvaluator` directly when evaluating many notes against the same
framework — the instance is stateless between `evaluate()` calls and can be reused
freely.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_config` | `LLMConfig \| None` | `None` | LLM provider configuration. Passed to `BaseCalculator`. |
| `custom_instruction` | `str \| None` | `None` | Additional instruction appended to every element prompt. |
| `verbose` | `bool` | `False` | Populate `GapItem.notes` with raw LLM reasoning. |
| `pass_policy` | `PassPolicy \| None` | `None` | Pass/fail threshold configuration. Defaults to `PassPolicy()`. |

### `evaluate()`

```python
def evaluate(
    self,
    note_text: str,
    framework: Union[str, dict],
    mask_pii: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> GapReport:
```

Run the full evaluation pipeline and return a `GapReport`.

**Pipeline steps:**
1. Load and validate the framework definition.
2. Optionally mask PII in `note_text`.
3. Evaluate each framework element independently via a focused LLM call.
4. Generate a plain-English summary via a single additional LLM call.
5. Compute summary statistics and overall score.
6. Apply `PassPolicy` to determine pass/fail.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `note_text` | `str` | — | Full text of the compliance note. |
| `framework` | `str \| dict` | — | Framework ID, file path, or pre-loaded dict. |
| `mask_pii` | `bool` | `False` | Apply PII masking before evaluation. |
| `metadata` | `dict \| None` | `None` | Attached to `GapReport.metadata` unchanged. |

**Returns:** `GapReport`

---

## `PassPolicy`

```python
@dataclass
class PassPolicy:
    block_on_critical_missing: bool = True
    block_on_critical_partial: bool = True
    block_on_high_missing: bool = True
    critical_partial_threshold: float = 0.5
```

Controls the pass/fail logic applied to `GapReport.passed`. All fields are optional
with sensible strict defaults.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `block_on_critical_missing` | `bool` | `True` | Fail if any required **critical** element has `status == "missing"`. |
| `block_on_critical_partial` | `bool` | `True` | Fail if any required **critical** element has `status == "partial"` and `score < critical_partial_threshold`. |
| `block_on_high_missing` | `bool` | `True` | Fail if any required **high** element has `status == "missing"`. |
| `critical_partial_threshold` | `float` | `0.5` | Minimum score for a **critical/partial** element to not trigger a block. Elements scoring at or above this threshold are not considered a blocking partial. |

> **Note (END-51):** `critical_partial_threshold` was made configurable in END-51.
> Previously the threshold was hardcoded. Pass a custom `PassPolicy` to
> `evaluate_note()` or `NoteEvaluator` to override it.

Optional elements (where `GapItem.required == False`) never block a pass,
regardless of status or severity.

---

## `GapReport`

```python
@dataclass
class GapReport:
    framework_id: str
    framework_version: str
    passed: bool
    overall_score: float
    items: List[GapItem]
    summary: str
    stats: GapReportStats
    pii_masked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
```

Top-level evaluation result returned by `evaluate_note()` and `NoteEvaluator.evaluate()`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `framework_id` | `str` | ID of the framework used (e.g. `"fca_suitability_v1"`). |
| `framework_version` | `str` | Version string from the framework definition. |
| `passed` | `bool` | Overall pass/fail result. `True` only when no blocking gaps exist per `PassPolicy`. |
| `overall_score` | `float` | Weighted mean of element scores, 0.0–1.0. Required elements are weighted 2×, optional elements 1×. |
| `items` | `List[GapItem]` | One `GapItem` per framework element, in definition order. |
| `summary` | `str` | Plain-English summary of the evaluation, generated by the LLM. |
| `stats` | `GapReportStats` | Aggregate counts — see [`GapReportStats`](#gapreportstats). |
| `pii_masked` | `bool` | `True` if PII masking was applied to the note before evaluation. |
| `metadata` | `dict` | Arbitrary key/value pairs passed in via the `metadata` argument. |

---

## `GapItem`

```python
@dataclass
class GapItem:
    element_id: str
    status: ElementStatus          # "present" | "partial" | "missing"
    score: float                   # 0.0–1.0
    evidence: str
    severity: ElementSeverity      # "critical" | "high" | "medium" | "low"
    required: bool
    notes: Optional[str] = None
```

Evaluation result for a single framework element.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `element_id` | `str` | Element ID as defined in the framework YAML. |
| `status` | `"present" \| "partial" \| "missing"` | Whether the element is fully present, partially addressed, or absent. |
| `score` | `float` | Quality/confidence score, 0.0–1.0. `1.0` = fully present and well-documented; `0.5` = partial; `0.0` = absent. |
| `evidence` | `str` | Verbatim or paraphrased excerpt from the note supporting the status. Empty string if missing. |
| `severity` | `"critical" \| "high" \| "medium" \| "low"` | Compliance impact severity, copied from the framework element definition. |
| `required` | `bool` | Whether the element is required per the framework. |
| `notes` | `str \| None` | LLM reasoning for the assessment. Only populated when `verbose=True` is passed to `evaluate_note()` or `NoteEvaluator`. |

### `OverallRating` type alias

```python
ElementStatus = Literal["present", "partial", "missing"]
```

| Value | Meaning |
|-------|---------|
| `"present"` | The element is clearly and adequately documented in the note. |
| `"partial"` | The topic is raised but insufficiently documented. |
| `"missing"` | There is no meaningful mention of the element in the note. |

---

## `GapReportStats`

```python
@dataclass
class GapReportStats:
    total_elements: int
    required_elements: int
    present_count: int
    partial_count: int
    missing_count: int
    critical_gaps: int
    high_gaps: int
    medium_gaps: int
    low_gaps: int
    required_missing_count: int
```

Aggregate counts derived from the `GapReport.items` list.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `total_elements` | `int` | Total number of framework elements assessed. |
| `required_elements` | `int` | Number of elements marked as required in the framework. |
| `present_count` | `int` | Elements with `status == "present"`. |
| `partial_count` | `int` | Elements with `status == "partial"`. |
| `missing_count` | `int` | Elements with `status == "missing"`. |
| `critical_gaps` | `int` | Elements with `severity == "critical"` that are **not** `"present"`. |
| `high_gaps` | `int` | Elements with `severity == "high"` that are **not** `"present"`. |
| `medium_gaps` | `int` | Elements with `severity == "medium"` that are **not** `"present"`. |
| `low_gaps` | `int` | Elements with `severity == "low"` that are **not** `"present"`. |
| `required_missing_count` | `int` | Required elements whose status is `"missing"` or `"partial"`. |

---

## `LLMConfig`

```python
from assert_llm_tools import LLMConfig

config = LLMConfig(
    provider="openai",       # "openai" | "bedrock"
    model_id="gpt-4o",
    api_key="sk-...",        # OpenAI key (openai provider)
    region="us-east-1",      # AWS region (bedrock provider)
)
```

See the LLM configuration documentation for the full list of parameters.

---

## Code Examples

### Basic evaluation (default Bedrock LLM)

```python
from assert_llm_tools import evaluate_note

report = evaluate_note(
    note_text="Client John expressed a preference for low-risk investments...",
    framework="fca_suitability_v1",
)

print(f"Passed: {report.passed}")
print(f"Overall score: {report.overall_score:.2f}")
print(f"Summary: {report.summary}")
```

### OpenAI provider

```python
from assert_llm_tools import evaluate_note, LLMConfig

config = LLMConfig(
    provider="openai",
    model_id="gpt-4o",
    api_key="sk-...",
)

report = evaluate_note(
    note_text="...",
    framework="fca_suitability_v1",
    llm_config=config,
)
```

### Custom framework (dict or YAML path)

```python
from assert_llm_tools import evaluate_note

# Built-in ID
report = evaluate_note(note_text="...", framework="fca_wealth_v1")

# Custom YAML file
report = evaluate_note(note_text="...", framework="/path/to/my_framework.yaml")

# Pre-loaded dict
framework_dict = {
    "framework_id": "my_framework",
    "name": "My Framework",
    "version": "1.0",
    "regulator": "FCA",
    "elements": [
        {
            "id": "risk_profile",
            "description": "Client risk profile documented",
            "required": True,
            "severity": "critical",
        }
    ],
}
report = evaluate_note(note_text="...", framework=framework_dict)
```

### Meeting type override

```python
report = evaluate_note(
    note_text="...",
    framework="fca_suitability_v1",
    custom_instruction="This note covers a pension transfer review meeting.",
)
```

### Custom pass/fail thresholds (END-51)

```python
from assert_llm_tools import evaluate_note, PassPolicy

# Stricter: also block on high-severity partial elements
strict_policy = PassPolicy(
    block_on_critical_missing=True,
    block_on_critical_partial=True,
    block_on_high_missing=True,
    critical_partial_threshold=0.7,  # raise bar for critical partials
)

report = evaluate_note(
    note_text="...",
    framework="fca_suitability_v1",
    pass_policy=strict_policy,
)

# Lenient: only hard-block on critical missing
lenient_policy = PassPolicy(
    block_on_critical_missing=True,
    block_on_critical_partial=False,
    block_on_high_missing=False,
)

report = evaluate_note(
    note_text="...",
    framework="fca_suitability_v1",
    pass_policy=lenient_policy,
)
```

### PII masking

```python
report = evaluate_note(
    note_text="John Smith, DOB 01/01/1970, account 12345678 ...",
    framework="fca_suitability_v1",
    mask_pii=True,
)

print(f"PII was masked: {report.pii_masked}")  # True
```

### JSON / dict output

```python
import json
from assert_llm_tools import evaluate_note

report = evaluate_note(note_text="...", framework="fca_suitability_v1")

# Serialise the full report
report_dict = report.to_dict()
report_json = report.to_json()

print(json.dumps(report_dict, indent=2))
```

### Reusing an evaluator across multiple notes

```python
from assert_llm_tools import NoteEvaluator, LLMConfig, PassPolicy

config = LLMConfig(provider="openai", model_id="gpt-4o", api_key="sk-...")
policy = PassPolicy(critical_partial_threshold=0.6)

evaluator = NoteEvaluator(
    llm_config=config,
    pass_policy=policy,
    verbose=True,
)

notes = ["Note A text...", "Note B text...", "Note C text..."]
for note in notes:
    report = evaluator.evaluate(note, framework="fca_suitability_v1")
    print(f"Passed: {report.passed} | Score: {report.overall_score:.2f}")
```

### Inspecting gaps

```python
report = evaluate_note(note_text="...", framework="fca_suitability_v1")

for item in report.items:
    if item.status != "present":
        print(
            f"[{item.severity.upper()}] {item.element_id}: {item.status} "
            f"(score={item.score:.2f})"
        )
        if item.evidence:
            print(f"  Evidence: {item.evidence}")

print(f"\nStats: {report.stats.missing_count} missing, "
      f"{report.stats.partial_count} partial, "
      f"{report.stats.present_count} present "
      f"(of {report.stats.total_elements} total)")
```
