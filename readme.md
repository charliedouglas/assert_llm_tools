# ASSERT LLM Tools

> **⚠️ Deprecated — this package is no longer maintained.**
>
> `assert_llm_tools` has been superseded by focused, independently-versioned packages:
>
> | Capability | New package | Install |
> |-----------|-------------|---------|
> | Summary evaluation | **assert-eval** | `pip install assert-eval` |
> | Compliance note evaluation | **assert-review** | `pip install assert-review` |
>
> Version 1.0.0 is the final release. No further updates will be made. Please migrate to the packages above.

---

**A**utomated **S**ummary **S**coring & **E**valuation of **R**etained **T**ext

ASSERT LLM Tools is a lightweight Python library for LLM-based text evaluation. It provides two main capabilities:

- **Summary evaluation** — score a summary against source text for coverage, factual accuracy, coherence, and more
- **Compliance note evaluation** — evaluate adviser meeting notes against regulatory frameworks (FCA, MiFID II) and return a structured gap report

All evaluation is LLM-based. No PyTorch, no BERT, no heavy dependencies.

## Installation

```bash
pip install assert-llm-tools
```

## Quick Start

### Summary Evaluation

```python
from assert_llm_tools import evaluate_summary, LLMConfig

config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
)

results = evaluate_summary(
    full_text="Original long text goes here...",
    summary="Summary to evaluate goes here...",
    metrics=["coverage", "factual_consistency", "coherence"],
    llm_config=config,
)

print(results)
# {'coverage': 0.85, 'factual_consistency': 0.92, 'coherence': 0.88}
```

### Compliance Note Evaluation

```python
from assert_llm_tools import evaluate_note, LLMConfig

config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
)

report = evaluate_note(
    note_text="Client meeting note text goes here...",
    framework="fca_suitability_v1",
    llm_config=config,
)

print(report.overall_rating)   # "Compliant" / "Minor Gaps" / "Requires Attention" / "Non-Compliant"
print(report.overall_score)    # 0.0–1.0
print(report.passed)           # True / False

for item in report.items:
    print(f"{item.element_id}: {item.status} (score: {item.score:.2f})")
    if item.suggestions:
        for s in item.suggestions:
            print(f"  → {s}")
```

## Summary Evaluation

### Available Metrics

| Metric | Description |
|--------|-------------|
| `coverage` | How completely the summary captures claims from the source text |
| `factual_consistency` | Whether claims in the summary are supported by the source |
| `factual_alignment` | Combined coverage + consistency score |
| `topic_preservation` | How well the summary preserves the main topics |
| `conciseness` | Information density — does the summary avoid padding? |
| `redundancy` | Detects repetitive content within the summary |
| `coherence` | Logical flow and readability of the summary |

> **Deprecated names** (still accepted for backwards compatibility): `faithfulness` → use `coverage`; `hallucination` → use `factual_consistency`.

### Custom Evaluation Instructions

Tailor LLM evaluation criteria for your domain:

```python
results = evaluate_summary(
    full_text=text,
    summary=summary,
    metrics=["coverage", "factual_consistency"],
    llm_config=config,
    custom_prompt_instructions={
        "coverage": "Apply strict standards. Only mark a claim as covered if it is clearly and explicitly represented.",
        "factual_consistency": "Flag any claim that adds detail not present in the original text.",
    },
)
```

### Verbose Output

Pass `verbose=True` to include per-claim reasoning in the results:

```python
results = evaluate_summary(..., verbose=True)
```

## Compliance Note Evaluation

> ⚠️ **Experimental — do not use in live or production systems.**
>
> `evaluate_note()` is under active development. Outputs are non-deterministic (LLM-based), the API may change between releases, and results have not been validated against real regulatory decisions. This feature is intended for research, prototyping, and internal tooling only. It is not a substitute for qualified compliance review and must not be used to make or support live regulatory or client-facing decisions.

### evaluate_note()

```python
from assert_llm_tools import evaluate_note, LLMConfig
from assert_llm_tools.metrics.note.models import PassPolicy

report = evaluate_note(
    note_text=note,
    framework="fca_suitability_v1",   # built-in ID or path to a custom YAML
    llm_config=config,
    mask_pii=False,                    # mask client PII before sending to LLM
    verbose=False,                     # include LLM reasoning in GapItem.notes
    custom_instruction=None,           # additional instruction appended to all element prompts
    pass_policy=None,                  # custom PassPolicy (see below)
    metadata={"note_id": "N-001"},     # arbitrary key/value pairs, passed through to GapReport
)
```

### GapReport

| Field | Type | Description |
|-------|------|-------------|
| `framework_id` | `str` | Framework used for evaluation |
| `framework_version` | `str` | Framework version |
| `passed` | `bool` | Whether the note passes the framework's policy thresholds |
| `overall_score` | `float` | Weighted mean element score, 0.0–1.0 |
| `overall_rating` | `str` | Human-readable compliance rating (see below) |
| `items` | `List[GapItem]` | Per-element evaluation results |
| `summary` | `str` | LLM-generated narrative summary of the evaluation |
| `stats` | `GapReportStats` | Counts by status and severity |
| `pii_masked` | `bool` | Whether PII masking was applied |
| `metadata` | `dict` | Caller-supplied metadata, passed through unchanged |

**Overall rating values:**

| Rating | Meaning |
|--------|---------|
| `Compliant` | Passed — all elements fully present |
| `Minor Gaps` | Passed — but some elements are partial or optional elements missing |
| `Requires Attention` | Failed — high/medium gaps, no critical blockers |
| `Non-Compliant` | Failed — one or more critical required elements missing or below threshold |

### GapItem

| Field | Type | Description |
|-------|------|-------------|
| `element_id` | `str` | Element identifier from the framework |
| `status` | `str` | `"present"`, `"partial"`, or `"missing"` |
| `score` | `float` | 0.0–1.0 quality score for this element |
| `evidence` | `Optional[str]` | Quote or paraphrase from the note supporting the assessment. `None` when element is missing. |
| `severity` | `str` | `"critical"`, `"high"`, `"medium"`, or `"low"` |
| `required` | `bool` | Whether this element is required by the framework |
| `suggestions` | `List[str]` | Actionable remediation suggestions for gaps (empty when `status == "present"`) |
| `notes` | `Optional[str]` | LLM reasoning (only populated when `verbose=True`) |

### Built-in Frameworks

| Framework ID | Description |
|-------------|-------------|
| `fca_suitability_v1` | FCA suitability note requirements under COBS 9.2 / PS13/1 (9 elements) |

### Custom Frameworks

Pass a path to your own YAML file:

```python
report = evaluate_note(
    note_text=note,
    framework="/path/to/my_framework.yaml",
    llm_config=config,
)
```

The YAML schema mirrors the built-in frameworks. See `assert_llm_tools/frameworks/fca_suitability_v1.yaml` for a reference example.

### Configurable Pass Policy

```python
from assert_llm_tools.metrics.note.models import PassPolicy

policy = PassPolicy(
    critical_partial_threshold=0.5,     # partial critical element treated as blocker if score < this
    required_pass_threshold=0.6,        # required element must score >= this to pass
    score_correction_missing_cutoff=0.2,
    score_correction_present_min=0.5,
    score_correction_present_floor=0.7,
)

report = evaluate_note(note_text=note, framework="fca_suitability_v1", pass_policy=policy, llm_config=config)
```

## LLM Configuration

```python
from assert_llm_tools import LLMConfig

# AWS Bedrock
config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
    # api_key / api_secret / aws_session_token for explicit credentials (optional — uses ~/.aws by default)
)

# OpenAI
config = LLMConfig(
    provider="openai",
    model_id="gpt-4o",
    api_key="your-openai-api-key",
)
```

### Supported Bedrock Model Families

| Model Family | Example Model IDs |
|-------------|-------------------|
| Amazon Nova | `us.amazon.nova-pro-v1:0`, `amazon.nova-lite-v1:0` |
| Anthropic Claude | `anthropic.claude-3-sonnet-20240229-v1:0` |
| Meta Llama | `meta.llama3-70b-instruct-v1:0` |
| Mistral AI | `mistral.mistral-large-2402-v1:0` |
| Cohere Command | `cohere.command-r-plus-v1:0` |
| AI21 Labs | `ai21.jamba-1-5-large-v1:0` |

## Proxy Configuration

```python
# Single proxy
config = LLMConfig(provider="bedrock", model_id="...", region="us-east-1",
                   proxy_url="http://proxy.example.com:8080")

# Protocol-specific
config = LLMConfig(provider="bedrock", model_id="...", region="us-east-1",
                   http_proxy="http://proxy.example.com:8080",
                   https_proxy="http://proxy.example.com:8443")

# Authenticated proxy
config = LLMConfig(provider="bedrock", model_id="...", region="us-east-1",
                   proxy_url="http://username:password@proxy.example.com:8080")
```

Standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables are also respected.

## PII Masking

Apply PII detection and masking before any text is sent to the LLM:

```python
# Summary evaluation
results = evaluate_summary(
    full_text=text, summary=summary, metrics=["coverage"],
    llm_config=config, mask_pii=True,
)

# Note evaluation
report = evaluate_note(note_text=note, framework="fca_suitability_v1",
                       llm_config=config, mask_pii=True)
```

> **Note:** `mask_pii=False` is the default. For production use with real client data, set `mask_pii=True`. Output files (e.g. `--output report.json`) may contain verbatim evidence quotes — treat them accordingly.

## License

MIT
