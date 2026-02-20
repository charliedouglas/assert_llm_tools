# assert-review

LLM-based compliance note evaluation for financial services.

Evaluates adviser suitability notes against regulatory framework definitions (FCA, MiFID II, etc.), returning structured gap reports with per-element scores, evidence quotes, and actionable remediation suggestions. No PyTorch, no BERT, no heavy dependencies.

> ⚠️ **Experimental — do not use in live or production systems.**
>
> Outputs are non-deterministic (LLM-based) and have not been validated against real regulatory decisions. This package is intended for research, prototyping, and internal tooling only. It is not a substitute for qualified compliance review and must not be used to make or support live regulatory or client-facing decisions.

## Installation

```bash
pip install assert-review
```

## Quick Start

```python
from assert_review import evaluate_note, LLMConfig

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

## evaluate_note()

Full parameter reference:

```python
from assert_review import evaluate_note, LLMConfig, PassPolicy

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

## GapReport

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

## GapItem

| Field | Type | Description |
|-------|------|-------------|
| `element_id` | `str` | Element identifier from the framework |
| `status` | `str` | `"present"`, `"partial"`, or `"missing"` |
| `score` | `float` | 0.0–1.0 quality score for this element |
| `evidence` | `Optional[str]` | Quote or paraphrase from the note supporting the assessment. `None` when element is missing. |
| `severity` | `str` | `"critical"`, `"high"`, `"medium"`, or `"low"` |
| `required` | `bool` | Whether this element is required by the framework |
| `suggestions` | `List[str]` | Actionable remediation suggestions (empty when `status == "present"`) |
| `notes` | `Optional[str]` | LLM reasoning (only populated when `verbose=True`) |

## Verbose Output

Pass `verbose=True` to include per-element LLM reasoning in `GapItem.notes`:

```python
report = evaluate_note(
    note_text=note,
    framework="fca_suitability_v1",
    llm_config=config,
    verbose=True,
)

for item in report.items:
    if item.notes:
        print(f"{item.element_id}: {item.notes}")
```

## Custom Evaluation Instructions

Append additional instructions to all element prompts for domain-specific guidance:

```python
report = evaluate_note(
    note_text=note,
    framework="fca_suitability_v1",
    llm_config=config,
    custom_instruction="This note relates to a high-net-worth client with complex tax considerations. Apply stricter standards for risk and objectives documentation.",
)
```

## PII Masking

Pass `mask_pii=True` to detect and mask personally identifiable information before any text is sent to the LLM:

```python
report = evaluate_note(
    note_text=note,
    framework="fca_suitability_v1",
    llm_config=config,
    mask_pii=True,
)
```

`mask_pii=False` is the default. For production use with real client data, set `mask_pii=True`. Note that output fields like `GapItem.evidence` may contain verbatim quotes from the note — treat them accordingly.

## Configurable Pass Policy

Override the default pass/fail thresholds:

```python
from assert_review import PassPolicy

policy = PassPolicy(
    critical_partial_threshold=0.5,      # partial critical element treated as blocker if score < this
    required_pass_threshold=0.6,         # required element must score >= this to pass
    score_correction_missing_cutoff=0.2,
    score_correction_present_min=0.5,
    score_correction_present_floor=0.7,
)

report = evaluate_note(
    note_text=note,
    framework="fca_suitability_v1",
    llm_config=config,
    pass_policy=policy,
)
```

## Bundled Frameworks

| Framework ID | Description |
|---|---|
| `fca_suitability_v1` | FCA suitability note requirements under COBS 9.2 / PS13/1 (9 elements) |

## Custom Frameworks

Pass a path to your own YAML file:

```python
report = evaluate_note(
    note_text=note,
    framework="/path/to/my_framework.yaml",
    llm_config=config,
)
```

The YAML schema mirrors the built-in frameworks. See `packages/assert-review/assert_review/frameworks/fca_suitability_v1.yaml` in the [source repo](https://github.com/charliedouglas/assert_llm_tools) for a reference example.

## CLI

```bash
# Evaluate a single note
assert-review evaluate note.txt --framework fca_suitability_v1

# Output as JSON
assert-review evaluate note.txt --framework fca_suitability_v1 --output json

# Batch evaluate from CSV
assert-review batch notes.csv --framework fca_suitability_v1 --note-column text

# Use OpenAI instead of Bedrock
assert-review evaluate note.txt --framework fca_suitability_v1 \
  --provider openai --model gpt-4o --api-key $OPENAI_API_KEY
```

## LLM Configuration

```python
from assert_review import LLMConfig

# AWS Bedrock (uses ~/.aws credentials by default)
config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
)

# AWS Bedrock with explicit credentials
config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
    api_key="your-aws-access-key-id",
    api_secret="your-aws-secret-access-key",
    aws_session_token="your-session-token",  # optional
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
|---|---|
| Amazon Nova | `us.amazon.nova-pro-v1:0`, `amazon.nova-lite-v1:0` |
| Anthropic Claude | `anthropic.claude-3-sonnet-20240229-v1:0` |
| Meta Llama | `meta.llama3-70b-instruct-v1:0` |
| Mistral AI | `mistral.mistral-large-2402-v1:0` |
| Cohere Command | `cohere.command-r-plus-v1:0` |
| AI21 Labs | `ai21.jamba-1-5-large-v1:0` |

## Proxy Configuration

```python
# Single proxy
config = LLMConfig(
    provider="bedrock", model_id="us.amazon.nova-pro-v1:0", region="us-east-1",
    proxy_url="http://proxy.example.com:8080",
)

# Protocol-specific proxies
config = LLMConfig(
    provider="bedrock", model_id="us.amazon.nova-pro-v1:0", region="us-east-1",
    http_proxy="http://proxy.example.com:8080",
    https_proxy="http://proxy.example.com:8443",
)

# Authenticated proxy
config = LLMConfig(
    provider="bedrock", model_id="us.amazon.nova-pro-v1:0", region="us-east-1",
    proxy_url="http://username:password@proxy.example.com:8080",
)
```

Standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables are also respected.

## Public API

```python
from assert_review import (
    evaluate_note,     # main entry point
    NoteEvaluator,     # evaluator class for advanced use
    GapReport,         # full evaluation result
    GapItem,           # per-element result
    GapReportStats,    # summary statistics
    PassPolicy,        # pass/fail threshold configuration
    LLMConfig,         # re-exported from assert-core
)
```

## Dependencies

- [assert-core](https://pypi.org/p/assert-core) — shared LLM provider layer (AWS Bedrock, OpenAI)
- PyYAML — framework loading

## Migrating from assert_llm_tools

`assert-review` replaces the compliance note evaluation functionality of `assert_llm_tools`, which is now deprecated. Swap the imports:

```python
# Before
from assert_llm_tools import evaluate_note, LLMConfig
from assert_llm_tools.metrics.note.models import PassPolicy, GapReport, GapItem

# After
from assert_review import evaluate_note, LLMConfig, PassPolicy, GapReport, GapItem
```

## License

MIT
