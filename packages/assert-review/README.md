# assert-review

LLM-based compliance note evaluation for financial services.

Evaluates adviser suitability notes against regulatory framework definitions
(FCA, MiFID II, etc.), returning structured gap reports with per-element scores,
evidence, and actionable remediation suggestions.

## Installation

```bash
pip install assert-review
```

## Quick Start

```python
from assert_review import evaluate_note, LLMConfig

llm_config = LLMConfig(
    provider="bedrock",
    model_id="anthropic.claude-v2",
    region="us-east-1",
)

report = evaluate_note(
    note_text=open("note.txt").read(),
    framework="fca_suitability_v1",
    llm_config=llm_config,
)

print(f"Result: {'PASS' if report.passed else 'FAIL'} ({report.overall_rating})")
print(f"Score: {report.overall_score:.2%}")
```

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

## Bundled Frameworks

- `fca_suitability_v1` — FCA COBS 9.2 Suitability Note Framework

Custom frameworks can be supplied as a YAML file path or pre-loaded dict.

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

## GapReport Output

```json
{
  "framework_id": "fca_suitability_v1",
  "framework_version": "1.0.0",
  "passed": true,
  "overall_score": 0.82,
  "overall_rating": "Compliant",
  "summary": "The note meets all 9 FCA suitability requirements...",
  "stats": { "total_elements": 9, "present_count": 9, ... },
  "items": [
    {
      "element_id": "client_objectives",
      "status": "present",
      "score": 0.9,
      "evidence": "Client states retirement goal in 15 years",
      "severity": "critical",
      "required": true,
      "suggestions": []
    }
  ]
}
```

## Dependencies

- [assert-core](https://pypi.org/p/assert-core) — shared LLM provider layer
- PyYAML — framework loading

Supports AWS Bedrock and OpenAI providers via `assert-core`.
