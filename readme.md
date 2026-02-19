# assert-llm-tools

**LLM-powered evaluation of financial adviser notes against FCA regulatory frameworks.**

assert-llm-tools checks whether a suitability note contains the elements required under FCA rules — objectively, consistently, and at scale. Feed it a note; it returns a structured gap report identifying what is present, what is partial, and what is missing, with evidence drawn directly from the text.

This is a tool to assist compliance review. It is not a substitute for legal advice or professional regulatory judgement.

---

## Who it's for

- **Compliance officers and heads of compliance** running T&C reviews or file audits
- **RegTech and oversight teams** building automated QA workflows into advice processes
- **Compliance consultants** assessing suitability note quality across adviser panels
- **Technology teams** integrating compliance checks into advice platforms or back-office systems

---

## Installation

```bash
pip install assert-llm-tools
```

Requires Python 3.8+. An LLM provider API key is required for evaluation (OpenAI or AWS Bedrock — see [LLM configuration](#llm-configuration)).

---

## Quick start

### Evaluate a note from Python

```python
from assert_llm_tools import evaluate_note, LLMConfig

llm_config = LLMConfig(
    provider="openai",
    model_id="gpt-4o",
    api_key="sk-..."
)

note = """
    Client: Mr. James Hargreaves, age 58. Seeking to consolidate pension pots
    ahead of planned retirement at 65. Risk questionnaire completed — scored
    balanced (4/7). Capacity for loss assessed as moderate; client has £12,000
    in instant-access savings and no essential expenditure dependent on these funds.
    Recommended: Fidelity Multi-Asset Balanced Fund (OCF 0.35%). Adviser fee: 1%
    initial, 0.5% ongoing. Recommendation chosen over cautious alternative due
    to client's 7-year horizon and stated growth objective.
"""

report = evaluate_note(
    note_text=note,
    framework="fca_suitability_v1",
    llm_config=llm_config
)

print(f"Result:  {'PASS' if report.passed else 'FAIL'}")
print(f"Score:   {report.overall_score:.0%}")
print(f"Summary: {report.summary}")

for item in report.items:
    if item.status != "present":
        print(f"  [{item.severity.upper()}] {item.element_id}: {item.status}")
```

### Evaluate a note from the command line

```bash
python - <<'EOF'
from assert_llm_tools import evaluate_note, LLMConfig
import json, dataclasses

config = LLMConfig(provider="openai", model_id="gpt-4o", api_key="sk-...")

with open("note.txt") as f:
    note = f.read()

report = evaluate_note(note, framework="fca_suitability_v1", llm_config=config)

# Print gap items as JSON
print(json.dumps(
    [dataclasses.asdict(item) for item in report.items if item.status != "present"],
    indent=2
))
EOF
```

---

## Output example

### Terminal summary

```
Result:  FAIL
Score:   61%
Summary: The note addresses client objectives, risk attitude, capacity for loss,
         and charges adequately. Key gaps: knowledge and experience is not
         documented, and the recommendation rationale lacks explicit linkage to
         the client's risk profile. One critical gap and one high gap were
         identified in required elements.

Gaps:
  [CRITICAL] recommendation_rationale: partial
  [HIGH]     knowledge_and_experience: missing
  [MEDIUM]   alternatives_considered: missing
```

### JSON gap report (excerpt)

```json
{
  "framework_id": "fca_suitability_v1",
  "framework_version": "1.0.0",
  "passed": false,
  "overall_score": 0.61,
  "summary": "The note addresses client objectives, risk attitude, capacity for loss, and charges adequately. Key gaps: knowledge and experience is not documented, and the recommendation rationale lacks explicit linkage to the client's risk profile. One critical gap and one high gap were identified in required elements.",
  "stats": {
    "total_elements": 9,
    "required_elements": 7,
    "present_count": 5,
    "partial_count": 1,
    "missing_count": 3,
    "critical_gaps": 1,
    "high_gaps": 1,
    "medium_gaps": 1,
    "low_gaps": 0,
    "required_missing_count": 2
  },
  "items": [
    {
      "element_id": "recommendation_rationale",
      "status": "partial",
      "score": 0.45,
      "severity": "critical",
      "required": true,
      "evidence": "Recommendation chosen over cautious alternative due to client's 7-year horizon",
      "notes": "Rationale references time horizon but does not link back to risk profile or capacity for loss."
    },
    {
      "element_id": "knowledge_and_experience",
      "status": "missing",
      "score": 0.0,
      "severity": "high",
      "required": true,
      "evidence": "",
      "notes": "No assessment of client's prior investment experience or product familiarity."
    }
  ]
}
```

---

## Key features

| Feature | Detail |
|---|---|
| **Built-in FCA frameworks** | Ship-ready frameworks for COBS 9.2 suitability and wealth advice scenarios |
| **Custom framework support** | Define your own elements in YAML — house rules, jurisdiction variations, internal standards |
| **Structured JSON output** | Machine-readable `GapReport` with per-element status, score, and evidence |
| **Meeting type context** | Promote or demote framework elements depending on meeting type (initial, review, drawdown) |
| **PII masking** | Strip client-identifiable data before sending to an LLM provider |
| **Pass/fail policy** | Configurable thresholds — block on critical gaps, allow partial high elements, etc. |
| **Multi-provider LLM** | OpenAI and AWS Bedrock supported; pluggable for private model deployments |

---

## Available frameworks

| Framework ID | Regulation | Description |
|---|---|---|
| `fca_suitability_v1` | COBS 9.2 / PS13/1 | Core suitability note requirements for retail investment advice: objectives, risk, capacity for loss, financial situation, knowledge and experience, rationale, charges, and alternatives. |
| `fca_wealth` | COBS 9.2 (wealth variant) | Coming soon — extended element set for discretionary and wealth management scenarios. |

To use a built-in framework, pass its ID as a string:

```python
report = evaluate_note(note_text=note, framework="fca_suitability_v1", llm_config=config)
```

To use a custom framework, pass a path to your YAML file:

```python
report = evaluate_note(note_text=note, framework="/path/to/my_framework.yaml", llm_config=config)
```

---

## LLM configuration

```python
# OpenAI
from assert_llm_tools import LLMConfig

config = LLMConfig(
    provider="openai",
    model_id="gpt-4o",
    api_key="sk-..."
)

# AWS Bedrock
config = LLMConfig(
    provider="bedrock",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1"
)
```

For corporate proxy environments, pass `proxy_url`, `http_proxy`, or `https_proxy` to `LLMConfig`.

---

## Pass/fail policy

By default, a note fails if any critical required element is missing or if any high required element is absent. This can be overridden:

```python
from assert_llm_tools import evaluate_note, PassPolicy

policy = PassPolicy(
    block_on_critical_missing=True,
    block_on_critical_partial=False,   # allow partial critical elements
    block_on_high_missing=True,
    critical_partial_threshold=0.5
)

report = evaluate_note(note, framework="fca_suitability_v1", llm_config=config, pass_policy=policy)
```

---

## PII masking

To prevent client-identifiable information from being sent to a third-party LLM provider:

```python
report = evaluate_note(
    note_text=note,
    framework="fca_suitability_v1",
    llm_config=config,
    mask_pii=True
)

print(report.pii_masked)  # True
```

---

## Documentation

- [API Reference](docs/api-reference.md) *(coming soon)*
- [Custom Framework Guide](docs/custom-frameworks.md) — create your own evaluation framework

---

## Regulatory context

assert-llm-tools is designed to assist compliance teams in reviewing adviser notes at scale. Evaluation results reflect an LLM's interpretation of a note against a defined set of elements and should be treated as a first-pass screening tool. They do not constitute a regulatory determination, legal opinion, or assurance of compliance.

Firms remain responsible for their own compliance processes and for exercising appropriate professional judgement on individual cases. This tool is not approved or endorsed by the FCA.

---

## License

MIT
