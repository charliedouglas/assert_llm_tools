# Custom Framework Guide

This guide explains how to create your own evaluation framework for `evaluate_note()`. A framework is a YAML file that defines the elements a compliance note must contain. The library ships with built-in FCA frameworks, but you can write your own to reflect house standards, jurisdiction-specific requirements, or internal quality criteria.

**Time to first working framework: approximately 15 minutes.**

---

## When to create a custom framework

- **House rules** — your firm has additional documentation standards beyond the regulatory minimum (e.g. mandatory fact-find reference numbers, investment committee sign-off notes)
- **Jurisdiction-specific requirements** — you operate under rules that differ from, or extend, the built-in FCA frameworks (e.g. CBI requirements in Ireland, MAS rules in Singapore)
- **Product-specific standards** — evaluation criteria for protection advice, mortgage recommendations, or drawdown reviews differ meaningfully from the standard investment suitability template
- **Internal quality scoring** — you want to assess notes against a higher bar than the regulatory floor, tracking firm-defined quality indicators rather than pass/fail compliance
- **Piloting new requirements** — draft a framework for incoming regulatory changes before they take effect, so your team can assess readiness ahead of the go-live date

---

## YAML schema reference

A framework file contains a top-level metadata block and a list of `elements`. Every field is described below.

### Top-level fields

```yaml
framework_id: string          # Required. Unique identifier. Use snake_case. Example: my_firm_suitability_v1
name: string                  # Required. Human-readable name shown in reports.
version: string               # Required. Semantic version string. Example: 1.0.0
regulator: string             # Required. Regulator or authority this maps to. Example: FCA, CBI, internal
description: string           # Recommended. One or two sentences describing scope and purpose.
effective_date: string        # Optional. ISO date (YYYY-MM-DD) when this framework applies from.
reference: string             # Optional. Regulation or rulebook reference. Example: COBS 9.2 / PS13/1
```

### Element fields

Each entry in the `elements` list represents one thing the note must contain.

```yaml
elements:
  - id: string                # Required. Unique identifier within this framework. Example: client_objectives
    description: string       # Required. What the element requires — written clearly enough for the LLM to act on.
    required: boolean         # Required. true = absence can block a pass. false = optional/recommended.
    severity: string          # Required. Impact level if absent. One of: critical | high | medium | low
    guidance: string          # Recommended. Evaluation tips for the LLM — see below.
    examples: list[string]    # Recommended. Phrases/patterns that signal this element is present.
    anti_patterns: list[string]  # Recommended. Phrases that look relevant but do not satisfy the requirement.
    meeting_type_overrides:   # Optional. Adjust severity or required status per meeting type.
      <meeting_type>:
        severity: string      # Override severity for this meeting type.
        required: boolean     # Override required flag for this meeting type.
```

### Severity levels

| Severity | Meaning |
|---|---|
| `critical` | Absence is a near-certain compliance breach. Blocks a pass by default. |
| `high` | Absence is a significant gap. Blocks a pass by default. |
| `medium` | Absence is a quality concern but not an automatic failure. |
| `low` | Absence is minor or contextually acceptable. |

---

## The `examples` and `anti_patterns` fields

These two fields are the most important levers for improving LLM accuracy.

### Why they matter

The LLM's evaluation of each element is based on a focused prompt that includes your `description` and `guidance`. Without examples, the model has to infer what "adequate documentation of X" looks like. This leads to:

- False positives — the model accepts a vague mention as "present" when it falls short of the required standard
- False negatives — the model marks a well-written element as "partial" because it doesn't recognise the firm's preferred phrasing

`examples` and `anti_patterns` reduce both problems by anchoring the model to what good (and insufficiently good) evidence actually looks like in practice.

### `examples`

A list of phrases, sentences, or patterns drawn from real notes that clearly satisfy the requirement. These should be representative, not exhaustive.

```yaml
  - id: capacity_for_loss
    description: >
      The note must document whether the client can sustain capital loss
      without material impact to their standard of living.
    examples:
      - "Client holds six months' emergency fund and has no dependents; loss of invested capital would not affect day-to-day living."
      - "CFL assessed as low — client's income is solely from the investment portfolio."
      - "Client confirmed they could tolerate a full loss of this amount without affecting essential expenditure."
```

### `anti_patterns`

Phrases that appear related to the requirement but do not adequately satisfy it. Including these prevents the model from accepting boilerplate.

```yaml
    anti_patterns:
      - "Capacity for loss discussed."          # Vague — no substance
      - "Client understands the risks."         # Conflates CFL with ATR
      - "Risk acknowledged."                    # No CFL-specific assessment
```

**Guideline:** aim for three to five examples and two to four anti_patterns per element. Drawn from your own note archive, these will reflect your adviser population's actual writing patterns.

---

## The `meeting_type_overrides` section

Some elements are essential in an initial advice meeting but less critical — or inappropriate — in a review or drawdown meeting. `meeting_type_overrides` lets you adjust the `severity` and `required` flag for specific meeting types without duplicating the entire framework.

### How it works

When you call `evaluate_note()` with a `metadata` dict containing `meeting_type`, the evaluator applies overrides for that meeting type before running the assessment.

```python
report = evaluate_note(
    note_text=note,
    framework="my_framework.yaml",
    llm_config=config,
    metadata={"meeting_type": "annual_review"}
)
```

### Promoting and demoting elements

```yaml
  - id: knowledge_and_experience
    description: >
      The note must record the client's knowledge of and experience with the
      relevant investment type.
    required: true
    severity: high
    meeting_type_overrides:
      annual_review:
        severity: medium    # Less critical at review if recorded at inception
        required: false     # Don't block a pass if unchanged from initial assessment
      drawdown_review:
        severity: high
        required: true      # Still important when recommending drawdown access
```

Valid values for `severity` and `required` within overrides are the same as at the top level. Any field you omit in an override inherits the element-level default.

---

## Step-by-step: create your first framework in 15 minutes

### Step 1 — Identify your elements (5 minutes)

List the things a note *must* contain to meet your standard. For a first pass, aim for five to ten elements. Draw from:

- Your existing file review checklist
- Your T&C framework or competency standards
- Any firm-specific documentation requirements in your compliance manual

### Step 2 — Draft the YAML (5 minutes)

Create a file, e.g. `frameworks/my_firm_initial_v1.yaml`:

```yaml
framework_id: my_firm_initial_v1
name: My Firm Initial Advice Framework
version: 1.0.0
regulator: FCA
description: >
  Internal documentation standard for initial investment advice notes,
  extending COBS 9.2 with firm-specific quality requirements.
effective_date: "2025-01-01"
reference: COBS 9.2 / My Firm TCF Policy v3.2

elements:

  - id: client_objectives
    description: >
      The note must document the client's investment objectives including
      time horizon, purpose, and any specific goals.
    required: true
    severity: critical
    guidance: >
      Look for explicit statements of what the client wants to achieve.
      Generic phrases like "to grow their money" without a time horizon
      or purpose are insufficient.
    examples:
      - "Client aims to retire at 62 with a target pension pot of £400,000."
      - "Investment objective: income of £1,200/month from age 67, 12-year horizon."
    anti_patterns:
      - "Client wants to invest."
      - "Growth objective noted."

  - id: risk_attitude
    description: >
      The note must record the client's attitude to risk, including the
      assigned risk category from a completed risk questionnaire.
    required: true
    severity: critical
    guidance: >
      A named risk category (cautious, balanced, adventurous) or numerical
      score from a risk tool must be present. Adviser-only assessment without
      a tool reference is insufficient under firm policy.
    examples:
      - "Risk questionnaire completed — Dynamic Planner score 4/10, Balanced."
      - "ATR: Cautious (Finametrica score 38)."
    anti_patterns:
      - "Risk discussed."
      - "Client is comfortable with some risk."

  - id: capacity_for_loss
    description: >
      The note must separately assess capacity for loss, distinct from attitude
      to risk, with reference to the client's financial resilience.
    required: true
    severity: critical
    examples:
      - "CFL: Moderate. Client holds 6 months' emergency fund; loss of invested capital would not affect living standards."
    anti_patterns:
      - "Client understands the risks."
      - "Capacity for loss: acceptable."

  - id: recommendation_rationale
    description: >
      The note must explain why the recommended product is suitable for this
      specific client, with explicit links to objectives, risk profile, and
      financial situation.
    required: true
    severity: critical
    guidance: >
      Generic product descriptions do not qualify. The rationale must be
      personalised. Look for causal language linking client attributes to
      the chosen product.
    examples:
      - "This fund was selected because its balanced growth profile matches the client's ATR (4/7), 10-year horizon, and moderate CFL."
    anti_patterns:
      - "This is a suitable product for the client."
      - "Recommended product meets client needs."

  - id: charges_and_costs
    description: >
      The note must disclose all charges including product OCF/AMC, platform
      fee, and adviser fee (initial and ongoing).
    required: true
    severity: high
    examples:
      - "OCF 0.22%, platform 0.15%, adviser initial 1%, adviser ongoing 0.5%."
    anti_patterns:
      - "Costs discussed and agreed."
      - "Fees as per schedule of services."
```

### Step 3 — Validate your framework (2 minutes)

```python
from assert_llm_tools.metrics.note.loader import load_framework

# This raises ValueError if anything is wrong
framework = load_framework("/path/to/my_firm_initial_v1.yaml")
print(f"Loaded: {framework['name']} — {len(framework['elements'])} elements")
```

### Step 4 — Run a test evaluation (3 minutes)

```python
from assert_llm_tools import evaluate_note, LLMConfig

config = LLMConfig(provider="openai", model_id="gpt-4o", api_key="sk-...")

with open("sample_note.txt") as f:
    note = f.read()

report = evaluate_note(
    note_text=note,
    framework="/path/to/my_firm_initial_v1.yaml",
    llm_config=config,
    verbose=True   # include LLM reasoning in each GapItem.notes
)

for item in report.items:
    print(f"{item.element_id:30s} {item.status:10s} ({item.severity})")
    if item.notes:
        print(f"  → {item.notes}")
```

---

## How to use your framework

Pass a path string to `evaluate_note()`:

```python
from assert_llm_tools import evaluate_note, LLMConfig

report = evaluate_note(
    note_text=note,
    framework="/path/to/my_framework.yaml",
    llm_config=config
)
```

You can also pass a pre-loaded dict (useful in test environments or when building frameworks programmatically):

```python
import yaml

with open("/path/to/my_framework.yaml") as f:
    framework_dict = yaml.safe_load(f)

report = evaluate_note(note_text=note, framework=framework_dict, llm_config=config)
```

---

## Minimal working example

The smallest valid framework has five elements and covers the required YAML structure:

```yaml
framework_id: minimal_example_v1
name: Minimal Example Framework
version: 1.0.0
regulator: internal

elements:

  - id: client_name_and_date
    description: The note must identify the client and the date of the meeting.
    required: true
    severity: high

  - id: investment_objective
    description: The note must state the client's investment objective and time horizon.
    required: true
    severity: critical

  - id: risk_category
    description: The note must record the client's assigned risk category.
    required: true
    severity: critical

  - id: product_recommended
    description: The note must name the specific product or fund recommended.
    required: true
    severity: high

  - id: adviser_signature_reference
    description: The note should reference the adviser sign-off or report reference number.
    required: false
    severity: low
```

---

## Common mistakes and how to avoid them

### Description is too vague

**Problem:** `description: Client risk discussed.`

The LLM will have difficulty deciding what "discussed" means in practice. Is a sentence enough? A paragraph?

**Fix:** Write the description as a testable requirement: `The note must record the client's attitude to risk, including the outcome of a risk profiling exercise and the assigned risk category.`

---

### Missing guidance on edge cases

**Problem:** Your `charges_and_costs` element keeps marking notes as "present" when they only say "fees as per schedule of services".

**Fix:** Add this to `anti_patterns`:

```yaml
anti_patterns:
  - "Fees as per schedule of services."
  - "Costs confirmed separately."
```

And add to `guidance`: "A cross-reference to a separate costs disclosure document is only acceptable if the document reference is explicitly named in the note."

---

### Severity mismatch with your pass policy

**Problem:** You mark `alternatives_considered` as `severity: critical`, but it's not a hard regulatory requirement and you don't want it to block a pass.

**Fix:** Either set `severity: medium` or set `required: false`. The `PassPolicy` only blocks on critical and high elements that are `required: true`.

---

### Duplicating elements across frameworks instead of using overrides

**Problem:** You maintain three separate YAML files (initial, review, drawdown) that are 90% identical.

**Fix:** Maintain one base framework and use `meeting_type_overrides` to adjust the handful of elements that differ per meeting type.

---

### Using examples that are too long

**Problem:** Examples are full paragraphs copied from notes. The LLM context window fills up when the framework has many elements.

**Fix:** Keep examples to one or two sentences — enough to show the pattern, not a transcript.

---

## Related documentation

- [API Reference](api-reference.md) *(coming soon)*
- [Built-in frameworks source](../assert_llm_tools/frameworks/) — review `fca_suitability_v1.yaml` as a worked example of a production-quality framework
