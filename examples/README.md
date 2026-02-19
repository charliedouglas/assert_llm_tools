# Examples

This directory contains synthetic example adviser notes with their expected evaluation output.
They serve two purposes:

1. **Documentation** — illustrating what a well-written suitability note looks like versus a
   note with gaps, so developers and compliance teams can calibrate expectations.

2. **Informal integration test fixtures** — the expected JSON files give an approximate reference
   output you can compare against when evaluating these notes with `assert_llm_tools`. Because
   LLM outputs are non-deterministic, exact match is not expected; the fixtures document the
   intended direction (scores, statuses, pass/fail) for regression-style checking.

---

## Directory structure

```
examples/
└── fca_suitability_v1/
    ├── compliant_note.txt              # Well-written note → Compliant
    ├── compliant_note_expected.json    # Expected GapReport (approximate)
    ├── minor_gaps_note.txt             # Note with partial elements → Minor Gaps
    ├── minor_gaps_note_expected.json   # Expected GapReport (approximate)
    ├── non_compliant_note.txt          # Note missing critical elements → Non-Compliant
    └── non_compliant_note_expected.json
```

---

## The notes

All three notes use fictional client details and are set against the
`fca_suitability_v1` framework (COBS 9.2 suitability elements).

| File | Client | Scenario | Expected rating | Expected `passed` | Expected score |
|------|--------|----------|-----------------|-------------------|----------------|
| `compliant_note.txt` | Margaret Thornton, 58 | Annual pension review, ISA subscription, full fact-find completed | Compliant | `true` | ~0.94 |
| `minor_gaps_note.txt` | James Whitfield, 42 | Initial advice, video call, financial fact-find deferred | Minor Gaps | `false` | ~0.58 |
| `non_compliant_note.txt` | David Patel, 35 | Initial advice, inheritance, very thin record | Non-Compliant | `false` | ~0.07 |

### What makes each note what it is

**`compliant_note.txt`** covers all nine framework elements. Objectives are specific and
quantified. ATR is evidenced with a named questionnaire and score. CfL is assessed
separately from ATR and treated as the binding constraint. Financial situation includes
income, expenditure, all assets and liabilities. Knowledge and experience is assessed with
duration and product familiarity noted. The recommendation rationale is client-specific with
explicit causal language. Charges are disclosed in both % and £ with a CCI cross-reference.
Alternatives were considered with reasoning. Client confirmation and next steps are recorded.

**`minor_gaps_note.txt`** covers the headline elements but has three notable gaps:
- **Knowledge and experience** is entirely absent — a required element under the framework.
- **Capacity for loss** relies on unverified client self-attestation; the adviser explicitly
  acknowledges the limitation and defers a full assessment. This is partial, not present.
- **Charges** are disclosed as percentages only, with no monetary illustration and no formal
  CCI issued at the meeting.

The adviser's note quality is adequate for an initial call but insufficient as a complete
suitability record. The gaps are remediable via the fact-find and suitability report.

**`non_compliant_note.txt`** is a bare transaction note. It lacks documented client
objectives, risk profiling, capacity for loss assessment, financial fact-find, and any
personalised recommendation rationale. Charges are mentioned without quantification. It reads
as a product sales note, not a suitability record. Five of seven required elements are missing.

---

## How to run

Evaluate a note against the framework using `assert_llm_tools`:

```python
from assert_llm_tools import evaluate_note

with open("examples/fca_suitability_v1/compliant_note.txt") as f:
    note_text = f.read()

report = evaluate_note(
    note_text=note_text,
    framework="fca_suitability_v1",
)

print(f"Passed: {report.passed}")
print(f"Score:  {report.overall_score:.2f}")
print(f"Summary: {report.summary}")
for item in report.items:
    print(f"  {item.element_id}: {item.status} ({item.score:.2f})")
```

Or via the CLI (if installed):

```bash
assert-llm evaluate-note \
  --note examples/fca_suitability_v1/compliant_note.txt \
  --framework fca_suitability_v1
```

### Comparing to expected output

The `*_expected.json` files contain illustrative `GapReport` JSON. To do a loose comparison:

```python
import json
from assert_llm_tools import evaluate_note

with open("examples/fca_suitability_v1/minor_gaps_note.txt") as f:
    note_text = f.read()

with open("examples/fca_suitability_v1/minor_gaps_note_expected.json") as f:
    expected = json.load(f)

report = evaluate_note(note_text=note_text, framework="fca_suitability_v1")

# Loose checks — direction of travel, not exact match
assert report.passed == expected["passed"], "Pass/fail mismatch"
assert abs(report.overall_score - expected["overall_score"]) < 0.25, "Score out of expected range"

# Check element statuses (LLM output may vary — use as a sanity check)
actual_statuses = {item.element_id: item.status for item in report.items}
for expected_item in expected["items"]:
    eid = expected_item["element_id"]
    if eid in actual_statuses:
        print(f"{eid}: expected={expected_item['status']}, actual={actual_statuses[eid]}")
```

---

## Notes on accuracy

The expected JSON files are **hand-crafted illustrative outputs**, not the result of running
`evaluate_note()` against these notes. They document the intended evaluation direction:

- Which elements should be `present`, `partial`, or `missing`
- Approximate score ranges
- Whether the overall report should `pass` or `fail`

Actual LLM output will differ in wording, exact scores, and evidence excerpts. These fixtures
are useful for sanity-checking that the model is directionally correct, not for exact
regression testing.
