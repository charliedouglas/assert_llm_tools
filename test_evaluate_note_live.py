from assert_review import evaluate_note
from assert_review import LLMConfig

# Select one of Bedrock or OpenAI
llm_config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
    # http_proxy="http://username:password@localhost:8080",
)

# Realistic FCA suitability note — covers most required elements
note_text = """
Client Meeting Note — Margaret Thornton
Date: 14 February 2026
Adviser: James Patel
Meeting type: Annual review

Client profile:
Margaret is 58 years old and retired from teaching in 2023. She holds an existing
stocks and shares ISA (current value approx. £142,000) invested in a balanced
managed fund, and a defined contribution pension pot with Aviva (current value
approx. £310,000). She has no outstanding mortgage or significant liabilities.
Monthly income from state pension and part-time tutoring totals £2,400.

Objectives:
Margaret's primary goal is to preserve capital while generating a modest income
supplement of £800–£1,000 per month from age 62. She has a secondary objective
to pass on residual assets to her two adult children and has expressed interest
in tax-efficient gifting strategies.

Risk profile:
Margaret completed the firm's risk questionnaire at the start of the meeting and
scored 34/60, placing her in the Balanced risk category. She understands that
her investments may fall in value and confirmed she is comfortable with short-term
volatility provided her long-term goals are preserved. Capacity for loss was
assessed as moderate — she has sufficient liquid savings (£28,000 in cash ISA
and current account) to cover 18 months of essential expenditure without drawing
on investments.

Knowledge and experience:
Margaret has held investment ISAs for over 12 years and previously worked through
an adviser to consolidate two legacy pension plans. She is familiar with fund-based
investing but has no direct equity experience. Assessed as having adequate knowledge
for the recommended product type.

Recommendation:
I recommended Margaret maintain her existing balanced managed fund ISA but switch
provider to reduce the ongoing charge from 0.85% to 0.45% (saving approximately
£640 per year at current value). For the pension, I recommended a phased drawdown
strategy beginning at age 62, drawing £1,000 per month initially, reviewed annually.
This recommendation is based on her balanced risk profile, moderate capacity for
loss, 4-year time horizon to drawdown, and income supplement objective.

Charges and costs:
Platform fee: 0.20% p.a. Ongoing adviser charge: 0.50% p.a. Fund OCF: 0.45% p.a.
Total ongoing cost: 1.15% p.a. (approx. £1,638/year at current combined value of
£452,000). One-off switching cost: £nil (in-specie transfer). Costs were discussed
and documented in the accompanying costs and charges illustration.

Client acknowledgement:
Margaret confirmed she had read and understood the suitability report issued prior
to the meeting. She agreed to proceed with the ISA provider switch and authorised
the drawdown nomination on the pension. She was reminded that she has a 30-day
cooling-off period.
"""

print("Evaluating note against fca_suitability_v1...")
print("=" * 60)

report = evaluate_note(
    note_text=note_text,
    framework="fca_suitability_v1",
    llm_config=llm_config,
    verbose=True,
)

print(f"Framework:      {report.framework_id} v{report.framework_version}")
print(f"Overall rating: {report.overall_rating}")
print(f"Overall score:  {report.overall_score:.2f}")
print(f"Passed:         {report.passed}")
print(f"Stats:          {report.stats.present_count} present / {report.stats.partial_count} partial / {report.stats.missing_count} missing")
print()
print("=" * 60)
print("ELEMENT BREAKDOWN")
print("=" * 60)

for item in report.items:
    status_icon = {"present": "✓", "partial": "◐", "missing": "●"}.get(item.status, "?")
    print(f"\n{status_icon} [{item.severity.upper()}] {item.element_id}")
    print(f"  Status:   {item.status}  |  Score: {item.score:.2f}")
    if item.evidence:
        print(f"  Evidence: {item.evidence[:120]}")
    elif item.evidence is None:
        print(f"  Evidence: (none — element missing)")
    if item.suggestions:
        print(f"  Suggestions:")
        for s in item.suggestions:
            print(f"    - {s}")
    if item.notes:
        print(f"  Notes:    {item.notes[:120]}")

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(report.summary)
