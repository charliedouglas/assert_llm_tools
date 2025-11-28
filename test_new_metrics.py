"""
Test script demonstrating the new metric names (coverage, factual_consistency, factual_alignment).

This test shows the improved metrics introduced in Phase 1 of the metric improvements:
- coverage (replaces faithfulness): measures source claim coverage in summary (recall)
- factual_consistency (replaces hallucination): measures summary claim support (precision)
- factual_alignment: F1 score combining coverage and factual_consistency
- Improved redundancy detection using semantic similarity
"""

from assert_llm_tools.core import evaluate_summary
from assert_llm_tools.llm.config import LLMConfig

# Configure LLM
llm_config = LLMConfig(
    provider="bedrock",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1",
)

# Example text and summary
full_text = """
The James Webb Space Telescope (JWST) has revolutionized our understanding of the cosmos since its launch in 2021.
As the largest and most powerful space telescope ever built, it has provided unprecedented views of distant galaxies,
exoplanets, and cosmic phenomena. The telescope's infrared capabilities allow it to peer through cosmic dust and gas,
revealing previously hidden details about star formation and galaxy evolution. Scientists have already used JWST data
to make groundbreaking discoveries, including observations of some of the earliest galaxies formed after the Big Bang
and detailed atmospheric analysis of potentially habitable exoplanets.
"""

summary = """
The James Webb Space Telescope, launched in 2021, is revolutionizing space observation with its powerful infrared
capabilities, enabling scientists to study early galaxies and exoplanets in unprecedented detail.
"""

# Example 1: Test coverage (what % of source claims are in summary)
print("=" * 80)
print("Example 1: Coverage Metric (Source Claim Coverage)")
print("=" * 80)
print("\nCoverage measures: What percentage of claims from the source appear in the summary?")
print("This is a RECALL metric - higher means more complete coverage of source material.")
results_coverage = evaluate_summary(
    full_text=full_text,
    summary=summary,
    metrics=["coverage"],
    llm_config=llm_config,
    show_progress=False
)
print("\nCoverage Results:")
for key, value in results_coverage.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Example 2: Test factual_consistency (what % of summary claims are supported)
print("\n" + "=" * 80)
print("Example 2: Factual Consistency Metric (Summary Claim Support)")
print("=" * 80)
print("\nFactual Consistency measures: What percentage of summary claims are supported by the source?")
print("This is a PRECISION metric - higher means more accurate/grounded summary.")

# Create a summary with potential unsupported claims
summary_with_hallucination = """
The James Webb Space Telescope, launched in 2021, is revolutionizing space observation with its powerful infrared
capabilities. It cost over $50 billion to build and can detect planets in other solar systems with 99% accuracy.
The telescope has discovered alien life on several exoplanets.
"""

results_consistency = evaluate_summary(
    full_text=full_text,
    summary=summary_with_hallucination,
    metrics=["factual_consistency"],
    llm_config=llm_config,
    show_progress=False
)
print("\nFactual Consistency Results (for summary with potential unsupported claims):")
for key, value in results_consistency.items():
    if key == "debug_info":
        print(f"  {key}:")
        for item in value:
            print(f"    - {item['claim'][:60]}... : {'SUPPORTED' if item['is_supported'] else 'UNSUPPORTED'}")
    elif isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Example 3: Test factual_alignment (F1 of coverage and consistency)
print("\n" + "=" * 80)
print("Example 3: Factual Alignment Metric (F1 Score)")
print("=" * 80)
print("\nFactual Alignment combines coverage and factual_consistency into a single F1 score.")
print("This provides a balanced measure of completeness AND accuracy.")

results_alignment = evaluate_summary(
    full_text=full_text,
    summary=summary,
    metrics=["factual_alignment"],
    llm_config=llm_config,
    show_progress=False
)
print("\nFactual Alignment Results:")
for key, value in results_alignment.items():
    if isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")
    elif key not in ["debug_info"]:
        print(f"  {key}: {value}")

print("\nInterpretation:")
print(f"  Coverage (recall): {results_alignment.get('coverage', 0):.2f} - How much of source is in summary")
print(f"  Factual Consistency (precision): {results_alignment.get('factual_consistency', 0):.2f} - How much of summary is supported")
print(f"  Factual Alignment (F1): {results_alignment.get('factual_alignment', 0):.2f} - Balanced score")

# Example 4: Test improved redundancy detection
print("\n" + "=" * 80)
print("Example 4: Improved Redundancy Detection (Semantic Similarity)")
print("=" * 80)
print("\nRedundancy now uses semantic similarity instead of string matching.")
print("This catches paraphrased redundancy more effectively.")

redundant_summary = """
The James Webb Space Telescope was launched in 2021. JWST began operations in 2021.
It has infrared capabilities. The telescope can observe in the infrared spectrum.
Scientists are using it to study galaxies. Researchers utilize JWST to examine distant galaxies.
"""

results_redundancy = evaluate_summary(
    full_text=full_text,
    summary=redundant_summary,
    metrics=["redundancy"],
    llm_config=llm_config,
    show_progress=False
)
print("\nRedundancy Results (for intentionally redundant summary):")
for key, value in results_redundancy.items():
    if key == "redundant_pairs":
        print(f"  {key}: (showing first 3)")
        for i, pair in enumerate(value[:3]):
            print(f"    Pair {i+1} (similarity: {pair['similarity']:.3f}):")
            print(f"      Sentence {pair['sentence_1_index']}: {pair['sentence_1'][:60]}...")
            print(f"      Sentence {pair['sentence_2_index']}: {pair['sentence_2'][:60]}...")
    elif isinstance(value, (int, float)):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Example 5: Compare old vs new metric names (backwards compatibility)
print("\n" + "=" * 80)
print("Example 5: Backwards Compatibility (Deprecated Metric Names)")
print("=" * 80)
print("\nOld metric names still work but emit deprecation warnings:")
print("  'faithfulness' -> use 'coverage'")
print("  'hallucination' -> use 'factual_consistency'")

import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    results_old = evaluate_summary(
        full_text=full_text,
        summary=summary,
        metrics=["faithfulness", "hallucination"],  # Old names
        llm_config=llm_config,
        show_progress=False
    )
    if w:
        print("\nDeprecation warnings emitted:")
        for warning in w:
            print(f"  - {warning.message}")

print("\n" + "=" * 80)
print("Done! New metrics provide clearer semantics and better implementation.")
print("=" * 80)
