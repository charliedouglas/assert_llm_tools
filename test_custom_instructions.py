"""
Test script demonstrating custom prompt instructions feature for LLM-based metrics.
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

# Example 1: Using default evaluation (no custom instructions)
print("=" * 80)
print("Example 1: Default Evaluation")
print("=" * 80)
results_default = evaluate_summary(
    full_text=full_text,
    summary=summary,
    metrics=["faithfulness", "coherence", "hallucination"],
    llm_config=llm_config,
    show_progress=False
)
print("\nDefault Results:")
for metric, score in results_default.items():
    if isinstance(score, (int, float)):
        print(f"  {metric}: {score:.4f}")
    else:
        print(f"  {metric}: {score}")

# Example 2: Using custom instructions for specific metrics
print("\n" + "=" * 80)
print("Example 2: With Custom Instructions")
print("=" * 80)
results_custom = evaluate_summary(
    full_text=full_text,
    summary=summary,
    metrics=["faithfulness", "coherence", "hallucination"],
    llm_config=llm_config,
    show_progress=False,
    custom_prompt_instructions={
        "faithfulness": "Apply strict scientific standards. Only mark claims as true if they are explicitly stated in the context, not just implied.",
        "coherence": "Focus on whether the text flows naturally for a technical/scientific audience. Expect precise, formal language.",
        "hallucination": "Be extremely strict. Flag any claim that adds details not present in the original text, even if plausible."
    }
)
print("\nCustom Instructions Results:")
for metric, score in results_custom.items():
    if isinstance(score, (int, float)):
        print(f"  {metric}: {score:.4f}")
    else:
        print(f"  {metric}: {score}")

# Example 3: Different custom instructions for creative writing
print("\n" + "=" * 80)
print("Example 3: Creative Writing Context")
print("=" * 80)

creative_text = """
The old bookstore on Main Street had a magical quality about it. Every shelf seemed to whisper stories,
and the smell of aged paper filled the air like perfume. Sarah loved spending her afternoons there,
getting lost between the stacks, discovering forgotten classics and hidden gems.
"""

creative_summary = """
Sarah enjoyed visiting the enchanting old bookstore on Main Street, where she discovered forgotten books among whispered stories and the scent of aged paper.
"""

results_creative = evaluate_summary(
    full_text=creative_text,
    summary=creative_summary,
    metrics=["coherence", "topic_preservation", "conciseness"],
    llm_config=llm_config,
    show_progress=False,
    custom_prompt_instructions={
        "coherence": "Evaluate for creative writing style. Look for natural narrative flow and evocative language.",
        "topic_preservation": "Consider emotional atmosphere and sensory details as important topics, not just facts.",
        "conciseness": "For creative writing, some descriptive language is valuable. Don't penalize evocative phrasing."
    }
)
print("\nCreative Writing Results:")
for metric, score in results_creative.items():
    if isinstance(score, (int, float)):
        print(f"  {metric}: {score:.4f}")
    else:
        print(f"  {metric}: {score}")

print("\n" + "=" * 80)
print("Done! Custom instructions allow you to tailor evaluation to your specific use case.")
print("=" * 80)
