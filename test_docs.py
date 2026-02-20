from assert_eval import evaluate_summary
from assert_eval import LLMConfig

metrics = [
    "coverage",
    "factual_consistency",
    "factual_alignment",
]


# Select one of Bedrock or OpenAI
llm_config = LLMConfig(
    provider="bedrock",
    model_id="us.amazon.nova-pro-v1:0",
    region="us-east-1",
    # http_proxy="http://username:password@localhost:8080",
)


full_text = """The James Webb Space Telescope (JWST) has revolutionized our understanding of the cosmos since its launch in 2021. As the largest and most powerful space telescope ever built, it has provided unprecedented views of distant galaxies, exoplanets, and cosmic phenomena. The telescope's infrared capabilities allow it to peer through cosmic dust and gas, revealing previously hidden details about star formation and galaxy evolution. Scientists have already used JWST data to make groundbreaking discoveries, including observations of some of the earliest galaxies formed after the Big Bang and detailed atmospheric analysis of potentially habitable exoplanets."""
summary = (
    "The James Webb Space Telescope, launched in 2021, it is located near Jupiter."
)

metrics = evaluate_summary(
    full_text,
    summary,
    metrics=metrics,
    llm_config=llm_config,
)

print(metrics)
