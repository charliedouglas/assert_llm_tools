from assert_llm_tools.core import evaluate_summary
from assert_llm_tools.llm.config import LLMConfig

metrics = ["topic_preservation"]


# Select one of Bedrock or OpenAI
llm_config = LLMConfig(
    provider="bedrock", model_id="anthropic.claude-v2", region="us-east-1"
)


full_text = "full text"
summary = "summary text"

metrics = evaluate_summary(
    full_text,
    summary,
    metrics=metrics,
    llm_config=llm_config,
)

print("\nEvaluation Metrics:")
for metric, score in metrics.items():
    print(f"{metric}: {score:.4f}")
