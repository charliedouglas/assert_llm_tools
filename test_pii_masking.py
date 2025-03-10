from assert_llm_tools.core import evaluate_summary
from assert_llm_tools.llm.config import LLMConfig
from assert_llm_tools.utils import detect_and_mask_pii

# Sample text with PII
full_text = """
John Smith lives at 123 Main Street, New York, NY 10001. His email is john.smith@example.com 
and his phone number is 555-123-4567. He has a social security number 123-45-6789 and 
credit card 4111-1111-1111-1111 that expires on 09/2025.
"""

summary = """
John Smith, who lives in New York and can be reached at john.smith@example.com, 
has a credit card and SSN on file.
"""

# Test standalone PII masking
def test_pii_masking():
    print("Testing standalone PII masking...")
    
    # Full masking
    masked_text, pii_info = detect_and_mask_pii(full_text)
    print("\nFull masking:")
    print(f"Original text: {full_text}")
    print(f"Masked text: {masked_text}")
    print(f"Detected PII types: {list(pii_info.keys())}")
    
    # Partial masking
    masked_text, pii_info = detect_and_mask_pii(full_text, preserve_partial=True)
    print("\nPartial masking:")
    print(f"Masked text: {masked_text}")
    
    # Custom masking character
    masked_text, pii_info = detect_and_mask_pii(full_text, mask_char="X")
    print("\nCustom mask character:")
    print(f"Masked text: {masked_text}")
    
    return True

# Test PII masking with evaluation
def test_evaluate_with_pii_masking():
    print("\nTesting PII masking with summary evaluation...")
    
    # Configure LLM (typically would use an actual API, but using mock for this test)
    config = LLMConfig(
        provider="openai",
        model_id="gpt-4",
        api_key="mock-key-for-testing"
    )
    
    # Process and evaluate WITH masking and return PII info
    metrics, pii_info = evaluate_summary(
        full_text=full_text,
        summary=summary,
        metrics=["rouge"],  # Using ROUGE since it doesn't require a real LLM connection
        llm_config=config,
        mask_pii=True,
        mask_pii_char="*",
        return_pii_info=True
    )
    
    print("\nEvaluation results with PII masking:")
    print(f"Metrics: {metrics}")
    print(f"PII detected in full text: {list(pii_info.get('full_text_pii', {}).keys())}")
    print(f"PII detected in summary: {list(pii_info.get('summary_pii', {}).keys())}")
    
    return True

if __name__ == "__main__":
    test_pii_masking()
    test_evaluate_with_pii_masking()