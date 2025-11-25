"""Test LLM model loading and basic inference"""
from llama_cpp import Llama

print("="*60)
print("TEST 1: Model Loading")
print("="*60)

print("\n[1/3] Loading model...")
llm = Llama(
    model_path="../models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    verbose=False
)
print("✓ Model loaded successfully!")

print("\n[2/3] Testing inference...")
response = llm(
    "Q: What is 5 + 7? A:",
    max_tokens=50,
    stop=["Q:", "\n"],
    temperature=0.7
)
print(f"Response: {response['choices'][0]['text']}")

print("\n[3/3] Testing with longer prompt...")
response = llm(
    "Explain what Python is in one sentence:",
    max_tokens=100,
    temperature=0.7
)
print(f"Response: {response['choices'][0]['text']}")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED - Model is working!")
print("="*60)
