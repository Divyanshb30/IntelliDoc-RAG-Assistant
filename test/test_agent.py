"""Test RAG Agent with all tools"""
import sys
sys.path.append('..')

from llama_cpp import Llama
from rag import RAGPipeline
from tools import get_tools
from agent import RAGAgent

print("="*60)
print("TEST 2: RAG Agent Integration")
print("="*60)

print("\n[1/4] Loading components...")
# Load RAG
rag = RAGPipeline()
rag.load_index("../vector_store")

# Load LLM
llm = Llama(
    model_path="../models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    verbose=False
)

# Initialize tools and agent
tools = get_tools(rag)
agent = RAGAgent(llm, tools)
print("✓ All components loaded!")

# Test queries
test_queries = [
    ("What products does TechCorp offer?", None),
    ("Summarize the company overview", None),
]

print("\n[2/4] Testing document search...")
for query, file_ctx in test_queries:
    print(f"\nQuery: {query}")
    print("-" * 60)
    
    result = agent.execute(query, file_context=file_ctx)
    
    print(f"Tool Used: {result['tool_used']}")
    print(f"Success: {result['success']}")
    print(f"Answer: {result['answer'][:200]}...")

print("\n[3/4] Testing code analysis...")
# This will only work if code file exists
print("(Requires code file upload - test manually in UI)")

print("\n[4/4] Testing intent detection...")
test_intents = [
    ("What is the API endpoint?", "document_search"),
    ("Summarize this document", "summarizer"),
    ("Analyze the code", "code_analyzer"),
    ("Check security vulnerabilities", "security_scanner"),
]

for query, expected in test_intents:
    detected = agent.detect_intent(query)
    status = "✓" if detected == expected else "✗"
    print(f"{status} '{query}' -> {detected} (expected: {expected})")

print("\n" + "="*60)
print("✓ Agent integration tests complete!")
print("="*60)
