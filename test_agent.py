from llama_cpp import Llama
from rag import RAGPipeline
from tools import get_tools
from agent import RAGAgent

print("Loading components...")

# Load RAG
rag = RAGPipeline()
rag.load_index("vector_store")

# Load LLM
llm = Llama(
    model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=2048,
    verbose=False
)

# Initialize tools and agent
tools = get_tools(rag)
agent = RAGAgent(llm, tools)

print("✓ All components loaded!\n")

# Test queries
test_queries = [
    "What products does TechCorp offer?",
    "Summarize the company overview",
    "Create tasks to implement a new API feature"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    result = agent.execute(query)
    
    print(f"Tool Used: {result['tool_used']}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")
