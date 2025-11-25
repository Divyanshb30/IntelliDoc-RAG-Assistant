import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle

class RAGPipeline:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """Initialize RAG pipeline with sentence transformer for embeddings"""
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks = []
        self.chunk_size = 500
        self.chunk_overlap = 50
        print("✓ Embedding model loaded")
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def load_documents(self, data_dir: str = "data"):
        """Load and chunk all text files from data directory"""
        all_text = ""
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    all_text += f.read() + "\n\n"
        
        self.chunks = self.chunk_text(all_text)
        print(f"✓ Loaded {len(self.chunks)} chunks from documents")
        
    def build_index(self):
        """Build FAISS index from document chunks"""
        if not self.chunks:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        print("Creating embeddings...")
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
        
        # Convert to float32 for FAISS
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"✓ Built FAISS index with {self.index.ntotal} vectors")
        
    def save_index(self, path: str = "vector_store"):
        """Save FAISS index and chunks to disk"""
        os.makedirs(path, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "chunks.pkl"), 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"✓ Saved index to {path}")
        
    def load_index(self, path: str = "vector_store"):
        """Load FAISS index and chunks from disk"""
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "chunks.pkl"), 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k relevant chunks for a query"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load_index() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": self.chunks[idx],
                "score": float(distances[0][i])
            })
        
        return results

# Test the RAG pipeline
if __name__ == "__main__":
    rag = RAGPipeline()
    rag.load_documents("data")
    rag.build_index()
    rag.save_index()
    
    # Test retrieval
    query = "What are the products?"
    results = rag.retrieve(query, top_k=2)
    
    print(f"\nQuery: {query}")
    print("\nTop results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text'][:200]}...")
        print(f"   Score: {result['score']:.4f}")
