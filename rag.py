from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import glob

class RAGPipeline:
    """RAG pipeline for document indexing and retrieval"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize RAG pipeline"""
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("✓ Embedding model loaded")
        
        self.index = None
        self.documents = []
        self.chunk_size = 500
        self.chunk_overlap = 50
    
    def build_index(self, data_dir: str = "data"):
        """Build FAISS index from documents in data directory"""
        try:
            # Check if directory exists
            if not os.path.exists(data_dir):
                print(f"Creating directory: {data_dir}")
                os.makedirs(data_dir, exist_ok=True)
                return
            
            # Get all text files
            text_files = glob.glob(os.path.join(data_dir, "*.txt"))
            
            if not text_files:
                print("No text files found in data directory")
                return
            
            print(f"Found {len(text_files)} files to index")
            
            # Load and chunk documents
            all_chunks = []
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    chunks = self._chunk_text(content)
                    all_chunks.extend(chunks)
                    print(f"✓ Processed: {os.path.basename(file_path)} ({len(chunks)} chunks)")
                except Exception as e:
                    print(f"✗ Error reading {file_path}: {e}")
            
            if not all_chunks:
                print("No content to index")
                return
            
            self.documents = all_chunks
            
            # Create embeddings
            print(f"Creating embeddings for {len(all_chunks)} chunks...")
            embeddings = self.model.encode(all_chunks, show_progress_bar=True)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            print(f"✓ Index built successfully with {len(all_chunks)} chunks")
            
        except Exception as e:
            print(f"Error building index: {e}")
            raise
    
    def _chunk_text(self, text: str) -> list:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def retrieve(self, query: str, top_k: int = 5) -> list:
        """Retrieve top-k most relevant chunks"""
        try:
            if self.index is None or len(self.documents) == 0:
                print("Index not built yet. Please build index first.")
                return []
            
            # Encode query
            query_embedding = self.model.encode([query])
            
            # Search
            distances, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(top_k, len(self.documents))
            )
            
            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    results.append({
                        "text": self.documents[idx],
                        "score": float(1 / (1 + distance))  # Convert distance to similarity score
                    })
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def add_documents(self, documents: list):
        """Add new documents to existing index"""
        try:
            if not documents:
                return
            
            # Chunk new documents
            new_chunks = []
            for doc in documents:
                chunks = self._chunk_text(doc)
                new_chunks.extend(chunks)
            
            if not new_chunks:
                return
            
            # Create embeddings
            new_embeddings = self.model.encode(new_chunks)
            
            # Add to documents list
            self.documents.extend(new_chunks)
            
            # Add to index
            if self.index is None:
                dimension = new_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)
            
            self.index.add(new_embeddings.astype('float32'))
            
            print(f"✓ Added {len(new_chunks)} new chunks to index")
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise
