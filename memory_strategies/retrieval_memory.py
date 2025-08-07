"""
Retrieval-based Memory Strategy

This strategy implements the core concept of Retrieval-Augmented Generation (RAG).
It converts conversations into vector embeddings and uses similarity search to find
the most relevant historical interactions for any given query.
"""

import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base_memory import BaseMemoryStrategy
from .utils import generate_embedding, get_openai_client


class RetrievalMemory(BaseMemoryStrategy):
    """
    Retrieval-based memory strategy using vector embeddings and similarity search.
    
    Advantages:
    - Semantic understanding of queries
    - Efficient retrieval of relevant information
    - Scalable to large conversation histories
    - Industry standard for RAG applications
    
    Disadvantages:
    - Complex implementation
    - Requires embedding model
    - Dependent on embedding quality
    - Additional computational overhead
    """
    
    def __init__(self, k: int = 2, embedding_dim: int = 1536, client: Optional[OpenAI] = None):
        """
        Initialize retrieval memory system.
        
        Args:
            k: Number of most relevant documents to retrieve for a given query
            embedding_dim: Dimension of embedding vectors (1536 for text-embedding-3-small)
            client: Optional OpenAI client instance
        """
        self.k = k
        self.embedding_dim = embedding_dim
        self.client = client or get_openai_client()
        
        # List to store original text content of each document
        self.documents: List[str] = []
        
        # Initialize FAISS index for similarity search
        # IndexFlatL2 uses L2 (Euclidean) distance for exhaustive search
        self.index = faiss.IndexFlatL2(self.embedding_dim)
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add new conversation turn to memory.
        
        Each part of the turn (user input and AI response) is embedded
        and indexed separately for fine-grained retrieval.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        # Store each part of the turn as separate documents for precise matching
        docs_to_add = [
            f"User said: {user_input}",
            f"AI responded: {ai_response}"
        ]
        
        for doc in docs_to_add:
            # Generate numerical vector representation of the document
            embedding = generate_embedding(doc, self.client)
            
            # Only proceed if embedding was successfully created
            if embedding:
                # Store original text - index will correspond to vector index in FAISS
                self.documents.append(doc)
                
                # FAISS requires input vectors to be float32 2D numpy arrays
                vector = np.array([embedding], dtype='float32')
                
                # Add vector to FAISS index, making it searchable
                self.index.add(vector)
    
    def get_context(self, query: str) -> str:
        """
        Find k most relevant documents from memory based on semantic similarity to query.
        
        Args:
            query: Current user query to find relevant context for
            
        Returns:
            Formatted string containing most relevant retrieved information
        """
        # If index has no vectors, there's nothing to search
        if self.index.ntotal == 0:
            return "No information in memory yet."
        
        # Convert user query to embedding vector
        query_embedding = generate_embedding(query, self.client)
        if not query_embedding:
            return "Could not process query for retrieval."
        
        # Convert query embedding to format required by FAISS
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Perform search - returns distances and indices of k nearest neighbors
        distances, indices = self.index.search(query_vector, self.k)
        
        # Use returned indices to retrieve original text documents
        # Check for i != -1 because FAISS may return -1 for invalid indices
        retrieved_docs = [
            self.documents[i] for i in indices[0] 
            if i != -1 and i < len(self.documents)
        ]
        
        if not retrieved_docs:
            return "Could not find any relevant information in memory."
        
        # Format retrieved documents as string for use as context
        return "### Relevant Information Retrieved from Memory:\n" + "\n---\n".join(retrieved_docs)
    
    def clear(self) -> None:
        """Reset both document storage and FAISS index."""
        self.documents = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        print("Retrieval memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        num_documents = len(self.documents)
        num_vectors = self.index.ntotal
        
        return {
            "strategy_type": "RetrievalMemory",
            "k": self.k,
            "embedding_dim": self.embedding_dim,
            "num_documents": num_documents,
            "num_vectors": num_vectors,
            "memory_size": f"{num_documents} documents, {num_vectors} vectors",
            "advantages": ["Semantic search", "Scalable", "Relevant retrieval"],
            "disadvantages": ["Complex implementation", "Embedding dependent"]
        }
