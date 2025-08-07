"""
Hierarchical Memory Strategy

This strategy combines multiple memory types into a layered system that mimics
human memory patterns with working memory (short-term) and long-term memory layers.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base_memory import BaseMemoryStrategy
from .sliding_window_memory import SlidingWindowMemory
from .retrieval_memory import RetrievalMemory
from .utils import get_openai_client


class HierarchicalMemory(BaseMemoryStrategy):
    """
    Hierarchical memory strategy combining working memory and long-term memory.
    
    Advantages:
    - Multi-level information processing
    - Intelligent information promotion
    - Combines strengths of multiple strategies
    - Resembles human cognitive patterns
    
    Disadvantages:
    - Complex implementation
    - Multiple memory systems to manage
    - Promotion logic may need tuning
    - Higher computational overhead
    """
    
    def __init__(
        self, 
        window_size: int = 2, 
        k: int = 2, 
        embedding_dim: int = 1536,
        client: Optional[OpenAI] = None
    ):
        """
        Initialize hierarchical memory system.
        
        Args:
            window_size: Size of short-term working memory (in turns)
            k: Number of documents to retrieve from long-term memory
            embedding_dim: Embedding vector dimension for long-term memory
            client: Optional OpenAI client instance
        """
        print("Initializing Hierarchical Memory...")
        self.client = client or get_openai_client()
        
        # Level 1: Fast, short-term working memory using sliding window
        self.working_memory = SlidingWindowMemory(window_size=window_size)
        
        # Level 2: Slower, persistent long-term memory using retrieval system
        self.long_term_memory = RetrievalMemory(k=k, embedding_dim=embedding_dim, client=self.client)
        
        # Simple heuristic: keywords that trigger promotion from working to long-term memory
        self.promotion_keywords = ["remember", "rule", "preference", "always", "never", "allergic", "important"]
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add messages to working memory and conditionally promote to long-term memory.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        # All interactions are added to fast, short-term working memory
        self.working_memory.add_message(user_input, ai_response)
        
        # Promotion logic: check if user input contains keywords indicating
        # information is important and should be stored long-term
        if any(keyword in user_input.lower() for keyword in self.promotion_keywords):
            print(f"--- [Hierarchical Memory: Promoting message to long-term storage.] ---")
            # If keywords found, also add interaction to long-term retrieval memory
            self.long_term_memory.add_message(user_input, ai_response)
    
    def get_context(self, query: str) -> str:
        """
        Construct rich context by combining relevant information from both memory layers.
        
        Args:
            query: Current user query
            
        Returns:
            Combined context from long-term and short-term memory
        """
        # Get recent context from working memory
        working_context = self.working_memory.get_context(query)
        
        # Retrieve relevant content from long-term memory
        long_term_context = self.long_term_memory.get_context(query)
        
        # If no relevant content in long-term memory, use only working memory
        if ("No information in memory yet" in long_term_context or 
            "Could not find any relevant information" in long_term_context):
            return f"### Recent Context:\n{working_context}"
        else:
            # Otherwise, combine both memory layers
            return f"### Long-Term Context:\n{long_term_context}\n\n### Recent Context:\n{working_context}"
    
    def clear(self) -> None:
        """Reset both working memory and long-term memory."""
        self.working_memory.clear()
        self.long_term_memory.clear()
        print("Hierarchical memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage from both layers.
        
        Returns:
            Dictionary containing memory statistics
        """
        working_stats = self.working_memory.get_memory_stats()
        long_term_stats = self.long_term_memory.get_memory_stats()
        
        return {
            "strategy_type": "HierarchicalMemory",
            "promotion_keywords": self.promotion_keywords,
            "working_memory_stats": working_stats,
            "long_term_memory_stats": long_term_stats,
            "memory_size": f"Working: {working_stats['memory_size']}, Long-term: {long_term_stats['memory_size']}",
            "advantages": ["Multi-level processing", "Intelligent promotion", "Human-like patterns"],
            "disadvantages": ["Complex implementation", "Multiple systems", "Overhead"]
        }
