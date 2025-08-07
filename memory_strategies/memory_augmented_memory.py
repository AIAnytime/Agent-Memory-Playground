"""
Memory-Augmented Memory Strategy

This strategy simulates memory-enhanced transformer behavior by maintaining
a short-term sliding window of recent conversations and a separate list of
"memory tokens" - important facts extracted from conversations.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base_memory import BaseMemoryStrategy
from .sliding_window_memory import SlidingWindowMemory
from .utils import generate_text, get_openai_client


class MemoryAugmentedMemory(BaseMemoryStrategy):
    """
    Memory-augmented strategy combining sliding window with persistent memory tokens.
    
    Advantages:
    - Excellent long-term retention of key information
    - Suitable for evolving long-term conversations
    - Intelligent fact extraction mechanism
    - Strong foundation for personal assistants
    
    Disadvantages:
    - More complex implementation
    - Additional LLM calls increase cost
    - Depends on fact extraction quality
    - May increase response time
    """
    
    def __init__(self, window_size: int = 2, client: Optional[OpenAI] = None):
        """
        Initialize memory-augmented system.
        
        Args:
            window_size: Number of recent turns to retain in short-term memory
            client: Optional OpenAI client instance
        """
        self.client = client or get_openai_client()
        
        # Use SlidingWindowMemory instance to manage recent conversation history
        self.recent_memory = SlidingWindowMemory(window_size=window_size)
        
        # List to store special, persistent "sticky notes" or key facts
        self.memory_tokens: List[str] = []
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add latest turn to recent memory, then use LLM call to decide
        if new persistent memory tokens should be created from this interaction.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        # First, add new interaction to short-term sliding window memory
        self.recent_memory.add_message(user_input, ai_response)
        
        # Construct prompt for LLM to analyze conversation turn and
        # determine if it contains core facts worth remembering long-term
        fact_extraction_prompt = (
            f"Analyze the following conversation turn. Does it contain a core fact, preference, or decision that should be remembered long-term? "
            f"Examples include user preferences ('I hate flying'), key decisions ('The budget is $1000'), or important facts ('My user ID is 12345').\n\n"
            f"Conversation Turn:\nUser: {user_input}\nAI: {ai_response}\n\n"
            f"If it contains such a fact, state the fact concisely in one sentence. Otherwise, respond with 'No important fact.'"
        )
        
        # Call LLM to perform fact extraction
        extracted_fact = generate_text(
            "You are a fact-extraction expert.", 
            fact_extraction_prompt,
            self.client
        )
        
        # Check if LLM's response indicates an important fact was found
        if "no important fact" not in extracted_fact.lower():
            # If fact found, print debug message and add to memory tokens list
            print(f"--- [Memory Augmentation: New memory token created: '{extracted_fact}'] ---")
            self.memory_tokens.append(extracted_fact)
    
    def get_context(self, query: str) -> str:
        """
        Construct context by combining short-term recent conversation
        with list of all long-term, persistent memory tokens.
        
        Args:
            query: Current user query
            
        Returns:
            Combined context from memory tokens and recent conversation
        """
        # Get context from short-term sliding window
        recent_context = self.recent_memory.get_context(query)
        
        # Format memory tokens list as readable string
        if self.memory_tokens:
            memory_token_context = "\n".join([f"- {token}" for token in self.memory_tokens])
            return f"### Key Memory Tokens (Long-Term Facts):\n{memory_token_context}\n\n### Recent Conversation:\n{recent_context}"
        else:
            return f"### Recent Conversation:\n{recent_context}"
    
    def clear(self) -> None:
        """Reset both recent memory and memory tokens."""
        self.recent_memory.clear()
        self.memory_tokens = []
        print("Memory-augmented memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        recent_stats = self.recent_memory.get_memory_stats()
        num_tokens = len(self.memory_tokens)
        
        return {
            "strategy_type": "MemoryAugmentedMemory",
            "memory_tokens": num_tokens,
            "recent_memory_stats": recent_stats,
            "memory_size": f"{num_tokens} memory tokens + recent window",
            "advantages": ["Long-term retention", "Intelligent extraction", "Personal assistant ready"],
            "disadvantages": ["Complex implementation", "Additional LLM calls", "Fact extraction dependent"]
        }
