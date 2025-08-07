"""
Sequential Memory Strategy

This is the most basic memory strategy that stores the entire conversation
history in chronological order. While it provides perfect recall, it's not
scalable as the context grows linearly with each conversation turn.
"""

from typing import List, Dict, Any
from .base_memory import BaseMemoryStrategy
from .utils import count_tokens


class SequentialMemory(BaseMemoryStrategy):
    """
    Sequential memory strategy that stores all conversation history.
    
    Advantages:
    - Simple implementation
    - Perfect recall of all conversations
    - Complete context preservation
    
    Disadvantages:
    - Linear token growth with conversation length
    - Expensive for long conversations
    - May hit token limits quickly
    """
    
    def __init__(self):
        """Initialize memory with empty list to store conversation history."""
        self.history: List[Dict[str, str]] = []
        self.total_content_tokens = 0  # Track cumulative content token usage
        self.total_prompt_tokens = 0   # Track cumulative prompt tokens sent to LLM
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add new user-AI interaction to history.
        
        Each interaction is stored as two dictionary entries in the list.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": ai_response})
        
        # Update content token count (just the message content)
        self.total_content_tokens += count_tokens(user_input + ai_response)
    
    def get_context(self, query: str) -> str:
        """
        Retrieve entire conversation history formatted as a single string.
        
        The 'query' parameter is ignored since this strategy always
        returns the complete history.
        
        Args:
            query: Current user query (ignored in this strategy)
            
        Returns:
            Complete conversation history as formatted string
        """
        if not self.history:
            return "No conversation history yet."
        
        # Join all messages into a single string separated by newlines
        return "\n".join([
            f"{turn['role'].capitalize()}: {turn['content']}" 
            for turn in self.history
        ])
    
    def clear(self) -> None:
        """Reset conversation history by clearing the list."""
        self.history = []
        self.total_content_tokens = 0
        self.total_prompt_tokens = 0
        print("Sequential memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        total_messages = len(self.history)
        total_turns = total_messages // 2  # Each turn has user + assistant message
        
        return {
            "strategy_type": "SequentialMemory",
            "total_messages": total_messages,
            "total_turns": total_turns,
            "total_content_tokens": self.total_content_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "memory_size": f"{total_messages} messages",
            "advantages": ["Perfect recall", "Simple implementation"],
            "disadvantages": ["Linear token growth", "Not scalable"]
        }
