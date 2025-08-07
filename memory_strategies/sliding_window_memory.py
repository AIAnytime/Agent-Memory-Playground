"""
Sliding Window Memory Strategy

This strategy maintains only the most recent N conversation turns using a fixed-size
window. It prevents unbounded context growth but may lose important historical information.
"""

from collections import deque
from typing import List, Dict, Any
from .base_memory import BaseMemoryStrategy
from .utils import count_tokens


class SlidingWindowMemory(BaseMemoryStrategy):
    """
    Sliding window memory strategy that keeps only recent N conversation turns.
    
    Advantages:
    - Controlled memory usage
    - Predictable token consumption
    - Scalable for long conversations
    
    Disadvantages:
    - Loses old information
    - May forget important early context
    - Fixed window size may not suit all scenarios
    """
    
    def __init__(self, window_size: int = 4):
        """
        Initialize memory with fixed-size deque.
        
        Args:
            window_size: Number of conversation turns to retain in memory.
                        A single turn includes one user message and one AI response.
        """
        self.window_size = window_size
        # Deque with maxlen automatically discards oldest items when full
        self.history = deque(maxlen=window_size)
        self.total_content_tokens = 0  # Track cumulative content token usage
        self.total_prompt_tokens = 0   # Track cumulative prompt tokens sent to LLM
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add new conversation turn to history.
        
        If deque is full, the oldest turn is automatically removed.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        # Each turn (user input + AI response) is stored as a single element
        # This makes it easy to manage window size by turns
        turn_data = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ]
        self.history.append(turn_data)
        
        # Update content token count (just the message content)
        self.total_content_tokens += count_tokens(user_input + ai_response)
    
    def get_context(self, query: str) -> str:
        """
        Retrieve conversation history within current window.
        
        The 'query' parameter is ignored in this strategy.
        
        Args:
            query: Current user query (ignored in this strategy)
            
        Returns:
            Recent conversation history as formatted string
        """
        if not self.history:
            return "No conversation history yet."
        
        # Create temporary list to hold formatted messages
        context_list = []
        
        # Iterate through each turn stored in the deque
        for turn in self.history:
            # Iterate through user and assistant messages in the turn
            for message in turn:
                # Format message and add to our list
                context_list.append(f"{message['role'].capitalize()}: {message['content']}")
        
        # Join all formatted messages into a single string
        return "\n".join(context_list)
    
    def clear(self) -> None:
        """Reset conversation history by clearing the deque."""
        self.history.clear()
        self.total_content_tokens = 0
        self.total_prompt_tokens = 0
        print("Sliding window memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        current_turns = len(self.history)
        total_messages = sum(len(turn) for turn in self.history)
        
        return {
            "strategy_type": "SlidingWindowMemory",
            "window_size": self.window_size,
            "current_turns": current_turns,
            "total_messages": total_messages,
            "total_content_tokens": self.total_content_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "memory_size": f"{current_turns}/{self.window_size} turns",
            "advantages": ["Controlled memory", "Predictable tokens", "Scalable"],
            "disadvantages": ["Loses old info", "Fixed window size"]
        }
