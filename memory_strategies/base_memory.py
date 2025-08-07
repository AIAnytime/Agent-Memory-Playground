"""
Base Memory Strategy Abstract Class

This module defines the abstract base class that all memory strategies must implement.
It ensures consistency and interchangeability between different memory optimization techniques.
"""

import abc
from typing import Any, Dict, List, Optional


class BaseMemoryStrategy(abc.ABC):
    """Abstract base class for all memory strategies."""
    
    @abc.abstractmethod
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add a new user-AI interaction to the memory storage.
        
        Args:
            user_input: The user's message
            ai_response: The AI's response
        """
        pass
    
    @abc.abstractmethod
    def get_context(self, query: str) -> str:
        """
        Retrieve and format relevant context from memory for the LLM.
        
        Args:
            query: The current user query to find relevant context for
            
        Returns:
            Formatted context string to send to the LLM
        """
        pass
    
    @abc.abstractmethod
    def clear(self) -> None:
        """
        Reset the memory storage, useful for starting new conversations.
        """
        pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        return {
            "strategy_type": self.__class__.__name__,
            "memory_size": "Unknown"
        }
