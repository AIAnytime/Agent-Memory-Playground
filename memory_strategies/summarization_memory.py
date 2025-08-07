"""
Summarization Memory Strategy

This strategy manages long conversations by periodically summarizing conversation history.
It maintains a buffer of recent messages and triggers summarization when the buffer
reaches a threshold, using LLM to compress historical information intelligently.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base_memory import BaseMemoryStrategy
from .utils import generate_text, get_openai_client


class SummarizationMemory(BaseMemoryStrategy):
    """
    Summarization memory strategy that compresses conversation history using LLM.
    
    Advantages:
    - Manages long conversations efficiently
    - Retains key information through intelligent compression
    - Scalable token usage
    - Maintains conversation flow
    
    Disadvantages:
    - May lose details during summarization
    - Depends on LLM summarization quality
    - Additional LLM calls increase cost
    - Information decay over time
    """
    
    def __init__(self, summary_threshold: int = 4, client: Optional[OpenAI] = None):
        """
        Initialize summarization memory.
        
        Args:
            summary_threshold: Number of messages to accumulate before triggering summary
            client: Optional OpenAI client instance
        """
        self.summary_threshold = summary_threshold
        self.client = client or get_openai_client()
        
        # Store continuously updated summary of conversation so far
        self.running_summary = ""
        
        # Temporary list to hold recent messages before summarization
        self.buffer: List[Dict[str, str]] = []
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add new user-AI interaction to buffer.
        
        If buffer size reaches threshold, triggers memory consolidation process.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        # Append latest user and AI messages to temporary buffer
        self.buffer.append({"role": "user", "content": user_input})
        self.buffer.append({"role": "assistant", "content": ai_response})
        
        # Check if buffer has reached its capacity
        if len(self.buffer) >= self.summary_threshold:
            # If so, call method to summarize buffer contents
            self._consolidate_memory()
    
    def _consolidate_memory(self) -> None:
        """
        Use LLM to summarize buffer contents and merge with existing summary.
        
        This is the core innovation of the summarization strategy.
        """
        print("\n--- [Memory Consolidation Triggered] ---")
        
        # Convert buffered message list to single formatted string
        buffer_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self.buffer
        ])
        
        # Construct specific prompt for LLM to perform summarization task
        summarization_prompt = (
            f"You are a summarization expert. Your task is to create a concise summary of a conversation. "
            f"Combine the 'Previous Summary' with the 'New Conversation' into a single, updated summary. "
            f"Capture all key facts, names, decisions, and important details.\n\n"
            f"### Previous Summary:\n{self.running_summary}\n\n"
            f"### New Conversation:\n{buffer_text}\n\n"
            f"### Updated Summary:"
        )
        
        # Call LLM with specific system prompt to get new summary
        new_summary = generate_text(
            "You are an expert summarization engine.", 
            summarization_prompt,
            self.client
        )
        
        # Replace old summary with newly generated merged summary
        self.running_summary = new_summary
        
        # Clear buffer since its contents are now merged into summary
        self.buffer = []
        
        print(f"--- [New Summary Generated] ---")
        print(f"Summary: {self.running_summary[:100]}...")
    
    def get_context(self, query: str) -> str:
        """
        Construct context to send to LLM by combining long-term summary
        with short-term buffer of recent messages.
        
        Args:
            query: Current user query (ignored in this strategy)
            
        Returns:
            Combined context from summary and recent messages
        """
        # Format current messages in buffer
        buffer_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self.buffer
        ])
        
        # Return combination of historical summary and recent unsummarized messages
        if self.running_summary:
            return f"### Summary of Past Conversation:\n{self.running_summary}\n\n### Recent Messages:\n{buffer_text}"
        else:
            return f"### Recent Messages:\n{buffer_text}" if buffer_text else "No conversation history yet."
    
    def clear(self) -> None:
        """Reset both summary and buffer."""
        self.running_summary = ""
        self.buffer = []
        print("Summarization memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        buffer_messages = len(self.buffer)
        has_summary = bool(self.running_summary)
        
        return {
            "strategy_type": "SummarizationMemory",
            "summary_threshold": self.summary_threshold,
            "buffer_messages": buffer_messages,
            "has_summary": has_summary,
            "summary_length": len(self.running_summary) if has_summary else 0,
            "memory_size": f"Summary + {buffer_messages} buffered messages",
            "advantages": ["Efficient compression", "Retains key info", "Scalable"],
            "disadvantages": ["May lose details", "LLM dependent", "Additional cost"]
        }
