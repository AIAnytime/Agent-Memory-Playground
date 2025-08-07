"""
AI Agent Class

This module contains the core AI Agent that coordinates conversation flow
and works with different memory strategies using the strategy pattern.
"""

import time
from typing import Optional
from openai import OpenAI

from .base_memory import BaseMemoryStrategy
from .utils import generate_text, count_tokens, get_openai_client


class AIAgent:
    """
    Main AI Agent class designed to work with any memory strategy.
    
    Uses the strategy pattern to allow switching between different
    memory management approaches at runtime.
    """
    
    def __init__(
        self, 
        memory_strategy: BaseMemoryStrategy, 
        system_prompt: str = "You are a helpful AI assistant.",
        client: Optional[OpenAI] = None
    ):
        """
        Initialize the AI agent.
        
        Args:
            memory_strategy: Instance of a class inheriting from BaseMemoryStrategy
            system_prompt: Initial instructions for the LLM defining its personality
            client: Optional OpenAI client instance
        """
        self.memory = memory_strategy
        self.system_prompt = system_prompt
        self.client = client or get_openai_client()
        print(f"Agent initialized with {type(memory_strategy).__name__}.")
    
    def chat(self, user_input: str, verbose: bool = True) -> dict:
        """
        Process a single conversation turn.
        
        Args:
            user_input: The user's latest message
            verbose: Whether to print detailed debug information
            
        Returns:
            Dictionary containing response and performance metrics
        """
        if verbose:
            print(f"\n{'='*25} NEW INTERACTION {'='*25}")
            print(f"User > {user_input}")
        
        # Step 1: Retrieve context from the agent's memory strategy
        start_time = time.time()
        context = self.memory.get_context(query=user_input)
        retrieval_time = time.time() - start_time
        
        # Step 2: Build complete prompt for the LLM
        full_user_prompt = f"### MEMORY CONTEXT\n{context}\n\n### CURRENT REQUEST\n{user_input}"
        
        # Step 3: Calculate token usage for debugging
        prompt_tokens = count_tokens(self.system_prompt + full_user_prompt)
        
        if verbose:
            print("\n--- Agent Debug Info ---")
            print(f"Memory Retrieval Time: {retrieval_time:.4f} seconds")
            print(f"Estimated Prompt Tokens: {prompt_tokens}")
            print(f"\n[Context Retrieved]:\n{context}\n")
        
        # Step 4: Call LLM to get response
        start_time = time.time()
        ai_response = generate_text(self.system_prompt, full_user_prompt, self.client)
        generation_time = time.time() - start_time
        
        # Step 5: Update memory with the latest interaction
        self.memory.add_message(user_input, ai_response)
        
        # Step 6: Update prompt token tracking if memory strategy supports it
        if hasattr(self.memory, 'total_prompt_tokens'):
            self.memory.total_prompt_tokens += prompt_tokens
        
        # Step 7: Display AI response and performance metrics
        if verbose:
            print(f"\nAgent > {ai_response}")
            print(f"(LLM Generation Time: {generation_time:.4f} seconds)")
            print(f"{'='*70}")
        
        return {
            "user_input": user_input,
            "ai_response": ai_response,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "prompt_tokens": prompt_tokens,
            "context": context
        }
    
    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary containing memory usage statistics
        """
        return self.memory.get_memory_stats()
    
    def clear_memory(self) -> None:
        """
        Clear the agent's memory.
        """
        self.memory.clear()
        print("Agent memory cleared.")
    
    def set_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt.
        
        Args:
            new_prompt: New system prompt to use
        """
        self.system_prompt = new_prompt
        print(f"System prompt updated to: {new_prompt[:50]}...")
