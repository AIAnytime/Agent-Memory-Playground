"""
Utility Functions for Memory Strategies

This module provides core utility functions used across all memory strategies,
including text generation, embedding generation, and token counting.
"""

import os
import time
import tiktoken
from typing import List, Optional
from openai import OpenAI


# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

# Model configurations
GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_client() -> OpenAI:
    """
    Initialize and return OpenAI client with API key from environment.
    
    Returns:
        Configured OpenAI client instance
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    return OpenAI(api_key=api_key)


def generate_text(system_prompt: str, user_prompt: str, client: Optional[OpenAI] = None) -> str:
    """
    Generate text response using the LLM API.
    
    Args:
        system_prompt: System instructions defining AI role and behavior
        user_prompt: User input that AI should respond to
        client: Optional OpenAI client instance
        
    Returns:
        Generated text content from the AI
    """
    if client is None:
        client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating text: {str(e)}"


def generate_embedding(text: str, client: Optional[OpenAI] = None) -> List[float]:
    """
    Generate embedding vector for given text using the embedding model.
    
    Args:
        text: Input text to convert to embedding vector
        client: Optional OpenAI client instance
        
    Returns:
        List of floats representing the embedding vector
    """
    if client is None:
        client = get_openai_client()
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return []


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text string.
    
    Args:
        text: String to tokenize and count
        
    Returns:
        Integer count of tokens
    """
    return len(tokenizer.encode(text))


def format_conversation_turn(user_input: str, ai_response: str) -> str:
    """
    Format a conversation turn into a standardized string format.
    
    Args:
        user_input: User's message
        ai_response: AI's response
        
    Returns:
        Formatted conversation turn string
    """
    return f"User: {user_input}\nAssistant: {ai_response}"


def measure_time(func):
    """
    Decorator to measure execution time of functions.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapper function that measures execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[TIMING] {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper
