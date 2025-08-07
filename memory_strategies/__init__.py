"""
Memory Strategies Package

This package contains 9 different memory optimization techniques for AI agents,
ranging from simple sequential storage to complex operating system-like memory management.
"""

from .base_memory import BaseMemoryStrategy
from .ai_agent import AIAgent
from .utils import generate_text, generate_embedding, count_tokens, get_openai_client

# Basic Memory Strategies
from .sequential_memory import SequentialMemory
from .sliding_window_memory import SlidingWindowMemory
from .summarization_memory import SummarizationMemory

# Advanced Memory Strategies
from .retrieval_memory import RetrievalMemory
from .memory_augmented_memory import MemoryAugmentedMemory
from .hierarchical_memory import HierarchicalMemory

# Complex Memory Strategies
from .graph_memory import GraphMemory
from .compression_memory import CompressionMemory
from .os_memory import OSMemory

__all__ = [
    # Base classes
    "BaseMemoryStrategy",
    "AIAgent",
    
    # Utilities
    "generate_text",
    "generate_embedding", 
    "count_tokens",
    "get_openai_client",
    
    # Basic strategies
    "SequentialMemory",
    "SlidingWindowMemory", 
    "SummarizationMemory",
    
    # Advanced strategies
    "RetrievalMemory",
    "MemoryAugmentedMemory",
    "HierarchicalMemory",
    
    # Complex strategies
    "GraphMemory",
    "CompressionMemory",
    "OSMemory"
]

# Strategy metadata for easy reference
STRATEGY_INFO = {
    "SequentialMemory": {
        "complexity": "Basic",
        "description": "Stores all conversation history chronologically",
        "best_for": "Simple, short-term chatbots"
    },
    "SlidingWindowMemory": {
        "complexity": "Basic", 
        "description": "Maintains only the most recent N conversations",
        "best_for": "Controlled memory usage scenarios"
    },
    "SummarizationMemory": {
        "complexity": "Basic",
        "description": "Compresses conversation history using LLM summarization", 
        "best_for": "Long-term creative conversations"
    },
    "RetrievalMemory": {
        "complexity": "Advanced",
        "description": "Uses vector embeddings and similarity search (RAG)",
        "best_for": "Accurate long-term recall, industry standard"
    },
    "MemoryAugmentedMemory": {
        "complexity": "Advanced",
        "description": "Combines sliding window with persistent memory tokens",
        "best_for": "Personal assistants requiring fact retention"
    },
    "HierarchicalMemory": {
        "complexity": "Advanced", 
        "description": "Multi-layered system with working + long-term memory",
        "best_for": "Human-like cognitive patterns"
    },
    "GraphMemory": {
        "complexity": "Complex",
        "description": "Treats conversations as nodes with relationship edges",
        "best_for": "Expert systems and knowledge bases"
    },
    "CompressionMemory": {
        "complexity": "Complex",
        "description": "Intelligent compression and integration of historical data",
        "best_for": "Space-constrained environments"
    },
    "OSMemory": {
        "complexity": "Complex",
        "description": "Simulates RAM/disk with active/passive memory",
        "best_for": "Large-scale systems with unlimited memory needs"
    }
}
