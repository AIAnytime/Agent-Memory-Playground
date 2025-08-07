"""
Memory Compression and Integration Strategy

This strategy compresses and integrates historical conversations through intelligent
algorithms, significantly reducing storage space and processing overhead while
retaining key information through multi-level compression mechanisms.
"""

import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from .base_memory import BaseMemoryStrategy
from .utils import generate_text, get_openai_client, count_tokens


class CompressionMemory(BaseMemoryStrategy):
    """
    Memory compression strategy with intelligent information integration.
    
    Advantages:
    - Significant storage space reduction
    - Intelligent information merging
    - Dynamic importance scoring
    - Automatic redundancy filtering
    
    Disadvantages:
    - Complex compression algorithms
    - Potential information loss
    - Computational overhead for compression
    - Tuning required for optimal performance
    """
    
    def __init__(
        self, 
        compression_ratio: float = 0.5,
        importance_threshold: float = 0.7,
        client: Optional[OpenAI] = None
    ):
        """
        Initialize compression memory system.
        
        Args:
            compression_ratio: Target compression ratio (0.5 = 50% compression)
            importance_threshold: Threshold for importance scoring (0-1)
            client: Optional OpenAI client instance
        """
        self.compression_ratio = compression_ratio
        self.importance_threshold = importance_threshold
        self.client = client or get_openai_client()
        
        # Store conversation segments with metadata
        self.memory_segments: List[Dict[str, Any]] = []
        
        # Compressed memory storage
        self.compressed_memory: List[Dict[str, Any]] = []
        
        # Track compression statistics
        self.compression_stats = {
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_count": 0
        }
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add new conversation turn with importance scoring and compression triggers.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        # Calculate importance score for this conversation turn
        importance_score = self._calculate_importance_score(user_input, ai_response)
        
        # Create memory segment with metadata
        segment = {
            "user_input": user_input,
            "ai_response": ai_response,
            "importance_score": importance_score,
            "timestamp": len(self.memory_segments),
            "token_count": count_tokens(user_input + ai_response),
            "compressed": False
        }
        
        self.memory_segments.append(segment)
        self.compression_stats["original_tokens"] += segment["token_count"]
        
        # Trigger compression if we have enough segments
        if len(self.memory_segments) >= 6:  # Compress every 6 segments
            self._compress_memory_segments()
    
    def _calculate_importance_score(self, user_input: str, ai_response: str) -> float:
        """
        Calculate importance score for a conversation turn using LLM.
        
        Args:
            user_input: User's message
            ai_response: AI's response
            
        Returns:
            Importance score between 0 and 1
        """
        scoring_prompt = (
            f"Rate the importance of this conversation turn on a scale of 0.0 to 1.0. "
            f"Consider factors like: factual information, user preferences, decisions, "
            f"emotional significance, and future relevance. "
            f"Respond with only a number between 0.0 and 1.0.\n\n"
            f"User: {user_input}\n"
            f"AI: {ai_response}"
        )
        
        try:
            score_text = generate_text(
                "You are an importance scoring expert.",
                scoring_prompt,
                self.client
            )
            # Extract numeric score from response
            score = float(score_text.strip())
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1
        except:
            return 0.5  # Default moderate importance
    
    def _compress_memory_segments(self) -> None:
        """
        Compress memory segments using intelligent algorithms.
        """
        print("--- [Memory Compression: Compressing memory segments] ---")
        
        # Separate high and low importance segments
        high_importance = [s for s in self.memory_segments if s["importance_score"] >= self.importance_threshold]
        low_importance = [s for s in self.memory_segments if s["importance_score"] < self.importance_threshold]
        
        # Compress low importance segments
        if low_importance:
            compressed_segment = self._semantic_compression(low_importance)
            self.compressed_memory.append(compressed_segment)
        
        # Keep high importance segments with minimal compression
        for segment in high_importance:
            segment["compressed"] = True
            self.compressed_memory.append({
                "type": "high_importance",
                "content": f"User: {segment['user_input']}\nAI: {segment['ai_response']}",
                "importance_score": segment["importance_score"],
                "timestamp": segment["timestamp"]
            })
        
        # Clear processed segments
        self.memory_segments = []
        self.compression_stats["compression_count"] += 1
    
    def _semantic_compression(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform semantic-level compression on low importance segments.
        
        Args:
            segments: List of memory segments to compress
            
        Returns:
            Compressed segment dictionary
        """
        # Combine all low importance conversations
        combined_text = "\n".join([
            f"User: {s['user_input']}\nAI: {s['ai_response']}"
            for s in segments
        ])
        
        # Use LLM to create compressed summary
        compression_prompt = (
            f"Compress the following conversations into a concise summary that retains "
            f"the key information while reducing length by approximately {int(self.compression_ratio * 100)}%. "
            f"Focus on facts, decisions, and context that might be relevant later.\n\n"
            f"Conversations:\n{combined_text}\n\n"
            f"Compressed Summary:"
        )
        
        compressed_content = generate_text(
            "You are a memory compression expert.",
            compression_prompt,
            self.client
        )
        
        compressed_tokens = count_tokens(compressed_content)
        original_tokens = sum(s["token_count"] for s in segments)
        
        self.compression_stats["compressed_tokens"] += compressed_tokens
        
        return {
            "type": "compressed",
            "content": compressed_content,
            "original_segments": len(segments),
            "compression_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 0,
            "timestamp_range": (segments[0]["timestamp"], segments[-1]["timestamp"])
        }
    
    def get_context(self, query: str) -> str:
        """
        Retrieve relevant context from both active segments and compressed memory.
        
        Args:
            query: Current user query
            
        Returns:
            Relevant context from compressed and active memory
        """
        context_parts = []
        
        # Add relevant compressed memory
        for compressed_segment in self.compressed_memory:
            if self._is_relevant_to_query(compressed_segment["content"], query):
                context_parts.append(f"[Compressed Memory]: {compressed_segment['content']}")
        
        # Add recent active segments
        for segment in self.memory_segments[-3:]:  # Last 3 active segments
            context_parts.append(f"User: {segment['user_input']}\nAI: {segment['ai_response']}")
        
        if not context_parts:
            return "No relevant information in memory yet."
        
        return "### Memory Context:\n" + "\n---\n".join(context_parts)
    
    def _is_relevant_to_query(self, content: str, query: str) -> bool:
        """
        Simple relevance check based on keyword overlap.
        
        Args:
            content: Memory content to check
            query: User query
            
        Returns:
            True if content is relevant to query
        """
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Check for word overlap (simple heuristic)
        overlap = len(query_words.intersection(content_words))
        return overlap >= 2  # At least 2 words in common
    
    def clear(self) -> None:
        """Reset all memory storage and statistics."""
        self.memory_segments = []
        self.compressed_memory = []
        self.compression_stats = {
            "original_tokens": 0,
            "compressed_tokens": 0,
            "compression_count": 0
        }
        print("Compression memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about memory compression.
        
        Returns:
            Dictionary containing memory statistics
        """
        active_segments = len(self.memory_segments)
        compressed_segments = len(self.compressed_memory)
        
        overall_compression_ratio = (
            self.compression_stats["compressed_tokens"] / self.compression_stats["original_tokens"]
            if self.compression_stats["original_tokens"] > 0 else 0
        )
        
        return {
            "strategy_type": "CompressionMemory",
            "compression_ratio_target": self.compression_ratio,
            "importance_threshold": self.importance_threshold,
            "active_segments": active_segments,
            "compressed_segments": compressed_segments,
            "compression_stats": self.compression_stats,
            "overall_compression_ratio": overall_compression_ratio,
            "memory_size": f"{active_segments} active + {compressed_segments} compressed",
            "advantages": ["Space reduction", "Intelligent merging", "Redundancy filtering"],
            "disadvantages": ["Complex algorithms", "Information loss", "Computational overhead"]
        }
