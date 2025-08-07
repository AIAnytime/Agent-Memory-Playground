"""
Operating System-like Memory Management Strategy

This strategy simulates how computer operating systems manage memory with
RAM (active memory) and disk (passive memory), implementing paging mechanisms
for intelligent memory management.
"""

from collections import deque
from typing import Dict, Any, Optional, Tuple
from .base_memory import BaseMemoryStrategy


class OSMemory(BaseMemoryStrategy):
    """
    OS-like memory management strategy simulating RAM and disk storage.
    
    Advantages:
    - Scalable memory management
    - Intelligent paging system
    - Efficient active context
    - Nearly unlimited memory capacity
    
    Disadvantages:
    - Complex paging logic
    - May miss relevant passive information
    - Requires tuning of RAM size
    - Page fault overhead
    """
    
    def __init__(self, ram_size: int = 2):
        """
        Initialize OS-like memory system.
        
        Args:
            ram_size: Maximum number of conversation turns to retain in active memory (RAM)
        """
        self.ram_size = ram_size
        
        # 'RAM' is a deque that holds recent turns
        self.active_memory: deque = deque()
        
        # 'Hard disk' is a dictionary for storing paged-out turns
        self.passive_memory: Dict[int, str] = {}
        
        # Counter to give each turn a unique ID
        self.turn_count = 0
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add turn to active memory, page out oldest turn to passive memory if RAM is full.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        turn_id = self.turn_count
        turn_data = f"User: {user_input}\nAI: {ai_response}"
        
        # Check if active memory (RAM) is full
        if len(self.active_memory) >= self.ram_size:
            # If so, remove least recently used (oldest) item from active memory
            lru_turn_id, lru_turn_data = self.active_memory.popleft()
            
            # Move it to passive memory (hard disk)
            self.passive_memory[lru_turn_id] = lru_turn_data
            print(f"--- [OS Memory: Paging out Turn {lru_turn_id} to passive storage.] ---")
        
        # Add new turn to active memory
        self.active_memory.append((turn_id, turn_data))
        self.turn_count += 1
    
    def get_context(self, query: str) -> str:
        """
        Provide RAM context and simulate 'page faults' by pulling from passive memory if needed.
        
        Args:
            query: Current user query
            
        Returns:
            Context from active memory and any paged-in passive memory
        """
        # Base context is always what's in active memory
        active_context = "\n".join([data for _, data in self.active_memory])
        
        # Simulate page fault: check if any words in query match content in passive memory
        paged_in_context = ""
        query_words = [word.lower() for word in query.split() if len(word) > 3]
        
        for turn_id, data in self.passive_memory.items():
            # Check for keyword matches in passive memory
            if any(word in data.lower() for word in query_words):
                paged_in_context += f"\n(Paged in from Turn {turn_id}): {data}"
                print(f"--- [OS Memory: Page fault! Paging in Turn {turn_id} from passive storage.] ---")
        
        # Combine active context with any paged-in context
        if paged_in_context:
            return f"### Active Memory (RAM):\n{active_context}\n\n### Paged-In from Passive Memory (Disk):\n{paged_in_context}"
        else:
            return f"### Active Memory (RAM):\n{active_context}" if active_context else "No information in memory yet."
    
    def clear(self) -> None:
        """Clear both active and passive memory storage."""
        self.active_memory.clear()
        self.passive_memory = {}
        self.turn_count = 0
        print("OS-like memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory usage.
        
        Returns:
            Dictionary containing memory statistics
        """
        active_turns = len(self.active_memory)
        passive_turns = len(self.passive_memory)
        total_turns = self.turn_count
        
        return {
            "strategy_type": "OSMemory",
            "ram_size": self.ram_size,
            "active_turns": active_turns,
            "passive_turns": passive_turns,
            "total_turns": total_turns,
            "memory_size": f"{active_turns} in RAM, {passive_turns} on disk",
            "advantages": ["Scalable management", "Intelligent paging", "Unlimited capacity"],
            "disadvantages": ["Complex paging", "May miss passive info", "Page fault overhead"]
        }
