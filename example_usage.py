"""
AI Agent Memory Design & Optimization - Example Usage

This file demonstrates how to use multiple memory optimization techniques
in a plug-and-play manner. Each strategy can be easily swapped and tested.
"""

import os
import time
from memory_strategies import (
    AIAgent,
    SequentialMemory,
    SlidingWindowMemory, 
    SummarizationMemory,
    RetrievalMemory,
    MemoryAugmentedMemory,
    HierarchicalMemory,
    GraphMemory,
    CompressionMemory,
    OSMemory,
    STRATEGY_INFO
)


def demo_strategy(strategy_class, strategy_name, test_conversations, **kwargs):
    """
    Demonstrate a specific memory strategy with test conversations.
    
    Args:
        strategy_class: Memory strategy class to test
        strategy_name: Name of the strategy for display
        test_conversations: List of user inputs to test
        **kwargs: Additional arguments for strategy initialization
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {strategy_name}")
    print(f"{'='*60}")
    
    # Display strategy information
    info = STRATEGY_INFO.get(strategy_class.__name__, {})
    print(f"Complexity: {info.get('complexity', 'Unknown')}")
    print(f"Description: {info.get('description', 'No description')}")
    print(f"Best for: {info.get('best_for', 'General use')}")
    print()
    
    try:
        # Initialize strategy and agent
        memory_strategy = strategy_class(**kwargs)
        agent = AIAgent(memory_strategy, system_prompt="You are a helpful AI assistant with memory.")
        
        # Run test conversations
        for i, user_input in enumerate(test_conversations, 1):
            print(f"\n--- Conversation Turn {i} ---")
            result = agent.chat(user_input, verbose=True)
            
            # Add small delay for readability
            time.sleep(0.5)
        
        # Display memory statistics
        print(f"\nMemory Statistics:")
        stats = agent.get_memory_stats()
        for key, value in stats.items():
            if key not in ['advantages', 'disadvantages']:
                print(f"   {key}: {value}")
        
        print(f"\nAdvantages: {', '.join(stats.get('advantages', []))}")
        print(f"Disadvantages: {', '.join(stats.get('disadvantages', []))}")
        
    except Exception as e:
        print(f"Error testing {strategy_name}: {str(e)}")
    
    print(f"\n{'='*60}")


def run_comprehensive_demo():
    """
    Run comprehensive demonstration of all memory strategies.
    """
    print("AI Agent Memory Design & Optimization - Comprehensive Demo")
    print("=" * 60)
    
    # Test conversations that showcase different memory capabilities
    test_conversations = [
        "Hi! My name is Alex and I'm a software engineer.",
        "I'm working on a machine learning project about natural language processing.",
        "My favorite programming language is Python, and I prefer coffee over tea.",
        "Can you remember what my name is and what I'm working on?",
        "What do you know about my preferences?"
    ]
    
    # Test each strategy
    strategies_to_test = [
        (SequentialMemory, "Sequential Memory", {}),
        (SlidingWindowMemory, "Sliding Window Memory", {"window_size": 3}),
        (SummarizationMemory, "Summarization Memory", {"summary_threshold": 4}),
        (RetrievalMemory, "Retrieval Memory", {"k": 2}),
        (MemoryAugmentedMemory, "Memory-Augmented Memory", {"window_size": 2}),
        (HierarchicalMemory, "Hierarchical Memory", {"window_size": 2, "k": 2}),
        (GraphMemory, "Graph Memory", {}),
        (CompressionMemory, "Compression Memory", {"compression_ratio": 0.6}),
        (OSMemory, "OS-like Memory", {"ram_size": 2})
    ]
    
    for strategy_class, strategy_name, kwargs in strategies_to_test:
        demo_strategy(strategy_class, strategy_name, test_conversations, **kwargs)
        
        # Ask user if they want to continue
        user_input = input("\n🤔 Continue to next strategy? (y/n/q to quit): ").lower()
        if user_input == 'q':
            break
        elif user_input == 'n':
            continue


def interactive_strategy_tester():
    """
    Interactive mode for testing specific strategies.
    """
    print("\nInteractive Strategy Tester")
    print("=" * 40)
    
    # Display available strategies
    strategies = {
        "1": (SequentialMemory, "Sequential Memory", {}),
        "2": (SlidingWindowMemory, "Sliding Window Memory", {"window_size": 3}),
        "3": (SummarizationMemory, "Summarization Memory", {"summary_threshold": 4}),
        "4": (RetrievalMemory, "Retrieval Memory", {"k": 2}),
        "5": (MemoryAugmentedMemory, "Memory-Augmented Memory", {"window_size": 2}),
        "6": (HierarchicalMemory, "Hierarchical Memory", {"window_size": 2, "k": 2}),
        "7": (GraphMemory, "Graph Memory", {}),
        "8": (CompressionMemory, "Compression Memory", {"compression_ratio": 0.6}),
        "9": (OSMemory, "OS-like Memory", {"ram_size": 2})
    }
    
    print("Available Memory Strategies:")
    for key, (_, name, _) in strategies.items():
        print(f"  {key}. {name}")
    
    while True:
        choice = input("\nSelect a strategy (1-9) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            break
        
        if choice in strategies:
            strategy_class, strategy_name, kwargs = strategies[choice]
            
            try:
                # Initialize strategy and agent
                memory_strategy = strategy_class(**kwargs)
                agent = AIAgent(memory_strategy, system_prompt="You are a helpful AI assistant.")
                
                print(f"\nNow using: {strategy_name}")
                print("Type 'stats' to see memory statistics, 'clear' to clear memory, 'back' to choose another strategy")
                
                while True:
                    user_input = input("\nYou: ").strip()
                    
                    if user_input.lower() == 'back':
                        break
                    elif user_input.lower() == 'stats':
                        stats = agent.get_memory_stats()
                        print("\nMemory Statistics:")
                        for key, value in stats.items():
                            print(f"   {key}: {value}")
                    elif user_input.lower() == 'clear':
                        agent.clear_memory()
                    elif user_input:
                        result = agent.chat(user_input, verbose=False)
                        print(f"AI: {result['ai_response']}")
                        print(f"Response time: {result['generation_time']:.2f}s | Tokens: {result['prompt_tokens']}")
                
            except Exception as e:
                print(f"Error with {strategy_name}: {str(e)}")
        else:
            print("Invalid choice. Please select 1-9 or 'q'.")


def quick_comparison_demo():
    """
    Quick comparison of key strategies on the same conversation.
    """
    print("\n⚡ Quick Comparison Demo")
    print("=" * 40)
    
    # Single conversation to test all strategies
    test_conversation = [
        "Remember this important fact: I am allergic to peanuts.",
        "I love traveling and have been to Japan, France, and Italy.",
        "My favorite hobby is photography, especially landscape photography.",
        "What do you know about my allergy and travel experiences?"
    ]
    
    # Key strategies to compare
    comparison_strategies = [
        (SequentialMemory, "Sequential", {}),
        (SlidingWindowMemory, "Sliding Window", {"window_size": 2}),
        (RetrievalMemory, "Retrieval (RAG)", {"k": 2}),
        (HierarchicalMemory, "Hierarchical", {"window_size": 2, "k": 2})
    ]
    
    results = {}
    
    for strategy_class, strategy_name, kwargs in comparison_strategies:
        print(f"\nTesting {strategy_name}...")
        
        try:
            memory_strategy = strategy_class(**kwargs)
            agent = AIAgent(memory_strategy, system_prompt="You are a helpful assistant.")
            
            # Run all conversations
            for user_input in test_conversation:
                result = agent.chat(user_input, verbose=False)
            
            # Store final result for comparison
            results[strategy_name] = {
                "final_response": result['ai_response'],
                "final_tokens": result['prompt_tokens'],
                "memory_stats": agent.get_memory_stats()
            }
            
        except Exception as e:
            results[strategy_name] = {"error": str(e)}
    
    # Display comparison
    print(f"\nCOMPARISON RESULTS")
    print("=" * 50)
    
    for strategy_name, data in results.items():
        print(f"\n{strategy_name}:")
        if "error" in data:
            print(f"   Error: {data['error']}")
        else:
            print(f"   Response: {data['final_response'][:100]}...")
            print(f"   Tokens: {data['final_tokens']}")
            print(f"   Memory: {data['memory_stats'].get('memory_size', 'Unknown')}")


def main():
    """
    Main function with menu-driven interface.
    """
    print("AI Agent Memory Design & Optimization - Demo Suite")
    print("=" * 50)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        return
    
    while True:
        print("\nChoose a demo mode:")
        print("1. Comprehensive Demo (all strategies)")
        print("2. Interactive Tester (choose strategy)")
        print("3. Quick Comparison (key strategies)")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_comprehensive_demo()
        elif choice == "2":
            interactive_strategy_tester()
        elif choice == "3":
            quick_comparison_demo()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please select 1-4.")


if __name__ == "__main__":
    main()
