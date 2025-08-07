"""
AI Agent Memory Design & Optimization - Streamlit Playground
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import json
from typing import Dict, Any, List
import os

# Import memory strategies
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

# Page configuration
st.set_page_config(
    page_title="AI Agent Memory Design & Optimization",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .strategy-card {
        background: #009AFE;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #0080d6;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 154, 254, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .chat-bubble-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
    }
    
    .chat-bubble-ai {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'agents' not in st.session_state:
        st.session_state.agents = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    if 'api_key_set' not in st.session_state:
        st.session_state.api_key_set = False

def setup_openai_client(api_key: str):
    """Setup OpenAI client with provided API key."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key_set = True
        return True
    return False

def get_strategy_class_and_config(strategy_name: str) -> tuple:
    """Get strategy class and default configuration."""
    strategy_mapping = {
        "Sequential Memory": (SequentialMemory, {}),
        "Sliding Window Memory": (SlidingWindowMemory, {"window_size": 4}),
        "Summarization Memory": (SummarizationMemory, {"summary_threshold": 4}),
        "Retrieval Memory (RAG)": (RetrievalMemory, {"k": 2}),
        "Memory-Augmented Memory": (MemoryAugmentedMemory, {"window_size": 2}),
        "Hierarchical Memory": (HierarchicalMemory, {"window_size": 2, "k": 2}),
        "Graph Memory": (GraphMemory, {}),
        "Compression Memory": (CompressionMemory, {"compression_ratio": 0.5}),
        "OS-like Memory": (OSMemory, {"ram_size": 2})
    }
    return strategy_mapping.get(strategy_name, (SequentialMemory, {}))

def create_agent(strategy_name: str, config: Dict[str, Any]) -> AIAgent:
    """Create an AI agent with specified strategy and configuration."""
    strategy_class, default_config = get_strategy_class_and_config(strategy_name)
    
    # Merge configurations
    final_config = {**default_config, **config}
    
    # Create strategy instance
    memory_strategy = strategy_class(**final_config)
    
    # Create agent
    agent = AIAgent(
        memory_strategy=memory_strategy,
        system_prompt="You are a helpful AI assistant with advanced memory capabilities."
    )
    
    return agent

def render_sidebar():
    """Render the sidebar with API key input and strategy information."""
    st.sidebar.markdown("""
    <div class="sidebar-info">
        <h3>🔑 API Configuration</h3>
        <p>Enter your OpenAI API key to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key. This is required for all memory strategies."
    )
    
    if api_key:
        if setup_openai_client(api_key):
            st.sidebar.success("✅ API Key configured successfully!")
        else:
            st.sidebar.error("❌ Invalid API Key")
    elif not st.session_state.api_key_set:
        st.sidebar.warning("⚠️ Please enter your OpenAI API key to continue")
        return False
    
    st.sidebar.markdown("---")
    
    # Strategy information
    st.sidebar.markdown("""
    <div class="sidebar-info">
        <h3>Memory Strategies</h3>
        <p>Explore multiple memory optimization techniques for AI agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Strategy complexity legend
    st.sidebar.markdown("### Complexity Levels")
    st.sidebar.markdown("🟢 **Basic** - Simple implementation")
    st.sidebar.markdown("🟡 **Advanced** - Moderate complexity")
    st.sidebar.markdown("🔴 **Complex** - High complexity")
    
    # Attribution section
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center; font-size: 12px; color: #666; margin-top: 20px;">
            <p>Built by <strong>AI Anytime</strong> ❤️</p>
            <p><a href="https://aianytime.net" target="_blank" style="color: #009AFE; text-decoration: none;">aianytime.net</a></p>
            <p>Creator Portfolio: <a href="https://sonukumar.site" target="_blank" style="color: #009AFE; text-decoration: none;">sonukumar.site</a></p>
            <p>YouTube: <a href="https://www.youtube.com/@AIAnytime" target="_blank" style="color: #009AFE; text-decoration: none;">@AIAnytime</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    return True

def render_main_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>AI Agent Memory Design & Optimization Playground</h1>
        <p>Interactive testing environment for multiple memory optimization techniques</p>
    </div>
    """, unsafe_allow_html=True)

def render_strategy_overview():
    """Render strategy overview cards."""
    st.markdown("## Available Memory Strategies")
    
    # Create columns for strategy cards
    col1, col2, col3 = st.columns(3)
    
    strategies = [
        ("Sequential Memory", "🟢", "Stores all conversation history"),
        ("Sliding Window Memory", "🟢", "Recent N conversations only"),
        ("Summarization Memory", "🟢", "LLM-based compression"),
        ("Retrieval Memory (RAG)", "🟡", "Vector similarity search"),
        ("Memory-Augmented Memory", "🟡", "Persistent memory tokens"),
        ("Hierarchical Memory", "🟡", "Multi-layered memory"),
        ("Graph Memory", "🔴", "Relationship modeling"),
        ("Compression Memory", "🔴", "Intelligent compression"),
        ("OS-like Memory", "🔴", "RAM/disk simulation")
    ]
    
    for i, (name, complexity, desc) in enumerate(strategies):
        col = [col1, col2, col3][i % 3]
        with col:
            st.markdown(f"""
            <div class="strategy-card">
                <h4>{complexity} {name}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

def render_single_strategy_tester():
    """Render single strategy testing interface."""
    st.markdown("## Single Strategy Tester")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Strategy selection
        strategy_name = st.selectbox(
            "Choose Memory Strategy",
            [
                "Sequential Memory",
                "Sliding Window Memory", 
                "Summarization Memory",
                "Retrieval Memory (RAG)",
                "Memory-Augmented Memory",
                "Hierarchical Memory",
                "Graph Memory",
                "Compression Memory",
                "OS-like Memory"
            ]
        )
        
        # Strategy configuration
        st.markdown("### Configuration")
        config = {}
        
        if strategy_name == "Sliding Window Memory":
            config["window_size"] = st.slider("Window Size", 1, 10, 4)
        elif strategy_name == "Summarization Memory":
            config["summary_threshold"] = st.slider("Summary Threshold", 2, 10, 4)
        elif strategy_name == "Retrieval Memory (RAG)":
            config["k"] = st.slider("Retrieval Count (k)", 1, 5, 2)
        elif strategy_name == "Memory-Augmented Memory":
            config["window_size"] = st.slider("Window Size", 1, 5, 2)
        elif strategy_name == "Hierarchical Memory":
            config["window_size"] = st.slider("Working Memory Size", 1, 5, 2)
            config["k"] = st.slider("Long-term Retrieval (k)", 1, 5, 2)
        elif strategy_name == "Compression Memory":
            config["compression_ratio"] = st.slider("Compression Ratio", 0.1, 0.9, 0.5)
        elif strategy_name == "OS-like Memory":
            config["ram_size"] = st.slider("RAM Size", 1, 5, 2)
        
        # Initialize agent button
        if st.button("🚀 Initialize Agent", type="primary"):
            try:
                agent = create_agent(strategy_name, config)
                st.session_state.agents[strategy_name] = agent
                st.session_state.chat_history[strategy_name] = []
                st.session_state.performance_metrics[strategy_name] = []
                st.success(f"✅ {strategy_name} agent initialized!")
            except Exception as e:
                st.error(f"❌ Error initializing agent: {str(e)}")
    
    with col2:
        if strategy_name in st.session_state.agents:
            # Chat interface
            st.markdown("### Chat Interface")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for msg in st.session_state.chat_history[strategy_name]:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div class="chat-bubble-user">
                            <strong>You:</strong> {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-bubble-ai">
                            <strong>AI:</strong> {msg["content"]}
                            <br><small>⏱️ {msg.get('time', 0):.2f}s | 🔢 {msg.get('tokens', 0)} tokens</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Chat input
            user_input = st.text_input("Your message:", key=f"input_{strategy_name}")
            
            col_send, col_clear = st.columns([1, 1])
            
            with col_send:
                if st.button("Send", key=f"send_{strategy_name}"):
                    if user_input:
                        try:
                            agent = st.session_state.agents[strategy_name]
                            result = agent.chat(user_input, verbose=False)
                            
                            # Add to chat history
                            st.session_state.chat_history[strategy_name].extend([
                                {"role": "user", "content": user_input},
                                {
                                    "role": "assistant", 
                                    "content": result["ai_response"],
                                    "time": result["generation_time"],
                                    "tokens": result["prompt_tokens"]
                                }
                            ])
                            
                            # Add to performance metrics
                            st.session_state.performance_metrics[strategy_name].append({
                                "turn": len(st.session_state.performance_metrics[strategy_name]) + 1,
                                "tokens": result["prompt_tokens"],
                                "retrieval_time": result["retrieval_time"],
                                "generation_time": result["generation_time"]
                            })
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col_clear:
                if st.button("🗑️ Clear", key=f"clear_{strategy_name}"):
                    if strategy_name in st.session_state.agents:
                        st.session_state.agents[strategy_name].clear_memory()
                        st.session_state.chat_history[strategy_name] = []
                        st.session_state.performance_metrics[strategy_name] = []
                        st.success("🧹 Memory cleared!")
                        st.rerun()
            
            # Memory statistics
            if strategy_name in st.session_state.agents:
                st.markdown("### Memory Statistics")
                try:
                    stats = st.session_state.agents[strategy_name].get_memory_stats()
                    
                    # Display key metrics
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Strategy Type</h4>
                            <p>{stats.get('strategy_type', 'Unknown')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[1]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Memory Size</h4>
                            <p>{stats.get('memory_size', 'Unknown')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_cols[2]:
                        turns = len(st.session_state.chat_history[strategy_name]) // 2
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4>Conversation Turns</h4>
                            <p>{turns}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed stats
                    with st.expander("📈 Detailed Statistics"):
                        st.json(stats)
                        
                except Exception as e:
                    st.error(f"Error getting stats: {str(e)}")
        else:
            st.info("👈 Please initialize an agent first to start chatting!")

def render_performance_dashboard():
    """Render performance comparison dashboard."""
    st.markdown("## Performance Dashboard")
    
    if not st.session_state.performance_metrics:
        st.info("No performance data yet. Test some strategies to see metrics!")
        return
    
    # Create performance comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Token Usage Over Time")
        
        # Prepare data for token usage chart
        token_data = []
        for strategy, metrics in st.session_state.performance_metrics.items():
            for metric in metrics:
                token_data.append({
                    "Strategy": strategy,
                    "Turn": metric["turn"],
                    "Tokens": metric["tokens"]
                })
        
        if token_data:
            df_tokens = pd.DataFrame(token_data)
            fig_tokens = px.line(
                df_tokens, 
                x="Turn", 
                y="Tokens", 
                color="Strategy",
                title="Token Usage Comparison",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_tokens.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_tokens, use_container_width=True)
    
    with col2:
        st.markdown("### Response Time Analysis")
        
        # Prepare data for response time chart
        time_data = []
        for strategy, metrics in st.session_state.performance_metrics.items():
            for metric in metrics:
                time_data.append({
                    "Strategy": strategy,
                    "Turn": metric["turn"],
                    "Generation Time": metric["generation_time"],
                    "Retrieval Time": metric["retrieval_time"]
                })
        
        if time_data:
            df_times = pd.DataFrame(time_data)
            fig_times = px.bar(
                df_times, 
                x="Strategy", 
                y=["Generation Time", "Retrieval Time"],
                title="Average Response Times",
                color_discrete_sequence=["#667eea", "#764ba2"]
            )
            fig_times.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_times, use_container_width=True)

def render_batch_tester():
    """Render batch testing interface for comparing multiple strategies."""
    st.markdown("## Batch Strategy Comparison")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Test Configuration")
        
        # Strategy selection
        strategies_to_test = st.multiselect(
            "Select Strategies to Compare",
            [
                "Sequential Memory",
                "Sliding Window Memory", 
                "Retrieval Memory (RAG)",
                "Hierarchical Memory"
            ],
            default=["Sequential Memory", "Retrieval Memory (RAG)"]
        )
        
        # Test conversations
        st.markdown("### Test Conversations")
        test_conversations = st.text_area(
            "Enter test messages (one per line)",
            value="Hi! My name is Alex and I'm a software engineer.\nI'm working on a machine learning project.\nI prefer Python and love coffee.\nWhat do you remember about me?",
            height=150
        ).split('\n')
        
        if st.button("🚀 Run Batch Test", type="primary"):
            if strategies_to_test and test_conversations:
                run_batch_test(strategies_to_test, test_conversations)
    
    with col2:
        if 'batch_results' in st.session_state:
            st.markdown("### Batch Test Results")
            display_batch_results()

def run_batch_test(strategies: List[str], conversations: List[str]):
    """Run batch test on multiple strategies."""
    results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(strategies) * len(conversations)
    current_step = 0
    
    for strategy_name in strategies:
        status_text.text(f"Testing {strategy_name}...")
        
        try:
            # Create agent
            agent = create_agent(strategy_name, {})
            
            strategy_results = {
                "responses": [],
                "metrics": [],
                "final_stats": {}
            }
            
            # Run conversations
            for i, conversation in enumerate(conversations):
                if conversation.strip():
                    result = agent.chat(conversation.strip(), verbose=False)
                    
                    strategy_results["responses"].append({
                        "turn": i + 1,
                        "user": conversation.strip(),
                        "ai": result["ai_response"],
                        "tokens": result["prompt_tokens"],
                        "time": result["generation_time"]
                    })
                    
                    strategy_results["metrics"].append({
                        "turn": i + 1,
                        "tokens": result["prompt_tokens"],
                        "generation_time": result["generation_time"],
                        "retrieval_time": result["retrieval_time"]
                    })
                
                current_step += 1
                progress_bar.progress(current_step / total_steps)
            
            # Get final memory stats
            strategy_results["final_stats"] = agent.get_memory_stats()
            results[strategy_name] = strategy_results
            
        except Exception as e:
            st.error(f"Error testing {strategy_name}: {str(e)}")
    
    st.session_state.batch_results = results
    status_text.text("✅ Batch test completed!")
    progress_bar.progress(1.0)

def display_batch_results():
    """Display batch test results."""
    results = st.session_state.batch_results
    
    # Summary metrics
    st.markdown("#### Summary Metrics")
    
    summary_data = []
    for strategy, data in results.items():
        if data["metrics"]:
            avg_tokens = sum(m["tokens"] for m in data["metrics"]) / len(data["metrics"])
            avg_time = sum(m["generation_time"] for m in data["metrics"]) / len(data["metrics"])
            
            summary_data.append({
                "Strategy": strategy,
                "Avg Tokens": f"{avg_tokens:.0f}",
                "Avg Response Time": f"{avg_time:.2f}s",
                "Memory Size": data["final_stats"].get("memory_size", "Unknown")
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    # Detailed results
    for strategy, data in results.items():
        with st.expander(f"{strategy} - Detailed Results"):
            st.markdown("**Final Response:**")
            if data["responses"]:
                final_response = data["responses"][-1]
                st.markdown(f"*User:* {final_response['user']}")
                st.markdown(f"*AI:* {final_response['ai']}")
            
            st.markdown("**Memory Statistics:**")
            st.json(data["final_stats"])

def main():
    """Main application function."""
    initialize_session_state()
    
    # Render sidebar
    if not render_sidebar():
        st.warning("⚠️ Please configure your OpenAI API key in the sidebar to continue.")
        return
    
    # Render main content
    render_main_header()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", 
        "Single Tester", 
        "Batch Comparison", 
        "Performance Dashboard"
    ])
    
    with tab1:
        render_strategy_overview()
        
        st.markdown("## Getting Started")
        st.markdown("""
        1. **Configure API Key**: Enter your OpenAI API key in the sidebar
        2. **Choose Strategy**: Select a memory strategy to test
        3. **Configure Settings**: Adjust strategy parameters as needed
        4. **Start Chatting**: Initialize an agent and begin testing
        5. **Compare Performance**: Use batch testing to compare strategies
        """)
        
        st.markdown("## Strategy Guide")
        
        guide_col1, guide_col2 = st.columns(2)
        
        with guide_col1:
            st.markdown("""
            **🟢 Basic Strategies (Easy to implement)**
            - **Sequential**: Perfect recall, but expensive for long chats
            - **Sliding Window**: Fixed memory, loses old information
            - **Summarization**: Compresses history, may lose details
            """)
        
        with guide_col2:
            st.markdown("""
            **🟡🔴 Advanced Strategies (Production-ready)**
            - **Retrieval (RAG)**: Industry standard, semantic search
            - **Hierarchical**: Human-like memory patterns
            - **Graph**: Complex relationship modeling
            """)
    
    with tab2:
        render_single_strategy_tester()
    
    with tab3:
        render_batch_tester()
    
    with tab4:
        render_performance_dashboard()

if __name__ == "__main__":
    main()
