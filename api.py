"""
FastAPI Application for AI Agent Memory Strategies

This API provides endpoints to interact with all 9 memory optimization techniques
through RESTful endpoints. Each strategy can be used independently via the API.
"""

import os
import uuid
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

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
    STRATEGY_INFO,
    get_openai_client
)


# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    user_input: str
    retrieval_time: float
    generation_time: float
    prompt_tokens: int
    session_id: str
    strategy_type: str


class SessionCreateRequest(BaseModel):
    strategy_type: str
    strategy_config: Optional[Dict[str, Any]] = {}
    system_prompt: Optional[str] = "You are a helpful AI assistant."
    api_key: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    strategy_type: str
    strategy_config: Dict[str, Any]
    created: bool


class MemoryStatsResponse(BaseModel):
    session_id: str
    strategy_type: str
    memory_stats: Dict[str, Any]


class StrategyInfoResponse(BaseModel):
    strategy_name: str
    complexity: str
    description: str
    best_for: str
    default_config: Dict[str, Any]


# Global storage for active sessions
active_sessions: Dict[str, AIAgent] = {}

# Available strategies with their default configurations
AVAILABLE_STRATEGIES = {
    "sequential": {
        "class": SequentialMemory,
        "default_config": {},
        "description": "Stores all conversation history chronologically"
    },
    "sliding_window": {
        "class": SlidingWindowMemory,
        "default_config": {"window_size": 4},
        "description": "Maintains only the most recent N conversations"
    },
    "summarization": {
        "class": SummarizationMemory,
        "default_config": {"summary_threshold": 4},
        "description": "Compresses conversation history using LLM summarization"
    },
    "retrieval": {
        "class": RetrievalMemory,
        "default_config": {"k": 2, "embedding_dim": 1536},
        "description": "Uses vector embeddings and similarity search (RAG)"
    },
    "memory_augmented": {
        "class": MemoryAugmentedMemory,
        "default_config": {"window_size": 2},
        "description": "Combines sliding window with persistent memory tokens"
    },
    "hierarchical": {
        "class": HierarchicalMemory,
        "default_config": {"window_size": 2, "k": 2, "embedding_dim": 1536},
        "description": "Multi-layered system with working + long-term memory"
    },
    "graph": {
        "class": GraphMemory,
        "default_config": {},
        "description": "Treats conversations as nodes with relationship edges"
    },
    "compression": {
        "class": CompressionMemory,
        "default_config": {"compression_ratio": 0.5, "importance_threshold": 0.7},
        "description": "Intelligent compression and integration of historical data"
    },
    "os_memory": {
        "class": OSMemory,
        "default_config": {"ram_size": 2},
        "description": "Simulates RAM/disk with active/passive memory"
    }
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting AI Agent Memory Strategies API...")
    yield
    # Shutdown
    print("Shutting down API...")
    active_sessions.clear()


# Initialize FastAPI app
app = FastAPI(
    title="AI Agent Memory Design & Optimization API",
    description="RESTful API for testing and using multiple AI agent memory optimization techniques",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_openai_client_with_key(api_key: Optional[str] = None):
    """Get OpenAI client with provided API key or from environment."""
    if api_key:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    return get_openai_client()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Agent Memory Strategies API",
        "version": "1.0.0",
        "available_strategies": list(AVAILABLE_STRATEGIES.keys()),
        "endpoints": {
            "GET /strategies": "List all available memory strategies",
            "POST /sessions": "Create a new chat session with a memory strategy",
            "POST /sessions/{session_id}/chat": "Send a message to a specific session",
            "GET /sessions/{session_id}/stats": "Get memory statistics for a session",
            "DELETE /sessions/{session_id}": "Delete a session",
            "GET /sessions": "List all active sessions"
        }
    }


@app.get("/strategies", response_model=List[StrategyInfoResponse])
async def list_strategies():
    """List all available memory strategies with their information."""
    strategies = []
    
    for strategy_key, strategy_data in AVAILABLE_STRATEGIES.items():
        strategy_class = strategy_data["class"]
        strategy_name = strategy_class.__name__
        
        # Get strategy info from STRATEGY_INFO
        info = STRATEGY_INFO.get(strategy_name, {})
        
        strategies.append(StrategyInfoResponse(
            strategy_name=strategy_key,
            complexity=info.get("complexity", "Unknown"),
            description=info.get("description", strategy_data["description"]),
            best_for=info.get("best_for", "General use"),
            default_config=strategy_data["default_config"]
        ))
    
    return strategies


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session with specified memory strategy."""
    if request.strategy_type not in AVAILABLE_STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy type. Available: {list(AVAILABLE_STRATEGIES.keys())}"
        )
    
    try:
        # Get strategy configuration
        strategy_info = AVAILABLE_STRATEGIES[request.strategy_type]
        strategy_class = strategy_info["class"]
        
        # Merge default config with user config
        config = {**strategy_info["default_config"], **request.strategy_config}
        
        # Get OpenAI client
        client = get_openai_client_with_key(request.api_key)
        
        # Add client to config if strategy supports it
        if hasattr(strategy_class, '__init__'):
            import inspect
            sig = inspect.signature(strategy_class.__init__)
            if 'client' in sig.parameters:
                config['client'] = client
        
        # Initialize memory strategy
        memory_strategy = strategy_class(**config)
        
        # Create AI agent
        agent = AIAgent(
            memory_strategy=memory_strategy,
            system_prompt=request.system_prompt,
            client=client
        )
        
        # Generate session ID and store
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = agent
        
        return SessionResponse(
            session_id=session_id,
            strategy_type=request.strategy_type,
            strategy_config=config,
            created=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat_with_session(session_id: str, request: ChatRequest):
    """Send a message to a specific session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent = active_sessions[session_id]
        
        # Process the chat message
        result = agent.chat(request.message, verbose=False)
        
        return ChatResponse(
            response=result["ai_response"],
            user_input=result["user_input"],
            retrieval_time=result["retrieval_time"],
            generation_time=result["generation_time"],
            prompt_tokens=result["prompt_tokens"],
            session_id=session_id,
            strategy_type=type(agent.memory).__name__
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/sessions/{session_id}/stats", response_model=MemoryStatsResponse)
async def get_session_stats(session_id: str):
    """Get memory statistics for a specific session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent = active_sessions[session_id]
        stats = agent.get_memory_stats()
        
        return MemoryStatsResponse(
            session_id=session_id,
            strategy_type=type(agent.memory).__name__,
            memory_stats=stats
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del active_sessions[session_id]
    return {"message": f"Session {session_id} deleted successfully"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    sessions = []
    for session_id, agent in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "strategy_type": type(agent.memory).__name__,
            "system_prompt": agent.system_prompt[:50] + "..." if len(agent.system_prompt) > 50 else agent.system_prompt
        })
    
    return {"active_sessions": len(sessions), "sessions": sessions}


@app.post("/sessions/{session_id}/clear")
async def clear_session_memory(session_id: str):
    """Clear memory for a specific session."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        agent = active_sessions[session_id]
        agent.clear_memory()
        return {"message": f"Memory cleared for session {session_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY"))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
