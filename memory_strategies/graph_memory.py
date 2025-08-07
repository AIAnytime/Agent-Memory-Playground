"""
Graph Memory Network Strategy

This strategy treats conversation elements as nodes and their relationships as edges,
enabling complex reasoning and relationship understanding. Particularly suited for
expert systems and knowledge base applications.
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Set
from openai import OpenAI
from .base_memory import BaseMemoryStrategy
from .utils import generate_text, get_openai_client


class GraphMemory(BaseMemoryStrategy):
    """
    Graph-based memory strategy using NetworkX for relationship modeling.
    
    Advantages:
    - Models complex relationships between information
    - Supports logical reasoning queries
    - Structured knowledge representation
    - Excellent for expert systems
    
    Disadvantages:
    - Complex implementation and maintenance
    - Requires relationship extraction
    - May be overkill for simple conversations
    - Computational overhead for large graphs
    """
    
    def __init__(self, client: Optional[OpenAI] = None):
        """
        Initialize graph memory system.
        
        Args:
            client: Optional OpenAI client instance
        """
        self.client = client or get_openai_client()
        
        # Initialize directed graph to store conversation elements and relationships
        self.knowledge_graph = nx.DiGraph()
        
        # Counter for generating unique node IDs
        self.node_counter = 0
        
        # Store raw conversation history for fallback
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_message(self, user_input: str, ai_response: str) -> None:
        """
        Add conversation turn to graph by extracting entities and relationships.
        
        Args:
            user_input: User's message
            ai_response: AI's response
        """
        # Store raw conversation for fallback
        self.conversation_history.append({
            "user": user_input,
            "assistant": ai_response,
            "turn_id": self.node_counter
        })
        
        # Extract entities and relationships from the conversation turn
        self._extract_and_add_entities(user_input, "user", self.node_counter)
        self._extract_and_add_entities(ai_response, "assistant", self.node_counter)
        
        self.node_counter += 1
    
    def _extract_and_add_entities(self, text: str, speaker: str, turn_id: int) -> None:
        """
        Extract entities and relationships from text and add to knowledge graph.
        
        Args:
            text: Text to extract entities from
            speaker: Who said the text (user/assistant)
            turn_id: Turn identifier
        """
        # Use LLM to extract key entities and relationships
        extraction_prompt = (
            f"Extract key entities (people, places, concepts, facts) and relationships from this text. "
            f"Format as: ENTITIES: entity1, entity2, entity3... RELATIONSHIPS: entity1->relationship->entity2, etc.\n\n"
            f"Text: {text}\n\n"
            f"If no clear entities or relationships, respond with 'ENTITIES: none RELATIONSHIPS: none'"
        )
        
        extracted_info = generate_text(
            "You are an entity and relationship extraction expert.",
            extraction_prompt,
            self.client
        )
        
        # Parse extracted information and add to graph
        self._parse_and_add_to_graph(extracted_info, speaker, turn_id, text)
    
    def _parse_and_add_to_graph(self, extracted_info: str, speaker: str, turn_id: int, original_text: str) -> None:
        """
        Parse extracted entities and relationships and add them to the knowledge graph.
        
        Args:
            extracted_info: LLM-extracted entities and relationships
            speaker: Who said the text
            turn_id: Turn identifier
            original_text: Original text for context
        """
        try:
            # Simple parsing of the extraction format
            if "ENTITIES:" in extracted_info and "RELATIONSHIPS:" in extracted_info:
                parts = extracted_info.split("RELATIONSHIPS:")
                entities_part = parts[0].replace("ENTITIES:", "").strip()
                relationships_part = parts[1].strip() if len(parts) > 1 else ""
                
                # Add entities as nodes
                if entities_part.lower() != "none":
                    entities = [e.strip() for e in entities_part.split(",") if e.strip()]
                    for entity in entities:
                        if entity:
                            # Add entity node with metadata
                            self.knowledge_graph.add_node(
                                entity,
                                type="entity",
                                speaker=speaker,
                                turn_id=turn_id,
                                context=original_text[:100]  # First 100 chars for context
                            )
                
                # Add relationships as edges
                if relationships_part.lower() != "none":
                    relationships = [r.strip() for r in relationships_part.split(",") if r.strip()]
                    for rel in relationships:
                        if "->" in rel:
                            parts = rel.split("->")
                            if len(parts) == 3:
                                source, relation, target = [p.strip() for p in parts]
                                if source and target and relation:
                                    # Add relationship edge
                                    self.knowledge_graph.add_edge(
                                        source, target,
                                        relationship=relation,
                                        turn_id=turn_id,
                                        speaker=speaker
                                    )
        except Exception as e:
            print(f"Error parsing extracted info: {e}")
    
    def get_context(self, query: str) -> str:
        """
        Retrieve relevant context by traversing the knowledge graph.
        
        Args:
            query: Current user query
            
        Returns:
            Relevant context from knowledge graph and conversation history
        """
        if self.knowledge_graph.number_of_nodes() == 0:
            return "No information in memory yet."
        
        # Extract entities from the query
        query_extraction_prompt = (
            f"Extract key entities (people, places, concepts) from this query. "
            f"List them separated by commas. If no clear entities, respond with 'none'.\n\n"
            f"Query: {query}"
        )
        
        query_entities = generate_text(
            "You are an entity extraction expert.",
            query_extraction_prompt,
            self.client
        )
        
        relevant_info = []
        
        # Find relevant nodes and relationships
        if query_entities.lower() != "none":
            entities = [e.strip() for e in query_entities.split(",") if e.strip()]
            
            for entity in entities:
                # Find exact matches or similar entities in graph
                for node in self.knowledge_graph.nodes():
                    if entity.lower() in node.lower() or node.lower() in entity.lower():
                        # Get node information
                        node_data = self.knowledge_graph.nodes[node]
                        relevant_info.append(f"Entity: {node} (from {node_data.get('speaker', 'unknown')})")
                        
                        # Get relationships involving this node
                        for neighbor in self.knowledge_graph.neighbors(node):
                            edge_data = self.knowledge_graph.edges[node, neighbor]
                            relationship = edge_data.get('relationship', 'related to')
                            relevant_info.append(f"  → {relationship} → {neighbor}")
        
        # Fallback to recent conversation if no graph matches
        if not relevant_info:
            recent_turns = self.conversation_history[-3:]  # Last 3 turns
            for turn in recent_turns:
                relevant_info.append(f"Turn {turn['turn_id']}: User: {turn['user']}")
                relevant_info.append(f"Turn {turn['turn_id']}: Assistant: {turn['assistant']}")
        
        return "### Knowledge Graph Context:\n" + "\n".join(relevant_info) if relevant_info else "No relevant information found."
    
    def clear(self) -> None:
        """Reset the knowledge graph and conversation history."""
        self.knowledge_graph.clear()
        self.conversation_history = []
        self.node_counter = 0
        print("Graph memory cleared.")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary containing memory statistics
        """
        num_nodes = self.knowledge_graph.number_of_nodes()
        num_edges = self.knowledge_graph.number_of_edges()
        num_turns = len(self.conversation_history)
        
        return {
            "strategy_type": "GraphMemory",
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_turns": num_turns,
            "memory_size": f"{num_nodes} nodes, {num_edges} edges, {num_turns} turns",
            "advantages": ["Relationship modeling", "Complex reasoning", "Structured knowledge"],
            "disadvantages": ["Complex implementation", "Extraction dependent", "Computational overhead"]
        }
