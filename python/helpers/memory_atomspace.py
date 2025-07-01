"""
Memory AtomSpace Integration

Integrates the hypergraph AtomSpace with the existing memory system,
providing both vector-based and graph-based memory capabilities.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from .atomspace import AtomSpace, DistributedAtomSpaceAPI, Node, Link, AtomType


class MemoryAtomSpaceWrapper:
    """
    Wrapper that integrates AtomSpace hypergraph capabilities with memory storage.
    Provides distributed memory capabilities as standalone component.
    """
    
    def __init__(self, agent_id: str = "default", atomspace_subdir: str = "atomspace"):
        self.agent_id = agent_id
        self.atomspace_subdir = atomspace_subdir
        
        # Initialize AtomSpace storage path
        memory_path = Path("memory") / agent_id
        atomspace_path = memory_path / atomspace_subdir
        atomspace_path.mkdir(parents=True, exist_ok=True)
        
        db_path = atomspace_path / "hypergraph.db"
        self.atomspace = AtomSpace(str(db_path))
        self.api = DistributedAtomSpaceAPI(self.atomspace)
    
    @staticmethod
    async def get(agent_id: str = "default") -> 'MemoryAtomSpaceWrapper':
        """Get or create MemoryAtomSpaceWrapper for agent"""
        return MemoryAtomSpaceWrapper(agent_id, "atomspace")
    
    # AtomSpace-specific methods
    async def add_knowledge_node(self, name: str, concept_type: str = "knowledge", 
                                content: str = "", metadata: Optional[Dict[str, Any]] = None) -> Node:
        """Add a knowledge node to the hypergraph"""
        if metadata is None:
            metadata = {}
        metadata['content'] = content
        metadata['source'] = 'agent'
        
        return await self.atomspace.add_node(
            name=name,
            concept_type=concept_type,
            metadata=metadata
        )
    
    async def add_relationship_link(self, name: str, source_id: str, target_id: str,
                                  link_type: str = "relates_to", 
                                  metadata: Optional[Dict[str, Any]] = None) -> Link:
        """Add a relationship link between atoms"""
        return await self.atomspace.add_link(
            name=name,
            outgoing=[source_id, target_id],
            link_type=link_type,
            metadata=metadata or {}
        )
    
    async def find_related_concepts(self, concept_name: str, relationship_type: str = None) -> List[Dict[str, Any]]:
        """Find concepts related to a given concept"""
        # First find the concept node
        pattern = {'atom_type': AtomType.NODE.value, 'name': concept_name}
        nodes = await self.atomspace.pattern_match(pattern, limit=1)
        
        if not nodes:
            return []
        
        concept_node = nodes[0]
        
        # Find links that include this node
        link_pattern = {'atom_type': AtomType.LINK.value}
        if relationship_type:
            link_pattern['link_type'] = relationship_type
            
        links = await self.atomspace.pattern_match(link_pattern, limit=100)
        
        related = []
        for link in links:
            if isinstance(link, Link) and concept_node.id in link.outgoing:
                # Get the other atoms in this link
                for atom_id in link.outgoing:
                    if atom_id != concept_node.id:
                        related_atom = await self.atomspace.get_atom(atom_id)
                        if related_atom:
                            related.append({
                                'atom': related_atom.to_dict(),
                                'relationship': link.link_type,
                                'link_name': link.name
                            })
        
        return related
    
    async def create_memory_snapshot(self, description: str = "Agent memory snapshot") -> str:
        """Create a snapshot of the current memory state"""
        return await self.atomspace.create_snapshot(description)
    
    async def get_memory_tensor_state(self, snapshot_id: Optional[str] = None) -> Dict[str, Any]:
        """Get tensor representation of memory state"""
        return await self.api.get_memory_state(snapshot_id)
    
    # Distributed API methods
    async def read_hypergraph_fragment(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read hypergraph fragment for distributed agents"""
        return await self.api.read_fragment(pattern)
    
    async def write_hypergraph_fragment(self, atoms_data: List[Dict[str, Any]]) -> List[str]:
        """Write hypergraph fragment from distributed agents"""
        return await self.api.write_fragment(atoms_data)
    
    # Vector similarity methods (simplified interface for compatibility)
    async def search_similar_concepts(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar concepts using name-based matching"""
        # Simple pattern matching based on query words
        query_words = query.lower().split()
        results = []
        
        for word in query_words:
            if len(word) > 2:  # Skip very short words
                pattern = {"atom_type": AtomType.NODE.value, "name": word}
                matches = await self.atomspace.pattern_match(pattern, limit)
                for match in matches:
                    results.append({
                        'atom': match.to_dict(),
                        'relevance': 1.0,  # Simplified relevance score
                        'query_term': word
                    })
        
        return results[:limit]
    
    async def store_text_as_concept(self, text: str, concept_name: str = None, metadata: dict = None):
        """Store text as a concept node"""
        if concept_name is None:
            # Generate concept name from text
            words = text.split()[:3]  # First 3 words
            concept_name = "_".join(words).lower()
        
        return await self.add_knowledge_node(concept_name, "text_concept", text, metadata)
    
    # Hybrid methods for knowledge storage
    async def store_knowledge_with_relations(self, content: str, concept_name: str,
                                           related_concepts: List[str] = None,
                                           metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store knowledge as hypergraph representation"""
        results = {}
        
        # Store as hypergraph node
        node = await self.add_knowledge_node(concept_name, "knowledge", content, metadata)
        results['node_id'] = node.id
        
        # Create relationships if specified
        if related_concepts:
            link_ids = []
            for related_concept in related_concepts:
                # Find or create related concept node
                pattern = {'atom_type': AtomType.NODE.value, 'name': related_concept}
                existing = await self.atomspace.pattern_match(pattern, limit=1)
                
                if existing:
                    related_node_id = existing[0].id
                else:
                    related_node = await self.add_knowledge_node(related_concept, "concept")
                    related_node_id = related_node.id
                
                # Create relationship link
                link = await self.add_relationship_link(
                    f"{concept_name}_relates_to_{related_concept}",
                    node.id,
                    related_node_id,
                    "relates_to"
                )
                link_ids.append(link.id)
            
            results['link_ids'] = link_ids
        
        return results
    
    async def query_knowledge_hybrid(self, query: str, use_relations: bool = True, 
                                   limit: int = 10) -> Dict[str, Any]:
        """Query knowledge using graph relationships and simple matching"""
        results = {'concept_results': [], 'graph_results': []}
        
        # Simple concept search
        concept_results = await self.search_similar_concepts(query, limit)
        results['concept_results'] = concept_results
        
        # Graph pattern search if relations are requested
        if use_relations:
            # Extract potential concept names from query (simplified)
            query_words = query.lower().split()
            for word in query_words:
                if len(word) > 3:  # Skip short words
                    related = await self.find_related_concepts(word)
                    if related:
                        results['graph_results'].extend(related)
        
        return results


class HypergraphMemoryAgent:
    """
    Agent wrapper that provides hypergraph memory capabilities
    """
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.memory_wrapper: Optional[MemoryAtomSpaceWrapper] = None
    
    async def initialize_hypergraph_memory(self) -> MemoryAtomSpaceWrapper:
        """Initialize hypergraph memory for this agent"""
        if self.memory_wrapper is None:
            self.memory_wrapper = await MemoryAtomSpaceWrapper.get(self.agent_id)
        return self.memory_wrapper
    
    async def remember_with_context(self, content: str, context_concepts: List[str] = None,
                                  memory_type: str = "experience") -> str:
        """Remember content with contextual relationships"""
        if self.memory_wrapper is None:
            await self.initialize_hypergraph_memory()
        
        # Create unique concept name from content
        concept_name = f"{memory_type}_{hash(content) % 10000}"
        
        result = await self.memory_wrapper.store_knowledge_with_relations(
            content=content,
            concept_name=concept_name,
            related_concepts=context_concepts or [],
            metadata={'type': memory_type, 'agent_id': self.agent_id}
        )
        
        return result['node_id']
    
    async def recall_by_association(self, query: str, include_related: bool = True) -> Dict[str, Any]:
        """Recall memories using both similarity and associative relationships"""
        if self.memory_wrapper is None:
            await self.initialize_hypergraph_memory()
        
        return await self.memory_wrapper.query_knowledge_hybrid(
            query=query,
            use_relations=include_related,
            limit=10
        )
    
    async def create_memory_checkpoint(self, description: str = None) -> str:
        """Create a checkpoint of current memory state"""
        if self.memory_wrapper is None:
            await self.initialize_hypergraph_memory()
        
        if description is None:
            description = f"Memory checkpoint for agent {self.agent_id}"
        
        return await self.memory_wrapper.create_memory_snapshot(description)