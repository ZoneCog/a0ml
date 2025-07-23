"""
Cognitive AtomSpace Integration

Integrates Scheme-based cognitive representation with the distributed AtomSpace
hypergraph, enabling sophisticated cognitive pattern storage and retrieval.
"""

import uuid
import json
import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

from .atomspace import AtomSpace, Node, Link, AtomType
from .scheme_grammar import SchemeCognitiveGrammarRegistry, CognitiveGrammar, CognitiveOperator
from .pattern_matcher import HypergraphPatternMatcher, Pattern, MatchType
from .pln_reasoning import PLNInferenceEngine, TruthValue
from .memory_atomspace import MemoryAtomSpaceWrapper


@dataclass
class CognitivePattern:
    """Represents a cognitive pattern stored in the AtomSpace"""
    id: str
    name: str
    scheme_expression: str
    grammar_id: str
    atom_ids: List[str] = field(default_factory=list)
    truth_value: TruthValue = field(default_factory=lambda: TruthValue(1.0, 0.8))
    usage_count: int = 0
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "scheme_expression": self.scheme_expression,
            "grammar_id": self.grammar_id,
            "atom_ids": self.atom_ids,
            "truth_value": self.truth_value.to_dict(),
            "usage_count": self.usage_count,
            "created": self.created.isoformat(),
            "metadata": self.metadata
        }


class CognitiveAtomSpaceIntegration:
    """
    Integration layer between Scheme cognitive grammars and AtomSpace hypergraph
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scheme_registry = SchemeCognitiveGrammarRegistry(atomspace)
        self.pln_engine = PLNInferenceEngine(atomspace)
        self.pattern_matcher = HypergraphPatternMatcher(atomspace, self.pln_engine)
        
        # Cognitive pattern storage
        self.cognitive_patterns: Dict[str, CognitivePattern] = {}
        self.pattern_atom_map: Dict[str, str] = {}  # pattern_id -> atom_id
        
    async def store_cognitive_pattern(self, name: str, scheme_expression: str, 
                                    grammar_id: str = None, 
                                    metadata: Dict[str, Any] = None) -> CognitivePattern:
        """
        Store a cognitive pattern as atoms in the hypergraph
        
        Args:
            name: Name of the cognitive pattern
            scheme_expression: Scheme expression defining the pattern
            grammar_id: ID of the grammar used (optional)
            metadata: Additional metadata
            
        Returns:
            CognitivePattern: The stored pattern
        """
        try:
            pattern_id = str(uuid.uuid4())
            
            # Parse the Scheme expression using the grammar registry
            if grammar_id and grammar_id in self.scheme_registry.grammars:
                grammar = self.scheme_registry.grammars[grammar_id]
            else:
                # Create a temporary grammar for this pattern
                temp_grammar_id = f"temp_grammar_{pattern_id[:8]}"
                grammar = self.scheme_registry.register_grammar(
                    temp_grammar_id,
                    f"Temporary grammar for {name}",
                    f"Auto-generated grammar for pattern {name}",
                    scheme_expression,
                    [CognitiveOperator.COMPOSE]
                )
                grammar_id = temp_grammar_id
            
            # Create atoms to represent the cognitive pattern
            atom_ids = await self._create_pattern_atoms(pattern_id, name, scheme_expression, grammar_id)
            
            # Create cognitive pattern
            cognitive_pattern = CognitivePattern(
                id=pattern_id,
                name=name,
                scheme_expression=scheme_expression,
                grammar_id=grammar_id,
                atom_ids=atom_ids,
                metadata=metadata or {}
            )
            
            # Store pattern
            self.cognitive_patterns[pattern_id] = cognitive_pattern
            
            self.logger.info(f"Stored cognitive pattern: {name}")
            return cognitive_pattern
            
        except Exception as e:
            self.logger.error(f"Failed to store cognitive pattern {name}: {e}")
            raise
    
    async def _create_pattern_atoms(self, pattern_id: str, name: str, 
                                  scheme_expression: str, grammar_id: str) -> List[str]:
        """Create atoms to represent a cognitive pattern in the hypergraph"""
        atom_ids = []
        
        # Create main pattern node
        pattern_node = await self.atomspace.add_node(
            name=f"cognitive_pattern_{name}",
            concept_type="cognitive_pattern",
            truth_value=1.0,
            confidence=0.8,
            metadata={
                "pattern_id": pattern_id,
                "scheme_expression": scheme_expression,
                "grammar_id": grammar_id,
                "created": datetime.now(timezone.utc).isoformat()
            }
        )
        atom_ids.append(pattern_node.id)
        
        # Create grammar reference node
        grammar_node = await self.atomspace.add_node(
            name=f"grammar_{grammar_id}",
            concept_type="cognitive_grammar",
            truth_value=1.0,
            confidence=0.9,
            metadata={"grammar_id": grammar_id}
        )
        atom_ids.append(grammar_node.id)
        
        # Create link between pattern and grammar
        pattern_grammar_link = await self.atomspace.add_link(
            name=f"pattern_uses_grammar_{pattern_id}",
            outgoing=[pattern_node.id, grammar_node.id],
            link_type="uses_grammar",
            truth_value=1.0,
            confidence=0.9
        )
        atom_ids.append(pattern_grammar_link.id)
        
        # Parse Scheme expression and create atoms for components
        try:
            parsed_tree = self.scheme_registry.parser.parse(scheme_expression)
            component_atoms = await self._create_atoms_from_scheme_tree(parsed_tree, pattern_id)
            atom_ids.extend(component_atoms)
            
            # Link pattern to its components
            if component_atoms:
                pattern_components_link = await self.atomspace.add_link(
                    name=f"pattern_components_{pattern_id}",
                    outgoing=[pattern_node.id] + component_atoms,
                    link_type="has_components",
                    truth_value=1.0,
                    confidence=0.8
                )
                atom_ids.append(pattern_components_link.id)
                
        except Exception as e:
            self.logger.warning(f"Could not parse Scheme expression for pattern {name}: {e}")
        
        return atom_ids
    
    async def _create_atoms_from_scheme_tree(self, scheme_node, pattern_id: str) -> List[str]:
        """Create atoms from a parsed Scheme tree"""
        atom_ids = []
        
        # Create node for this Scheme element
        node_name = f"scheme_{scheme_node.node_type.value}_{pattern_id}_{len(atom_ids)}"
        
        node = await self.atomspace.add_node(
            name=node_name,
            concept_type="scheme_element",
            truth_value=1.0,
            confidence=0.7,
            metadata={
                "scheme_type": scheme_node.node_type.value,
                "scheme_value": str(scheme_node.value),
                "pattern_id": pattern_id
            }
        )
        atom_ids.append(node.id)
        
        # Recursively create atoms for children
        child_atom_ids = []
        for child in scheme_node.children:
            child_atoms = await self._create_atoms_from_scheme_tree(child, pattern_id)
            child_atom_ids.extend(child_atoms)
            atom_ids.extend(child_atoms)
        
        # Create links to children if any
        if child_atom_ids:
            child_link = await self.atomspace.add_link(
                name=f"scheme_children_{node.id}",
                outgoing=[node.id] + child_atom_ids,
                link_type="scheme_children",
                truth_value=1.0,
                confidence=0.7
            )
            atom_ids.append(child_link.id)
        
        return atom_ids
    
    async def evaluate_cognitive_pattern(self, pattern_id: str, 
                                       bindings: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate a cognitive pattern with variable bindings
        
        Args:
            pattern_id: ID of the pattern to evaluate
            bindings: Variable bindings for evaluation
            
        Returns:
            Evaluation result
        """
        if pattern_id not in self.cognitive_patterns:
            return {"error": "Pattern not found"}
        
        pattern = self.cognitive_patterns[pattern_id]
        
        try:
            # Evaluate using the grammar registry
            result = self.scheme_registry.evaluate_grammar_expression(
                pattern.grammar_id, 
                bindings or {}
            )
            
            # Update pattern usage
            pattern.usage_count += 1
            
            # Enhance result with AtomSpace context
            if result.get("success"):
                atomspace_context = await self._get_pattern_atomspace_context(pattern)
                result["atomspace_context"] = atomspace_context
            
            return result
            
        except Exception as e:
            return {"error": f"Evaluation failed: {e}"}
    
    async def _get_pattern_atomspace_context(self, pattern: CognitivePattern) -> Dict[str, Any]:
        """Get AtomSpace context for a pattern"""
        context = {
            "pattern_atoms": len(pattern.atom_ids),
            "related_patterns": [],
            "inference_paths": []
        }
        
        # Find related patterns through the hypergraph
        for atom_id in pattern.atom_ids:
            # Get neighbors of this atom
            atom = await self.atomspace.get_atom(atom_id)
            if atom:
                neighbors = await self.pattern_matcher._get_neighbors(atom)
                for neighbor in neighbors:
                    if neighbor.id not in pattern.atom_ids:
                        # Check if this neighbor belongs to another pattern
                        for other_pattern_id, other_pattern in self.cognitive_patterns.items():
                            if (other_pattern_id != pattern.id and 
                                neighbor.id in other_pattern.atom_ids):
                                context["related_patterns"].append({
                                    "pattern_id": other_pattern_id,
                                    "pattern_name": other_pattern.name,
                                    "connection_atom": neighbor.id
                                })
        
        return context
    
    async def find_cognitive_patterns(self, query: str, 
                                    max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find cognitive patterns matching a query
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching patterns with scores
        """
        results = []
        
        # Search by pattern name and Scheme expression
        for pattern in self.cognitive_patterns.values():
            score = 0.0
            
            # Name similarity
            if query.lower() in pattern.name.lower():
                score += 0.5
            
            # Scheme expression similarity  
            if query.lower() in pattern.scheme_expression.lower():
                score += 0.3
            
            # Metadata similarity
            for key, value in pattern.metadata.items():
                if query.lower() in str(value).lower():
                    score += 0.2
            
            if score > 0:
                results.append({
                    "pattern": pattern.to_dict(),
                    "score": score,
                    "match_type": "text_similarity"
                })
        
        # Also search the hypergraph for semantic matches
        try:
            # Create a pattern for hypergraph search
            search_pattern = Pattern(
                id=f"search_{uuid.uuid4().hex[:8]}",
                pattern_type=MatchType.SEMANTIC,
                template={"concept_type": "cognitive_pattern"},
                weights={"truth_value": 0.8}
            )
            
            await self.pattern_matcher.register_pattern(search_pattern)
            matches = await self.pattern_matcher.match_pattern(search_pattern.id, max_matches=max_results)
            
            for match in matches:
                # Find the corresponding cognitive pattern
                for atom_id in match.matched_atoms:
                    for pattern in self.cognitive_patterns.values():
                        if atom_id in pattern.atom_ids:
                            results.append({
                                "pattern": pattern.to_dict(),
                                "score": match.confidence,
                                "match_type": "semantic_hypergraph"
                            })
                            break
                            
        except Exception as e:
            self.logger.warning(f"Hypergraph search failed: {e}")
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
    
    async def reason_with_patterns(self, pattern_ids: List[str], 
                                 reasoning_type: str = "forward_chaining") -> Dict[str, Any]:
        """
        Perform reasoning using cognitive patterns
        
        Args:
            pattern_ids: List of pattern IDs to use as premises
            reasoning_type: Type of reasoning to perform
            
        Returns:
            Reasoning results
        """
        try:
            # Get atoms for all patterns
            premise_atoms = []
            for pattern_id in pattern_ids:
                if pattern_id in self.cognitive_patterns:
                    pattern = self.cognitive_patterns[pattern_id]
                    premise_atoms.extend(pattern.atom_ids)
            
            if not premise_atoms:
                return {"error": "No valid patterns found"}
            
            # Perform PLN reasoning
            if reasoning_type == "forward_chaining":
                result = await self.pln_engine.forward_chaining(premise_atoms, max_iterations=5)
            else:
                return {"error": f"Unsupported reasoning type: {reasoning_type}"}
            
            # Enhance result with pattern context
            enhanced_result = dict(result)
            enhanced_result["pattern_context"] = {
                "input_patterns": [
                    {
                        "pattern_id": pid,
                        "pattern_name": self.cognitive_patterns[pid].name
                    }
                    for pid in pattern_ids if pid in self.cognitive_patterns
                ],
                "reasoning_type": reasoning_type
            }
            
            return enhanced_result
            
        except Exception as e:
            return {"error": f"Reasoning failed: {e}"}
    
    async def get_cognitive_statistics(self) -> Dict[str, Any]:
        """Get statistics about cognitive patterns and integration"""
        grammar_stats = self.scheme_registry.get_grammar_statistics()
        pln_stats = self.pln_engine.get_statistics()
        pattern_stats = self.pattern_matcher.get_statistics()
        
        return {
            "cognitive_patterns": len(self.cognitive_patterns),
            "total_pattern_atoms": sum(len(p.atom_ids) for p in self.cognitive_patterns.values()),
            "grammar_statistics": grammar_stats,
            "pln_statistics": pln_stats,
            "pattern_matching_statistics": pattern_stats,
            "most_used_patterns": [
                {
                    "pattern_id": p.id,
                    "pattern_name": p.name,
                    "usage_count": p.usage_count
                }
                for p in sorted(self.cognitive_patterns.values(), 
                              key=lambda x: x.usage_count, reverse=True)[:5]
            ]
        }
    
    async def export_cognitive_knowledge(self) -> Dict[str, Any]:
        """Export all cognitive knowledge to dictionary format"""
        return {
            "cognitive_patterns": {
                pid: pattern.to_dict() 
                for pid, pattern in self.cognitive_patterns.items()
            },
            "grammars": self.scheme_registry.export_grammars(),
            "statistics": await self.get_cognitive_statistics(),
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
    
    async def import_cognitive_knowledge(self, data: Dict[str, Any]) -> bool:
        """Import cognitive knowledge from dictionary format"""
        try:
            # Import grammars first
            if "grammars" in data:
                self.scheme_registry.import_grammars(data["grammars"])
            
            # Import cognitive patterns
            if "cognitive_patterns" in data:
                for pattern_data in data["cognitive_patterns"].values():
                    pattern = CognitivePattern(
                        id=pattern_data["id"],
                        name=pattern_data["name"],
                        scheme_expression=pattern_data["scheme_expression"],
                        grammar_id=pattern_data["grammar_id"],
                        atom_ids=pattern_data["atom_ids"],
                        truth_value=TruthValue.from_dict(pattern_data["truth_value"]),
                        usage_count=pattern_data.get("usage_count", 0),
                        metadata=pattern_data.get("metadata", {})
                    )
                    self.cognitive_patterns[pattern.id] = pattern
            
            self.logger.info(f"Imported {len(data.get('cognitive_patterns', {}))} cognitive patterns")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import cognitive knowledge: {e}")
            return False


class EnhancedMemoryAtomSpaceWrapper(MemoryAtomSpaceWrapper):
    """
    Enhanced memory wrapper with cognitive integration
    """
    
    def __init__(self, agent_id: str = "default", atomspace_subdir: str = "atomspace"):
        super().__init__(agent_id, atomspace_subdir)
        self.cognitive_integration = CognitiveAtomSpaceIntegration(self.atomspace)
    
    async def store_cognitive_memory(self, content: str, scheme_pattern: str = None,
                                   context_concepts: List[str] = None,
                                   memory_type: str = "experience") -> Dict[str, Any]:
        """Store memory using cognitive patterns"""
        # Store as regular memory first
        memory_result = await self.store_knowledge_with_relations(
            content=content,
            concept_name=f"{memory_type}_{hash(content) % 10000}",
            related_concepts=context_concepts or [],
            metadata={"type": memory_type, "agent_id": self.agent_id}
        )
        
        # If a Scheme pattern is provided, store as cognitive pattern
        if scheme_pattern:
            try:
                cognitive_pattern = await self.cognitive_integration.store_cognitive_pattern(
                    name=f"memory_pattern_{memory_result['node_id'][:8]}",
                    scheme_expression=scheme_pattern,
                    metadata={
                        "content": content,
                        "memory_node_id": memory_result["node_id"],
                        "agent_id": self.agent_id
                    }
                )
                memory_result["cognitive_pattern_id"] = cognitive_pattern.id
            except Exception as e:
                self.logger.warning(f"Failed to create cognitive pattern: {e}")
        
        return memory_result
    
    async def cognitive_recall(self, query: str, use_scheme_reasoning: bool = True) -> Dict[str, Any]:
        """Recall memories using cognitive reasoning"""
        # Regular hypergraph recall
        regular_results = await self.query_knowledge_hybrid(query, use_relations=True)
        
        # Cognitive pattern search if requested
        cognitive_results = []
        if use_scheme_reasoning:
            try:
                cognitive_results = await self.cognitive_integration.find_cognitive_patterns(query)
            except Exception as e:
                self.logger.warning(f"Cognitive search failed: {e}")
        
        return {
            "query": query,
            "regular_results": regular_results,
            "cognitive_results": cognitive_results,
            "cognitive_integration_available": True
        }