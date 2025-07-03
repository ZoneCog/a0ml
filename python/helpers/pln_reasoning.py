"""
Probabilistic Logic Networks (PLN) for Neural-Symbolic Reasoning

Implements PLN inference engine for uncertain reasoning over the hypergraph
AtomSpace with truth values, confidence measures, and logical inference.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid

from .atomspace import AtomSpace, Node, Link, Atom, AtomType


class LogicalOperator(Enum):
    """Logical operators for PLN inference"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    EQUIVALENT = "equivalent"


@dataclass
class TruthValue:
    """PLN Truth Value with strength and confidence"""
    strength: float = 1.0  # [0, 1] - probability/certainty
    confidence: float = 1.0  # [0, 1] - amount of evidence
    
    def __post_init__(self):
        """Ensure valid truth value ranges"""
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {"strength": self.strength, "confidence": self.confidence}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'TruthValue':
        """Create from dictionary"""
        return cls(strength=data.get("strength", 1.0), confidence=data.get("confidence", 1.0))


class PLNInferenceEngine:
    """
    Probabilistic Logic Networks inference engine for neural-symbolic reasoning
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.inference_history: List[Dict[str, Any]] = []
        self.truth_cache: Dict[str, TruthValue] = {}
        
    async def infer_truth_value(self, atom_id: str, visited: Optional[Set[str]] = None) -> TruthValue:
        """
        Infer truth value of an atom using PLN rules
        
        Args:
            atom_id: ID of the atom to evaluate
            visited: Set of visited atoms to prevent cycles
            
        Returns:
            TruthValue: Computed truth value
        """
        if visited is None:
            visited = set()
            
        if atom_id in visited:
            return TruthValue(0.5, 0.1)  # Default for cycles
            
        if atom_id in self.truth_cache:
            return self.truth_cache[atom_id]
            
        visited.add(atom_id)
        
        atom = await self.atomspace.get_atom(atom_id)
        if not atom:
            return TruthValue(0.0, 0.0)
        
        # Base case: atom has explicit truth value
        if hasattr(atom, 'truth_value') and hasattr(atom, 'confidence'):
            truth_value = TruthValue(atom.truth_value, atom.confidence)
            self.truth_cache[atom_id] = truth_value
            return truth_value
        
        # Inference case: compute from structure
        if atom.atom_type == AtomType.LINK:
            truth_value = await self._infer_link_truth_value(atom, visited)
        else:
            truth_value = await self._infer_node_truth_value(atom, visited)
        
        self.truth_cache[atom_id] = truth_value
        return truth_value
    
    async def _infer_link_truth_value(self, link: Link, visited: Set[str]) -> TruthValue:
        """Infer truth value for a link based on its type and outgoing atoms"""
        if not hasattr(link, 'outgoing') or not link.outgoing:
            return TruthValue(0.5, 0.1)
        
        outgoing_atoms = link.outgoing
        link_type = getattr(link, 'link_type', 'unknown')
        
        # Get truth values of outgoing atoms
        outgoing_truth_values = []
        for atom_id in outgoing_atoms:
            tv = await self.infer_truth_value(atom_id, visited.copy())
            outgoing_truth_values.append(tv)
        
        # Apply PLN inference rules based on link type
        if link_type == LogicalOperator.AND.value:
            return self._pln_and(outgoing_truth_values)
        elif link_type == LogicalOperator.OR.value:
            return self._pln_or(outgoing_truth_values)
        elif link_type == LogicalOperator.NOT.value:
            return self._pln_not(outgoing_truth_values[0]) if outgoing_truth_values else TruthValue(0.5, 0.1)
        elif link_type == LogicalOperator.IMPLIES.value:
            return self._pln_implies(outgoing_truth_values[0], outgoing_truth_values[1]) if len(outgoing_truth_values) >= 2 else TruthValue(0.5, 0.1)
        elif link_type == LogicalOperator.EQUIVALENT.value:
            return self._pln_equivalent(outgoing_truth_values[0], outgoing_truth_values[1]) if len(outgoing_truth_values) >= 2 else TruthValue(0.5, 0.1)
        else:
            # Default: combine with weighted average
            return self._pln_weighted_average(outgoing_truth_values)
    
    async def _infer_node_truth_value(self, node: Node, visited: Set[str]) -> TruthValue:
        """Infer truth value for a node based on incoming links"""
        # Find links that point to this node
        incoming_links = await self._find_incoming_links(node.id)
        
        if not incoming_links:
            # No incoming evidence, use default
            return TruthValue(0.5, 0.1)
        
        # Combine evidence from incoming links
        truth_values = []
        for link in incoming_links:
            tv = await self.infer_truth_value(link.id, visited.copy())
            truth_values.append(tv)
        
        return self._pln_weighted_average(truth_values)
    
    async def _find_incoming_links(self, node_id: str) -> List[Link]:
        """Find all links that have this node as an outgoing atom"""
        # Search for links that contain this node
        pattern = {"atom_type": AtomType.LINK.value}
        all_links = await self.atomspace.storage.get_atoms_by_pattern(pattern, limit=1000)
        
        incoming_links = []
        for atom in all_links:
            if isinstance(atom, Link) and hasattr(atom, 'outgoing') and atom.outgoing:
                if node_id in atom.outgoing:
                    incoming_links.append(atom)
        
        return incoming_links
    
    def _pln_and(self, truth_values: List[TruthValue]) -> TruthValue:
        """PLN AND operation: minimum strength, combined confidence"""
        if not truth_values:
            return TruthValue(0.5, 0.1)
        
        min_strength = min(tv.strength for tv in truth_values)
        combined_confidence = np.mean([tv.confidence for tv in truth_values])
        
        return TruthValue(min_strength, combined_confidence)
    
    def _pln_or(self, truth_values: List[TruthValue]) -> TruthValue:
        """PLN OR operation: maximum strength, combined confidence"""
        if not truth_values:
            return TruthValue(0.5, 0.1)
        
        max_strength = max(tv.strength for tv in truth_values)
        combined_confidence = np.mean([tv.confidence for tv in truth_values])
        
        return TruthValue(max_strength, combined_confidence)
    
    def _pln_not(self, truth_value: TruthValue) -> TruthValue:
        """PLN NOT operation: complement strength, same confidence"""
        return TruthValue(1.0 - truth_value.strength, truth_value.confidence)
    
    def _pln_implies(self, antecedent: TruthValue, consequent: TruthValue) -> TruthValue:
        """PLN IMPLIES operation: logical implication"""
        # P(B|A) = P(A→B) using PLN implication formula
        strength = 1.0 - antecedent.strength + (antecedent.strength * consequent.strength)
        confidence = min(antecedent.confidence, consequent.confidence)
        
        return TruthValue(strength, confidence)
    
    def _pln_equivalent(self, tv1: TruthValue, tv2: TruthValue) -> TruthValue:
        """PLN EQUIVALENT operation: bidirectional implication"""
        # P(A↔B) = P(A→B) ∧ P(B→A)
        impl1 = self._pln_implies(tv1, tv2)
        impl2 = self._pln_implies(tv2, tv1)
        
        return self._pln_and([impl1, impl2])
    
    def _pln_weighted_average(self, truth_values: List[TruthValue]) -> TruthValue:
        """Weighted average of truth values by confidence"""
        if not truth_values:
            return TruthValue(0.5, 0.1)
        
        total_weight = sum(tv.confidence for tv in truth_values)
        if total_weight == 0:
            return TruthValue(0.5, 0.1)
        
        weighted_strength = sum(tv.strength * tv.confidence for tv in truth_values) / total_weight
        avg_confidence = np.mean([tv.confidence for tv in truth_values])
        
        return TruthValue(weighted_strength, avg_confidence)
    
    async def forward_chaining(self, premises: List[str], max_iterations: int = 10) -> Dict[str, Any]:
        """
        Forward chaining inference from premises
        
        Args:
            premises: List of premise atom IDs
            max_iterations: Maximum inference iterations
            
        Returns:
            Dictionary with inference results
        """
        inference_session = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "premises": premises,
            "derived_facts": [],
            "iterations": 0
        }
        
        active_facts = set(premises)
        
        for iteration in range(max_iterations):
            new_facts = set()
            
            for fact_id in active_facts:
                # Apply inference rules to derive new facts
                derived = await self._apply_inference_rules(fact_id)
                new_facts.update(derived)
            
            # Add newly derived facts
            if new_facts - active_facts:
                inference_session["derived_facts"].extend(list(new_facts - active_facts))
                active_facts.update(new_facts)
            else:
                # No new facts derived, stop
                break
            
            inference_session["iterations"] = iteration + 1
        
        self.inference_history.append(inference_session)
        return inference_session
    
    async def _apply_inference_rules(self, fact_id: str) -> Set[str]:
        """Apply inference rules to derive new facts"""
        derived_facts = set()
        
        # Find all links where this fact appears
        incoming_links = await self._find_incoming_links(fact_id)
        
        for link in incoming_links:
            # Check if we can derive something from this link
            if hasattr(link, 'link_type') and link.link_type == LogicalOperator.IMPLIES.value:
                # Modus ponens: if we have A and A→B, derive B
                if hasattr(link, 'outgoing') and len(link.outgoing) == 2:
                    antecedent, consequent = link.outgoing
                    if antecedent == fact_id:
                        # We have A and A→B, so derive B
                        derived_facts.add(consequent)
        
        return derived_facts
    
    async def get_inference_explanation(self, atom_id: str) -> Dict[str, Any]:
        """
        Get explanation for how an atom's truth value was derived
        
        Args:
            atom_id: ID of the atom to explain
            
        Returns:
            Dictionary with explanation structure
        """
        explanation = {
            "atom_id": atom_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "explanation_tree": await self._build_explanation_tree(atom_id, set())
        }
        
        return explanation
    
    async def _build_explanation_tree(self, atom_id: str, visited: Set[str]) -> Dict[str, Any]:
        """Build recursive explanation tree for truth value derivation"""
        if atom_id in visited:
            return {"atom_id": atom_id, "type": "cycle", "truth_value": None}
        
        visited.add(atom_id)
        
        atom = await self.atomspace.get_atom(atom_id)
        if not atom:
            return {"atom_id": atom_id, "type": "missing", "truth_value": None}
        
        truth_value = await self.infer_truth_value(atom_id, visited.copy())
        
        explanation_node = {
            "atom_id": atom_id,
            "atom_name": atom.name,
            "atom_type": atom.atom_type.value,
            "truth_value": truth_value.to_dict(),
            "children": []
        }
        
        # Add children for links
        if atom.atom_type == AtomType.LINK and hasattr(atom, 'outgoing') and atom.outgoing:
            for child_id in atom.outgoing:
                child_explanation = await self._build_explanation_tree(child_id, visited.copy())
                explanation_node["children"].append(child_explanation)
        
        return explanation_node
    
    def clear_cache(self):
        """Clear truth value cache"""
        self.truth_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get PLN inference statistics"""
        return {
            "cached_truth_values": len(self.truth_cache),
            "inference_sessions": len(self.inference_history),
            "total_inferences": sum(len(session["derived_facts"]) for session in self.inference_history)
        }