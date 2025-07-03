"""
Hypergraph Pattern Matcher for Neural-Symbolic Reasoning

Implements sophisticated pattern matching and semantic relation extraction
for the AtomSpace hypergraph with support for complex graph traversal.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid
from collections import defaultdict, deque

from .atomspace import AtomSpace, Node, Link, Atom, AtomType
from .pln_reasoning import PLNInferenceEngine, TruthValue


class MatchType(Enum):
    """Types of pattern matching"""
    EXACT = "exact"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    FUZZY = "fuzzy"
    TEMPORAL = "temporal"


class TraversalStrategy(Enum):
    """Graph traversal strategies"""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    BEST_FIRST = "best_first"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class Pattern:
    """Represents a pattern to match in the hypergraph"""
    id: str
    pattern_type: MatchType
    template: Dict[str, Any]
    constraints: Dict[str, Any] = field(default_factory=dict)
    variables: Set[str] = field(default_factory=set)
    weights: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "template": self.template,
            "constraints": self.constraints,
            "variables": list(self.variables),
            "weights": self.weights,
            "metadata": self.metadata
        }


@dataclass
class Match:
    """Represents a pattern match result"""
    pattern_id: str
    match_id: str
    matched_atoms: List[str]
    variable_bindings: Dict[str, str]
    confidence: float
    strength: float
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "pattern_id": self.pattern_id,
            "match_id": self.match_id,
            "matched_atoms": self.matched_atoms,
            "variable_bindings": self.variable_bindings,
            "confidence": self.confidence,
            "strength": self.strength,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


class HypergraphPatternMatcher:
    """
    Advanced pattern matcher for hypergraph traversal and semantic relation extraction
    """
    
    def __init__(self, atomspace: AtomSpace, pln_engine: PLNInferenceEngine):
        self.atomspace = atomspace
        self.pln_engine = pln_engine
        self.patterns: Dict[str, Pattern] = {}
        self.match_cache: Dict[str, List[Match]] = {}
        self.traversal_cache: Dict[str, List[str]] = {}
        self.similarity_threshold = 0.7
        self.max_traversal_depth = 5
        
    async def register_pattern(self, pattern: Pattern) -> bool:
        """Register a pattern for matching"""
        try:
            self.patterns[pattern.id] = pattern
            return True
        except Exception as e:
            print(f"Error registering pattern {pattern.id}: {e}")
            return False
    
    async def match_pattern(self, pattern_id: str, 
                          match_type: Optional[MatchType] = None,
                          max_matches: int = 100) -> List[Match]:
        """
        Match a pattern in the hypergraph
        
        Args:
            pattern_id: ID of the pattern to match
            match_type: Optional override for match type
            max_matches: Maximum number of matches to return
            
        Returns:
            List of matches
        """
        if pattern_id not in self.patterns:
            return []
        
        pattern = self.patterns[pattern_id]
        effective_match_type = match_type or pattern.pattern_type
        
        # Check cache first
        cache_key = f"{pattern_id}_{effective_match_type.value}_{max_matches}"
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]
        
        matches = []
        
        if effective_match_type == MatchType.EXACT:
            matches = await self._exact_match(pattern, max_matches)
        elif effective_match_type == MatchType.STRUCTURAL:
            matches = await self._structural_match(pattern, max_matches)
        elif effective_match_type == MatchType.SEMANTIC:
            matches = await self._semantic_match(pattern, max_matches)
        elif effective_match_type == MatchType.FUZZY:
            matches = await self._fuzzy_match(pattern, max_matches)
        elif effective_match_type == MatchType.TEMPORAL:
            matches = await self._temporal_match(pattern, max_matches)
        
        # Cache results
        self.match_cache[cache_key] = matches
        return matches
    
    async def _exact_match(self, pattern: Pattern, max_matches: int) -> List[Match]:
        """Exact pattern matching"""
        matches = []
        template = pattern.template
        
        # Find atoms that match the template exactly
        search_pattern = {}
        for key, value in template.items():
            if not isinstance(value, str) or not value.startswith("$"):
                search_pattern[key] = value
        
        candidate_atoms = await self.atomspace.storage.get_atoms_by_pattern(search_pattern, limit=max_matches * 2)
        
        for atom in candidate_atoms:
            if await self._atom_matches_template(atom, template):
                match = Match(
                    pattern_id=pattern.id,
                    match_id=str(uuid.uuid4()),
                    matched_atoms=[atom.id],
                    variable_bindings=await self._extract_variable_bindings(atom, template),
                    confidence=1.0,
                    strength=1.0,
                    context={"match_type": "exact"}
                )
                matches.append(match)
                
                if len(matches) >= max_matches:
                    break
        
        return matches
    
    async def _structural_match(self, pattern: Pattern, max_matches: int) -> List[Match]:
        """Structural pattern matching based on graph structure"""
        matches = []
        template = pattern.template
        
        # Find starting nodes
        start_nodes = await self._find_starting_nodes(template)
        
        for start_node in start_nodes:
            # Traverse from this node following the pattern structure
            traversal_matches = await self._traverse_structural_pattern(start_node, pattern, max_matches)
            matches.extend(traversal_matches)
            
            if len(matches) >= max_matches:
                break
        
        return matches[:max_matches]
    
    async def _semantic_match(self, pattern: Pattern, max_matches: int) -> List[Match]:
        """Semantic pattern matching using PLN truth values"""
        matches = []
        template = pattern.template
        
        # Get all candidate atoms
        all_atoms = await self._get_candidate_atoms(template)
        
        # Score each atom for semantic similarity
        scored_atoms = []
        for atom in all_atoms:
            semantic_score = await self._compute_semantic_similarity(atom, pattern)
            if semantic_score >= self.similarity_threshold:
                scored_atoms.append((atom, semantic_score))
        
        # Sort by semantic score
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        
        # Create matches
        for atom, score in scored_atoms[:max_matches]:
            match = Match(
                pattern_id=pattern.id,
                match_id=str(uuid.uuid4()),
                matched_atoms=[atom.id],
                variable_bindings=await self._extract_variable_bindings(atom, template),
                confidence=score,
                strength=score,
                context={"match_type": "semantic", "semantic_score": score}
            )
            matches.append(match)
        
        return matches
    
    async def _fuzzy_match(self, pattern: Pattern, max_matches: int) -> List[Match]:
        """Fuzzy pattern matching with tolerance"""
        matches = []
        template = pattern.template
        fuzzy_threshold = pattern.constraints.get("fuzzy_threshold", 0.6)
        
        # Get candidate atoms
        candidates = await self._get_candidate_atoms(template)
        
        for atom in candidates:
            fuzzy_score = await self._compute_fuzzy_similarity(atom, pattern)
            if fuzzy_score >= fuzzy_threshold:
                match = Match(
                    pattern_id=pattern.id,
                    match_id=str(uuid.uuid4()),
                    matched_atoms=[atom.id],
                    variable_bindings=await self._extract_variable_bindings(atom, template),
                    confidence=fuzzy_score,
                    strength=fuzzy_score,
                    context={"match_type": "fuzzy", "fuzzy_score": fuzzy_score}
                )
                matches.append(match)
                
                if len(matches) >= max_matches:
                    break
        
        return matches
    
    async def _temporal_match(self, pattern: Pattern, max_matches: int) -> List[Match]:
        """Temporal pattern matching based on time constraints"""
        matches = []
        template = pattern.template
        time_window = pattern.constraints.get("time_window", 3600)  # 1 hour default
        
        # Get atoms within time window
        now = datetime.now(timezone.utc)
        time_constraint = (now - timezone.utc).total_seconds() - time_window
        
        search_pattern = dict(template)
        search_pattern["timestamp_after"] = time_constraint
        
        candidate_atoms = await self.atomspace.storage.get_atoms_by_pattern(search_pattern, limit=max_matches * 2)
        
        for atom in candidate_atoms:
            if await self._atom_matches_template(atom, template):
                match = Match(
                    pattern_id=pattern.id,
                    match_id=str(uuid.uuid4()),
                    matched_atoms=[atom.id],
                    variable_bindings=await self._extract_variable_bindings(atom, template),
                    confidence=1.0,
                    strength=1.0,
                    context={"match_type": "temporal", "time_window": time_window}
                )
                matches.append(match)
                
                if len(matches) >= max_matches:
                    break
        
        return matches
    
    async def _atom_matches_template(self, atom: Atom, template: Dict[str, Any]) -> bool:
        """Check if an atom matches a template"""
        for key, value in template.items():
            if key == "variables":
                continue
                
            if isinstance(value, str) and value.startswith("$"):
                # Variable - always matches
                continue
            
            atom_value = getattr(atom, key, None)
            if atom_value != value:
                return False
        
        return True
    
    async def _extract_variable_bindings(self, atom: Atom, template: Dict[str, Any]) -> Dict[str, str]:
        """Extract variable bindings from an atom match"""
        bindings = {}
        
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("$"):
                variable_name = value[1:]  # Remove $ prefix
                atom_value = getattr(atom, key, None)
                if atom_value is not None:
                    bindings[variable_name] = str(atom_value)
        
        return bindings
    
    async def _find_starting_nodes(self, template: Dict[str, Any]) -> List[Atom]:
        """Find starting nodes for structural traversal"""
        # Look for specific node patterns in template
        node_patterns = {}
        for key, value in template.items():
            if key.startswith("node_") and not (isinstance(value, str) and value.startswith("$")):
                node_patterns[key.replace("node_", "")] = value
        
        if node_patterns:
            return await self.atomspace.storage.get_atoms_by_pattern(node_patterns, limit=100)
        
        # Default: get sample of nodes
        return await self.atomspace.storage.get_atoms_by_pattern({"atom_type": AtomType.NODE.value}, limit=50)
    
    async def _traverse_structural_pattern(self, start_atom: Atom, pattern: Pattern, max_matches: int) -> List[Match]:
        """Traverse structural pattern from starting atom"""
        matches = []
        template = pattern.template
        
        # Use BFS to explore the structure
        queue = deque([(start_atom, [start_atom.id], {})])
        visited = set()
        
        while queue and len(matches) < max_matches:
            current_atom, path, bindings = queue.popleft()
            
            if current_atom.id in visited:
                continue
            visited.add(current_atom.id)
            
            # Check if current path satisfies pattern
            if await self._path_matches_pattern(path, pattern):
                match = Match(
                    pattern_id=pattern.id,
                    match_id=str(uuid.uuid4()),
                    matched_atoms=path,
                    variable_bindings=bindings,
                    confidence=0.8,
                    strength=0.8,
                    context={"match_type": "structural", "path_length": len(path)}
                )
                matches.append(match)
            
            # Explore neighbors
            if len(path) < self.max_traversal_depth:
                neighbors = await self._get_neighbors(current_atom)
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        new_path = path + [neighbor.id]
                        new_bindings = dict(bindings)
                        queue.append((neighbor, new_path, new_bindings))
        
        return matches
    
    async def _path_matches_pattern(self, path: List[str], pattern: Pattern) -> bool:
        """Check if a path matches the structural pattern"""
        template = pattern.template
        
        # Simple check: path length constraints
        min_length = template.get("min_path_length", 1)
        max_length = template.get("max_path_length", 10)
        
        if not (min_length <= len(path) <= max_length):
            return False
        
        # Check structural constraints
        if "required_types" in template:
            required_types = template["required_types"]
            path_types = []
            for atom_id in path:
                atom = await self.atomspace.get_atom(atom_id)
                if atom:
                    path_types.append(atom.atom_type.value)
            
            # Check if all required types are present
            for req_type in required_types:
                if req_type not in path_types:
                    return False
        
        return True
    
    async def _get_neighbors(self, atom: Atom) -> List[Atom]:
        """Get neighboring atoms in the hypergraph"""
        neighbors = []
        
        if atom.atom_type == AtomType.NODE:
            # Find links containing this node
            pattern = {"atom_type": AtomType.LINK.value}
            links = await self.atomspace.storage.get_atoms_by_pattern(pattern, limit=200)
            
            for link in links:
                if hasattr(link, 'outgoing') and link.outgoing and atom.id in link.outgoing:
                    neighbors.append(link)
                    # Also add other atoms in the link
                    for other_atom_id in link.outgoing:
                        if other_atom_id != atom.id:
                            other_atom = await self.atomspace.get_atom(other_atom_id)
                            if other_atom:
                                neighbors.append(other_atom)
        
        elif atom.atom_type == AtomType.LINK:
            # Add all outgoing atoms
            if hasattr(atom, 'outgoing') and atom.outgoing:
                for atom_id in atom.outgoing:
                    neighbor = await self.atomspace.get_atom(atom_id)
                    if neighbor:
                        neighbors.append(neighbor)
        
        return neighbors
    
    async def _get_candidate_atoms(self, template: Dict[str, Any]) -> List[Atom]:
        """Get candidate atoms for pattern matching"""
        # Build search pattern from non-variable parts of template
        search_pattern = {}
        for key, value in template.items():
            if not isinstance(value, str) or not value.startswith("$"):
                search_pattern[key] = value
        
        if search_pattern:
            return await self.atomspace.storage.get_atoms_by_pattern(search_pattern, limit=500)
        
        # Default: get sample of all atoms
        nodes = await self.atomspace.storage.get_atoms_by_pattern({"atom_type": AtomType.NODE.value}, limit=250)
        links = await self.atomspace.storage.get_atoms_by_pattern({"atom_type": AtomType.LINK.value}, limit=250)
        
        return nodes + links
    
    async def _compute_semantic_similarity(self, atom: Atom, pattern: Pattern) -> float:
        """Compute semantic similarity between atom and pattern"""
        try:
            # Get truth value for semantic assessment
            truth_value = await self.pln_engine.infer_truth_value(atom.id)
            
            # Base similarity on truth value and pattern weights
            base_similarity = truth_value.strength * truth_value.confidence
            
            # Apply pattern-specific weights
            if pattern.weights:
                weighted_similarity = 0.0
                total_weight = 0.0
                
                for key, weight in pattern.weights.items():
                    if hasattr(atom, key):
                        attr_value = getattr(atom, key)
                        if isinstance(attr_value, (int, float)):
                            weighted_similarity += attr_value * weight
                            total_weight += weight
                
                if total_weight > 0:
                    base_similarity = (base_similarity + weighted_similarity / total_weight) / 2
            
            return base_similarity
            
        except Exception:
            return 0.1
    
    async def _compute_fuzzy_similarity(self, atom: Atom, pattern: Pattern) -> float:
        """Compute fuzzy similarity between atom and pattern"""
        similarity_score = 0.0
        total_criteria = 0
        
        template = pattern.template
        
        for key, expected_value in template.items():
            if isinstance(expected_value, str) and expected_value.startswith("$"):
                continue  # Skip variables
            
            total_criteria += 1
            atom_value = getattr(atom, key, None)
            
            if atom_value == expected_value:
                similarity_score += 1.0
            elif isinstance(atom_value, str) and isinstance(expected_value, str):
                # String similarity (simple)
                string_similarity = self._string_similarity(atom_value, expected_value)
                similarity_score += string_similarity
            elif isinstance(atom_value, (int, float)) and isinstance(expected_value, (int, float)):
                # Numeric similarity
                max_val = max(abs(atom_value), abs(expected_value), 1.0)
                numeric_similarity = 1.0 - abs(atom_value - expected_value) / max_val
                similarity_score += max(0.0, numeric_similarity)
        
        return similarity_score / total_criteria if total_criteria > 0 else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity using character overlap"""
        if not s1 or not s2:
            return 0.0
        
        set1 = set(s1.lower())
        set2 = set(s2.lower())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    async def traverse_hypergraph(self, start_atom_id: str, 
                                strategy: TraversalStrategy = TraversalStrategy.BREADTH_FIRST,
                                max_depth: int = 5,
                                max_nodes: int = 100) -> List[str]:
        """
        Traverse the hypergraph starting from a given atom
        
        Args:
            start_atom_id: Starting atom ID
            strategy: Traversal strategy
            max_depth: Maximum traversal depth
            max_nodes: Maximum nodes to visit
            
        Returns:
            List of atom IDs in traversal order
        """
        cache_key = f"{start_atom_id}_{strategy.value}_{max_depth}_{max_nodes}"
        if cache_key in self.traversal_cache:
            return self.traversal_cache[cache_key]
        
        if strategy == TraversalStrategy.BREADTH_FIRST:
            traversal = await self._bfs_traverse(start_atom_id, max_depth, max_nodes)
        elif strategy == TraversalStrategy.DEPTH_FIRST:
            traversal = await self._dfs_traverse(start_atom_id, max_depth, max_nodes)
        elif strategy == TraversalStrategy.BEST_FIRST:
            traversal = await self._best_first_traverse(start_atom_id, max_depth, max_nodes)
        else:
            traversal = await self._bfs_traverse(start_atom_id, max_depth, max_nodes)
        
        self.traversal_cache[cache_key] = traversal
        return traversal
    
    async def _bfs_traverse(self, start_atom_id: str, max_depth: int, max_nodes: int) -> List[str]:
        """Breadth-first traversal"""
        visited = set()
        queue = deque([(start_atom_id, 0)])
        traversal_order = []
        
        while queue and len(traversal_order) < max_nodes:
            atom_id, depth = queue.popleft()
            
            if atom_id in visited or depth > max_depth:
                continue
            
            visited.add(atom_id)
            traversal_order.append(atom_id)
            
            # Add neighbors
            atom = await self.atomspace.get_atom(atom_id)
            if atom:
                neighbors = await self._get_neighbors(atom)
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, depth + 1))
        
        return traversal_order
    
    async def _dfs_traverse(self, start_atom_id: str, max_depth: int, max_nodes: int) -> List[str]:
        """Depth-first traversal"""
        visited = set()
        stack = [(start_atom_id, 0)]
        traversal_order = []
        
        while stack and len(traversal_order) < max_nodes:
            atom_id, depth = stack.pop()
            
            if atom_id in visited or depth > max_depth:
                continue
            
            visited.add(atom_id)
            traversal_order.append(atom_id)
            
            # Add neighbors
            atom = await self.atomspace.get_atom(atom_id)
            if atom:
                neighbors = await self._get_neighbors(atom)
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        stack.append((neighbor.id, depth + 1))
        
        return traversal_order
    
    async def _best_first_traverse(self, start_atom_id: str, max_depth: int, max_nodes: int) -> List[str]:
        """Best-first traversal using truth values as heuristic"""
        visited = set()
        # Priority queue: (priority, atom_id, depth)
        import heapq
        pq = [(0.0, start_atom_id, 0)]
        traversal_order = []
        
        while pq and len(traversal_order) < max_nodes:
            neg_priority, atom_id, depth = heapq.heappop(pq)
            
            if atom_id in visited or depth > max_depth:
                continue
            
            visited.add(atom_id)
            traversal_order.append(atom_id)
            
            # Add neighbors with priorities
            atom = await self.atomspace.get_atom(atom_id)
            if atom:
                neighbors = await self._get_neighbors(atom)
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        try:
                            tv = await self.pln_engine.infer_truth_value(neighbor.id)
                            priority = tv.strength * tv.confidence
                            heapq.heappush(pq, (-priority, neighbor.id, depth + 1))
                        except Exception:
                            heapq.heappush(pq, (-0.5, neighbor.id, depth + 1))
        
        return traversal_order
    
    async def extract_semantic_relations(self, atom_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract semantic relations between atoms
        
        Args:
            atom_ids: List of atom IDs to analyze
            
        Returns:
            Dictionary of relation types and their instances
        """
        relations = defaultdict(list)
        
        for i, atom_id1 in enumerate(atom_ids):
            for j, atom_id2 in enumerate(atom_ids):
                if i != j:
                    relation = await self._analyze_relation(atom_id1, atom_id2)
                    if relation:
                        relations[relation["type"]].append(relation)
        
        return dict(relations)
    
    async def _analyze_relation(self, atom_id1: str, atom_id2: str) -> Optional[Dict[str, Any]]:
        """Analyze relation between two atoms"""
        atom1 = await self.atomspace.get_atom(atom_id1)
        atom2 = await self.atomspace.get_atom(atom_id2)
        
        if not atom1 or not atom2:
            return None
        
        # Find connecting links
        connecting_links = await self._find_connecting_links(atom_id1, atom_id2)
        
        if connecting_links:
            # Direct connection
            link_types = [getattr(link, 'link_type', 'unknown') for link in connecting_links]
            return {
                "type": "direct",
                "source": atom_id1,
                "target": atom_id2,
                "link_types": link_types,
                "strength": 1.0
            }
        
        # Check for indirect connection (up to 2 hops)
        path = await self._find_shortest_path(atom_id1, atom_id2, max_hops=2)
        if path and len(path) <= 3:
            return {
                "type": "indirect",
                "source": atom_id1,
                "target": atom_id2,
                "path": path,
                "strength": 0.5
            }
        
        # Check semantic similarity
        try:
            tv1 = await self.pln_engine.infer_truth_value(atom_id1)
            tv2 = await self.pln_engine.infer_truth_value(atom_id2)
            
            similarity = abs(tv1.strength - tv2.strength) + abs(tv1.confidence - tv2.confidence)
            if similarity < 0.3:  # Similar truth values
                return {
                    "type": "semantic_similarity",
                    "source": atom_id1,
                    "target": atom_id2,
                    "similarity": 1.0 - similarity,
                    "strength": 0.3
                }
        except Exception:
            pass
        
        return None
    
    async def _find_connecting_links(self, atom_id1: str, atom_id2: str) -> List[Link]:
        """Find links that connect two atoms"""
        links = []
        
        # Get all links
        pattern = {"atom_type": AtomType.LINK.value}
        all_links = await self.atomspace.storage.get_atoms_by_pattern(pattern, limit=1000)
        
        for link in all_links:
            if (hasattr(link, 'outgoing') and link.outgoing and 
                atom_id1 in link.outgoing and atom_id2 in link.outgoing):
                links.append(link)
        
        return links
    
    async def _find_shortest_path(self, start_id: str, end_id: str, max_hops: int) -> Optional[List[str]]:
        """Find shortest path between two atoms"""
        if start_id == end_id:
            return [start_id]
        
        visited = set()
        queue = deque([(start_id, [start_id])])
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_hops + 1:
                continue
            
            if current_id in visited:
                continue
            visited.add(current_id)
            
            # Get neighbors
            current_atom = await self.atomspace.get_atom(current_id)
            if current_atom:
                neighbors = await self._get_neighbors(current_atom)
                
                for neighbor in neighbors:
                    if neighbor.id == end_id:
                        return path + [neighbor.id]
                    
                    if neighbor.id not in visited:
                        queue.append((neighbor.id, path + [neighbor.id]))
        
        return None
    
    def clear_cache(self):
        """Clear all caches"""
        self.match_cache.clear()
        self.traversal_cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pattern matcher statistics"""
        return {
            "registered_patterns": len(self.patterns),
            "cached_matches": len(self.match_cache),
            "cached_traversals": len(self.traversal_cache),
            "similarity_threshold": self.similarity_threshold,
            "max_traversal_depth": self.max_traversal_depth
        }