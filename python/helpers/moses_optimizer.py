"""
MOSES (Meta-Optimizing Semantic Evolutionary Search) for Neural-Symbolic Reasoning

Implements evolutionary program optimization for learning and adapting
cognitive programs in the hypergraph AtomSpace.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid
import random
import copy

from .atomspace import AtomSpace, Node, Link, Atom, AtomType
from .pln_reasoning import PLNInferenceEngine, TruthValue


class ProgramType(Enum):
    """Types of programs that can be evolved"""
    INFERENCE_RULE = "inference_rule"
    PATTERN_MATCHER = "pattern_matcher"
    COGNITIVE_KERNEL = "cognitive_kernel"
    BEHAVIOR_TREE = "behavior_tree"


@dataclass
class Program:
    """Represents an evolvable program in the AtomSpace"""
    id: str
    program_type: ProgramType
    atoms: List[str]  # List of atom IDs that make up the program
    fitness: float = 0.0
    complexity: int = 0
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "program_type": self.program_type.value,
            "atoms": self.atoms,
            "fitness": self.fitness,
            "complexity": self.complexity,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Program':
        """Create from dictionary"""
        return cls(
            id=data["id"],
            program_type=ProgramType(data["program_type"]),
            atoms=data["atoms"],
            fitness=data.get("fitness", 0.0),
            complexity=data.get("complexity", 0),
            generation=data.get("generation", 0),
            parent_ids=data.get("parent_ids", []),
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc)
        )


class MOSESOptimizer:
    """
    Meta-Optimizing Semantic Evolutionary Search for program evolution
    """
    
    def __init__(self, atomspace: AtomSpace, pln_engine: PLNInferenceEngine):
        self.atomspace = atomspace
        self.pln_engine = pln_engine
        self.population: List[Program] = []
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.selection_pressure = 0.3
        self.max_complexity = 20
        self.generation_count = 0
        self.evolution_history: List[Dict[str, Any]] = []
        
    async def initialize_population(self, program_type: ProgramType, 
                                  seed_programs: Optional[List[Program]] = None) -> List[Program]:
        """
        Initialize population of programs
        
        Args:
            program_type: Type of programs to evolve
            seed_programs: Optional seed programs to start with
            
        Returns:
            List of initialized programs
        """
        if seed_programs:
            self.population = seed_programs[:self.population_size]
        else:
            self.population = []
        
        # Fill remaining population with random programs
        while len(self.population) < self.population_size:
            program = await self._create_random_program(program_type)
            self.population.append(program)
        
        # Evaluate initial population
        await self._evaluate_population()
        
        return self.population
    
    async def _create_random_program(self, program_type: ProgramType) -> Program:
        """Create a random program of the specified type"""
        program_id = str(uuid.uuid4())
        
        # Get random atoms from the AtomSpace
        available_atoms = await self._get_available_atoms()
        
        # Create program with random atoms
        num_atoms = random.randint(1, min(10, len(available_atoms)))
        selected_atoms = random.sample(available_atoms, num_atoms)
        
        program = Program(
            id=program_id,
            program_type=program_type,
            atoms=selected_atoms,
            complexity=len(selected_atoms),
            generation=0
        )
        
        return program
    
    async def _get_available_atoms(self) -> List[str]:
        """Get available atoms from the AtomSpace"""
        # Get a sample of atoms for program construction
        node_pattern = {"atom_type": AtomType.NODE.value}
        link_pattern = {"atom_type": AtomType.LINK.value}
        
        nodes = await self.atomspace.storage.get_atoms_by_pattern(node_pattern, limit=100)
        links = await self.atomspace.storage.get_atoms_by_pattern(link_pattern, limit=100)
        
        all_atoms = nodes + links
        return [atom.id for atom in all_atoms]
    
    async def evolve_generation(self) -> Dict[str, Any]:
        """
        Evolve one generation of programs
        
        Returns:
            Dictionary with generation statistics
        """
        generation_start = datetime.now(timezone.utc)
        
        # Evaluate current population
        await self._evaluate_population()
        
        # Selection
        selected_programs = await self._selection()
        
        # Crossover and mutation
        new_generation = []
        
        # Keep best programs (elitism)
        elite_count = int(self.population_size * 0.1)
        elite_programs = sorted(self.population, key=lambda p: p.fitness, reverse=True)[:elite_count]
        new_generation.extend(elite_programs)
        
        # Generate offspring
        while len(new_generation) < self.population_size:
            if random.random() < self.crossover_rate and len(selected_programs) >= 2:
                # Crossover
                parent1, parent2 = random.sample(selected_programs, 2)
                offspring = await self._crossover(parent1, parent2)
            else:
                # Mutation only
                parent = random.choice(selected_programs)
                offspring = await self._mutate(parent)
            
            # Apply mutation
            if random.random() < self.mutation_rate:
                offspring = await self._mutate(offspring)
            
            offspring.generation = self.generation_count + 1
            new_generation.append(offspring)
        
        # Update population
        self.population = new_generation[:self.population_size]
        self.generation_count += 1
        
        # Record generation statistics
        generation_stats = {
            "generation": self.generation_count,
            "timestamp": generation_start.isoformat(),
            "duration": (datetime.now(timezone.utc) - generation_start).total_seconds(),
            "best_fitness": max(p.fitness for p in self.population),
            "average_fitness": np.mean([p.fitness for p in self.population]),
            "average_complexity": np.mean([p.complexity for p in self.population]),
            "population_size": len(self.population)
        }
        
        self.evolution_history.append(generation_stats)
        return generation_stats
    
    async def _evaluate_population(self):
        """Evaluate fitness of all programs in population"""
        for program in self.population:
            program.fitness = await self._evaluate_program(program)
    
    async def _evaluate_program(self, program: Program) -> float:
        """
        Evaluate fitness of a single program
        
        Args:
            program: Program to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        if not program.atoms:
            return 0.0
        
        fitness_components = []
        
        # Component 1: Semantic coherence
        semantic_fitness = await self._evaluate_semantic_coherence(program)
        fitness_components.append(semantic_fitness)
        
        # Component 2: Utility/performance
        utility_fitness = await self._evaluate_utility(program)
        fitness_components.append(utility_fitness)
        
        # Component 3: Complexity penalty
        complexity_penalty = max(0, 1.0 - (program.complexity / self.max_complexity))
        fitness_components.append(complexity_penalty)
        
        # Combine fitness components
        total_fitness = np.mean(fitness_components)
        
        return total_fitness
    
    async def _evaluate_semantic_coherence(self, program: Program) -> float:
        """Evaluate semantic coherence of program atoms"""
        if len(program.atoms) < 2:
            return 0.5
        
        coherence_scores = []
        
        # Check truth value consistency
        for atom_id in program.atoms:
            try:
                truth_value = await self.pln_engine.infer_truth_value(atom_id)
                coherence_scores.append(truth_value.confidence)
            except Exception:
                coherence_scores.append(0.1)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    async def _evaluate_utility(self, program: Program) -> float:
        """Evaluate utility/performance of program"""
        # This is a simplified utility evaluation
        # In practice, this would test the program on specific tasks
        
        utility_score = 0.0
        
        # Program type specific evaluation
        if program.program_type == ProgramType.INFERENCE_RULE:
            # Test inference capability
            utility_score = await self._test_inference_rule(program)
        elif program.program_type == ProgramType.PATTERN_MATCHER:
            # Test pattern matching capability
            utility_score = await self._test_pattern_matcher(program)
        elif program.program_type == ProgramType.COGNITIVE_KERNEL:
            # Test cognitive processing
            utility_score = await self._test_cognitive_kernel(program)
        else:
            # Default utility based on atom diversity
            utility_score = min(1.0, len(set(program.atoms)) / 10.0)
        
        return utility_score
    
    async def _test_inference_rule(self, program: Program) -> float:
        """Test inference rule program"""
        # Simple test: can it derive something meaningful?
        if len(program.atoms) < 2:
            return 0.1
        
        # Test forward chaining with program atoms as premises
        try:
            inference_result = await self.pln_engine.forward_chaining(program.atoms[:3], max_iterations=3)
            derived_facts = len(inference_result.get("derived_facts", []))
            return min(1.0, derived_facts / 5.0)
        except Exception:
            return 0.1
    
    async def _test_pattern_matcher(self, program: Program) -> float:
        """Test pattern matcher program"""
        # Test ability to find patterns in AtomSpace
        if not program.atoms:
            return 0.1
        
        try:
            # Count successful pattern matches
            match_count = 0
            for atom_id in program.atoms:
                atom = await self.atomspace.get_atom(atom_id)
                if atom:
                    match_count += 1
            
            return match_count / len(program.atoms)
        except Exception:
            return 0.1
    
    async def _test_cognitive_kernel(self, program: Program) -> float:
        """Test cognitive kernel program"""
        # Test cognitive processing capability
        if not program.atoms:
            return 0.1
        
        # Simple test: connectivity and truth value propagation
        try:
            total_truth_strength = 0.0
            for atom_id in program.atoms:
                tv = await self.pln_engine.infer_truth_value(atom_id)
                total_truth_strength += tv.strength
            
            return total_truth_strength / len(program.atoms)
        except Exception:
            return 0.1
    
    async def _selection(self) -> List[Program]:
        """Select programs for reproduction"""
        # Tournament selection
        tournament_size = max(2, int(self.population_size * self.selection_pressure))
        selected = []
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda p: p.fitness)
            selected.append(winner)
        
        return selected
    
    async def _crossover(self, parent1: Program, parent2: Program) -> Program:
        """Create offspring through crossover"""
        offspring_id = str(uuid.uuid4())
        
        # Combine atoms from both parents
        combined_atoms = list(set(parent1.atoms + parent2.atoms))
        
        # Select subset for offspring
        offspring_size = min(self.max_complexity, random.randint(1, len(combined_atoms)))
        offspring_atoms = random.sample(combined_atoms, offspring_size)
        
        offspring = Program(
            id=offspring_id,
            program_type=parent1.program_type,
            atoms=offspring_atoms,
            complexity=len(offspring_atoms),
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[parent1.id, parent2.id]
        )
        
        return offspring
    
    async def _mutate(self, program: Program) -> Program:
        """Mutate a program"""
        mutated_program = copy.deepcopy(program)
        mutated_program.id = str(uuid.uuid4())
        mutated_program.parent_ids = [program.id]
        
        # Mutation operations
        mutation_type = random.choice(["add", "remove", "replace"])
        
        if mutation_type == "add" and len(mutated_program.atoms) < self.max_complexity:
            # Add a random atom
            available_atoms = await self._get_available_atoms()
            if available_atoms:
                new_atom = random.choice(available_atoms)
                if new_atom not in mutated_program.atoms:
                    mutated_program.atoms.append(new_atom)
        
        elif mutation_type == "remove" and len(mutated_program.atoms) > 1:
            # Remove a random atom
            mutated_program.atoms.remove(random.choice(mutated_program.atoms))
        
        elif mutation_type == "replace" and mutated_program.atoms:
            # Replace a random atom
            available_atoms = await self._get_available_atoms()
            if available_atoms:
                old_atom = random.choice(mutated_program.atoms)
                new_atom = random.choice(available_atoms)
                idx = mutated_program.atoms.index(old_atom)
                mutated_program.atoms[idx] = new_atom
        
        mutated_program.complexity = len(mutated_program.atoms)
        return mutated_program
    
    async def optimize_program(self, program_type: ProgramType, 
                             generations: int = 10,
                             seed_programs: Optional[List[Program]] = None) -> Dict[str, Any]:
        """
        Optimize programs of a specific type
        
        Args:
            program_type: Type of programs to optimize
            generations: Number of generations to evolve
            seed_programs: Optional seed programs
            
        Returns:
            Dictionary with optimization results
        """
        optimization_session = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "program_type": program_type.value,
            "generations": generations,
            "results": []
        }
        
        # Initialize population
        await self.initialize_population(program_type, seed_programs)
        
        # Evolve generations
        for generation in range(generations):
            generation_stats = await self.evolve_generation()
            optimization_session["results"].append(generation_stats)
        
        # Get best program
        best_program = max(self.population, key=lambda p: p.fitness)
        optimization_session["best_program"] = best_program.to_dict()
        
        return optimization_session
    
    def get_best_programs(self, n: int = 5) -> List[Program]:
        """Get the n best programs from current population"""
        return sorted(self.population, key=lambda p: p.fitness, reverse=True)[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MOSES optimizer statistics"""
        if not self.population:
            return {
                "population_size": 0,
                "generation_count": self.generation_count,
                "best_fitness": 0.0,
                "average_fitness": 0.0
            }
        
        return {
            "population_size": len(self.population),
            "generation_count": self.generation_count,
            "best_fitness": max(p.fitness for p in self.population),
            "average_fitness": np.mean([p.fitness for p in self.population]),
            "average_complexity": np.mean([p.complexity for p in self.population]),
            "evolution_sessions": len(self.evolution_history)
        }
    
    async def save_program(self, program: Program) -> bool:
        """Save program to AtomSpace"""
        try:
            # Create a link to represent the program
            program_link = await self.atomspace.add_link(
                name=f"program_{program.id}",
                link_type="program",
                outgoing=program.atoms,
                truth_value=program.fitness,
                confidence=1.0,
                metadata=program.to_dict()
            )
            
            return True
        except Exception as e:
            print(f"Error saving program {program.id}: {e}")
            return False
    
    async def load_program(self, program_id: str) -> Optional[Program]:
        """Load program from AtomSpace"""
        try:
            # Find program link
            pattern = {"atom_type": AtomType.LINK.value, "link_type": "program"}
            links = await self.atomspace.storage.get_atoms_by_pattern(pattern, limit=1000)
            
            for link in links:
                if hasattr(link, 'metadata') and link.metadata:
                    try:
                        metadata = json.loads(link.metadata) if isinstance(link.metadata, str) else link.metadata
                        if metadata.get("id") == program_id:
                            return Program.from_dict(metadata)
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            return None
        except Exception as e:
            print(f"Error loading program {program_id}: {e}")
            return None