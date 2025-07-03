"""
Neural-Symbolic Reasoning Engine Demonstration

This script demonstrates the key capabilities of the neural-symbolic reasoning engine
including PLN inference, MOSES optimization, pattern matching, and integrated reasoning.
"""

import asyncio
import sys
import os
from datetime import datetime
import json

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from python.helpers.atomspace import AtomSpace
from python.helpers.neural_symbolic_reasoning import NeuralSymbolicReasoningEngine, ReasoningStage
from python.helpers.pln_reasoning import LogicalOperator
from python.helpers.moses_optimizer import ProgramType
from python.helpers.pattern_matcher import Pattern, MatchType


async def main():
    """Main demonstration of neural-symbolic reasoning capabilities"""
    
    print("🧠🔗 Neural-Symbolic Reasoning Engine Demonstration")
    print("=" * 60)
    
    # Initialize system
    print("\n1. 🔧 Initializing Neural-Symbolic Reasoning Engine...")
    
    # Create temporary AtomSpace
    db_path = "/tmp/demo_neural_symbolic.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    atomspace = AtomSpace(db_path)
    reasoning_engine = NeuralSymbolicReasoningEngine(atomspace)
    
    # Initialize the system
    init_result = await reasoning_engine.initialize_system()
    print(f"   ✅ System initialized with {init_result['default_kernels']} cognitive kernels")
    
    # Create knowledge base
    print("\n2. 📚 Creating Demonstration Knowledge Base...")
    
    # Add concepts
    animal = await atomspace.add_node("animal", "concept", 0.9, 0.8)
    dog = await atomspace.add_node("dog", "concept", 0.95, 0.9)
    mammal = await atomspace.add_node("mammal", "concept", 0.92, 0.85)
    intelligence = await atomspace.add_node("intelligence", "property", 0.8, 0.7)
    
    # Add relationships
    dog_is_mammal = await atomspace.add_link("dog_is_mammal", [dog.id, mammal.id], "is_a", 0.98, 0.95)
    mammal_is_animal = await atomspace.add_link("mammal_is_animal", [mammal.id, animal.id], "is_a", 0.99, 0.98)
    dog_has_intelligence = await atomspace.add_link("dog_has_intelligence", [dog.id, intelligence.id], "has_property", 0.85, 0.8)
    
    print(f"   ✅ Created knowledge base with {len([animal, dog, mammal, intelligence, dog_is_mammal, mammal_is_animal, dog_has_intelligence])} atoms")
    
    # Demonstrate PLN inference
    print("\n3. 🧠 PLN Probabilistic Logic Networks Demonstration...")
    
    # Infer truth values
    dog_truth = await reasoning_engine.pln_engine.infer_truth_value(dog.id)
    print(f"   📊 Dog concept truth value: strength={dog_truth.strength:.3f}, confidence={dog_truth.confidence:.3f}")
    
    # Forward chaining inference
    premises = [dog.id, mammal.id]
    inference_result = await reasoning_engine.pln_engine.forward_chaining(premises, max_iterations=5)
    print(f"   🔄 Forward chaining: derived {len(inference_result['derived_facts'])} new facts")
    
    # Get explanation
    explanation = await reasoning_engine.pln_engine.get_inference_explanation(dog.id)
    print(f"   📝 Generated explanation tree for dog concept")
    
    # Demonstrate MOSES optimization  
    print("\n4. 🧬 MOSES Evolutionary Program Optimization...")
    
    # Initialize population
    await reasoning_engine.moses_optimizer.initialize_population(ProgramType.INFERENCE_RULE)
    print(f"   🌱 Initialized population of {reasoning_engine.moses_optimizer.population_size} programs")
    
    # Run evolution for a few generations
    for gen in range(3):
        stats = await reasoning_engine.moses_optimizer.evolve_generation()
        print(f"   📈 Generation {stats['generation']}: best_fitness={stats['best_fitness']:.3f}, avg_fitness={stats['average_fitness']:.3f}")
    
    # Get best programs
    best_programs = reasoning_engine.moses_optimizer.get_best_programs(3)
    print(f"   🏆 Top program fitness: {best_programs[0].fitness:.3f} with complexity {best_programs[0].complexity}")
    
    # Demonstrate pattern matching
    print("\n5. 🔍 Advanced Pattern Matching Demonstration...")
    
    # Create and register patterns
    concept_pattern = Pattern(
        id="concept_pattern",
        pattern_type=MatchType.EXACT,
        template={"atom_type": "node", "concept_type": "concept"},
        constraints={"truth_value_min": 0.8}
    )
    
    await reasoning_engine.pattern_matcher.register_pattern(concept_pattern)
    print(f"   📋 Registered concept pattern")
    
    # Find matches
    matches = await reasoning_engine.pattern_matcher.match_pattern("concept_pattern", max_matches=10)
    print(f"   🎯 Found {len(matches)} concept matches")
    
    # Semantic pattern matching
    semantic_pattern = Pattern(
        id="semantic_pattern", 
        pattern_type=MatchType.SEMANTIC,
        template={"concept_type": "concept"},
        weights={"truth_value": 0.8, "confidence": 0.2}
    )
    
    await reasoning_engine.pattern_matcher.register_pattern(semantic_pattern)
    semantic_matches = await reasoning_engine.pattern_matcher.match_pattern("semantic_pattern", max_matches=5)
    print(f"   🧠 Found {len(semantic_matches)} semantic matches")
    
    # Hypergraph traversal
    traversal = await reasoning_engine.pattern_matcher.traverse_hypergraph(dog.id, max_depth=3, max_nodes=10)
    print(f"   🌐 Hypergraph traversal from 'dog': visited {len(traversal)} nodes")
    
    # Extract semantic relations
    atom_ids = [dog.id, mammal.id, animal.id]
    relations = await reasoning_engine.pattern_matcher.extract_semantic_relations(atom_ids)
    print(f"   🔗 Extracted {len(relations)} types of semantic relations")
    
    # Demonstrate integrated neural-symbolic reasoning
    print("\n6. 🧠🔗 Integrated Neural-Symbolic Reasoning...")
    
    # Reasoning query
    query = {
        "type": "infer",
        "concepts": ["dog", "mammal", "animal"],
        "include_details": True
    }
    
    reasoning_result = await reasoning_engine.reason(query)
    print(f"   🤔 Reasoning session: {reasoning_result['session_id'][:8]}...")
    print(f"   ⚡ Executed {len(reasoning_result['stages'])} cognitive stages")
    print(f"   ✅ Success: {reasoning_result['result']['success']}")
    print(f"   📊 Confidence: {reasoning_result['result']['confidence']:.3f}")
    print(f"   ⏱️  Duration: {reasoning_result['duration']:.3f} seconds")
    
    # Show stage results
    for stage in reasoning_result['stages']:
        stage_name = stage['stage']
        success = "✅" if stage['success'] else "❌"
        print(f"      {success} {stage_name}: {stage['duration']:.3f}s")
    
    # Demonstrate cognitive tensor encoding
    print("\n7. 🧮 Cognitive Tensor Encoding T_ai[n_patterns, n_reasoning, l_stages]...")
    
    # Get system tensor
    system_tensor = await reasoning_engine.get_system_tensor()
    print(f"   📐 System tensor shape: {system_tensor.shape}")
    print(f"   📊 Tensor statistics:")
    print(f"      - Mean: {system_tensor.mean():.6f}")
    print(f"      - Std:  {system_tensor.std():.6f}")
    print(f"      - Min:  {system_tensor.min():.6f}")
    print(f"      - Max:  {system_tensor.max():.6f}")
    
    # Get specific kernel tensor
    inference_tensor = await reasoning_engine.get_cognitive_tensor("inference_kernel")
    if inference_tensor is not None:
        print(f"   🧠 Inference kernel tensor shape: {inference_tensor.shape}")
    
    # System statistics
    print("\n8. 📊 Final System Statistics...")
    stats = reasoning_engine.get_statistics()
    
    print(f"   🧠 Cognitive Kernels: {stats['cognitive_kernels']}")
    print(f"   💭 Reasoning Sessions: {stats['reasoning_sessions']}")
    print(f"   📈 PLN Statistics:")
    print(f"      - Cached truth values: {stats['pln_statistics']['cached_truth_values']}")
    print(f"      - Inference sessions: {stats['pln_statistics']['inference_sessions']}")
    print(f"   🧬 MOSES Statistics:")
    print(f"      - Population size: {stats['moses_statistics']['population_size']}")
    print(f"      - Best fitness: {stats['moses_statistics']['best_fitness']:.3f}")
    print(f"   🔍 Pattern Matcher Statistics:")
    print(f"      - Registered patterns: {stats['pattern_matcher_statistics']['registered_patterns']}")
    print(f"      - Cached matches: {stats['pattern_matcher_statistics']['cached_matches']}")
    
    print("\n9. 🎉 Demonstration Complete!")
    print("\n" + "=" * 60)
    print("✨ Neural-Symbolic Reasoning Engine Successfully Demonstrated!")
    print("\n🚀 Key Capabilities Shown:")
    print("   ✅ PLN inference with uncertain reasoning")
    print("   ✅ MOSES evolutionary program optimization") 
    print("   ✅ Advanced pattern matching and hypergraph traversal")
    print("   ✅ Integrated multi-stage reasoning pipeline")
    print("   ✅ Tensor-based cognitive kernel encoding")
    print("   ✅ Real AtomSpace data processing (no mockups)")
    print("\n🔗 Ready for integration with Agent Zero distributed systems!")
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)


if __name__ == "__main__":
    asyncio.run(main())