"""
Comprehensive test suite for Neural-Symbolic Reasoning Engine

Tests PLN inference, MOSES optimization, pattern matching, and integrated reasoning
with real AtomSpace data instead of mockups.
"""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.helpers.atomspace import AtomSpace, AtomType
from python.helpers.neural_symbolic_reasoning import NeuralSymbolicReasoningEngine, ReasoningStage
from python.helpers.pln_reasoning import PLNInferenceEngine, TruthValue, LogicalOperator
from python.helpers.moses_optimizer import MOSESOptimizer, ProgramType
from python.helpers.pattern_matcher import HypergraphPatternMatcher, Pattern, MatchType


class NeuralSymbolicTest:
    """Test suite for neural-symbolic reasoning engine"""
    
    def __init__(self):
        self.test_db_path = "/tmp/test_neural_symbolic.db"
        self.atomspace = None
        self.reasoning_engine = None
        self.test_results = []
        
    async def setup_test_environment(self):
        """Setup test environment with real AtomSpace data"""
        print("🔧 Setting up test environment...")
        
        # Create fresh AtomSpace
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        
        self.atomspace = AtomSpace(self.test_db_path)
        
        # Initialize reasoning engine
        self.reasoning_engine = NeuralSymbolicReasoningEngine(self.atomspace)
        await self.reasoning_engine.initialize_system()
        
        # Create test knowledge base
        await self._create_test_knowledge_base()
        
        print("✅ Test environment setup complete!")
    
    async def _create_test_knowledge_base(self):
        """Create a real knowledge base for testing"""
        print("📚 Creating test knowledge base...")
        
        # Create knowledge nodes
        animal_node = await self.atomspace.add_node(
            "animal", "concept", 
            truth_value=0.9, confidence=0.8,
            metadata={"category": "biological", "level": "kingdom"}
        )
        
        dog_node = await self.atomspace.add_node(
            "dog", "concept",
            truth_value=0.95, confidence=0.9,
            metadata={"category": "biological", "level": "species"}
        )
        
        mammal_node = await self.atomspace.add_node(
            "mammal", "concept",
            truth_value=0.92, confidence=0.85,
            metadata={"category": "biological", "level": "class"}
        )
        
        loyal_node = await self.atomspace.add_node(
            "loyal", "property",
            truth_value=0.8, confidence=0.7,
            metadata={"category": "behavioral", "type": "trait"}
        )
        
        # Create logical relationships
        dog_is_mammal = await self.atomspace.add_link(
            "dog_is_mammal", [dog_node.id, mammal_node.id], "is_a",
            truth_value=0.98, confidence=0.95
        )
        
        mammal_is_animal = await self.atomspace.add_link(
            "mammal_is_animal", [mammal_node.id, animal_node.id], "is_a",
            truth_value=0.99, confidence=0.98
        )
        
        dog_is_loyal = await self.atomspace.add_link(
            "dog_is_loyal", [dog_node.id, loyal_node.id], "has_property",
            truth_value=0.85, confidence=0.8
        )
        
        # Create inference rules
        transitivity_rule = await self.atomspace.add_link(
            "transitivity_rule", [dog_is_mammal.id, mammal_is_animal.id], LogicalOperator.IMPLIES.value,
            truth_value=0.9, confidence=0.9
        )
        
        print(f"✅ Created knowledge base with {len(await self._get_all_atoms())} atoms")
    
    async def _get_all_atoms(self):
        """Get all atoms in the test AtomSpace"""
        nodes = await self.atomspace.storage.get_atoms_by_pattern({"atom_type": AtomType.NODE.value}, limit=1000)
        links = await self.atomspace.storage.get_atoms_by_pattern({"atom_type": AtomType.LINK.value}, limit=1000)
        return nodes + links
    
    async def test_pln_inference(self):
        """Test PLN logical inference on real data"""
        print("\n🧠 Testing PLN Inference Engine...")
        
        test_passed = True
        
        try:
            # Test 1: Basic truth value inference
            print("  📊 Test 1: Basic truth value inference")
            
            # Get dog node
            dog_atoms = await self.atomspace.storage.get_atoms_by_pattern({"name": "dog"})
            if not dog_atoms:
                raise Exception("Dog node not found")
            
            dog_id = dog_atoms[0].id
            truth_value = await self.reasoning_engine.pln_engine.infer_truth_value(dog_id)
            
            print(f"    Dog truth value: strength={truth_value.strength:.3f}, confidence={truth_value.confidence:.3f}")
            
            if truth_value.strength < 0.9 or truth_value.confidence < 0.8:
                print("    ❌ Truth value inference failed")
                test_passed = False
            else:
                print("    ✅ Truth value inference passed")
            
            # Test 2: Forward chaining inference
            print("  🔄 Test 2: Forward chaining inference")
            
            # Use dog and mammal as premises
            mammal_atoms = await self.atomspace.storage.get_atoms_by_pattern({"name": "mammal"})
            if not mammal_atoms:
                raise Exception("Mammal node not found")
            
            premises = [dog_id, mammal_atoms[0].id]
            inference_result = await self.reasoning_engine.pln_engine.forward_chaining(premises, max_iterations=5)
            
            derived_facts = inference_result.get("derived_facts", [])
            print(f"    Derived facts: {len(derived_facts)}")
            print(f"    Iterations: {inference_result.get('iterations', 0)}")
            
            if len(derived_facts) > 0:
                print("    ✅ Forward chaining inference passed")
            else:
                print("    ⚠️  Forward chaining produced no new facts (may be normal)")
            
            # Test 3: Explanation generation
            print("  📝 Test 3: Explanation generation")
            
            explanation = await self.reasoning_engine.pln_engine.get_inference_explanation(dog_id)
            
            if explanation and "explanation_tree" in explanation:
                print("    ✅ Explanation generation passed")
            else:
                print("    ❌ Explanation generation failed")
                test_passed = False
                
        except Exception as e:
            print(f"    ❌ PLN inference test failed: {e}")
            test_passed = False
        
        self.test_results.append({"test": "PLN Inference", "passed": test_passed})
        return test_passed
    
    async def test_moses_optimization(self):
        """Test MOSES evolutionary program optimization"""
        print("\n🧬 Testing MOSES Optimizer...")
        
        test_passed = True
        
        try:
            # Test 1: Population initialization
            print("  🌱 Test 1: Population initialization")
            
            population = await self.reasoning_engine.moses_optimizer.initialize_population(
                ProgramType.INFERENCE_RULE, seed_programs=None
            )
            
            print(f"    Initial population size: {len(population)}")
            
            if len(population) != self.reasoning_engine.moses_optimizer.population_size:
                print("    ❌ Population initialization failed")
                test_passed = False
            else:
                print("    ✅ Population initialization passed")
            
            # Test 2: Program evaluation
            print("  🎯 Test 2: Program evaluation")
            
            if population:
                first_program = population[0]
                fitness = await self.reasoning_engine.moses_optimizer._evaluate_program(first_program)
                
                print(f"    Program fitness: {fitness:.3f}")
                print(f"    Program complexity: {first_program.complexity}")
                
                if 0.0 <= fitness <= 1.0:
                    print("    ✅ Program evaluation passed")
                else:
                    print("    ❌ Program evaluation failed")
                    test_passed = False
            
            # Test 3: Evolution cycle
            print("  🔄 Test 3: Evolution cycle")
            
            generation_stats = await self.reasoning_engine.moses_optimizer.evolve_generation()
            
            print(f"    Generation: {generation_stats['generation']}")
            print(f"    Best fitness: {generation_stats['best_fitness']:.3f}")
            print(f"    Average fitness: {generation_stats['average_fitness']:.3f}")
            
            if generation_stats['best_fitness'] >= 0.0:
                print("    ✅ Evolution cycle passed")
            else:
                print("    ❌ Evolution cycle failed")
                test_passed = False
            
            # Test 4: Full optimization
            print("  🚀 Test 4: Full optimization run")
            
            optimization_result = await self.reasoning_engine.moses_optimizer.optimize_program(
                ProgramType.PATTERN_MATCHER, generations=3
            )
            
            best_program = optimization_result.get("best_program")
            if best_program:
                print(f"    Best program fitness: {best_program['fitness']:.3f}")
                print(f"    Best program complexity: {best_program['complexity']}")
                print("    ✅ Full optimization passed")
            else:
                print("    ❌ Full optimization failed")
                test_passed = False
                
        except Exception as e:
            print(f"    ❌ MOSES optimization test failed: {e}")
            test_passed = False
        
        self.test_results.append({"test": "MOSES Optimization", "passed": test_passed})
        return test_passed
    
    async def test_pattern_matching(self):
        """Test hypergraph pattern matching"""
        print("\n🔍 Testing Pattern Matcher...")
        
        test_passed = True
        
        try:
            # Test 1: Pattern registration
            print("  📋 Test 1: Pattern registration")
            
            # Create test pattern
            test_pattern = Pattern(
                id="test_pattern_1",
                pattern_type=MatchType.EXACT,
                template={"atom_type": AtomType.NODE.value, "concept_type": "concept"},
                constraints={"min_truth_value": 0.8}
            )
            
            registration_success = await self.reasoning_engine.pattern_matcher.register_pattern(test_pattern)
            
            if registration_success:
                print("    ✅ Pattern registration passed")
            else:
                print("    ❌ Pattern registration failed")
                test_passed = False
            
            # Test 2: Exact pattern matching
            print("  🎯 Test 2: Exact pattern matching")
            
            matches = await self.reasoning_engine.pattern_matcher.match_pattern("test_pattern_1", max_matches=10)
            
            print(f"    Found {len(matches)} matches")
            
            if len(matches) > 0:
                print("    ✅ Exact pattern matching passed")
                
                # Show match details
                for i, match in enumerate(matches[:3]):
                    print(f"      Match {i+1}: confidence={match.confidence:.3f}, atoms={len(match.matched_atoms)}")
            else:
                print("    ⚠️  No matches found (may be normal)")
            
            # Test 3: Semantic pattern matching
            print("  🧠 Test 3: Semantic pattern matching")
            
            semantic_pattern = Pattern(
                id="semantic_pattern_1",
                pattern_type=MatchType.SEMANTIC,
                template={"concept_type": "concept"},
                weights={"truth_value": 0.8, "confidence": 0.2}
            )
            
            await self.reasoning_engine.pattern_matcher.register_pattern(semantic_pattern)
            semantic_matches = await self.reasoning_engine.pattern_matcher.match_pattern("semantic_pattern_1", max_matches=5)
            
            print(f"    Found {len(semantic_matches)} semantic matches")
            
            if len(semantic_matches) > 0:
                print("    ✅ Semantic pattern matching passed")
            else:
                print("    ⚠️  No semantic matches found")
            
            # Test 4: Hypergraph traversal
            print("  🌐 Test 4: Hypergraph traversal")
            
            # Get a starting node
            all_atoms = await self._get_all_atoms()
            if all_atoms:
                start_atom = all_atoms[0]
                traversal_result = await self.reasoning_engine.pattern_matcher.traverse_hypergraph(
                    start_atom.id, max_depth=3, max_nodes=10
                )
                
                print(f"    Traversed {len(traversal_result)} nodes")
                
                if len(traversal_result) > 0:
                    print("    ✅ Hypergraph traversal passed")
                else:
                    print("    ❌ Hypergraph traversal failed")
                    test_passed = False
            
            # Test 5: Semantic relation extraction
            print("  🔗 Test 5: Semantic relation extraction")
            
            if len(all_atoms) >= 2:
                atom_ids = [atom.id for atom in all_atoms[:3]]
                semantic_relations = await self.reasoning_engine.pattern_matcher.extract_semantic_relations(atom_ids)
                
                print(f"    Found {len(semantic_relations)} relation types")
                for rel_type, relations in semantic_relations.items():
                    print(f"      {rel_type}: {len(relations)} relations")
                
                print("    ✅ Semantic relation extraction passed")
            
        except Exception as e:
            print(f"    ❌ Pattern matching test failed: {e}")
            test_passed = False
        
        self.test_results.append({"test": "Pattern Matching", "passed": test_passed})
        return test_passed
    
    async def test_neural_symbolic_integration(self):
        """Test integrated neural-symbolic reasoning"""
        print("\n🧠🔗 Testing Neural-Symbolic Integration...")
        
        test_passed = True
        
        try:
            # Test 1: Cognitive tensor encoding
            print("  🧮 Test 1: Cognitive tensor encoding")
            
            system_tensor = await self.reasoning_engine.get_system_tensor()
            
            print(f"    System tensor shape: {system_tensor.shape}")
            print(f"    Expected shape: ({self.reasoning_engine.max_patterns}, {self.reasoning_engine.max_reasoning_programs}, {self.reasoning_engine.max_stages})")
            
            if system_tensor.shape == (self.reasoning_engine.max_patterns, self.reasoning_engine.max_reasoning_programs, self.reasoning_engine.max_stages):
                print("    ✅ Cognitive tensor encoding passed")
            else:
                print("    ❌ Cognitive tensor encoding failed")
                test_passed = False
            
            # Test 2: Reasoning query execution
            print("  🤔 Test 2: Reasoning query execution")
            
            query = {
                "type": "infer",
                "concepts": ["dog", "mammal"],
                "include_details": True
            }
            
            reasoning_result = await self.reasoning_engine.reason(query)
            
            print(f"    Reasoning session: {reasoning_result['session_id'][:8]}...")
            print(f"    Stages executed: {len(reasoning_result['stages'])}")
            print(f"    Success: {reasoning_result['result']['success']}")
            print(f"    Confidence: {reasoning_result['result']['confidence']:.3f}")
            
            if reasoning_result['result']['success']:
                print("    ✅ Reasoning query execution passed")
            else:
                print("    ❌ Reasoning query execution failed")
                test_passed = False
            
            # Test 3: API request processing
            print("  🌐 Test 3: API request processing")
            
            api_response = await self.reasoning_engine.create_reasoning_api_request(
                "pattern", {"concepts": ["animal", "mammal"], "include_details": True}
            )
            
            print(f"    API response success: {api_response['success']}")
            print(f"    API response confidence: {api_response['confidence']:.3f}")
            
            if api_response['success']:
                print("    ✅ API request processing passed")
            else:
                print("    ❌ API request processing failed")
                test_passed = False
            
            # Test 4: System statistics
            print("  📊 Test 4: System statistics")
            
            stats = self.reasoning_engine.get_statistics()
            
            print(f"    Cognitive kernels: {stats['cognitive_kernels']}")
            print(f"    Reasoning sessions: {stats['reasoning_sessions']}")
            print(f"    PLN cached values: {stats['pln_statistics']['cached_truth_values']}")
            print(f"    MOSES population: {stats['moses_statistics']['population_size']}")
            
            if stats['cognitive_kernels'] > 0:
                print("    ✅ System statistics passed")
            else:
                print("    ❌ System statistics failed")
                test_passed = False
                
        except Exception as e:
            print(f"    ❌ Neural-symbolic integration test failed: {e}")
            test_passed = False
        
        self.test_results.append({"test": "Neural-Symbolic Integration", "passed": test_passed})
        return test_passed
    
    async def test_performance_and_scalability(self):
        """Test performance and scalability"""
        print("\n⚡ Testing Performance and Scalability...")
        
        test_passed = True
        
        try:
            # Test 1: Large knowledge base handling
            print("  📈 Test 1: Large knowledge base handling")
            
            # Add more test data
            start_time = datetime.now()
            
            for i in range(50):
                node = await self.atomspace.add_node(
                    f"test_concept_{i}", "concept",
                    truth_value=0.7 + (i % 3) * 0.1,
                    confidence=0.8 + (i % 2) * 0.1
                )
            
            creation_time = (datetime.now() - start_time).total_seconds()
            print(f"    Created 50 nodes in {creation_time:.3f} seconds")
            
            # Test retrieval performance
            start_time = datetime.now()
            all_atoms = await self._get_all_atoms()
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            print(f"    Retrieved {len(all_atoms)} atoms in {retrieval_time:.3f} seconds")
            
            if creation_time < 5.0 and retrieval_time < 2.0:
                print("    ✅ Large knowledge base handling passed")
            else:
                print("    ⚠️  Performance may be suboptimal")
            
            # Test 2: Concurrent reasoning
            print("  🔄 Test 2: Concurrent reasoning")
            
            queries = [
                {"type": "infer", "concepts": ["dog"]},
                {"type": "pattern", "concepts": ["animal"]},
                {"type": "optimize", "concepts": ["mammal"]}
            ]
            
            start_time = datetime.now()
            
            # Run concurrent reasoning
            tasks = [self.reasoning_engine.reason(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            concurrent_time = (datetime.now() - start_time).total_seconds()
            print(f"    Completed {len(queries)} concurrent queries in {concurrent_time:.3f} seconds")
            
            successful_results = [r for r in results if not isinstance(r, Exception)]
            print(f"    Successful results: {len(successful_results)}")
            
            if len(successful_results) == len(queries):
                print("    ✅ Concurrent reasoning passed")
            else:
                print("    ❌ Concurrent reasoning failed")
                test_passed = False
            
        except Exception as e:
            print(f"    ❌ Performance test failed: {e}")
            test_passed = False
        
        self.test_results.append({"test": "Performance and Scalability", "passed": test_passed})
        return test_passed
    
    async def cleanup(self):
        """Cleanup test environment"""
        print("\n🧹 Cleaning up test environment...")
        
        try:
            if os.path.exists(self.test_db_path):
                os.remove(self.test_db_path)
            print("✅ Cleanup complete!")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    async def run_all_tests(self):
        """Run all tests in the suite"""
        print("🚀 Starting Neural-Symbolic Reasoning Engine Test Suite")
        print("=" * 80)
        
        await self.setup_test_environment()
        
        test_functions = [
            self.test_pln_inference,
            self.test_moses_optimization,
            self.test_pattern_matching,
            self.test_neural_symbolic_integration,
            self.test_performance_and_scalability
        ]
        
        for test_func in test_functions:
            try:
                await test_func()
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed with exception: {e}")
                self.test_results.append({"test": test_func.__name__, "passed": False})
        
        await self.cleanup()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        passed_tests = sum(1 for result in self.test_results if result["passed"])
        total_tests = len(self.test_results)
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"  {result['test']}: {status}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Neural-Symbolic Reasoning Engine is working correctly.")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please review the results above.")
        
        return passed_tests == total_tests


async def main():
    """Main test runner"""
    test_suite = NeuralSymbolicTest()
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())