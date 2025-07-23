#!/usr/bin/env python3
"""
Test Suite for Scheme-based Cognitive Representation in Distributed AtomSpace

This test validates the complete integration of Scheme-based cognitive grammars
with the distributed hypergraph AtomSpace memory model.
"""

import asyncio
import json
import logging
import sys
import os
import time
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent))

from python.helpers.atomspace import AtomSpace
from python.helpers.cognitive_atomspace_integration import (
    CognitiveAtomSpaceIntegration, 
    EnhancedMemoryAtomSpaceWrapper
)
from python.helpers.scheme_grammar import CognitiveOperator


class CognitiveAtomSpaceTestSuite:
    """Comprehensive test suite for cognitive AtomSpace integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    async def setup_test_environment(self):
        """Setup test environment with temporary AtomSpace"""
        print("🔧 Setting up cognitive test environment...")
        
        # Create temporary test storage
        test_storage_path = "memory/test_cognitive_atomspace"
        Path(test_storage_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize AtomSpace and cognitive integration
        self.atomspace = AtomSpace(f"{test_storage_path}/test.db")
        self.cognitive_integration = CognitiveAtomSpaceIntegration(self.atomspace)
        self.enhanced_memory = EnhancedMemoryAtomSpaceWrapper("test_agent")
        
        print("✅ Cognitive test environment setup complete!")
        
    async def cleanup_test_environment(self):
        """Clean up test environment"""
        print("🧹 Cleaning up cognitive test environment...")
        
        # Clean up temporary files
        import shutil
        test_storage_path = Path("memory/test_cognitive_atomspace")
        if test_storage_path.exists():
            shutil.rmtree(test_storage_path)
            
        print("✅ Cleanup complete!")
        
    async def run_test(self, test_name: str, test_func):
        """Run a single test and track results"""
        self.total_tests += 1
        try:
            print(f"  📋 Test: {test_name}")
            result = await test_func()
            if result:
                print(f"    ✅ {test_name} passed")
                self.passed_tests += 1
                self.test_results[test_name] = "PASSED"
            else:
                print(f"    ❌ {test_name} failed")
                self.test_results[test_name] = "FAILED"
        except Exception as e:
            print(f"    💥 {test_name} error: {e}")
            self.test_results[test_name] = f"ERROR: {e}"
    
    async def test_scheme_grammar_integration(self) -> bool:
        """Test Scheme grammar parsing and registration"""
        try:
            # Test grammar registration
            grammar_id = "test_perception_grammar"
            scheme_expr = "(perceive (object ?x) (property ?p) (context ?c))"
            
            grammar = self.cognitive_integration.scheme_registry.register_grammar(
                grammar_id=grammar_id,
                name="Test Perception Grammar",
                description="Test grammar for perception",
                scheme_expression=scheme_expr,
                cognitive_operators=[CognitiveOperator.PERCEIVE]
            )
            
            print(f"    • Registered grammar: {grammar.name}")
            
            # Test grammar evaluation
            bindings = {"x": "apple", "p": "red", "c": "kitchen"}
            result = self.cognitive_integration.scheme_registry.evaluate_grammar_expression(
                grammar_id, bindings
            )
            
            print(f"    • Evaluation result: {result.get('success', False)}")
            
            return grammar is not None and result.get("success", False)
            
        except Exception as e:
            print(f"    Error in scheme grammar test: {e}")
            return False
    
    async def test_cognitive_pattern_storage(self) -> bool:
        """Test storing cognitive patterns in AtomSpace"""
        try:
            # Store a complex cognitive pattern
            pattern_name = "learning_pattern"
            scheme_expr = "(learn (experience ?exp) (pattern ?pat) (update ?model))"
            
            pattern = await self.cognitive_integration.store_cognitive_pattern(
                name=pattern_name,
                scheme_expression=scheme_expr,
                metadata={"domain": "machine_learning", "complexity": "medium"}
            )
            
            print(f"    • Stored pattern: {pattern.name}")
            print(f"    • Pattern atoms: {len(pattern.atom_ids)}")
            
            # Verify atoms were created in AtomSpace
            atoms_exist = True
            for atom_id in pattern.atom_ids:
                atom = await self.atomspace.get_atom(atom_id)
                if not atom:
                    atoms_exist = False
                    break
            
            print(f"    • All atoms exist in AtomSpace: {atoms_exist}")
            
            return pattern is not None and atoms_exist and len(pattern.atom_ids) > 0
            
        except Exception as e:
            print(f"    Error in pattern storage test: {e}")
            return False
    
    async def test_cognitive_pattern_evaluation(self) -> bool:
        """Test evaluating cognitive patterns with bindings"""
        try:
            # First store a pattern for evaluation
            pattern_name = "decision_pattern"
            scheme_expr = "(decide (options ?opts) (criteria ?crit) (choice ?choice))"
            
            pattern = await self.cognitive_integration.store_cognitive_pattern(
                name=pattern_name,
                scheme_expression=scheme_expr
            )
            
            # Evaluate the pattern with bindings
            bindings = {
                "opts": ["option_a", "option_b", "option_c"],
                "crit": ["cost", "quality", "time"],
                "choice": "option_b"
            }
            
            result = await self.cognitive_integration.evaluate_cognitive_pattern(
                pattern.id, bindings
            )
            
            print(f"    • Pattern evaluation success: {result.get('success', False)}")
            print(f"    • Has AtomSpace context: {'atomspace_context' in result}")
            
            return result.get("success", False)
            
        except Exception as e:
            print(f"    Error in pattern evaluation test: {e}")
            return False
    
    async def test_cognitive_pattern_search(self) -> bool:
        """Test finding cognitive patterns by query"""
        try:
            # Store multiple patterns for search
            patterns = [
                ("reasoning_pattern", "(reason (premise ?p1) (premise ?p2) (conclusion ?c))"),
                ("learning_adaptation", "(learn (experience ?exp) (adapt ?model))"),
                ("perception_analysis", "(perceive (stimulus ?s) (analyze ?a))")
            ]
            
            stored_patterns = []
            for name, expr in patterns:
                pattern = await self.cognitive_integration.store_cognitive_pattern(
                    name=name,
                    scheme_expression=expr,
                    metadata={"category": "cognitive_process"}
                )
                stored_patterns.append(pattern)
            
            # Search for patterns
            search_results = await self.cognitive_integration.find_cognitive_patterns("learn")
            learning_patterns = [r for r in search_results if "learn" in r["pattern"]["name"]]
            
            print(f"    • Stored {len(stored_patterns)} patterns")
            print(f"    • Found {len(learning_patterns)} learning patterns")
            
            return len(learning_patterns) > 0
            
        except Exception as e:
            print(f"    Error in pattern search test: {e}")
            return False
    
    async def test_cognitive_reasoning(self) -> bool:
        """Test PLN reasoning with cognitive patterns"""
        try:
            # Create patterns that can be used for reasoning
            premise_pattern = await self.cognitive_integration.store_cognitive_pattern(
                name="premise_pattern",
                scheme_expression="(premise (fact ?f) (truth_value ?tv))",
                metadata={"reasoning_role": "premise"}
            )
            
            conclusion_pattern = await self.cognitive_integration.store_cognitive_pattern(
                name="conclusion_pattern", 
                scheme_expression="(conclusion (derived ?d) (from ?premises))",
                metadata={"reasoning_role": "conclusion"}
            )
            
            # Perform reasoning
            reasoning_result = await self.cognitive_integration.reason_with_patterns(
                pattern_ids=[premise_pattern.id, conclusion_pattern.id],
                reasoning_type="forward_chaining"
            )
            
            print(f"    • Reasoning completed: {'session_id' in reasoning_result}")
            print(f"    • Has pattern context: {'pattern_context' in reasoning_result}")
            
            return "session_id" in reasoning_result
            
        except Exception as e:
            print(f"    Error in cognitive reasoning test: {e}")
            return False
    
    async def test_enhanced_memory_integration(self) -> bool:
        """Test enhanced memory with cognitive patterns"""
        try:
            # Store memory with cognitive pattern
            memory_content = "Agent learned to optimize pathfinding using A* algorithm"
            scheme_pattern = "(learn (algorithm a_star) (domain pathfinding) (result optimization))"
            
            memory_result = await self.enhanced_memory.store_cognitive_memory(
                content=memory_content,
                scheme_pattern=scheme_pattern,
                context_concepts=["pathfinding", "optimization", "algorithms"],
                memory_type="learning"
            )
            
            print(f"    • Memory stored with node ID: {memory_result['node_id'][:8]}...")
            print(f"    • Has cognitive pattern: {'cognitive_pattern_id' in memory_result}")
            
            # Test cognitive recall
            recall_result = await self.enhanced_memory.cognitive_recall(
                "pathfinding optimization",
                use_scheme_reasoning=True
            )
            
            print(f"    • Cognitive recall available: {recall_result['cognitive_integration_available']}")
            print(f"    • Found cognitive results: {len(recall_result['cognitive_results'])}")
            
            return (memory_result['node_id'] is not None and 
                   recall_result['cognitive_integration_available'])
            
        except Exception as e:
            print(f"    Error in enhanced memory test: {e}")
            return False
    
    async def test_distributed_api_readiness(self) -> bool:
        """Test that cognitive components are ready for distributed API"""
        try:
            # Test statistics generation
            stats = await self.cognitive_integration.get_cognitive_statistics()
            
            print(f"    • Cognitive patterns: {stats['cognitive_patterns']}")
            print(f"    • Grammar statistics available: {'grammar_statistics' in stats}")
            print(f"    • PLN statistics available: {'pln_statistics' in stats}")
            
            # Test export/import functionality
            export_data = await self.cognitive_integration.export_cognitive_knowledge()
            
            print(f"    • Export data has patterns: {'cognitive_patterns' in export_data}")
            print(f"    • Export data has grammars: {'grammars' in export_data}")
            
            # Test import (simulate by re-importing same data)
            import_success = await self.cognitive_integration.import_cognitive_knowledge(export_data)
            
            print(f"    • Import successful: {import_success}")
            
            return (stats['cognitive_patterns'] >= 0 and 
                   'cognitive_patterns' in export_data and 
                   import_success)
            
        except Exception as e:
            print(f"    Error in distributed API test: {e}")
            return False
    
    async def test_hypergraph_traversal_with_schemes(self) -> bool:
        """Test hypergraph traversal finds cognitive patterns"""
        try:
            # Create a connected set of cognitive patterns
            pattern1 = await self.cognitive_integration.store_cognitive_pattern(
                name="perception_input",
                scheme_expression="(perceive (input ?i) (process ?p))"
            )
            
            pattern2 = await self.cognitive_integration.store_cognitive_pattern(
                name="reasoning_process", 
                scheme_expression="(reason (input ?i) (knowledge ?k) (output ?o))"
            )
            
            pattern3 = await self.cognitive_integration.store_cognitive_pattern(
                name="action_output",
                scheme_expression="(act (decision ?d) (action ?a) (result ?r))"
            )
            
            # Test pattern matcher can find these patterns
            all_patterns = [pattern1, pattern2, pattern3]
            pattern_atoms = []
            for pattern in all_patterns:
                pattern_atoms.extend(pattern.atom_ids)
            
            # Use pattern matcher to traverse from first pattern
            if pattern_atoms:
                traversal = await self.cognitive_integration.pattern_matcher.traverse_hypergraph(
                    start_atom_id=pattern_atoms[0],
                    max_depth=3,
                    max_nodes=50
                )
                
                print(f"    • Traversed {len(traversal)} atoms")
                print(f"    • Patterns connected in hypergraph: {len(traversal) > len(pattern1.atom_ids)}")
                
                return len(traversal) > 0
            
            return False
            
        except Exception as e:
            print(f"    Error in hypergraph traversal test: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all cognitive AtomSpace tests"""
        print("🚀 Starting Scheme-based Cognitive AtomSpace Test Suite")
        print("=" * 80)
        
        await self.setup_test_environment()
        
        # Test suite
        test_functions = [
            ("Scheme Grammar Integration", self.test_scheme_grammar_integration),
            ("Cognitive Pattern Storage", self.test_cognitive_pattern_storage),
            ("Cognitive Pattern Evaluation", self.test_cognitive_pattern_evaluation),
            ("Cognitive Pattern Search", self.test_cognitive_pattern_search),
            ("Cognitive Reasoning", self.test_cognitive_reasoning),
            ("Enhanced Memory Integration", self.test_enhanced_memory_integration),
            ("Distributed API Readiness", self.test_distributed_api_readiness),
            ("Hypergraph Traversal with Schemes", self.test_hypergraph_traversal_with_schemes),
        ]
        
        print("\n🧠 Testing Scheme-based Cognitive Representation...")
        for test_name, test_func in test_functions:
            await self.run_test(test_name, test_func)
        
        await self.cleanup_test_environment()
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success rate: {(self.passed_tests / self.total_tests * 100):.1f}%")
        
        print("\nDetailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == "PASSED" else "❌"
            print(f"  {status_icon} {test_name}: {result}")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! Scheme-based Cognitive AtomSpace is working correctly.")
            return True
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} tests failed. Please check the implementation.")
            return False


async def main():
    """Main test execution"""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test suite
    test_suite = CognitiveAtomSpaceTestSuite()
    success = await test_suite.run_all_tests()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)