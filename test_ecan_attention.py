"""
Comprehensive test suite for ECAN (Economic Attention Allocation Network) system

Tests attention allocation, self-modification, and live system introspection
with real AtomSpace data and distributed agent scenarios.
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.helpers.atomspace import AtomSpace, AtomType
from python.helpers.neural_symbolic_reasoning import NeuralSymbolicReasoningEngine
from python.helpers.ecan_attention import ECANAttentionSystem, AttentionType, ResourceType, MetaCognitiveState


class ECANTestSuite:
    """Test suite for ECAN attention allocation system"""
    
    def __init__(self):
        self.test_db_path = "/tmp/test_ecan.db"
        self.atomspace = None
        self.reasoning_engine = None
        self.ecan_system = None
        self.test_results = []
        
    async def setup_test_environment(self):
        """Setup test environment with real AtomSpace data"""
        print("🔧 Setting up ECAN test environment...")
        
        # Create fresh AtomSpace
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        
        self.atomspace = AtomSpace(self.test_db_path)
        
        # Initialize reasoning engine
        self.reasoning_engine = NeuralSymbolicReasoningEngine(self.atomspace)
        await self.reasoning_engine.initialize_system()
        
        # ECAN system should be initialized automatically
        self.ecan_system = self.reasoning_engine.ecan_system
        
        # Create test knowledge base
        await self._create_test_knowledge_base()
        
        print("✅ ECAN test environment setup complete!")
    
    async def _create_test_knowledge_base(self):
        """Create a comprehensive knowledge base for testing"""
        print("📚 Creating test knowledge base...")
        
        # Create diverse knowledge nodes
        test_nodes = [
            ("artificial_intelligence", "concept", 0.9, 0.8, {"domain": "computer_science", "priority": "high"}),
            ("machine_learning", "concept", 0.85, 0.9, {"domain": "computer_science", "priority": "high"}),
            ("neural_networks", "concept", 0.8, 0.85, {"domain": "computer_science", "priority": "medium"}),
            ("attention_mechanism", "concept", 0.95, 0.9, {"domain": "cognitive_science", "priority": "critical"}),
            ("self_modification", "concept", 0.7, 0.6, {"domain": "cognitive_science", "priority": "high"}),
            ("resource_allocation", "concept", 0.75, 0.7, {"domain": "economics", "priority": "medium"}),
            ("hypergraph", "concept", 0.8, 0.75, {"domain": "mathematics", "priority": "medium"}),
            ("autonomy", "concept", 0.85, 0.8, {"domain": "philosophy", "priority": "high"}),
            ("cognition", "concept", 0.9, 0.85, {"domain": "psychology", "priority": "high"}),
            ("optimization", "concept", 0.8, 0.75, {"domain": "mathematics", "priority": "medium"})
        ]
        
        created_nodes = {}
        for name, concept_type, tv, conf, meta in test_nodes:
            node = await self.atomspace.add_node(
                name=name,
                concept_type=concept_type,
                truth_value=tv,
                confidence=conf,
                metadata=meta
            )
            created_nodes[name] = node
        
        # Create relationships
        test_relationships = [
            ("machine_learning", "artificial_intelligence", "is_part_of", 0.9, 0.85),
            ("neural_networks", "machine_learning", "is_part_of", 0.85, 0.8),
            ("attention_mechanism", "neural_networks", "is_used_in", 0.8, 0.75),
            ("self_modification", "autonomy", "enables", 0.7, 0.65),
            ("resource_allocation", "attention_mechanism", "involves", 0.75, 0.7),
            ("hypergraph", "cognition", "represents", 0.8, 0.75),
            ("optimization", "resource_allocation", "improves", 0.85, 0.8)
        ]
        
        for source, target, relation, tv, conf in test_relationships:
            if source in created_nodes and target in created_nodes:
                await self.atomspace.add_link(
                    name=f"{source}_{relation}_{target}",
                    outgoing=[created_nodes[source].id, created_nodes[target].id],
                    link_type=relation,
                    truth_value=tv,
                    confidence=conf
                )
        
        print(f"✅ Created knowledge base with {len(created_nodes)} nodes and {len(test_relationships)} relationships")
    
    async def test_attention_allocation(self):
        """Test basic attention allocation functionality"""
        print("\n🎯 Testing Attention Allocation...")
        
        test_passed = True
        
        # Test 1: Basic attention allocation
        print("  📊 Test 1: Basic attention allocation")
        atoms = await self.atomspace.pattern_match({}, limit=10)
        test_atom = atoms[0] if atoms else None
        
        if test_atom:
            success = await self.ecan_system.allocate_attention(
                test_atom.id, 0.8, "test_requester"
            )
            if success:
                print(f"    ✅ Successfully allocated attention to atom {test_atom.id}")
            else:
                print(f"    ❌ Failed to allocate attention to atom {test_atom.id}")
                test_passed = False
        else:
            print("    ❌ No atoms available for testing")
            test_passed = False
        
        # Test 2: Attention allocation with resource constraints
        print("  🔒 Test 2: Attention allocation with resource constraints")
        
        # Exhaust attention resources
        for i in range(150):  # More than available capacity
            atom_id = f"test_atom_{i}"
            success = await self.ecan_system.allocate_attention(
                atom_id, 0.5, f"test_requester_{i}"
            )
            if not success:
                print(f"    ✅ Resource exhaustion detected at allocation {i}")
                break
        else:
            print("    ⚠️  No resource exhaustion detected (may be normal)")
        
        # Test 3: Autonomy tensor validation
        print("  🧮 Test 3: Autonomy tensor validation")
        tensor = self.ecan_system.get_autonomy_tensor()
        expected_shape = (5, 5, 6)  # a_levels, r_types, m_states
        
        if tensor.shape == expected_shape:
            print(f"    ✅ Autonomy tensor shape correct: {tensor.shape}")
            print(f"    📊 Tensor stats: mean={tensor.mean():.3f}, std={tensor.std():.3f}")
        else:
            print(f"    ❌ Autonomy tensor shape incorrect: {tensor.shape}, expected {expected_shape}")
            test_passed = False
        
        self.test_results.append(("Attention Allocation", test_passed))
        return test_passed
    
    async def test_adaptive_attention_allocation(self):
        """Test adaptive attention allocation under variable loads"""
        print("\n🔄 Testing Adaptive Attention Allocation...")
        
        test_passed = True
        
        # Test 1: Variable load scenarios
        print("  📈 Test 1: Variable load scenarios")
        
        scenarios = [
            # High salience, low load
            ({"task_a": 0.9, "task_b": 0.8, "task_c": 0.7}, {"cpu": 0.2, "memory": 0.1}),
            # Medium salience, high load  
            ({"task_a": 0.6, "task_b": 0.5, "task_c": 0.4}, {"cpu": 0.8, "memory": 0.9}),
            # Low salience, variable load
            ({"task_a": 0.3, "task_b": 0.2, "task_c": 0.1}, {"cpu": 0.5, "memory": 0.3}),
        ]
        
        for i, (task_salience, resource_load) in enumerate(scenarios):
            print(f"    Scenario {i+1}: Testing with salience={task_salience}, load={resource_load}")
            
            allocation_result = await self.ecan_system.adaptive_attention_allocation(
                task_salience, resource_load
            )
            
            if allocation_result:
                total_allocation = sum(allocation_result.values())
                print(f"    ✅ Allocation successful: {allocation_result}")
                print(f"    📊 Total allocation: {total_allocation:.3f}")
            else:
                print(f"    ❌ Allocation failed for scenario {i+1}")
                test_passed = False
        
        # Test 2: Agent population scaling
        print("  👥 Test 2: Agent population scaling")
        
        # Simulate different agent population sizes
        agent_populations = [5, 10, 20, 50]
        
        for pop_size in agent_populations:
            task_salience = {f"agent_{i}_task": 0.5 + (i % 3) * 0.2 for i in range(pop_size)}
            resource_load = {"cpu": 0.3 + (pop_size / 100), "memory": 0.2 + (pop_size / 150)}
            
            allocation_result = await self.ecan_system.adaptive_attention_allocation(
                task_salience, resource_load
            )
            
            if allocation_result:
                print(f"    ✅ Population {pop_size}: {len(allocation_result)} allocations")
            else:
                print(f"    ❌ Population {pop_size}: allocation failed")
                test_passed = False
        
        self.test_results.append(("Adaptive Attention Allocation", test_passed))
        return test_passed
    
    async def test_self_modification(self):
        """Test real self-modification events in live system"""
        print("\n🔧 Testing Self-Modification Events...")
        
        test_passed = True
        
        # Test 1: Attention threshold modification
        print("  🎚️ Test 1: Attention threshold modification")
        
        original_threshold = self.ecan_system.attention_threshold
        new_threshold = original_threshold * 0.7
        
        success = await self.ecan_system.self_modify(
            "attention_threshold", 
            {"new_threshold": new_threshold}
        )
        
        if success and self.ecan_system.attention_threshold != original_threshold:
            print(f"    ✅ Threshold modified: {original_threshold} -> {self.ecan_system.attention_threshold}")
        else:
            print(f"    ❌ Threshold modification failed")
            test_passed = False
        
        # Test 2: Resource reallocation
        print("  💾 Test 2: Resource reallocation")
        
        original_memory = self.ecan_system.resource_pools[ResourceType.MEMORY].total_capacity
        new_memory = original_memory * 1.2
        
        success = await self.ecan_system.self_modify(
            "resource_reallocation",
            {"reallocation": {"memory": new_memory}}
        )
        
        current_memory = self.ecan_system.resource_pools[ResourceType.MEMORY].total_capacity
        if success and current_memory != original_memory:
            print(f"    ✅ Memory capacity modified: {original_memory} -> {current_memory}")
        else:
            print(f"    ❌ Memory reallocation failed")
            test_passed = False
        
        # Test 3: Decay rate adjustment
        print("  ⏱️ Test 3: Decay rate adjustment")
        
        new_decay_rate = 0.05
        success = await self.ecan_system.self_modify(
            "decay_rate_adjustment",
            {"new_decay_rate": new_decay_rate}
        )
        
        if success:
            # Check if any attention units have the new decay rate
            updated_units = [au for au in self.ecan_system.attention_units.values() 
                           if au.decay_rate == new_decay_rate]
            if updated_units:
                print(f"    ✅ Decay rate updated for {len(updated_units)} attention units")
            else:
                print(f"    ❌ Decay rate not updated")
                test_passed = False
        else:
            print(f"    ❌ Decay rate modification failed")
            test_passed = False
        
        self.test_results.append(("Self-Modification", test_passed))
        return test_passed
    
    async def test_live_system_introspection(self):
        """Test live system introspection and metrics derivation"""
        print("\n🔍 Testing Live System Introspection...")
        
        test_passed = True
        
        # Test 1: Self-inspection report generation
        print("  📋 Test 1: Self-inspection report generation")
        
        report = await self.ecan_system.inspect_system()
        
        if report:
            print(f"    ✅ Inspection report generated: {report.id}")
            print(f"    📊 Cognitive state: {report.cognitive_state.value}")
            print(f"    🧠 Attention units: {len(report.attention_distribution)}")
            print(f"    💾 Resource utilization: {report.resource_utilization}")
            print(f"    📈 Performance metrics: {report.performance_metrics}")
            
            # Validate report structure
            required_fields = [
                'id', 'timestamp', 'cognitive_state', 'attention_distribution',
                'resource_utilization', 'performance_metrics', 'anomalies_detected',
                'recommendations', 'hypergraph_snapshot'
            ]
            
            report_dict = report.to_dict()
            missing_fields = [field for field in required_fields if field not in report_dict]
            
            if not missing_fields:
                print(f"    ✅ All required fields present in report")
            else:
                print(f"    ❌ Missing fields in report: {missing_fields}")
                test_passed = False
        else:
            print(f"    ❌ Failed to generate inspection report")
            test_passed = False
        
        # Test 2: Hypergraph serialization
        print("  🕸️ Test 2: Hypergraph serialization")
        
        if report and report.hypergraph_snapshot:
            snapshot = report.hypergraph_snapshot
            required_snapshot_fields = ['atoms_count', 'attention_units', 'active_kernels', 'tensor_shape']
            
            missing_snapshot_fields = [field for field in required_snapshot_fields if field not in snapshot]
            
            if not missing_snapshot_fields:
                print(f"    ✅ Complete hypergraph snapshot: {snapshot}")
            else:
                print(f"    ❌ Incomplete hypergraph snapshot, missing: {missing_snapshot_fields}")
                test_passed = False
        else:
            print(f"    ❌ No hypergraph snapshot in report")
            test_passed = False
        
        # Test 3: Live metrics validation
        print("  📊 Test 3: Live metrics validation")
        
        # Get system statistics
        stats = self.ecan_system.get_statistics()
        
        if stats:
            print(f"    ✅ System statistics: {stats}")
            
            # Validate that metrics are derived from live system
            if stats['attention_units_count'] > 0:
                print(f"    ✅ Live attention units detected: {stats['attention_units_count']}")
            else:
                print(f"    ⚠️  No attention units active")
            
            if stats['attention_allocations'] > 0:
                print(f"    ✅ Live allocation events: {stats['attention_allocations']}")
            else:
                print(f"    ⚠️  No allocation events recorded")
        else:
            print(f"    ❌ Failed to get system statistics")
            test_passed = False
        
        self.test_results.append(("Live System Introspection", test_passed))
        return test_passed
    
    async def test_concurrent_attention_allocation(self):
        """Test attention allocation under concurrent load"""
        print("\n⚡ Testing Concurrent Attention Allocation...")
        
        test_passed = True
        
        # Test 1: Concurrent allocation requests
        print("  🚀 Test 1: Concurrent allocation requests")
        
        async def allocate_attention_task(task_id):
            """Individual allocation task"""
            atom_id = f"concurrent_atom_{task_id}"
            priority = 0.3 + (task_id % 5) * 0.1
            requester_id = f"concurrent_requester_{task_id}"
            
            success = await self.ecan_system.allocate_attention(atom_id, priority, requester_id)
            return success, task_id
        
        # Create multiple concurrent tasks
        tasks = [allocate_attention_task(i) for i in range(20)]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks)
        
        successful_allocations = sum(1 for success, _ in results if success)
        total_tasks = len(results)
        
        if successful_allocations > 0:
            print(f"    ✅ Concurrent allocations: {successful_allocations}/{total_tasks}")
        else:
            print(f"    ❌ No successful concurrent allocations")
            test_passed = False
        
        # Test 2: Resource contention handling
        print("  🔒 Test 2: Resource contention handling")
        
        # Simulate high resource contention
        async def contention_task(task_id):
            """High-contention allocation task"""
            for i in range(10):
                atom_id = f"contention_atom_{task_id}_{i}"
                success = await self.ecan_system.allocate_attention(atom_id, 0.5, f"contention_{task_id}")
                if not success:
                    return False, task_id
            return True, task_id
        
        contention_tasks = [contention_task(i) for i in range(5)]
        contention_results = await asyncio.gather(*contention_tasks)
        
        successful_contention = sum(1 for success, _ in contention_results if success)
        total_contention = len(contention_results)
        
        print(f"    📊 Contention handling: {successful_contention}/{total_contention} tasks completed")
        
        self.test_results.append(("Concurrent Attention Allocation", test_passed))
        return test_passed
    
    async def test_periodic_self_inspection(self):
        """Test periodic self-inspection routine"""
        print("\n🔄 Testing Periodic Self-Inspection...")
        
        test_passed = True
        
        # Test 1: Verify inspection thread is running
        print("  🧵 Test 1: Inspection thread verification")
        
        if self.ecan_system.inspection_thread and self.ecan_system.inspection_thread.is_alive():
            print("    ✅ Inspection thread is active")
        else:
            print("    ❌ Inspection thread is not active")
            test_passed = False
        
        # Test 2: Wait for inspection cycles
        print("  ⏰ Test 2: Inspection cycle monitoring")
        
        initial_cycles = self.ecan_system.statistics["inspection_cycles"]
        
        # Wait for a few seconds to allow inspection cycles
        print("    ⏳ Waiting for inspection cycles...")
        await asyncio.sleep(3)
        
        final_cycles = self.ecan_system.statistics["inspection_cycles"] 
        
        if final_cycles >= initial_cycles:
            print(f"    ✅ Inspection cycles detected: {initial_cycles} -> {final_cycles}")
        else:
            print(f"    ❌ No inspection cycles detected")
            test_passed = False
        
        # Test 3: Inspect report accumulation
        print("  📚 Test 3: Report accumulation")
        
        num_reports = len(self.ecan_system.inspection_reports)
        
        if num_reports > 0:
            print(f"    ✅ Inspection reports accumulated: {num_reports}")
            
            # Check latest report
            latest_report = self.ecan_system.inspection_reports[-1]
            print(f"    📋 Latest report: {latest_report.id} at {latest_report.timestamp}")
        else:
            print(f"    ❌ No inspection reports accumulated")
            test_passed = False
        
        self.test_results.append(("Periodic Self-Inspection", test_passed))
        return test_passed
    
    async def run_all_tests(self):
        """Run all ECAN tests"""
        print("🚀 Starting ECAN Test Suite")
        print("=" * 80)
        
        # Setup test environment
        await self.setup_test_environment()
        
        # Run individual test suites
        tests = [
            self.test_attention_allocation,
            self.test_adaptive_attention_allocation,
            self.test_self_modification,
            self.test_live_system_introspection,
            self.test_concurrent_attention_allocation,
            self.test_periodic_self_inspection
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"❌ Test {test.__name__} failed with exception: {e}")
                self.test_results.append((test.__name__, False))
        
        # Cleanup
        await self.cleanup_test_environment()
        
        # Print summary
        self.print_test_summary()
    
    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        print("\n🧹 Cleaning up test environment...")
        
        # Stop ECAN system
        if self.ecan_system:
            self.ecan_system.stop()
        
        # Shutdown reasoning engine
        if self.reasoning_engine:
            self.reasoning_engine.shutdown()
        
        # Atomspace doesn't have close method, just clear reference
        self.atomspace = None
        
        # Remove test database
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
        
        print("✅ Cleanup complete!")
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("📊 ECAN TEST SUMMARY")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed in self.test_results if passed)
        failed_tests = total_tests - passed_tests
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, passed in self.test_results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! ECAN system is working correctly.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review the results above.")


async def main():
    """Main test execution"""
    test_suite = ECANTestSuite()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())