"""
ECAN (Economic Attention Allocation Network) Demonstration

This script demonstrates the autonomous meta-cognitive control capabilities
of the ECAN system, including attention allocation, self-modification, and
live system introspection with real hypergraph data.
"""

import asyncio
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.helpers.atomspace import AtomSpace
from python.helpers.neural_symbolic_reasoning import NeuralSymbolicReasoningEngine
from python.helpers.ecan_attention import ECANAttentionSystem, ResourceType, MetaCognitiveState


class ECANDemo:
    """Demonstration of ECAN autonomous meta-cognitive control"""
    
    def __init__(self):
        self.demo_db_path = "/tmp/demo_ecan.db"
        self.atomspace = None
        self.reasoning_engine = None
        self.ecan_system = None
        
    async def setup_demo_environment(self):
        """Setup comprehensive demo environment"""
        print("🚀 ECAN Autonomous Meta-Cognitive Control Demo")
        print("=" * 60)
        print("🔧 Setting up demonstration environment...")
        
        # Create fresh AtomSpace
        if os.path.exists(self.demo_db_path):
            os.remove(self.demo_db_path)
        
        self.atomspace = AtomSpace(self.demo_db_path)
        
        # Initialize reasoning engine with ECAN
        self.reasoning_engine = NeuralSymbolicReasoningEngine(self.atomspace)
        await self.reasoning_engine.initialize_system()
        
        # Get ECAN system reference
        self.ecan_system = self.reasoning_engine.ecan_system
        
        # Create rich knowledge base
        await self._create_rich_knowledge_base()
        
        print("✅ Demo environment ready!")
        
    async def _create_rich_knowledge_base(self):
        """Create a rich knowledge base for realistic demonstration"""
        print("📚 Creating rich knowledge base...")
        
        # Cognitive science concepts
        cognitive_concepts = [
            ("attention", "cognitive_process", 0.95, 0.9, {"priority": "critical", "domain": "cognition"}),
            ("memory", "cognitive_process", 0.9, 0.85, {"priority": "high", "domain": "cognition"}),
            ("perception", "cognitive_process", 0.85, 0.8, {"priority": "high", "domain": "cognition"}),
            ("reasoning", "cognitive_process", 0.9, 0.85, {"priority": "high", "domain": "cognition"}),
            ("decision_making", "cognitive_process", 0.8, 0.75, {"priority": "medium", "domain": "cognition"}),
            ("self_awareness", "meta_cognitive", 0.7, 0.6, {"priority": "high", "domain": "meta_cognition"}),
            ("self_monitoring", "meta_cognitive", 0.75, 0.7, {"priority": "high", "domain": "meta_cognition"}),
            ("self_modification", "meta_cognitive", 0.6, 0.5, {"priority": "critical", "domain": "meta_cognition"}),
        ]
        
        # AI and autonomy concepts
        ai_concepts = [
            ("artificial_intelligence", "technology", 0.9, 0.8, {"priority": "high", "domain": "ai"}),
            ("machine_learning", "technology", 0.85, 0.9, {"priority": "high", "domain": "ai"}),
            ("neural_networks", "technology", 0.8, 0.85, {"priority": "medium", "domain": "ai"}),
            ("autonomy", "property", 0.85, 0.8, {"priority": "critical", "domain": "ai"}),
            ("adaptation", "behavior", 0.8, 0.75, {"priority": "high", "domain": "ai"}),
            ("learning", "behavior", 0.9, 0.85, {"priority": "high", "domain": "ai"}),
        ]
        
        # Resource and economics concepts
        resource_concepts = [
            ("computational_resources", "resource", 0.8, 0.7, {"priority": "medium", "domain": "system"}),
            ("memory_bandwidth", "resource", 0.75, 0.7, {"priority": "medium", "domain": "system"}),
            ("processing_power", "resource", 0.8, 0.75, {"priority": "medium", "domain": "system"}),
            ("attention_economy", "concept", 0.7, 0.6, {"priority": "high", "domain": "economics"}),
            ("resource_allocation", "process", 0.75, 0.7, {"priority": "high", "domain": "economics"}),
            ("optimization", "process", 0.8, 0.75, {"priority": "medium", "domain": "mathematics"}),
        ]
        
        all_concepts = cognitive_concepts + ai_concepts + resource_concepts
        created_nodes = {}
        
        for name, concept_type, tv, conf, meta in all_concepts:
            node = await self.atomspace.add_node(
                name=name,
                concept_type=concept_type,
                truth_value=tv,
                confidence=conf,
                metadata=meta
            )
            created_nodes[name] = node
        
        # Create meaningful relationships
        relationships = [
            # Cognitive relationships
            ("attention", "memory", "interacts_with", 0.8, 0.75),
            ("attention", "perception", "guides", 0.85, 0.8),
            ("reasoning", "decision_making", "enables", 0.9, 0.85),
            ("self_awareness", "self_monitoring", "requires", 0.8, 0.75),
            ("self_monitoring", "self_modification", "enables", 0.75, 0.7),
            
            # AI relationships
            ("machine_learning", "artificial_intelligence", "is_part_of", 0.9, 0.85),
            ("neural_networks", "machine_learning", "implements", 0.85, 0.8),
            ("autonomy", "self_modification", "requires", 0.8, 0.75),
            ("adaptation", "learning", "involves", 0.85, 0.8),
            
            # Resource relationships
            ("attention_economy", "resource_allocation", "governs", 0.8, 0.75),
            ("computational_resources", "processing_power", "includes", 0.9, 0.85),
            ("optimization", "resource_allocation", "improves", 0.85, 0.8),
            
            # Cross-domain relationships
            ("attention", "attention_economy", "is_governed_by", 0.7, 0.65),
            ("autonomy", "attention", "depends_on", 0.8, 0.75),
            ("self_modification", "adaptation", "enables", 0.75, 0.7),
        ]
        
        for source, target, relation, tv, conf in relationships:
            if source in created_nodes and target in created_nodes:
                await self.atomspace.add_link(
                    name=f"{source}_{relation}_{target}",
                    outgoing=[created_nodes[source].id, created_nodes[target].id],
                    link_type=relation,
                    truth_value=tv,
                    confidence=conf
                )
        
        print(f"✅ Created rich knowledge base with {len(created_nodes)} concepts and {len(relationships)} relationships")
        
    async def demonstrate_autonomy_tensor(self):
        """Demonstrate autonomy metrics tensor T_auto[a_levels, r_types, m_states]"""
        print("\n🧮 Demonstrating Autonomy Metrics Tensor")
        print("-" * 40)
        
        tensor = self.ecan_system.get_autonomy_tensor()
        
        print(f"Tensor shape: {tensor.shape}")
        print(f"Dimensions: [{tensor.shape[0]} autonomy levels, {tensor.shape[1]} resource types, {tensor.shape[2]} meta-states]")
        print(f"Tensor statistics:")
        print(f"  Mean: {tensor.mean():.4f}")
        print(f"  Std:  {tensor.std():.4f}")
        print(f"  Min:  {tensor.min():.4f}")
        print(f"  Max:  {tensor.max():.4f}")
        
        # Show non-zero entries
        non_zero = np.nonzero(tensor)
        if len(non_zero[0]) > 0:
            print(f"Non-zero entries: {len(non_zero[0])}")
            for i in range(min(5, len(non_zero[0]))):
                level, resource, state = non_zero[0][i], non_zero[1][i], non_zero[2][i]
                value = tensor[level, resource, state]
                print(f"  T_auto[{level},{resource},{state}] = {value:.4f}")
        
        return tensor
        
    async def demonstrate_attention_allocation(self):
        """Demonstrate distributed attention allocation under varying conditions"""
        print("\n🎯 Demonstrating Attention Allocation")
        print("-" * 40)
        
        # Get some atoms for attention allocation
        atoms = await self.atomspace.pattern_match({}, limit=10)
        
        if not atoms:
            print("❌ No atoms available for attention allocation")
            return
        
        print(f"Found {len(atoms)} atoms for attention allocation")
        
        # Scenario 1: High priority critical concepts
        print("\n📊 Scenario 1: High priority allocation")
        critical_atoms = [atom for atom in atoms if atom.metadata.get("priority") == "critical"]
        
        for i, atom in enumerate(critical_atoms[:3]):
            priority = 0.9 - i * 0.1
            success = await self.ecan_system.allocate_attention(
                atom.id, priority, f"critical_system_{i}"
            )
            print(f"  Atom '{atom.name}': priority={priority:.1f}, success={success}")
        
        # Scenario 2: Adaptive allocation based on domain relevance
        print("\n🔄 Scenario 2: Adaptive allocation by domain")
        
        # Create task salience based on domain priorities
        domain_salience = {
            "meta_cognition": 0.9,
            "cognition": 0.8,
            "ai": 0.7,
            "system": 0.6,
            "economics": 0.5
        }
        
        task_salience = {}
        for atom in atoms:
            domain = atom.metadata.get("domain", "unknown")
            if domain in domain_salience:
                task_salience[atom.id] = domain_salience[domain]
        
        # Simulate varying resource load
        resource_load = {"cpu": 0.4, "memory": 0.3, "bandwidth": 0.2}
        
        allocation_result = await self.ecan_system.adaptive_attention_allocation(
            task_salience, resource_load
        )
        
        print(f"  Task salience: {len(task_salience)} tasks")
        print(f"  Resource load: {resource_load}")
        print(f"  Allocation result: {len(allocation_result)} allocations")
        
        # Show top allocations
        sorted_allocations = sorted(allocation_result.items(), key=lambda x: x[1], reverse=True)
        for atom_id, allocation in sorted_allocations[:5]:
            atom = next((a for a in atoms if a.id == atom_id), None)
            if atom:
                print(f"    '{atom.name}': {allocation:.4f}")
                
    async def demonstrate_self_modification(self):
        """Demonstrate real self-modification events"""
        print("\n🔧 Demonstrating Self-Modification")
        print("-" * 40)
        
        # Record initial state
        initial_threshold = self.ecan_system.attention_threshold
        initial_memory = self.ecan_system.resource_pools[ResourceType.MEMORY].total_capacity
        
        print(f"Initial state:")
        print(f"  Attention threshold: {initial_threshold}")
        print(f"  Memory capacity: {initial_memory}")
        
        # Modification 1: Adaptive threshold adjustment
        print("\n🎚️ Modification 1: Attention threshold adaptation")
        new_threshold = initial_threshold * 0.8
        success = await self.ecan_system.self_modify(
            "attention_threshold",
            {"new_threshold": new_threshold}
        )
        
        current_threshold = self.ecan_system.attention_threshold
        print(f"  Requested: {new_threshold:.4f}")
        print(f"  Actual: {current_threshold:.4f}")
        print(f"  Success: {success}")
        
        # Modification 2: Resource reallocation
        print("\n💾 Modification 2: Dynamic resource reallocation")
        new_memory = initial_memory * 1.3
        new_processing = 100 * 1.1
        
        success = await self.ecan_system.self_modify(
            "resource_reallocation",
            {
                "reallocation": {
                    "memory": new_memory,
                    "processing": new_processing
                }
            }
        )
        
        current_memory = self.ecan_system.resource_pools[ResourceType.MEMORY].total_capacity
        current_processing = self.ecan_system.resource_pools[ResourceType.PROCESSING].total_capacity
        
        print(f"  Memory: {initial_memory} -> {current_memory}")
        print(f"  Processing: 100 -> {current_processing}")
        print(f"  Success: {success}")
        
        # Modification 3: Decay rate optimization
        print("\n⏱️ Modification 3: Attention decay optimization")
        success = await self.ecan_system.self_modify(
            "decay_rate_adjustment",
            {"new_decay_rate": 0.02}
        )
        
        updated_units = sum(1 for au in self.ecan_system.attention_units.values() 
                          if au.decay_rate == 0.02)
        print(f"  Updated {updated_units} attention units")
        print(f"  Success: {success}")
        
    async def demonstrate_live_introspection(self):
        """Demonstrate live system introspection and metrics"""
        print("\n🔍 Demonstrating Live System Introspection")
        print("-" * 40)
        
        # Generate inspection report
        report = await self.ecan_system.inspect_system()
        
        print(f"Inspection Report: {report.id}")
        print(f"Timestamp: {report.timestamp}")
        print(f"Cognitive State: {report.cognitive_state.value}")
        
        print(f"\nAttention Distribution:")
        attention_items = list(report.attention_distribution.items())[:5]
        for atom_id, attention in attention_items:
            print(f"  {atom_id[:8]}...: {attention:.4f}")
        
        print(f"\nResource Utilization:")
        for resource, utilization in report.resource_utilization.items():
            print(f"  {resource}: {utilization:.2%}")
        
        print(f"\nPerformance Metrics:")
        for metric, value in report.performance_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        if report.anomalies_detected:
            print(f"\nAnomalies Detected:")
            for anomaly in report.anomalies_detected:
                print(f"  ⚠️  {anomaly}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  💡 {rec}")
        
        print(f"\nHypergraph Snapshot:")
        snapshot = report.hypergraph_snapshot
        print(f"  Atoms: {snapshot['atoms_count']}")
        print(f"  Attention units: {snapshot['attention_units']}")
        print(f"  Active kernels: {snapshot['active_kernels']}")
        print(f"  Tensor shape: {snapshot['tensor_shape']}")
        
        return report
        
    async def demonstrate_periodic_inspection(self):
        """Demonstrate periodic self-inspection routine"""
        print("\n🔄 Demonstrating Periodic Self-Inspection")
        print("-" * 40)
        
        initial_reports = len(self.ecan_system.inspection_reports)
        initial_cycles = self.ecan_system.statistics["inspection_cycles"]
        
        print(f"Initial state:")
        print(f"  Reports: {initial_reports}")
        print(f"  Cycles: {initial_cycles}")
        print(f"  Thread active: {self.ecan_system.inspection_thread.is_alive()}")
        
        print("\n⏳ Waiting for periodic inspection cycles...")
        
        # Wait for inspection cycles
        for i in range(3):
            await asyncio.sleep(2)
            current_cycles = self.ecan_system.statistics["inspection_cycles"]
            current_reports = len(self.ecan_system.inspection_reports)
            print(f"  After {(i+1)*2}s: {current_cycles} cycles, {current_reports} reports")
        
        final_reports = len(self.ecan_system.inspection_reports)
        final_cycles = self.ecan_system.statistics["inspection_cycles"]
        
        print(f"\nFinal state:")
        print(f"  Reports: {initial_reports} -> {final_reports}")
        print(f"  Cycles: {initial_cycles} -> {final_cycles}")
        
        if final_reports > initial_reports:
            latest_report = self.ecan_system.inspection_reports[-1]
            print(f"  Latest report: {latest_report.id}")
            print(f"  Cognitive state: {latest_report.cognitive_state.value}")
        
    async def demonstrate_system_statistics(self):
        """Show comprehensive system statistics"""
        print("\n📊 System Statistics")
        print("-" * 40)
        
        stats = self.ecan_system.get_statistics()
        
        print("Core Metrics:")
        core_metrics = [
            "attention_allocations", "resource_allocations", 
            "self_modifications", "inspection_cycles", "adaptation_events"
        ]
        for metric in core_metrics:
            if metric in stats:
                print(f"  {metric}: {stats[metric]}")
        
        print(f"\nSystem State:")
        print(f"  Attention units: {stats['attention_units_count']}")
        print(f"  Active reports: {stats['active_reports']}")
        print(f"  Modification events: {stats['modification_events']}")
        print(f"  Current state: {stats['current_state']}")
        
        print(f"\nResource Pool Status:")
        for resource, pool_stats in stats['resource_pools'].items():
            utilization = pool_stats['utilization']
            total = pool_stats['total_capacity']
            available = pool_stats['available_capacity']
            print(f"  {resource}: {utilization:.1%} used ({available:.1f}/{total:.1f})")
        
    async def run_complete_demo(self):
        """Run the complete ECAN demonstration"""
        try:
            await self.setup_demo_environment()
            
            # Run all demonstrations
            await self.demonstrate_autonomy_tensor()
            await self.demonstrate_attention_allocation()
            await self.demonstrate_self_modification()
            await self.demonstrate_live_introspection()
            await self.demonstrate_periodic_inspection()
            await self.demonstrate_system_statistics()
            
            print("\n🎉 ECAN Demo Complete!")
            print("=" * 60)
            print("Summary: Successfully demonstrated autonomous meta-cognitive control")
            print("Features: Attention allocation, self-modification, live introspection")
            print("Architecture: Integrated with neural-symbolic reasoning engine")
            print("Metrics: Live derivation from hypergraph AtomSpace structures")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup_demo()
    
    async def cleanup_demo(self):
        """Cleanup demo environment"""
        print("\n🧹 Cleaning up demo environment...")
        
        if self.ecan_system:
            self.ecan_system.stop()
        
        if self.reasoning_engine:
            self.reasoning_engine.shutdown()
        
        if os.path.exists(self.demo_db_path):
            os.remove(self.demo_db_path)
        
        print("✅ Demo cleanup complete!")


async def main():
    """Main demo execution"""
    demo = ECANDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())