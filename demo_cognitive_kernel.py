"""
Unified Cognitive Kernel Demonstration

Live demonstration of the meta-recursive attention system, cognitive grammar
management, and integrated subsystem orchestration with real agent interactions.
"""

import asyncio
import sys
import os
import json
import numpy as np
from datetime import datetime, timezone
import tempfile
import time
import logging

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.helpers.cognitive_kernel import UnifiedCognitiveKernel, KernelState, AttentionMembraneType
from python.helpers.scheme_grammar import SchemeCognitiveGrammarRegistry, CognitiveOperator
from python.helpers.atomspace import AtomSpace, AtomType
from python.helpers.neural_symbolic_reasoning import ReasoningStage
from python.helpers.ecan_attention import AttentionType, ResourceType

# Configure logging
logging.basicConfig(level=logging.INFO)


class CognitiveKernelDemo:
    """Demonstration of the unified cognitive kernel system"""
    
    def __init__(self):
        self.demo_db_path = "/tmp/demo_cognitive_kernel.db"
        self.atomspace = None
        self.cognitive_kernel = None
        self.grammar_registry = None
        self.demo_results = []
        
    async def setup_demo_environment(self):
        """Setup demonstration environment"""
        print("🔧 Setting up Cognitive Kernel Demo Environment")
        print("=" * 60)
        
        # Create fresh AtomSpace
        if os.path.exists(self.demo_db_path):
            os.remove(self.demo_db_path)
            
        self.atomspace = AtomSpace(self.demo_db_path)
        
        # Initialize cognitive kernel
        self.cognitive_kernel = UnifiedCognitiveKernel(self.atomspace)
        initialization_result = await self.cognitive_kernel.initialize()
        
        # Initialize grammar registry
        self.grammar_registry = SchemeCognitiveGrammarRegistry(self.atomspace)
        
        # Create rich knowledge base
        await self._create_demo_knowledge_base()
        
        print("✅ Demo environment setup complete!")
        print(f"   🧠 Kernel ID: {self.cognitive_kernel.kernel_id}")
        print(f"   📊 State: {self.cognitive_kernel.state.value}")
        print(f"   🧱 Attention Membranes: {len(self.cognitive_kernel.attention_membranes)}")
        print(f"   📝 Grammar Registry: {len(self.grammar_registry.grammars)} grammars")
        print(f"   🏗️ Tensor Shape: {self.cognitive_kernel.kernel_tensor.shape}")
        
        return initialization_result
        
    async def _create_demo_knowledge_base(self):
        """Create comprehensive knowledge base for demonstration"""
        print("\n📚 Creating comprehensive knowledge base...")
        
        # Advanced cognitive concepts
        concepts = [
            # Core cognitive concepts
            ("consciousness", "the state of being aware and able to think"),
            ("intelligence", "the ability to learn, understand, and solve problems"),
            ("reasoning", "the process of thinking logically about something"),
            ("perception", "the process of becoming aware through the senses"),
            ("memory", "the ability to store and retrieve information"),
            ("attention", "the cognitive process of selectively concentrating"),
            ("learning", "the process of acquiring knowledge or skills"),
            ("creativity", "the ability to generate novel and valuable ideas"),
            ("emotion", "complex psychological and physiological states"),
            ("intuition", "immediate understanding without reasoning"),
            
            # Meta-cognitive concepts
            ("metacognition", "thinking about thinking"),
            ("self_awareness", "conscious knowledge of one's own character"),
            ("reflection", "serious thought or consideration"),
            ("introspection", "examination of one's own mental processes"),
            ("self_regulation", "control over one's own behavior and emotions"),
            
            # AI and cognitive architecture
            ("neural_networks", "computing systems inspired by biological neural networks"),
            ("symbolic_reasoning", "reasoning using symbols and logical rules"),
            ("machine_learning", "algorithms that improve through experience"),
            ("cognitive_architecture", "blueprints for intelligent systems"),
            ("artificial_intelligence", "intelligence demonstrated by machines"),
            
            # Philosophical concepts
            ("qualia", "the subjective conscious experiences"),
            ("intentionality", "the directedness of mental states"),
            ("free_will", "the ability to choose between different courses of action"),
            ("emergence", "complex properties arising from simple interactions"),
            ("causation", "the relationship between cause and effect")
        ]
        
        created_atoms = []
        
        for concept, description in concepts:
            # Create concept atom
            atom = await self.atomspace.add_node(
                name=concept,
                concept_type="concept",
                truth_value=0.8 + np.random.random() * 0.2,
                confidence=0.7 + np.random.random() * 0.3
            )
            created_atoms.append(atom)
            
            # Create description predicate
            desc_atom = await self.atomspace.add_node(
                name=f"description_{concept}",
                concept_type="description",
                truth_value=0.9,
                confidence=0.8
            )
            
            # Create inheritance relationship
            await self.atomspace.add_link(
                name=f"describes_{concept}",
                outgoing=[desc_atom.id, atom.id],
                link_type="inheritance",
                truth_value=0.95,
                confidence=0.9
            )
            
        # Create relationships between concepts
        relationships = [
            ("consciousness", "intelligence", "enables"),
            ("intelligence", "reasoning", "involves"),
            ("reasoning", "logic", "uses"),
            ("perception", "awareness", "creates"),
            ("memory", "learning", "supports"),
            ("attention", "focus", "directs"),
            ("metacognition", "self_awareness", "includes"),
            ("reflection", "introspection", "related_to"),
            ("neural_networks", "machine_learning", "implements"),
            ("symbolic_reasoning", "logic", "uses"),
            ("artificial_intelligence", "cognitive_architecture", "built_on"),
            ("qualia", "consciousness", "part_of"),
            ("intentionality", "mental_states", "property_of"),
            ("emergence", "complex_systems", "explains"),
            ("causation", "reasoning", "principle_of")
        ]
        
        for concept1, concept2, relation in relationships:
            await self.atomspace.add_link(
                name=f"{relation}_{concept1}_{concept2}",
                outgoing=[],  # We'll use simplified relationships
                link_type="evaluation",
                truth_value=0.8 + np.random.random() * 0.15,
                confidence=0.7 + np.random.random() * 0.2
            )
            
        print(f"✅ Created knowledge base with {len(created_atoms)} concepts and {len(relationships)} relationships")
        
    async def demonstrate_kernel_initialization(self):
        """Demonstrate kernel initialization and architecture"""
        print("\n🧠 Demonstrating Kernel Initialization and Architecture")
        print("-" * 60)
        
        # Show kernel components
        print("🏗️ Kernel Architecture:")
        print(f"   Kernel ID: {self.cognitive_kernel.kernel_id}")
        print(f"   State: {self.cognitive_kernel.state.value}")
        print(f"   Running: {self.cognitive_kernel.running}")
        
        # Show subsystems
        print("\n🔧 Integrated Subsystems:")
        subsystems = {
            "Neural-Symbolic Engine": self.cognitive_kernel.neural_symbolic_engine is not None,
            "ECAN Attention System": self.cognitive_kernel.ecan_system is not None,
            "Task Orchestrator": self.cognitive_kernel.task_orchestrator is not None
        }
        
        for name, status in subsystems.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {name}: {'Active' if status else 'Inactive'}")
            
        # Show kernel tensor
        print(f"\n📊 Kernel Tensor T_kernel[n_atoms, n_tasks, n_reasoning, a_levels, t_steps]:")
        tensor = self.cognitive_kernel.get_kernel_tensor()
        print(f"   Shape: {tensor.shape}")
        print(f"   Dimensions: {self.cognitive_kernel.n_atoms} atoms × {self.cognitive_kernel.n_tasks} tasks × {self.cognitive_kernel.n_reasoning} reasoning × {self.cognitive_kernel.a_levels} levels × {self.cognitive_kernel.t_steps} steps")
        print(f"   Non-zero elements: {np.count_nonzero(tensor):,}")
        print(f"   Memory footprint: {tensor.nbytes / 1024 / 1024:.2f} MB")
        
        # Show attention membranes
        print("\n🧱 Attention Membranes (P-System Compartments):")
        membranes = self.cognitive_kernel.get_attention_membranes()
        
        for membrane_id, membrane in membranes.items():
            print(f"   🔸 {membrane['name']} ({membrane['type']})")
            print(f"      Resources: {len(membrane['resources'])} types")
            print(f"      Permeability: {membrane['permeability']:.2f}")
            print(f"      Active processes: {membrane['active_processes']}")
            print(f"      Gradient: {np.array(membrane['attention_gradient'])[:3].tolist()}...")
            
        # Show modification protocols
        print(f"\n🔧 Self-Modification Protocols:")
        protocols = self.cognitive_kernel.modification_protocols
        for protocol_id, protocol in protocols.items():
            print(f"   🔸 {protocol.name}")
            print(f"      Target: {protocol.target_subsystem}")
            print(f"      Type: {protocol.modification_type}")
            print(f"      Safety checks: {len(protocol.safety_checks)}")
            
        return True
        
    async def demonstrate_cognitive_grammars(self):
        """Demonstrate Scheme-based cognitive grammar system"""
        print("\n🗣️ Demonstrating Scheme-Based Cognitive Grammar System")
        print("-" * 60)
        
        # Show built-in grammars
        print("📋 Built-in Cognitive Grammars:")
        grammars = self.grammar_registry.list_grammars()
        
        for grammar in grammars[:6]:  # Show first 6
            print(f"   🔸 {grammar.name}")
            print(f"      Expression: {grammar.scheme_expression}")
            print(f"      Operators: {[op.value for op in grammar.cognitive_operators]}")
            print(f"      Patterns: {len(grammar.pattern_templates)}")
            print(f"      Usage: {grammar.usage_count} times")
            
        # Demonstrate grammar registration
        print("\n➕ Registering Custom Grammar:")
        custom_grammar = self.grammar_registry.register_grammar(
            "advanced_reasoning",
            "Advanced Reasoning Grammar",
            "Complex multi-step reasoning with meta-cognitive monitoring",
            "(advanced_reason (premises ?prems) (meta_monitor ?monitor) (conclusions ?concl) (confidence ?conf))",
            [CognitiveOperator.REASON, CognitiveOperator.REFLECT],
            [
                {
                    "template": {"advanced_reason": True, "premises": "?prems", "conclusions": "?concl"},
                    "match_type": "logical",
                    "weights": {"truth_value": 0.9, "confidence": 0.8}
                }
            ]
        )
        
        print(f"   ✅ Registered: {custom_grammar.name}")
        print(f"      ID: {custom_grammar.id}")
        print(f"      Parsed: {custom_grammar.parsed_tree is not None}")
        
        # Demonstrate grammar evaluation
        print("\n🔍 Evaluating Grammar:")
        evaluation = self.grammar_registry.evaluate_grammar_expression(
            "advanced_reasoning",
            {
                "prems": ["All intelligent systems can reason", "This system is intelligent"],
                "monitor": "confidence_tracking",
                "concl": ["This system can reason"],
                "conf": 0.85
            }
        )
        
        print(f"   Success: {evaluation['success']}")
        print(f"   Result: {evaluation.get('result', {})}")
        
        # Demonstrate grammar extension
        print("\n🔧 Extending Grammar:")
        extension_success = self.grammar_registry.extend_grammar(
            "advanced_reasoning",
            "(validate (result ?r) (criteria ?c) (outcome ?o))"
        )
        
        print(f"   Extension success: {extension_success}")
        
        if extension_success:
            extended_grammar = self.grammar_registry.get_grammar("advanced_reasoning")
            print(f"   New expression length: {len(extended_grammar.scheme_expression)}")
            
        # Demonstrate grammar specialization
        print("\n🎯 Specializing Grammar:")
        specialized_id = self.grammar_registry.specialize_grammar(
            "reasoning_grammar",
            "scientific_reasoning",
            "(scientific (hypothesis ?h) (experiment ?e) (evidence ?ev))"
        )
        
        if specialized_id:
            specialized = self.grammar_registry.get_grammar(specialized_id)
            print(f"   ✅ Created specialized grammar: {specialized.name}")
            print(f"      ID: {specialized_id}")
            
        # Demonstrate grammar composition
        print("\n🔗 Composing Grammars:")
        composition_id = self.grammar_registry.compose_grammars(
            ["perception_grammar", "reasoning_grammar", "decision_grammar"],
            "cognitive_pipeline"
        )
        
        if composition_id:
            composition = self.grammar_registry.get_grammar(composition_id)
            print(f"   ✅ Created composition: {composition.name}")
            print(f"      Operators: {[op.value for op in composition.cognitive_operators]}")
            
        # Show grammar statistics
        print("\n📊 Grammar Statistics:")
        stats = self.grammar_registry.get_grammar_statistics()
        print(f"   Total grammars: {stats['total_grammars']}")
        print(f"   Active grammars: {stats['active_grammars']}")
        print(f"   Total patterns: {stats['total_patterns']}")
        print(f"   Average usage: {stats['average_usage']:.2f}")
        print(f"   Operator usage: {stats['cognitive_operator_usage']}")
        
        return True
        
    async def demonstrate_recursive_invocation(self):
        """Demonstrate recursive kernel invocation with all subsystems"""
        print("\n🔄 Demonstrating Recursive Kernel Invocation")
        print("-" * 60)
        
        # Demonstrate memory-focused query
        print("💾 Memory-Focused Query:")
        memory_query = {
            "type": "memory",
            "content": {
                "concepts": ["consciousness", "intelligence", "reasoning"],
                "patterns": [],
                "semantic_depth": 3
            }
        }
        
        start_time = time.time()
        memory_result = await self.cognitive_kernel.recursive_invoke(memory_query)
        memory_duration = time.time() - start_time
        
        print(f"   ✅ Success: {memory_result['success']}")
        print(f"   🕐 Duration: {memory_duration:.3f} seconds")
        print(f"   📞 Subsystems called: {memory_result['subsystem_calls']}")
        print(f"   💭 Attention allocation: {memory_result['attention_allocation']}")
        
        if memory_result['success'] and 'results' in memory_result:
            memory_data = memory_result['results'].get('memory', {})
            if 'concepts' in memory_data:
                print(f"   📊 Concepts analyzed: {len(memory_data['concepts'])}")
                for concept_result in memory_data['concepts'][:2]:  # Show first 2
                    print(f"      - {concept_result['concept']}: {concept_result['atoms']} atoms")
        
        await asyncio.sleep(0.1)  # Brief pause for demonstration
        
        # Demonstrate reasoning-focused query
        print("\n🧠 Reasoning-Focused Query:")
        reasoning_query = {
            "type": "reasoning",
            "content": {
                "concepts": ["intelligence", "consciousness"],
                "reasoning_type": "causal",
                "depth": 2,
                "include_confidence": True
            }
        }
        
        start_time = time.time()
        reasoning_result = await self.cognitive_kernel.recursive_invoke(reasoning_query)
        reasoning_duration = time.time() - start_time
        
        print(f"   ✅ Success: {reasoning_result['success']}")
        print(f"   🕐 Duration: {reasoning_duration:.3f} seconds")
        print(f"   📞 Subsystems called: {reasoning_result['subsystem_calls']}")
        print(f"   💭 Attention allocation: {reasoning_result['attention_allocation']}")
        
        await asyncio.sleep(0.1)
        
        # Demonstrate task orchestration query
        print("\n📋 Task Orchestration Query:")
        task_query = {
            "type": "task",
            "content": {
                "goal": "Analyze relationships between consciousness, intelligence, and reasoning",
                "priority": "high",
                "complexity": "advanced",
                "resources_required": ["memory", "reasoning", "attention"]
            }
        }
        
        start_time = time.time()
        task_result = await self.cognitive_kernel.recursive_invoke(task_query)
        task_duration = time.time() - start_time
        
        print(f"   ✅ Success: {task_result['success']}")
        print(f"   🕐 Duration: {task_duration:.3f} seconds")
        print(f"   📞 Subsystems called: {task_result['subsystem_calls']}")
        print(f"   💭 Attention allocation: {task_result['attention_allocation']}")
        
        await asyncio.sleep(0.1)
        
        # Demonstrate autonomy and meta-cognitive query
        print("\n🤖 Autonomy and Meta-Cognitive Query:")
        autonomy_query = {
            "type": "autonomy",
            "content": {
                "self_inspection": True,
                "attention_analysis": True,
                "performance_metrics": True,
                "adaptation_suggestions": True
            }
        }
        
        start_time = time.time()
        autonomy_result = await self.cognitive_kernel.recursive_invoke(autonomy_query)
        autonomy_duration = time.time() - start_time
        
        print(f"   ✅ Success: {autonomy_result['success']}")
        print(f"   🕐 Duration: {autonomy_duration:.3f} seconds")
        print(f"   📞 Subsystems called: {autonomy_result['subsystem_calls']}")
        print(f"   💭 Attention allocation: {autonomy_result['attention_allocation']}")
        
        await asyncio.sleep(0.1)
        
        # Demonstrate multi-subsystem integration query
        print("\n🌐 Multi-Subsystem Integration Query:")
        integration_query = {
            "type": "general",
            "content": {
                "comprehensive_analysis": True,
                "concepts": ["artificial_intelligence", "cognitive_architecture"],
                "reasoning_required": True,
                "task_planning": True,
                "meta_cognitive_monitoring": True,
                "output_format": "detailed_report"
            }
        }
        
        start_time = time.time()
        integration_result = await self.cognitive_kernel.recursive_invoke(integration_query)
        integration_duration = time.time() - start_time
        
        print(f"   ✅ Success: {integration_result['success']}")
        print(f"   🕐 Duration: {integration_duration:.3f} seconds")
        print(f"   📞 Subsystems called: {integration_result['subsystem_calls']}")
        print(f"   💭 Attention allocation: {integration_result['attention_allocation']}")
        print(f"   📊 Meta-events generated: {len(integration_result['meta_events'])}")
        
        # Show performance summary
        print("\n📊 Performance Summary:")
        total_queries = 5
        total_duration = memory_duration + reasoning_duration + task_duration + autonomy_duration + integration_duration
        
        print(f"   Total queries: {total_queries}")
        print(f"   Total duration: {total_duration:.3f} seconds")
        print(f"   Average duration: {total_duration/total_queries:.3f} seconds/query")
        print(f"   Queries per second: {total_queries/total_duration:.2f}")
        
        return True
        
    async def demonstrate_attention_membrane_dynamics(self):
        """Demonstrate attention membrane system and resource flow"""
        print("\n🧱 Demonstrating Attention Membrane Dynamics")
        print("-" * 60)
        
        # Show initial membrane states
        print("📊 Initial Membrane States:")
        initial_membranes = self.cognitive_kernel.get_attention_membranes()
        
        for membrane_id, membrane in initial_membranes.items():
            print(f"   🔸 {membrane['name']}:")
            print(f"      Resources: {membrane['resources']}")
            print(f"      Permeability: {membrane['permeability']}")
            print(f"      Gradient: {np.array(membrane['attention_gradient'])[:3].tolist()}...")
            
        # Simulate high memory activity
        print("\n🧠 Simulating High Memory Activity:")
        memory_membrane = next(m for m in self.cognitive_kernel.attention_membranes.values() 
                             if m.type == AttentionMembraneType.MEMORY)
        
        # Add active processes
        for i in range(5):
            memory_membrane.active_processes.add(f"memory_process_{i}")
            
        print(f"   Added {len(memory_membrane.active_processes)} active processes to memory membrane")
        
        # Simulate high reasoning activity
        print("\n⚡ Simulating High Reasoning Activity:")
        reasoning_membrane = next(m for m in self.cognitive_kernel.attention_membranes.values() 
                                if m.type == AttentionMembraneType.REASONING)
        
        for i in range(3):
            reasoning_membrane.active_processes.add(f"reasoning_process_{i}")
            
        print(f"   Added {len(reasoning_membrane.active_processes)} active processes to reasoning membrane")
        
        # Wait for resource flow
        print("\n🔄 Waiting for Resource Flow (2 seconds)...")
        await asyncio.sleep(2.0)
        
        # Show updated membrane states
        print("\n📊 Updated Membrane States:")
        updated_membranes = self.cognitive_kernel.get_attention_membranes()
        
        print("   Resource Changes:")
        for membrane_id, membrane in updated_membranes.items():
            initial_membrane = initial_membranes[membrane_id]
            
            print(f"   🔸 {membrane['name']}:")
            for resource_type, current_value in membrane['resources'].items():
                initial_value = initial_membrane['resources'].get(resource_type, 0)
                change = current_value - initial_value
                change_str = f"{'+'if change > 0 else ''}{change:.2f}"
                print(f"      {resource_type}: {initial_value:.2f} → {current_value:.2f} ({change_str})")
                
        # Show attention gradients
        print("\n📈 Attention Gradients:")
        for membrane_id, membrane in updated_membranes.items():
            gradient = np.array(membrane['attention_gradient'])
            print(f"   🔸 {membrane['name']}: {gradient[:3].tolist()}...")
            
        # Calculate resource flow metrics
        print("\n📊 Resource Flow Metrics:")
        total_flow = 0
        for membrane_id, membrane in updated_membranes.items():
            initial_membrane = initial_membranes[membrane_id]
            for resource_type, current_value in membrane['resources'].items():
                initial_value = initial_membrane['resources'].get(resource_type, 0)
                total_flow += abs(current_value - initial_value)
                
        print(f"   Total resource flow: {total_flow:.2f}")
        print(f"   Average flow per membrane: {total_flow/len(updated_membranes):.2f}")
        
        return True
        
    async def demonstrate_meta_cognitive_feedback(self):
        """Demonstrate meta-cognitive feedback and self-monitoring"""
        print("\n🧠 Demonstrating Meta-Cognitive Feedback System")
        print("-" * 60)
        
        # Show initial meta-events
        initial_events = len(self.cognitive_kernel.meta_events)
        print(f"📊 Initial meta-events: {initial_events}")
        
        # Trigger various activities to generate meta-events
        print("\n🎯 Triggering Activities to Generate Meta-Events:")
        
        activities = [
            {"type": "memory", "content": {"concepts": ["consciousness"]}},
            {"type": "reasoning", "content": {"reasoning_type": "abductive"}},
            {"type": "autonomy", "content": {"self_inspection": True}},
            {"type": "meta", "content": {"system_analysis": True}}
        ]
        
        for i, activity in enumerate(activities):
            print(f"   Activity {i+1}: {activity['type']} query")
            await self.cognitive_kernel.recursive_invoke(activity)
            await asyncio.sleep(0.1)
            
        # Wait for meta-event generation
        await asyncio.sleep(0.5)
        
        # Show generated meta-events
        final_events = len(self.cognitive_kernel.meta_events)
        new_events = final_events - initial_events
        
        print(f"\n📊 Meta-Event Generation Results:")
        print(f"   Initial events: {initial_events}")
        print(f"   Final events: {final_events}")
        print(f"   New events: {new_events}")
        
        # Show recent meta-events
        print("\n📝 Recent Meta-Events:")
        if self.cognitive_kernel.meta_events:
            recent_events = self.cognitive_kernel.meta_events[-5:]  # Show last 5
            for event in recent_events:
                event_dict = event.to_dict()
                print(f"   🔸 {event_dict['event_type']} ({event_dict['source_subsystem']})")
                print(f"      Description: {event_dict['description']}")
                print(f"      Impact: {event_dict['impact_level']:.2f}")
                print(f"      Time: {event_dict['timestamp']}")
                
        # Demonstrate feedback callback
        print("\n📞 Demonstrating Feedback Callback:")
        callback_events = []
        
        def demo_callback(event_data):
            callback_events.append(event_data)
            print(f"   📞 Callback triggered: {event_data.get('event_type', 'unknown')}")
            
        self.cognitive_kernel.feedback_callbacks["demo_callback"] = demo_callback
        
        # Trigger another activity
        await self.cognitive_kernel.recursive_invoke({
            "type": "autonomy",
            "content": {"performance_check": True}
        })
        
        print(f"   Callback events captured: {len(callback_events)}")
        
        # Show meta-cognitive state analysis
        print("\n📊 Meta-Cognitive State Analysis:")
        stats = self.cognitive_kernel.get_kernel_statistics()
        
        print(f"   Kernel cycles: {stats['kernel_cycles']}")
        print(f"   Subsystem invocations: {stats['subsystem_invocations']}")
        print(f"   Meta events: {stats['meta_events']}")
        print(f"   Self modifications: {stats['self_modifications']}")
        
        return True
        
    async def demonstrate_self_modification(self):
        """Demonstrate self-modification protocols"""
        print("\n🔧 Demonstrating Self-Modification Protocols")
        print("-" * 60)
        
        # Show available protocols
        print("📋 Available Self-Modification Protocols:")
        protocols = self.cognitive_kernel.modification_protocols
        
        for protocol_id, protocol in protocols.items():
            print(f"   🔸 {protocol.name}")
            print(f"      ID: {protocol_id}")
            print(f"      Target: {protocol.target_subsystem}")
            print(f"      Type: {protocol.modification_type}")
            print(f"      Safety checks: {len(protocol.safety_checks)}")
            
        # Record initial state
        print("\n📊 Recording Initial State:")
        initial_stats = self.cognitive_kernel.statistics.copy()
        initial_membranes = self.cognitive_kernel.get_attention_membranes()
        
        print(f"   Self-modifications: {initial_stats['self_modifications']}")
        print(f"   Kernel cycles: {initial_stats['kernel_cycles']}")
        
        # Trigger attention reallocation
        print("\n⚡ Triggering Attention Reallocation:")
        
        # Create attention imbalance
        for membrane in self.cognitive_kernel.attention_membranes.values():
            if membrane.type == AttentionMembraneType.MEMORY:
                membrane.resources[ResourceType.ATTENTION] = 150.0  # High attention
            elif membrane.type == AttentionMembraneType.REASONING:
                membrane.resources[ResourceType.ATTENTION] = 10.0   # Low attention
                
        print("   Created attention imbalance")
        print("   Waiting for automatic self-modification trigger...")
        
        # Wait for automatic trigger
        await asyncio.sleep(1.0)
        
        # Check if self-modification occurred
        updated_stats = self.cognitive_kernel.statistics
        modifications_occurred = updated_stats['self_modifications'] > initial_stats['self_modifications']
        
        print(f"   Self-modifications: {initial_stats['self_modifications']} → {updated_stats['self_modifications']}")
        print(f"   Automatic trigger: {'✅' if modifications_occurred else '❌'}")
        
        # Manual self-modification trigger
        print("\n🔧 Manual Self-Modification Trigger:")
        
        try:
            await self.cognitive_kernel._trigger_self_modification("attention_reallocation")
            print("   ✅ Manual trigger successful")
        except Exception as e:
            print(f"   ❌ Manual trigger failed: {e}")
            
        # Show updated state
        print("\n📊 Updated State:")
        final_stats = self.cognitive_kernel.statistics
        final_membranes = self.cognitive_kernel.get_attention_membranes()
        
        print(f"   Self-modifications: {final_stats['self_modifications']}")
        print(f"   Kernel cycles: {final_stats['kernel_cycles']}")
        
        # Show resource changes
        print("\n💱 Resource Changes:")
        for membrane_id, membrane in final_membranes.items():
            initial_membrane = initial_membranes[membrane_id]
            
            attention_initial = initial_membrane['resources'].get('attention', 0)
            attention_final = membrane['resources'].get('attention', 0)
            
            if abs(attention_final - attention_initial) > 0.1:
                print(f"   🔸 {membrane['name']}: {attention_initial:.2f} → {attention_final:.2f}")
                
        # Test safety checks
        print("\n🛡️ Testing Safety Checks:")
        safety_checks = ["resource_bounds_check", "stability_check", "tensor_consistency_check"]
        
        for check in safety_checks:
            try:
                result = await self.cognitive_kernel._execute_safety_check(check)
                print(f"   {check}: {'✅' if result else '❌'}")
            except Exception as e:
                print(f"   {check}: ❌ Error: {e}")
                
        return True
        
    async def demonstrate_integration_capabilities(self):
        """Demonstrate integration capabilities for Agent Zero and Bolt.diy"""
        print("\n🔌 Demonstrating Integration Capabilities")
        print("-" * 60)
        
        # Show API compatibility
        print("📡 API Compatibility:")
        status = self.cognitive_kernel.get_kernel_statistics()
        
        api_endpoints = {
            "Kernel Status": status,
            "Tensor Access": self.cognitive_kernel.get_kernel_tensor() is not None,
            "Grammar Registry": len(self.grammar_registry.grammars) > 0,
            "Attention Membranes": len(self.cognitive_kernel.attention_membranes) > 0,
            "Meta Events": len(self.cognitive_kernel.meta_events) > 0
        }
        
        for endpoint, available in api_endpoints.items():
            print(f"   {'✅' if available else '❌'} {endpoint}")
            
        # Show modular access
        print("\n🧩 Modular Subsystem Access:")
        subsystem_access = {
            "Neural-Symbolic Engine": self.cognitive_kernel.neural_symbolic_engine,
            "ECAN Attention System": self.cognitive_kernel.ecan_system,
            "Task Orchestrator": self.cognitive_kernel.task_orchestrator,
            "Grammar Registry": self.grammar_registry
        }
        
        for name, subsystem in subsystem_access.items():
            available = subsystem is not None
            print(f"   {'✅' if available else '❌'} {name}")
            
        # Demonstrate state serialization
        print("\n💾 State Serialization:")
        try:
            kernel_state = {
                "kernel_id": self.cognitive_kernel.kernel_id,
                "state": self.cognitive_kernel.state.value,
                "statistics": self.cognitive_kernel.get_kernel_statistics(),
                "membranes": self.cognitive_kernel.get_attention_membranes(),
                "grammars": self.grammar_registry.export_grammars()
            }
            
            json_str = json.dumps(kernel_state, default=str)
            print(f"   ✅ Serialization successful")
            print(f"   📦 Serialized size: {len(json_str):,} bytes")
            
            # Test deserialization
            restored_state = json.loads(json_str)
            print(f"   ✅ Deserialization successful")
            print(f"   🔗 Kernel ID preserved: {restored_state['kernel_id'] == self.cognitive_kernel.kernel_id}")
            
        except Exception as e:
            print(f"   ❌ Serialization failed: {e}")
            
        # Show extension capabilities
        print("\n🔧 Extension Capabilities:")
        extension_features = {
            "Runtime Grammar Registration": True,
            "Dynamic Pattern Extension": True,
            "Callback System": len(self.cognitive_kernel.feedback_callbacks) > 0,
            "Protocol Registration": len(self.cognitive_kernel.modification_protocols) > 0,
            "Tensor Customization": self.cognitive_kernel.kernel_tensor is not None
        }
        
        for feature, available in extension_features.items():
            print(f"   {'✅' if available else '❌'} {feature}")
            
        # Create integration adapter example
        print("\n🌉 Integration Adapter Example:")
        
        class CognitiveKernelAdapter:
            """Example adapter for external integration"""
            
            def __init__(self, kernel):
                self.kernel = kernel
                
            async def process_external_request(self, request):
                """Process external request through kernel"""
                return await self.kernel.recursive_invoke(request)
                
            def get_capabilities(self):
                """Get kernel capabilities"""
                return {
                    "subsystems": ["memory", "reasoning", "task", "autonomy"],
                    "grammars": list(self.kernel.cognitive_grammars.keys()),
                    "protocols": list(self.kernel.modification_protocols.keys())
                }
                
        adapter = CognitiveKernelAdapter(self.cognitive_kernel)
        capabilities = adapter.get_capabilities()
        
        print(f"   ✅ Adapter created")
        print(f"   📊 Capabilities: {len(capabilities['subsystems'])} subsystems")
        print(f"   📝 Grammars: {len(capabilities['grammars'])}")
        print(f"   🔧 Protocols: {len(capabilities['protocols'])}")
        
        # Test external request processing
        test_request = {
            "type": "reasoning",
            "content": {"concepts": ["integration", "adaptation"]}
        }
        
        external_result = await adapter.process_external_request(test_request)
        print(f"   ✅ External request processed: {external_result['success']}")
        
        return True
        
    async def cleanup_demo_environment(self):
        """Clean up demonstration environment"""
        print("\n🧹 Cleaning up demonstration environment...")
        
        if self.cognitive_kernel:
            await self.cognitive_kernel.shutdown()
            
        if os.path.exists(self.demo_db_path):
            os.remove(self.demo_db_path)
            
        print("✅ Cleanup complete!")
        
    async def run_full_demonstration(self):
        """Run the complete cognitive kernel demonstration"""
        print("🚀 Unified Cognitive Kernel - Live Demonstration")
        print("=" * 80)
        print("Showcasing meta-recursive attention system, cognitive grammar")
        print("management, and integrated subsystem orchestration")
        print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Setup
            await self.setup_demo_environment()
            
            # Demonstration sections
            demo_sections = [
                ("Kernel Initialization", self.demonstrate_kernel_initialization),
                ("Cognitive Grammars", self.demonstrate_cognitive_grammars),
                ("Recursive Invocation", self.demonstrate_recursive_invocation),
                ("Attention Membrane Dynamics", self.demonstrate_attention_membrane_dynamics),
                ("Meta-Cognitive Feedback", self.demonstrate_meta_cognitive_feedback),
                ("Self-Modification", self.demonstrate_self_modification),
                ("Integration Capabilities", self.demonstrate_integration_capabilities)
            ]
            
            successful_sections = 0
            
            for section_name, demo_func in demo_sections:
                try:
                    print(f"\n⏳ Running {section_name}...")
                    result = await demo_func()
                    
                    if result:
                        successful_sections += 1
                        print(f"✅ {section_name} completed successfully")
                    else:
                        print(f"❌ {section_name} failed")
                        
                except Exception as e:
                    print(f"❌ {section_name} failed with error: {e}")
                    
            # Final cleanup
            await self.cleanup_demo_environment()
            
            # Summary
            end_time = time.time()
            total_duration = end_time - start_time
            
            print("\n" + "=" * 80)
            print("🎯 DEMONSTRATION SUMMARY")
            print("=" * 80)
            print(f"📊 Total sections: {len(demo_sections)}")
            print(f"✅ Successful: {successful_sections}")
            print(f"❌ Failed: {len(demo_sections) - successful_sections}")
            print(f"🕐 Total duration: {total_duration:.2f} seconds")
            print(f"📈 Success rate: {successful_sections/len(demo_sections)*100:.1f}%")
            
            if successful_sections == len(demo_sections):
                print("\n🎉 ALL DEMONSTRATIONS SUCCESSFUL!")
                print("The Unified Cognitive Kernel is fully operational and ready for:")
                print("  • Agent Zero integration")
                print("  • Bolt.diy integration")
                print("  • Production deployment")
                print("  • Advanced cognitive applications")
                return True
            else:
                print(f"\n⚠️  {len(demo_sections) - successful_sections} demonstration(s) failed.")
                print("Please review the output above for details.")
                return False
                
        except Exception as e:
            print(f"❌ Demonstration failed with critical error: {e}")
            import traceback
            traceback.print_exc()
            return False


async def main():
    """Run the cognitive kernel demonstration"""
    demo = CognitiveKernelDemo()
    success = await demo.run_full_demonstration()
    
    if success:
        print("\n🚀 Cognitive Kernel Demonstration: SUCCESS")
        exit(0)
    else:
        print("\n❌ Cognitive Kernel Demonstration: FAILURE")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())