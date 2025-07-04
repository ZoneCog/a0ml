"""
Unified Cognitive Kernel - Meta-Recursive Attention System

A unifying meta-kernel that orchestrates the interplay of distributed hypergraph
AtomSpace memory, ECAN-powered attention allocation, recursive task/AI orchestration,
and meta-cognitive feedback in a dynamically extensible, agentic grammar.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid
import threading
from collections import defaultdict
import logging

from .atomspace import AtomSpace
from .neural_symbolic_reasoning import NeuralSymbolicReasoningEngine, ReasoningStage, CognitiveKernel
from .ecan_attention import ECANAttentionSystem, AttentionType, ResourceType, MetaCognitiveState
from .pattern_matcher import HypergraphPatternMatcher
from .pln_reasoning import PLNInferenceEngine
from .moses_optimizer import MOSESOptimizer

# Import distributed orchestrator conditionally
try:
    from .distributed_orchestrator import DistributedOrchestrator
    DISTRIBUTED_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    DISTRIBUTED_ORCHESTRATOR_AVAILABLE = False
    DistributedOrchestrator = None


class KernelState(Enum):
    """States of the cognitive kernel"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    REFLECTING = "reflecting"
    SELF_MODIFYING = "self_modifying"
    HIBERNATING = "hibernating"
    ERROR = "error"


class AttentionMembraneType(Enum):
    """Types of attention membranes (P-System compartments)"""
    MEMORY = "memory"
    REASONING = "reasoning"
    TASK = "task"
    AUTONOMY = "autonomy"
    META = "meta"


@dataclass
class AttentionMembrane:
    """
    P-System compartment for attention allocation
    Encapsulates subsystem boundaries and enables dynamic resource flows
    """
    id: str
    type: AttentionMembraneType
    name: str
    resources: Dict[ResourceType, float] = field(default_factory=dict)
    permeability: float = 0.5  # How easily resources flow in/out
    active_processes: Set[str] = field(default_factory=set)
    attention_gradient: np.ndarray = field(default_factory=lambda: np.zeros(5))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_gradient(self, new_gradient: np.ndarray):
        """Update attention gradient with temporal decay"""
        decay_factor = 0.9
        self.attention_gradient = decay_factor * self.attention_gradient + (1 - decay_factor) * new_gradient
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class MetaCognitiveEvent:
    """Event for meta-cognitive feedback logging"""
    id: str
    timestamp: datetime
    event_type: str
    source_subsystem: str
    description: str
    data: Dict[str, Any]
    impact_level: float  # 0.0 to 1.0


@dataclass
class SelfModificationProtocol:
    """Protocol for self-modification operations"""
    id: str
    name: str
    target_subsystem: str
    modification_type: str
    parameters: Dict[str, Any]
    safety_checks: List[str]
    rollback_possible: bool = True
    
    
class UnifiedCognitiveKernel:
    """
    Unified cognitive kernel that orchestrates all subsystems
    
    Serves as the nexus for Memory (AtomSpace), Task (DistributedOrchestrator),
    AI (NeuralSymbolicReasoningEngine), and Autonomy (ECANAttentionSystem).
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.logger = logging.getLogger(__name__)
        
        # Core subsystems
        self.neural_symbolic_engine: Optional[NeuralSymbolicReasoningEngine] = None
        self.ecan_system: Optional[ECANAttentionSystem] = None
        self.task_orchestrator: Optional[DistributedOrchestrator] = None
        
        # Kernel state
        self.state = KernelState.INITIALIZING
        self.kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"
        
        # Attention membranes (P-System compartments)
        self.attention_membranes: Dict[str, AttentionMembrane] = {}
        
        # Meta-cognitive feedback system
        self.meta_events: List[MetaCognitiveEvent] = []
        self.feedback_callbacks: Dict[str, Callable] = {}
        
        # Self-modification protocols
        self.modification_protocols: Dict[str, SelfModificationProtocol] = {}
        
        # High-rank kernel tensor T_kernel[n_atoms, n_tasks, n_reasoning, a_levels, t_steps]
        self.n_atoms = 1000  # Maximum atoms to track
        self.n_tasks = 100   # Maximum tasks to track
        self.n_reasoning = 50  # Maximum reasoning programs
        self.a_levels = 5    # Attention/autonomy levels
        self.t_steps = 10    # Time steps for temporal dynamics
        
        self.kernel_tensor: Optional[np.ndarray] = None
        self.tensor_update_lock = threading.Lock()
        
        # Dynamic vocabulary registry for Scheme grammars
        self.cognitive_grammars: Dict[str, Any] = {}
        
        # Threading for kernel operations
        self.kernel_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Statistics
        self.statistics = {
            "kernel_cycles": 0,
            "subsystem_invocations": 0,
            "attention_reallocations": 0,
            "meta_events": 0,
            "self_modifications": 0
        }
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the unified cognitive kernel"""
        self.logger.info(f"Initializing unified cognitive kernel {self.kernel_id}")
        
        try:
            # Initialize core subsystems
            await self._initialize_subsystems()
            
            # Create attention membranes
            await self._create_attention_membranes()
            
            # Initialize kernel tensor
            self._initialize_kernel_tensor()
            
            # Register default modification protocols
            self._register_default_protocols()
            
            # Start kernel thread
            self.running = True
            self.kernel_thread = threading.Thread(target=self._kernel_loop, daemon=True)
            self.kernel_thread.start()
            
            self.state = KernelState.ACTIVE
            
            initialization_report = {
                "kernel_id": self.kernel_id,
                "state": self.state.value,
                "subsystems": {
                    "neural_symbolic": self.neural_symbolic_engine is not None,
                    "ecan": self.ecan_system is not None,
                    "task_orchestrator": self.task_orchestrator is not None
                },
                "attention_membranes": len(self.attention_membranes),
                "kernel_tensor_shape": self.kernel_tensor.shape if self.kernel_tensor is not None else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.logger.info(f"Kernel {self.kernel_id} initialized successfully")
            return initialization_report
            
        except Exception as e:
            self.state = KernelState.ERROR
            self.logger.error(f"Failed to initialize kernel: {e}")
            raise
            
    async def _initialize_subsystems(self):
        """Initialize all subsystems"""
        # Initialize neural-symbolic reasoning engine
        self.neural_symbolic_engine = NeuralSymbolicReasoningEngine(self.atomspace)
        await self.neural_symbolic_engine.initialize_system()
        
        # Initialize ECAN attention system
        self.ecan_system = ECANAttentionSystem(self.atomspace, self.neural_symbolic_engine)
        await self.ecan_system.initialize()
        
        # Initialize distributed task orchestrator if available
        if DISTRIBUTED_ORCHESTRATOR_AVAILABLE:
            self.task_orchestrator = DistributedOrchestrator()
            await self.task_orchestrator.initialize()
        else:
            self.task_orchestrator = None
            self.logger.warning("Distributed orchestrator not available - task subsystem disabled")
        
        self.logger.info("All subsystems initialized")
        
    async def _create_attention_membranes(self):
        """Create attention membranes for each subsystem"""
        membrane_configs = [
            (AttentionMembraneType.MEMORY, "Memory Membrane", {
                ResourceType.MEMORY: 100.0,
                ResourceType.ATTENTION: 25.0
            }),
            (AttentionMembraneType.REASONING, "Reasoning Membrane", {
                ResourceType.PROCESSING: 100.0,
                ResourceType.ATTENTION: 50.0
            }),
            (AttentionMembraneType.TASK, "Task Membrane", {
                ResourceType.PROCESSING: 50.0,
                ResourceType.BANDWIDTH: 100.0
            }),
            (AttentionMembraneType.AUTONOMY, "Autonomy Membrane", {
                ResourceType.ENERGY: 100.0,
                ResourceType.ATTENTION: 75.0
            }),
            (AttentionMembraneType.META, "Meta-Cognitive Membrane", {
                ResourceType.PROCESSING: 25.0,
                ResourceType.ATTENTION: 100.0
            })
        ]
        
        for membrane_type, name, resources in membrane_configs:
            membrane_id = f"membrane_{membrane_type.value}_{uuid.uuid4().hex[:8]}"
            
            membrane = AttentionMembrane(
                id=membrane_id,
                type=membrane_type,
                name=name,
                resources=resources,
                permeability=0.5,
                attention_gradient=np.random.random(5)
            )
            
            self.attention_membranes[membrane_id] = membrane
            
        self.logger.info(f"Created {len(self.attention_membranes)} attention membranes")
        
    def _initialize_kernel_tensor(self):
        """Initialize the high-rank kernel tensor"""
        self.kernel_tensor = np.zeros((
            self.n_atoms,
            self.n_tasks,
            self.n_reasoning,
            self.a_levels,
            self.t_steps
        ))
        
        # Initialize with small random values
        self.kernel_tensor += np.random.normal(0, 0.01, self.kernel_tensor.shape)
        
        self.logger.info(f"Initialized kernel tensor with shape {self.kernel_tensor.shape}")
        
    def _register_default_protocols(self):
        """Register default self-modification protocols"""
        protocols = [
            SelfModificationProtocol(
                id="attention_reallocation",
                name="Dynamic Attention Reallocation",
                target_subsystem="ecan",
                modification_type="resource_adjustment",
                parameters={"max_adjustment": 0.2},
                safety_checks=["resource_bounds_check", "stability_check"]
            ),
            SelfModificationProtocol(
                id="kernel_tensor_reshape",
                name="Kernel Tensor Reshape",
                target_subsystem="kernel",
                modification_type="tensor_modification",
                parameters={"max_dimension_change": 0.1},
                safety_checks=["tensor_consistency_check", "memory_bounds_check"]
            ),
            SelfModificationProtocol(
                id="cognitive_grammar_extension",
                name="Cognitive Grammar Extension",
                target_subsystem="neural_symbolic",
                modification_type="grammar_addition",
                parameters={"max_new_patterns": 10},
                safety_checks=["grammar_validity_check", "pattern_consistency_check"]
            )
        ]
        
        for protocol in protocols:
            self.modification_protocols[protocol.id] = protocol
            
        self.logger.info(f"Registered {len(protocols)} default modification protocols")
        
    def _kernel_loop(self):
        """Main kernel loop running in background thread"""
        while self.running:
            try:
                # Update kernel tensor
                self._update_kernel_tensor()
                
                # Process attention flow between membranes
                self._process_attention_flow()
                
                # Generate meta-cognitive events
                self._generate_meta_events()
                
                # Check for self-modification triggers
                self._check_modification_triggers()
                
                self.statistics["kernel_cycles"] += 1
                
            except Exception as e:
                self.logger.error(f"Error in kernel loop: {e}")
                
            # Sleep for kernel cycle interval
            import time
            time.sleep(0.1)
            
    def _update_kernel_tensor(self):
        """Update the kernel tensor with current subsystem states"""
        with self.tensor_update_lock:
            if self.kernel_tensor is None:
                return
                
            # Shift time dimension (temporal dynamics)
            self.kernel_tensor[:, :, :, :, 1:] = self.kernel_tensor[:, :, :, :, :-1]
            
            # Update current time step (t=0) with subsystem states
            current_step = np.zeros((self.n_atoms, self.n_tasks, self.n_reasoning, self.a_levels))
            
            # Integrate memory state (atoms)
            if self.neural_symbolic_engine and hasattr(self.neural_symbolic_engine, 'get_system_tensor'):
                try:
                    system_tensor = asyncio.run(self.neural_symbolic_engine.get_system_tensor())
                    if system_tensor is not None:
                        # Map system tensor to kernel tensor dimensions
                        n_patterns, n_reasoning_progs, n_stages = system_tensor.shape
                        atoms_slice = min(n_patterns, self.n_atoms)
                        reasoning_slice = min(n_reasoning_progs, self.n_reasoning)
                        
                        # Broadcast system tensor across task and attention dimensions
                        for t in range(self.n_tasks):
                            for a in range(self.a_levels):
                                current_step[:atoms_slice, t, :reasoning_slice, a] = np.mean(system_tensor[:atoms_slice, :reasoning_slice, :], axis=2)
                except Exception as e:
                    self.logger.warning(f"Could not update memory state in kernel tensor: {e}")
            
            # Integrate task state if available
            if self.task_orchestrator and hasattr(self.task_orchestrator, 'get_task_tensor'):
                try:
                    task_tensor = self.task_orchestrator.get_task_tensor()
                    if task_tensor is not None:
                        n_tasks_actual, n_agents, n_priorities = task_tensor.shape
                        tasks_slice = min(n_tasks_actual, self.n_tasks)
                        
                        # Map task tensor to kernel tensor
                        for a in range(min(n_agents, self.a_levels)):
                            current_step[:, :tasks_slice, :, a] += np.mean(task_tensor[:tasks_slice, a, :])
                except Exception as e:
                    self.logger.warning(f"Could not update task state in kernel tensor: {e}")
            
            # Integrate autonomy state
            if self.ecan_system and hasattr(self.ecan_system, 'get_autonomy_tensor'):
                try:
                    autonomy_tensor = self.ecan_system.get_autonomy_tensor()
                    if autonomy_tensor is not None:
                        a_levels_actual, r_types, m_states = autonomy_tensor.shape
                        levels_slice = min(a_levels_actual, self.a_levels)
                        
                        # Map autonomy tensor to kernel tensor
                        autonomy_values = np.mean(autonomy_tensor[:levels_slice, :, :], axis=(1, 2))
                        current_step[:, :, :, :levels_slice] += autonomy_values[None, None, None, :]
                except Exception as e:
                    self.logger.warning(f"Could not update autonomy state in kernel tensor: {e}")
            
            # Store current state
            self.kernel_tensor[:, :, :, :, 0] = current_step
            
    def _process_attention_flow(self):
        """Process attention flow between membranes"""
        for membrane_id, membrane in self.attention_membranes.items():
            # Calculate attention gradients
            gradient = np.zeros(5)
            
            # Update gradient based on membrane type and current load
            if membrane.type == AttentionMembraneType.MEMORY:
                # Memory membrane receives attention based on knowledge access patterns
                gradient[0] = len(membrane.active_processes) / 10.0
            elif membrane.type == AttentionMembraneType.REASONING:
                # Reasoning membrane receives attention based on inference activity
                gradient[1] = len(membrane.active_processes) / 20.0
            elif membrane.type == AttentionMembraneType.TASK:
                # Task membrane receives attention based on task load
                gradient[2] = len(membrane.active_processes) / 15.0
            elif membrane.type == AttentionMembraneType.AUTONOMY:
                # Autonomy membrane receives attention based on self-monitoring
                gradient[3] = len(membrane.active_processes) / 5.0
            elif membrane.type == AttentionMembraneType.META:
                # Meta membrane receives attention based on meta-cognitive events
                gradient[4] = len(self.meta_events) / 100.0
            
            # Update membrane gradient
            membrane.update_gradient(gradient)
            
            # Flow resources between membranes based on gradients
            self._flow_resources_between_membranes(membrane)
            
    def _flow_resources_between_membranes(self, source_membrane: AttentionMembrane):
        """Flow resources from source membrane to others based on gradients"""
        for target_id, target_membrane in self.attention_membranes.items():
            if target_id == source_membrane.id:
                continue
                
            # Calculate flow amount based on gradient difference and permeability
            gradient_diff = np.linalg.norm(target_membrane.attention_gradient - source_membrane.attention_gradient)
            flow_amount = gradient_diff * source_membrane.permeability * 0.1
            
            # Flow each resource type
            for resource_type in ResourceType:
                if resource_type in source_membrane.resources and resource_type in target_membrane.resources:
                    transfer_amount = min(flow_amount, source_membrane.resources[resource_type] * 0.1)
                    
                    source_membrane.resources[resource_type] -= transfer_amount
                    target_membrane.resources[resource_type] += transfer_amount
                    
    def _generate_meta_events(self):
        """Generate meta-cognitive events for monitoring"""
        # Generate events based on kernel state
        if len(self.meta_events) < 1000:  # Limit event history
            event = MetaCognitiveEvent(
                id=f"meta_{uuid.uuid4().hex[:8]}",
                timestamp=datetime.now(timezone.utc),
                event_type="kernel_cycle",
                source_subsystem="kernel",
                description=f"Kernel cycle {self.statistics['kernel_cycles']}",
                data={
                    "tensor_shape": self.kernel_tensor.shape if self.kernel_tensor is not None else None,
                    "active_membranes": len(self.attention_membranes),
                    "kernel_state": self.state.value
                },
                impact_level=0.1
            )
            self.meta_events.append(event)
            self.statistics["meta_events"] += 1
            
    def _check_modification_triggers(self):
        """Check for triggers that require self-modification"""
        # Check if attention distribution is highly imbalanced
        if len(self.attention_membranes) > 0:
            attention_values = []
            for membrane in self.attention_membranes.values():
                if ResourceType.ATTENTION in membrane.resources:
                    attention_values.append(membrane.resources[ResourceType.ATTENTION])
            
            if len(attention_values) > 1:
                attention_std = np.std(attention_values)
                attention_mean = np.mean(attention_values)
                
                if attention_std > attention_mean * 0.5:  # High imbalance
                    asyncio.create_task(self._trigger_self_modification("attention_reallocation"))
                    
    async def _trigger_self_modification(self, protocol_id: str):
        """Trigger a self-modification protocol"""
        if protocol_id not in self.modification_protocols:
            self.logger.warning(f"Unknown modification protocol: {protocol_id}")
            return
            
        protocol = self.modification_protocols[protocol_id]
        
        try:
            self.state = KernelState.SELF_MODIFYING
            
            # Execute safety checks
            for check in protocol.safety_checks:
                if not await self._execute_safety_check(check):
                    self.logger.warning(f"Safety check failed for protocol {protocol_id}: {check}")
                    self.state = KernelState.ACTIVE
                    return
                    
            # Execute modification
            success = await self._execute_modification(protocol)
            
            if success:
                self.statistics["self_modifications"] += 1
                self.logger.info(f"Successfully executed modification protocol: {protocol_id}")
                
                # Generate meta-cognitive event
                event = MetaCognitiveEvent(
                    id=f"modification_{uuid.uuid4().hex[:8]}",
                    timestamp=datetime.now(timezone.utc),
                    event_type="self_modification",
                    source_subsystem="kernel",
                    description=f"Executed modification protocol: {protocol.name}",
                    data={"protocol_id": protocol_id, "parameters": protocol.parameters},
                    impact_level=0.8
                )
                self.meta_events.append(event)
            else:
                self.logger.error(f"Failed to execute modification protocol: {protocol_id}")
                
        except Exception as e:
            self.logger.error(f"Error in self-modification: {e}")
            
        finally:
            self.state = KernelState.ACTIVE
            
    async def _execute_safety_check(self, check_name: str) -> bool:
        """Execute a safety check"""
        if check_name == "resource_bounds_check":
            # Check that no membrane resources go negative
            for membrane in self.attention_membranes.values():
                for resource_value in membrane.resources.values():
                    if resource_value < 0:
                        return False
            return True
            
        elif check_name == "stability_check":
            # Check that system is in stable state
            return self.state == KernelState.ACTIVE
            
        elif check_name == "tensor_consistency_check":
            # Check tensor integrity
            return self.kernel_tensor is not None and not np.any(np.isnan(self.kernel_tensor))
            
        elif check_name == "memory_bounds_check":
            # Check memory usage
            return True  # Simplified check
            
        elif check_name == "grammar_validity_check":
            # Check grammar validity
            return True  # Simplified check
            
        elif check_name == "pattern_consistency_check":
            # Check pattern consistency
            return True  # Simplified check
            
        return False
        
    async def _execute_modification(self, protocol: SelfModificationProtocol) -> bool:
        """Execute a modification protocol"""
        try:
            if protocol.modification_type == "resource_adjustment":
                # Adjust resources between membranes
                max_adjustment = protocol.parameters.get("max_adjustment", 0.1)
                
                for membrane in self.attention_membranes.values():
                    for resource_type in membrane.resources:
                        adjustment = np.random.uniform(-max_adjustment, max_adjustment)
                        membrane.resources[resource_type] *= (1 + adjustment)
                        membrane.resources[resource_type] = max(0, membrane.resources[resource_type])
                
                return True
                
            elif protocol.modification_type == "tensor_modification":
                # Modify kernel tensor shape if needed
                max_change = protocol.parameters.get("max_dimension_change", 0.1)
                
                # For now, just adjust tensor values slightly
                if self.kernel_tensor is not None:
                    noise = np.random.normal(0, max_change * 0.01, self.kernel_tensor.shape)
                    self.kernel_tensor += noise
                
                return True
                
            elif protocol.modification_type == "grammar_addition":
                # Add new cognitive grammar patterns
                max_patterns = protocol.parameters.get("max_new_patterns", 5)
                
                for i in range(max_patterns):
                    grammar_id = f"grammar_{uuid.uuid4().hex[:8]}"
                    self.cognitive_grammars[grammar_id] = {
                        "pattern": f"(scheme-pattern-{i})",
                        "created": datetime.now(timezone.utc).isoformat(),
                        "active": True
                    }
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error executing modification: {e}")
            return False
            
        return False
        
    async def recursive_invoke(self, query: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recursively invoke the kernel with a query
        
        This is the main API for kernel invocation that orchestrates all subsystems
        """
        self.statistics["subsystem_invocations"] += 1
        
        # Create session for this invocation
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        result = {
            "session_id": session_id,
            "kernel_id": self.kernel_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "context": context or {},
            "results": {},
            "subsystem_calls": [],
            "attention_allocation": {},
            "meta_events": []
        }
        
        try:
            # Extract query components
            query_type = query.get("type", "general")
            query_content = query.get("content", {})
            
            # Allocate attention for this query
            attention_allocation = await self._allocate_attention_for_query(query, session_id)
            result["attention_allocation"] = attention_allocation
            
            # Route query to appropriate subsystems based on query type and attention
            if query_type in ["memory", "knowledge", "pattern"] or attention_allocation.get("memory", 0) > 0.5:
                # Memory/Pattern matching query
                memory_result = await self._invoke_memory_subsystem(query_content, session_id)
                result["results"]["memory"] = memory_result
                result["subsystem_calls"].append("memory")
                
            if query_type in ["reasoning", "inference", "logic"] or attention_allocation.get("reasoning", 0) > 0.5:
                # Reasoning query
                reasoning_result = await self._invoke_reasoning_subsystem(query_content, session_id)
                result["results"]["reasoning"] = reasoning_result
                result["subsystem_calls"].append("reasoning")
                
            if query_type in ["task", "planning", "orchestration"] or attention_allocation.get("task", 0) > 0.5:
                # Task orchestration query
                task_result = await self._invoke_task_subsystem(query_content, session_id)
                result["results"]["task"] = task_result
                result["subsystem_calls"].append("task")
                
            if query_type in ["autonomy", "self", "meta"] or attention_allocation.get("autonomy", 0) > 0.5:
                # Autonomy/meta-cognitive query
                autonomy_result = await self._invoke_autonomy_subsystem(query_content, session_id)
                result["results"]["autonomy"] = autonomy_result
                result["subsystem_calls"].append("autonomy")
                
            # Generate meta-cognitive events for this invocation
            events = await self._generate_invocation_events(session_id, query, result)
            result["meta_events"] = events
            
            result["success"] = True
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            self.logger.error(f"Error in kernel invocation: {e}")
            
        return result
        
    async def _allocate_attention_for_query(self, query: Dict[str, Any], session_id: str) -> Dict[str, float]:
        """Allocate attention resources for a query"""
        query_type = query.get("type", "general")
        query_content = query.get("content", {})
        
        # Base attention allocation
        allocation = {
            "memory": 0.25,
            "reasoning": 0.25,
            "task": 0.25,
            "autonomy": 0.25
        }
        
        # Adjust based on query type
        if query_type in ["memory", "knowledge", "pattern"]:
            allocation["memory"] = 0.6
            allocation["reasoning"] = 0.3
            allocation["task"] = 0.05
            allocation["autonomy"] = 0.05
        elif query_type in ["reasoning", "inference", "logic"]:
            allocation["memory"] = 0.3
            allocation["reasoning"] = 0.6
            allocation["task"] = 0.05
            allocation["autonomy"] = 0.05
        elif query_type in ["task", "planning", "orchestration"]:
            allocation["memory"] = 0.1
            allocation["reasoning"] = 0.2
            allocation["task"] = 0.6
            allocation["autonomy"] = 0.1
        elif query_type in ["autonomy", "self", "meta"]:
            allocation["memory"] = 0.1
            allocation["reasoning"] = 0.1
            allocation["task"] = 0.1
            allocation["autonomy"] = 0.7
            
        # Use ECAN system for attention allocation if available
        if self.ecan_system:
            try:
                ecan_allocation = await self.ecan_system.adaptive_attention_allocation(
                    {"query": 1.0}, 
                    {"memory": allocation["memory"], "reasoning": allocation["reasoning"]}
                )
                if ecan_allocation:
                    allocation.update(ecan_allocation)
            except Exception as e:
                self.logger.warning(f"Could not use ECAN for attention allocation: {e}")
                
        return allocation
        
    async def _invoke_memory_subsystem(self, query_content: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Invoke memory/pattern matching subsystem"""
        if not self.neural_symbolic_engine:
            return {"error": "Neural-symbolic engine not available"}
            
        try:
            # Use pattern matcher for memory queries
            pattern_matcher = self.neural_symbolic_engine.pattern_matcher
            
            # Extract patterns from query
            patterns = query_content.get("patterns", [])
            concepts = query_content.get("concepts", [])
            
            results = {"patterns": [], "concepts": []}
            
            # Pattern matching
            for pattern in patterns:
                matches = await pattern_matcher.match_pattern(pattern, max_matches=10)
                results["patterns"].append({
                    "pattern": pattern,
                    "matches": len(matches),
                    "results": [match.to_dict() for match in matches[:5]]
                })
                
            # Concept queries
            for concept in concepts:
                # Use hypergraph traversal for concept exploration
                concept_atoms = await self.atomspace.pattern_match({"concept": concept}, limit=10)
                results["concepts"].append({
                    "concept": concept,
                    "atoms": len(concept_atoms),
                    "results": [atom.to_dict() for atom in concept_atoms[:5]]
                })
                
            return results
            
        except Exception as e:
            return {"error": f"Memory subsystem error: {e}"}
            
    async def _invoke_reasoning_subsystem(self, query_content: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Invoke reasoning subsystem"""
        if not self.neural_symbolic_engine:
            return {"error": "Neural-symbolic engine not available"}
            
        try:
            # Use neural-symbolic reasoning
            reasoning_query = {
                "type": "reasoning",
                "content": query_content
            }
            
            result = await self.neural_symbolic_engine.reason(reasoning_query)
            return result
            
        except Exception as e:
            return {"error": f"Reasoning subsystem error: {e}"}
            
    async def _invoke_task_subsystem(self, query_content: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Invoke task orchestration subsystem"""
        if not self.task_orchestrator:
            return {"error": "Task orchestrator not available"}
            
        try:
            # Handle task-related queries
            if "goal" in query_content:
                goal = query_content["goal"]
                if hasattr(self.task_orchestrator, 'decompose_goal'):
                    decomposition = await self.task_orchestrator.decompose_goal(goal)
                    return {"goal": goal, "decomposition": decomposition}
                else:
                    return {"goal": goal, "decomposition": "Task decomposition simulated"}
            elif "task_status" in query_content:
                if hasattr(self.task_orchestrator, 'get_orchestrator_status'):
                    status = self.task_orchestrator.get_orchestrator_status()
                    return {"task_status": status}
                else:
                    return {"task_status": "Task status simulated"}
            else:
                return {"error": "Unknown task query type"}
                
        except Exception as e:
            return {"error": f"Task subsystem error: {e}"}
            
    async def _invoke_autonomy_subsystem(self, query_content: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Invoke autonomy/ECAN subsystem"""
        if not self.ecan_system:
            return {"error": "ECAN system not available"}
            
        try:
            # Handle autonomy queries
            if "attention_status" in query_content:
                # Get attention status
                autonomy_tensor = self.ecan_system.get_autonomy_tensor()
                return {
                    "attention_status": "active",
                    "autonomy_tensor_shape": autonomy_tensor.shape if autonomy_tensor is not None else None,
                    "attention_units": len(self.ecan_system.attention_units)
                }
            elif "self_inspection" in query_content:
                # Trigger self-inspection
                await self.ecan_system.self_inspect()
                return {"self_inspection": "completed"}
            else:
                return {"error": "Unknown autonomy query type"}
                
        except Exception as e:
            return {"error": f"Autonomy subsystem error: {e}"}
            
    async def _generate_invocation_events(self, session_id: str, query: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate meta-cognitive events for this invocation"""
        events = []
        
        # Invocation event
        event = MetaCognitiveEvent(
            id=f"invocation_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            event_type="kernel_invocation",
            source_subsystem="kernel",
            description=f"Kernel invocation for query type: {query.get('type', 'general')}",
            data={
                "session_id": session_id,
                "query_type": query.get("type"),
                "subsystems_called": result.get("subsystem_calls", []),
                "success": result.get("success", False)
            },
            impact_level=0.5
        )
        events.append(event.to_dict())
        
        # Add to meta events
        self.meta_events.append(event)
        
        return events
        
    def get_kernel_tensor(self) -> Optional[np.ndarray]:
        """Get current kernel tensor"""
        return self.kernel_tensor.copy() if self.kernel_tensor is not None else None
        
    def get_attention_membranes(self) -> Dict[str, Dict[str, Any]]:
        """Get current attention membrane states"""
        return {
            membrane_id: {
                "id": membrane.id,
                "type": membrane.type.value,
                "name": membrane.name,
                "resources": membrane.resources,
                "permeability": membrane.permeability,
                "active_processes": len(membrane.active_processes),
                "attention_gradient": membrane.attention_gradient.tolist(),
                "last_updated": membrane.last_updated.isoformat()
            }
            for membrane_id, membrane in self.attention_membranes.items()
        }
        
    def get_kernel_statistics(self) -> Dict[str, Any]:
        """Get kernel statistics"""
        return {
            **self.statistics,
            "state": self.state.value,
            "kernel_id": self.kernel_id,
            "tensor_shape": self.kernel_tensor.shape if self.kernel_tensor is not None else None,
            "attention_membranes": len(self.attention_membranes),
            "meta_events": len(self.meta_events),
            "cognitive_grammars": len(self.cognitive_grammars),
            "modification_protocols": len(self.modification_protocols)
        }
        
    def register_cognitive_grammar(self, grammar_id: str, grammar_def: Dict[str, Any]):
        """Register a new cognitive grammar"""
        self.cognitive_grammars[grammar_id] = {
            **grammar_def,
            "created": datetime.now(timezone.utc).isoformat(),
            "active": True
        }
        
    def get_cognitive_grammars(self) -> Dict[str, Any]:
        """Get registered cognitive grammars"""
        return self.cognitive_grammars.copy()
        
    async def shutdown(self):
        """Shutdown the kernel"""
        self.running = False
        
        if self.kernel_thread and self.kernel_thread.is_alive():
            self.kernel_thread.join(timeout=5)
            
        self.state = KernelState.HIBERNATING
        self.logger.info(f"Kernel {self.kernel_id} shutdown complete")


# Helper functions for meta-cognitive events
def create_meta_event(event_type: str, source: str, description: str, data: Dict[str, Any] = None, impact: float = 0.5) -> MetaCognitiveEvent:
    """Create a meta-cognitive event"""
    return MetaCognitiveEvent(
        id=f"meta_{uuid.uuid4().hex[:8]}",
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        source_subsystem=source,
        description=description,
        data=data or {},
        impact_level=impact
    )


# Extension for MetaCognitiveEvent to support serialization
def meta_event_to_dict(event: MetaCognitiveEvent) -> Dict[str, Any]:
    """Convert MetaCognitiveEvent to dictionary"""
    return {
        "id": event.id,
        "timestamp": event.timestamp.isoformat(),
        "event_type": event.event_type,
        "source_subsystem": event.source_subsystem,
        "description": event.description,
        "data": event.data,
        "impact_level": event.impact_level
    }

# Add to_dict method to MetaCognitiveEvent
MetaCognitiveEvent.to_dict = meta_event_to_dict