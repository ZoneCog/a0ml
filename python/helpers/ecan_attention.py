"""
ECAN (Economic Attention Allocation Network) for Meta-Cognitive Control

Implements attention allocation and economic resource management for cognitive agents
based on AtomSpace hypergraph structures and neural-symbolic reasoning.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid
import threading
from collections import defaultdict

from .atomspace import AtomSpace
# Import at function level to avoid circular import
# from .neural_symbolic_reasoning import NeuralSymbolicReasoningEngine, ReasoningStage, CognitiveKernel


class AttentionType(Enum):
    """Types of attention allocation"""
    PERCEPTUAL = "perceptual"
    COGNITIVE = "cognitive"
    EXECUTIVE = "executive"
    SOCIAL = "social"
    SELF_MONITORING = "self_monitoring"


class ResourceType(Enum):
    """Types of cognitive resources"""
    MEMORY = "memory"
    PROCESSING = "processing"
    ATTENTION = "attention"
    BANDWIDTH = "bandwidth"
    ENERGY = "energy"


class MetaCognitiveState(Enum):
    """Meta-cognitive states for self-monitoring"""
    ANALYZING = "analyzing"
    LEARNING = "learning"
    ADAPTING = "adapting"
    OPTIMIZING = "optimizing"
    IDLE = "idle"
    SELF_MODIFYING = "self_modifying"


@dataclass
class AttentionUnit:
    """Represents a unit of attention with economic properties"""
    id: str
    atom_id: str  # Reference to AtomSpace atom
    attention_value: float  # Short-term importance
    importance: float  # Long-term importance
    confidence: float  # Confidence in importance
    resource_cost: float  # Cost to maintain attention
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.01
    
    def update_attention(self, delta: float, access_time: datetime = None):
        """Update attention value with decay"""
        if access_time is None:
            access_time = datetime.now(timezone.utc)
        
        # Apply decay based on time since last access
        time_delta = (access_time - self.last_accessed).total_seconds()
        decay = self.decay_rate * time_delta
        
        # Update values
        self.attention_value = max(0.0, self.attention_value - decay + delta)
        self.last_accessed = access_time
        self.access_count += 1


@dataclass
class ResourcePool:
    """Manages cognitive resource allocation"""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    allocation_history: List[Tuple[str, float, datetime]] = field(default_factory=list)
    
    def allocate(self, requester_id: str, amount: float) -> bool:
        """Attempt to allocate resources"""
        if self.available_capacity >= amount:
            self.available_capacity -= amount
            self.allocation_history.append((requester_id, amount, datetime.now(timezone.utc)))
            return True
        return False
    
    def release(self, amount: float):
        """Release allocated resources"""
        self.available_capacity = min(self.total_capacity, self.available_capacity + amount)


@dataclass
class SelfInspectionReport:
    """Report from self-inspection routine"""
    id: str
    timestamp: datetime
    cognitive_state: MetaCognitiveState
    attention_distribution: Dict[str, float]
    resource_utilization: Dict[str, float]
    performance_metrics: Dict[str, float]
    anomalies_detected: List[str]
    recommendations: List[str]
    hypergraph_snapshot: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "cognitive_state": self.cognitive_state.value,
            "attention_distribution": self.attention_distribution,
            "resource_utilization": self.resource_utilization,
            "performance_metrics": self.performance_metrics,
            "anomalies_detected": self.anomalies_detected,
            "recommendations": self.recommendations,
            "hypergraph_snapshot": self.hypergraph_snapshot
        }


class ECANAttentionSystem:
    """
    Economic Attention Allocation Network (ECAN) for meta-cognitive control
    
    Implements distributed attention allocation based on economic principles
    with self-monitoring and adaptive behavior modification.
    """
    
    def __init__(self, atomspace: AtomSpace, reasoning_engine=None):
        self.atomspace = atomspace
        self.reasoning_engine = reasoning_engine  # Can be None initially
        
        # Attention management
        self.attention_units: Dict[str, AttentionUnit] = {}
        self.attention_threshold = 0.1
        self.max_attention_units = 1000
        
        # Resource management
        self.resource_pools = {
            ResourceType.MEMORY: ResourcePool(ResourceType.MEMORY, 100.0, 100.0),
            ResourceType.PROCESSING: ResourcePool(ResourceType.PROCESSING, 100.0, 100.0),
            ResourceType.ATTENTION: ResourcePool(ResourceType.ATTENTION, 100.0, 100.0),
            ResourceType.BANDWIDTH: ResourcePool(ResourceType.BANDWIDTH, 100.0, 100.0),
            ResourceType.ENERGY: ResourcePool(ResourceType.ENERGY, 100.0, 100.0)
        }
        
        # Self-monitoring
        self.current_state = MetaCognitiveState.IDLE
        self.inspection_reports: List[SelfInspectionReport] = []
        self.self_modification_events: List[Dict[str, Any]] = []
        
        # Autonomy metrics tensor T_auto[a_levels, r_types, m_states]
        self.autonomy_levels = 5  # Different levels of autonomy
        self.num_resource_types = len(ResourceType)
        self.num_meta_states = len(MetaCognitiveState)
        self.autonomy_tensor = np.zeros((self.autonomy_levels, self.num_resource_types, self.num_meta_states))
        
        # Threading for periodic tasks
        self.inspection_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Statistics
        self.statistics = {
            "attention_allocations": 0,
            "resource_allocations": 0,
            "self_modifications": 0,
            "inspection_cycles": 0,
            "adaptation_events": 0
        }
    
    async def initialize(self):
        """Initialize the ECAN system"""
        # Initialize attention units from existing atoms
        await self._initialize_attention_units()
        
        # Start self-monitoring thread
        self.running = True
        self.inspection_thread = threading.Thread(target=self._periodic_inspection, daemon=True)
        self.inspection_thread.start()
    
    async def _initialize_attention_units(self):
        """Initialize attention units from atomspace"""
        # Get all atoms from atomspace using pattern matching
        atoms = await self.atomspace.pattern_match({}, limit=1000)
        
        for atom in atoms:
            attention_unit = AttentionUnit(
                id=f"au_{atom.id}",
                atom_id=atom.id,
                attention_value=atom.truth_value,
                importance=atom.confidence,
                confidence=atom.confidence,
                resource_cost=0.1,  # Base cost
                last_accessed=datetime.now(timezone.utc)
            )
            self.attention_units[attention_unit.id] = attention_unit
    
    def get_autonomy_tensor(self) -> np.ndarray:
        """Get current autonomy metrics tensor T_auto[a_levels, r_types, m_states]"""
        # Update tensor with current state
        self._update_autonomy_tensor()
        return self.autonomy_tensor.copy()
    
    def _update_autonomy_tensor(self):
        """Update autonomy tensor with current metrics"""
        # Reset tensor
        self.autonomy_tensor.fill(0.0)
        
        # Current state index
        state_idx = list(MetaCognitiveState).index(self.current_state)
        
        # Resource utilization
        for i, resource_type in enumerate(ResourceType):
            pool = self.resource_pools[resource_type]
            utilization = 1.0 - (pool.available_capacity / pool.total_capacity)
            
            # Distribute across autonomy levels based on utilization
            for level in range(self.autonomy_levels):
                if utilization > (level / self.autonomy_levels):
                    self.autonomy_tensor[level, i, state_idx] = utilization
    
    async def allocate_attention(self, atom_id: str, priority: float, 
                               requester_id: str) -> bool:
        """Allocate attention to a specific atom"""
        # Check if we have attention resources
        if not self.resource_pools[ResourceType.ATTENTION].allocate(requester_id, 1.0):
            return False
        
        # Find or create attention unit
        attention_unit = None
        for au in self.attention_units.values():
            if au.atom_id == atom_id:
                attention_unit = au
                break
        
        if attention_unit is None:
            # Create new attention unit
            attention_unit = AttentionUnit(
                id=f"au_{atom_id}_{uuid.uuid4().hex[:8]}",
                atom_id=atom_id,
                attention_value=priority,
                importance=priority,
                confidence=0.5,
                resource_cost=0.1,
                last_accessed=datetime.now(timezone.utc)
            )
            self.attention_units[attention_unit.id] = attention_unit
        
        # Update attention
        attention_unit.update_attention(priority)
        
        # Update statistics
        self.statistics["attention_allocations"] += 1
        
        return True
    
    async def adaptive_attention_allocation(self, task_salience: Dict[str, float], 
                                          resource_load: Dict[str, float]) -> Dict[str, float]:
        """Adaptive attention allocation based on task salience and resource load"""
        allocation_result = {}
        
        # Calculate attention budget based on resource load
        total_load = sum(resource_load.values())
        attention_budget = max(0.1, 1.0 - (total_load / len(resource_load)))
        
        # Sort tasks by salience
        sorted_tasks = sorted(task_salience.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate attention proportionally
        total_salience = sum(task_salience.values())
        for task_id, salience in sorted_tasks:
            if total_salience > 0:
                allocation = (salience / total_salience) * attention_budget
                allocation_result[task_id] = allocation
                
                # Actually allocate attention
                await self.allocate_attention(task_id, allocation, "adaptive_system")
        
        return allocation_result
    
    async def self_modify(self, modification_type: str, parameters: Dict[str, Any]) -> bool:
        """Perform self-modification based on introspection"""
        modification_event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": modification_type,
            "parameters": parameters,
            "success": False,
            "changes_made": []
        }
        
        try:
            if modification_type == "attention_threshold":
                old_threshold = self.attention_threshold
                new_threshold = parameters.get("new_threshold", self.attention_threshold)
                self.attention_threshold = max(0.01, min(1.0, new_threshold))
                modification_event["changes_made"].append(f"Attention threshold: {old_threshold} -> {self.attention_threshold}")
            
            elif modification_type == "resource_reallocation":
                # Redistribute resource capacities
                reallocation = parameters.get("reallocation", {})
                for resource_name, new_capacity in reallocation.items():
                    try:
                        resource_type = ResourceType(resource_name)
                        if resource_type in self.resource_pools:
                            old_capacity = self.resource_pools[resource_type].total_capacity
                            self.resource_pools[resource_type].total_capacity = max(1.0, new_capacity)
                            modification_event["changes_made"].append(f"{resource_name} capacity: {old_capacity} -> {new_capacity}")
                    except ValueError:
                        continue
            
            elif modification_type == "decay_rate_adjustment":
                new_decay_rate = parameters.get("new_decay_rate", 0.01)
                changes_count = 0
                for au in self.attention_units.values():
                    if au.decay_rate != new_decay_rate:
                        au.decay_rate = new_decay_rate
                        changes_count += 1
                modification_event["changes_made"].append(f"Updated decay rate for {changes_count} attention units")
            
            modification_event["success"] = True
            self.statistics["self_modifications"] += 1
            
        except Exception as e:
            modification_event["error"] = str(e)
        
        self.self_modification_events.append(modification_event)
        return modification_event["success"]
    
    async def inspect_system(self) -> SelfInspectionReport:
        """Perform comprehensive self-inspection"""
        report_id = str(uuid.uuid4())
        
        # Attention distribution
        attention_distribution = {}
        for au in self.attention_units.values():
            attention_distribution[au.atom_id] = au.attention_value
        
        # Resource utilization
        resource_utilization = {}
        for resource_type, pool in self.resource_pools.items():
            utilization = 1.0 - (pool.available_capacity / pool.total_capacity)
            resource_utilization[resource_type.value] = utilization
        
        # Performance metrics
        performance_metrics = {
            "attention_units_count": len(self.attention_units),
            "average_attention": np.mean(list(attention_distribution.values())) if attention_distribution else 0.0,
            "resource_efficiency": 1.0 - np.mean(list(resource_utilization.values())),
            "allocation_success_rate": self.statistics["attention_allocations"] / max(1, self.statistics["attention_allocations"])
        }
        
        # Anomaly detection
        anomalies = []
        if performance_metrics["average_attention"] < 0.1:
            anomalies.append("Low average attention levels detected")
        if performance_metrics["resource_efficiency"] < 0.3:
            anomalies.append("Poor resource efficiency detected")
        if len(self.attention_units) > self.max_attention_units * 0.9:
            anomalies.append("Approaching maximum attention units limit")
        
        # Recommendations
        recommendations = []
        if performance_metrics["average_attention"] < 0.2:
            recommendations.append("Consider increasing attention threshold")
        if performance_metrics["resource_efficiency"] < 0.5:
            recommendations.append("Optimize resource allocation strategies")
        if len(anomalies) > 0:
            recommendations.append("Investigate detected anomalies")
        
        # Hypergraph snapshot
        hypergraph_snapshot = {
            "atoms_count": len(await self.atomspace.pattern_match({}, limit=1000)),
            "attention_units": len(self.attention_units),
            "active_kernels": len(self.reasoning_engine.cognitive_kernels) if self.reasoning_engine else 0,
            "tensor_shape": self.autonomy_tensor.shape,
            "tensor_stats": {
                "mean": float(self.autonomy_tensor.mean()),
                "std": float(self.autonomy_tensor.std()),
                "max": float(self.autonomy_tensor.max())
            }
        }
        
        report = SelfInspectionReport(
            id=report_id,
            timestamp=datetime.now(timezone.utc),
            cognitive_state=self.current_state,
            attention_distribution=attention_distribution,
            resource_utilization=resource_utilization,
            performance_metrics=performance_metrics,
            anomalies_detected=anomalies,
            recommendations=recommendations,
            hypergraph_snapshot=hypergraph_snapshot
        )
        
        self.inspection_reports.append(report)
        self.statistics["inspection_cycles"] += 1
        
        # Trigger self-modification if needed
        if len(anomalies) > 0:
            await self._trigger_adaptive_modifications(report)
        
        return report
    
    async def _trigger_adaptive_modifications(self, report: SelfInspectionReport):
        """Trigger adaptive modifications based on inspection report"""
        if "Low average attention levels detected" in report.anomalies_detected:
            await self.self_modify("attention_threshold", {"new_threshold": self.attention_threshold * 0.8})
        
        if "Poor resource efficiency detected" in report.anomalies_detected:
            # Increase processing and memory capacity
            await self.self_modify("resource_reallocation", {
                "reallocation": {
                    "processing": self.resource_pools[ResourceType.PROCESSING].total_capacity * 1.2,
                    "memory": self.resource_pools[ResourceType.MEMORY].total_capacity * 1.1
                }
            })
        
        self.statistics["adaptation_events"] += 1
    
    def _periodic_inspection(self):
        """Periodic self-inspection routine (runs in separate thread)"""
        while self.running:
            try:
                # Run inspection every 30 seconds
                import time
                time.sleep(30)
                
                # Create event loop for async inspection
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Perform inspection
                report = loop.run_until_complete(self.inspect_system())
                
                # Log report (in real implementation, this would be logged properly)
                print(f"🔍 Self-inspection completed: {report.id}")
                
            except Exception as e:
                print(f"❌ Self-inspection error: {e}")
            finally:
                if 'loop' in locals():
                    loop.close()
    
    def stop(self):
        """Stop the ECAN system"""
        self.running = False
        if self.inspection_thread and self.inspection_thread.is_alive():
            self.inspection_thread.join(timeout=5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            **self.statistics,
            "attention_units_count": len(self.attention_units),
            "active_reports": len(self.inspection_reports),
            "modification_events": len(self.self_modification_events),
            "current_state": self.current_state.value,
            "resource_pools": {
                resource_type.value: {
                    "total_capacity": pool.total_capacity,
                    "available_capacity": pool.available_capacity,
                    "utilization": 1.0 - (pool.available_capacity / pool.total_capacity)
                }
                for resource_type, pool in self.resource_pools.items()
            }
        }