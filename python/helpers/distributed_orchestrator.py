import asyncio
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from enum import Enum
from dataclasses import dataclass, field
import uuid
import threading
from collections import defaultdict, deque
import heapq

from python.helpers.task_scheduler import (
    TaskScheduler, BaseTask, ScheduledTask, AdHocTask, PlannedTask, 
    TaskState, TaskType, TaskPlan
)
from agent import Agent, AgentContext, UserMessage


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class AtomicSubtask:
    """Represents an atomic subtask that cannot be further decomposed"""
    uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    parent_task_uuid: str = ""
    dependencies: List[str] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: int = 0  # in minutes
    required_skills: List[str] = field(default_factory=list)
    assigned_agent_id: Optional[str] = None
    state: TaskState = TaskState.IDLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[str] = None


@dataclass
class AgentCapability:
    """Describes an agent's capabilities for task assignment"""
    agent_id: str
    agent_context: AgentContext
    skills: Set[str] = field(default_factory=set)
    current_load: int = 0  # number of active tasks
    max_load: int = 5
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def can_handle_task(self, subtask: AtomicSubtask) -> bool:
        """Check if this agent can handle the given subtask"""
        if self.current_load >= self.max_load:
            return False
        
        # Check if agent has required skills
        required_skills = set(subtask.required_skills)
        if not required_skills:
            return True  # Task has no specific skill requirements
        
        # Check for exact skill matches first
        if required_skills.issubset(self.skills):
            return True
        
        # Check for skill category matches (more flexible matching)
        agent_skills_lower = {skill.lower().replace('_', ' ') for skill in self.skills}
        required_skills_lower = {skill.lower().replace('_', ' ') for skill in required_skills}
        
        # Define skill equivalencies
        skill_mappings = {
            'requirements analysis': ['planning', 'analysis', 'design'],
            'data collection': ['data analysis', 'statistics'],
            'data analysis': ['data collection', 'statistics', 'analysis'],
            'report writing': ['communication', 'documentation', 'visualization'],
            'coding': ['development', 'programming', 'implementation'],
            'development': ['coding', 'programming', 'implementation'],
            'testing': ['qa', 'validation', 'quality assurance'],
            'qa': ['testing', 'validation', 'quality assurance'],
            'design': ['planning', 'architecture', 'requirements analysis'],
            'planning': ['design', 'coordination', 'requirements analysis'],
            'general': ['coordination', 'management', 'planning']
        }
        
        # Check if agent has equivalent skills
        for required_skill in required_skills_lower:
            # Check direct match
            if required_skill in agent_skills_lower:
                return True
            
            # Check equivalent skills
            equivalent_skills = skill_mappings.get(required_skill, [])
            if any(equiv_skill in agent_skills_lower for equiv_skill in equivalent_skills):
                return True
        
        # Special case: agents with 'general' skill can handle basic tasks
        if 'general' in self.skills:
            return True
            
        return False


class DistributedOrchestrator:
    """
    Distributed orchestration agent for task decomposition and coordination
    """
    
    def __init__(self):
        self._registered_agents: Dict[str, AgentCapability] = {}
        self._task_queue: List[Tuple[int, AtomicSubtask]] = []  # priority queue
        self._subtasks: Dict[str, AtomicSubtask] = {}
        self._task_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._message_queue: deque = deque()
        
    def register_agent(self, agent_context: AgentContext, skills: Optional[List[str]] = None) -> str:
        """Register an agent with the orchestrator"""
        with self._lock:
            agent_id = agent_context.id
            capability = AgentCapability(
                agent_id=agent_id,
                agent_context=agent_context,
                skills=set(skills or []),
                current_load=0,
                max_load=5,
                last_heartbeat=datetime.now(timezone.utc)
            )
            self._registered_agents[agent_id] = capability
            return agent_id
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the orchestrator"""
        with self._lock:
            if agent_id in self._registered_agents:
                # Reassign any tasks that were assigned to this agent
                for subtask in self._subtasks.values():
                    if subtask.assigned_agent_id == agent_id and subtask.state == TaskState.RUNNING:
                        subtask.assigned_agent_id = None
                        subtask.state = TaskState.IDLE
                        self._enqueue_subtask(subtask)
                
                del self._registered_agents[agent_id]
    
    def update_agent_heartbeat(self, agent_id: str):
        """Update agent heartbeat to indicate it's still alive"""
        with self._lock:
            if agent_id in self._registered_agents:
                self._registered_agents[agent_id].last_heartbeat = datetime.now(timezone.utc)
    
    def decompose_goal(self, goal: str, context: str = "") -> List[AtomicSubtask]:
        """
        Decompose a high-level goal into atomic subtasks
        This is a simplified version - in practice this would use LLM reasoning
        """
        # For now, implement a simple rule-based decomposition
        # This would be enhanced with LLM-based decomposition in production
        
        subtasks = []
        
        # Simple patterns for common goals
        if "analyze" in goal.lower():
            subtasks.extend([
                AtomicSubtask(
                    name="Data Collection",
                    description=f"Collect relevant data for: {goal}",
                    priority=TaskPriority.HIGH,
                    estimated_duration=30,
                    required_skills=["data_collection"]
                ),
                AtomicSubtask(
                    name="Data Analysis", 
                    description=f"Analyze collected data for: {goal}",
                    priority=TaskPriority.HIGH,
                    estimated_duration=60,
                    required_skills=["data_analysis"],
                    dependencies=[]  # Will be set after all subtasks are created
                ),
                AtomicSubtask(
                    name="Report Generation",
                    description=f"Generate analysis report for: {goal}",
                    priority=TaskPriority.MEDIUM,
                    estimated_duration=30,
                    required_skills=["report_writing"]
                )
            ])
            
            # Set dependencies: Data Collection -> Data Analysis -> Report Generation
            if len(subtasks) >= 3:
                subtasks[1].dependencies = [subtasks[0].uuid]
                subtasks[2].dependencies = [subtasks[1].uuid]
        
        elif "develop" in goal.lower() or "create" in goal.lower():
            subtasks.extend([
                AtomicSubtask(
                    name="Requirements Analysis",
                    description=f"Analyze requirements for: {goal}",
                    priority=TaskPriority.HIGH,
                    estimated_duration=45,
                    required_skills=["requirements_analysis"]
                ),
                AtomicSubtask(
                    name="Design Phase",
                    description=f"Design solution for: {goal}",
                    priority=TaskPriority.HIGH,
                    estimated_duration=90,
                    required_skills=["design", "planning"]
                ),
                AtomicSubtask(
                    name="Implementation",
                    description=f"Implement solution for: {goal}",
                    priority=TaskPriority.MEDIUM,
                    estimated_duration=120,
                    required_skills=["coding", "development"]
                ),
                AtomicSubtask(
                    name="Testing",
                    description=f"Test solution for: {goal}",
                    priority=TaskPriority.MEDIUM,
                    estimated_duration=60,
                    required_skills=["testing", "qa"]
                )
            ])
            
            # Set dependencies: Requirements -> Design -> Implementation, Testing depends on Implementation
            if len(subtasks) >= 4:
                subtasks[1].dependencies = [subtasks[0].uuid]
                subtasks[2].dependencies = [subtasks[1].uuid]
                subtasks[3].dependencies = [subtasks[2].uuid]
        
        else:
            # Generic decomposition for other goals
            subtasks.append(AtomicSubtask(
                name="Goal Execution",
                description=f"Execute goal: {goal}",
                priority=TaskPriority.MEDIUM,
                estimated_duration=60,
                required_skills=["general"]
            ))
        
        # Store subtasks and enqueue ready ones
        with self._lock:
            for subtask in subtasks:
                self._subtasks[subtask.uuid] = subtask
                # Add to dependency tracking
                for dep in subtask.dependencies:
                    self._task_dependencies[dep].add(subtask.uuid)
                
                # Enqueue tasks that can be executed immediately (no dependencies)
                if not subtask.dependencies:
                    self._enqueue_subtask(subtask)
        
        return subtasks
    
    def _enqueue_subtask(self, subtask: AtomicSubtask):
        """Add subtask to priority queue"""
        priority_value = subtask.priority.value
        # Use creation time as tiebreaker to avoid comparison issues
        tiebreaker = subtask.created_at.timestamp()
        heapq.heappush(self._task_queue, (priority_value, tiebreaker, subtask))
    
    def _can_execute_subtask(self, subtask: AtomicSubtask) -> bool:
        """Check if all dependencies of a subtask are completed"""
        with self._lock:
            for dep_uuid in subtask.dependencies:
                if dep_uuid in self._subtasks:
                    dep_task = self._subtasks[dep_uuid]
                    if dep_task.state != TaskState.IDLE or dep_task.completed_at is None:
                        return False
        return True
    
    def assign_subtasks(self) -> List[Tuple[AtomicSubtask, str]]:
        """
        Assign available subtasks to capable agents
        Returns list of (subtask, agent_id) pairs that were assigned
        """
        assignments = []
        
        with self._lock:
            # Process queue in priority order
            remaining_queue = []
            
            while self._task_queue:
                priority, tiebreaker, subtask = heapq.heappop(self._task_queue)
                
                # Check if subtask can be executed (dependencies met)
                if not self._can_execute_subtask(subtask):
                    remaining_queue.append((priority, tiebreaker, subtask))
                    continue
                
                # Find capable agent
                best_agent = None
                best_score = float('inf')
                
                for agent_capability in self._registered_agents.values():
                    if agent_capability.can_handle_task(subtask):
                        # Score based on current load and skill match
                        score = agent_capability.current_load
                        if score < best_score:
                            best_score = score
                            best_agent = agent_capability
                
                if best_agent:
                    # Assign task to agent
                    subtask.assigned_agent_id = best_agent.agent_id
                    subtask.state = TaskState.RUNNING
                    subtask.started_at = datetime.now(timezone.utc)
                    best_agent.current_load += 1
                    
                    assignments.append((subtask, best_agent.agent_id))
                else:
                    # No capable agent available, put back in queue
                    remaining_queue.append((priority, tiebreaker, subtask))
            
            # Restore unassigned tasks to queue
            self._task_queue = remaining_queue
            heapq.heapify(self._task_queue)
        
        return assignments
    
    async def execute_subtask_on_agent(self, subtask: AtomicSubtask, agent_id: str) -> str:
        """Execute a subtask on the specified agent"""
        if agent_id not in self._registered_agents:
            raise ValueError(f"Agent {agent_id} not registered")
        
        agent_capability = self._registered_agents[agent_id]
        agent_context = agent_capability.agent_context
        
        # Create a user message for the subtask
        message = UserMessage(
            message=f"Execute subtask: {subtask.name}\n\nDescription: {subtask.description}",
            system_message=[f"You are executing a subtask as part of a distributed system. Focus on: {subtask.description}"]
        )
        
        # Execute on agent
        task = agent_context.communicate(message)
        
        # For now, we'll return immediately
        # In production, this would wait for completion and handle async execution
        return f"Subtask {subtask.name} assigned to agent {agent_id}"
    
    def mark_subtask_completed(self, subtask_uuid: str, result: str):
        """Mark a subtask as completed and update dependencies"""
        with self._lock:
            if subtask_uuid in self._subtasks:
                subtask = self._subtasks[subtask_uuid]
                subtask.state = TaskState.IDLE
                subtask.completed_at = datetime.now(timezone.utc)
                subtask.result = result
                
                # Update agent load
                if subtask.assigned_agent_id and subtask.assigned_agent_id in self._registered_agents:
                    self._registered_agents[subtask.assigned_agent_id].current_load -= 1
                
                # Check if any dependent tasks can now be executed
                if subtask_uuid in self._task_dependencies:
                    for dependent_uuid in self._task_dependencies[subtask_uuid]:
                        if dependent_uuid in self._subtasks:
                            dependent_task = self._subtasks[dependent_uuid]
                            if dependent_task.state == TaskState.IDLE and self._can_execute_subtask(dependent_task):
                                self._enqueue_subtask(dependent_task)
    
    def get_task_tensor(self) -> np.ndarray:
        """
        Encode task structures as tensors: T_task[n_tasks, n_agents, p_levels]
        
        Returns a 3D tensor where:
        - Dimension 0: tasks (subtasks)
        - Dimension 1: agents  
        - Dimension 2: priority levels (0=critical, 1=high, 2=medium, 3=low)
        """
        with self._lock:
            n_tasks = len(self._subtasks)
            n_agents = len(self._registered_agents)
            p_levels = len(TaskPriority)
            
            if n_tasks == 0 or n_agents == 0:
                return np.zeros((n_tasks, n_agents, p_levels))
            
            # Create tensor
            tensor = np.zeros((n_tasks, n_agents, p_levels))
            
            # Map subtasks and agents to indices
            task_to_idx = {uuid: idx for idx, uuid in enumerate(self._subtasks.keys())}
            agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(self._registered_agents.keys())}
            
            # Fill tensor
            for task_uuid, subtask in self._subtasks.items():
                task_idx = task_to_idx[task_uuid]
                priority_idx = subtask.priority.value
                
                if subtask.assigned_agent_id and subtask.assigned_agent_id in agent_to_idx:
                    agent_idx = agent_to_idx[subtask.assigned_agent_id]
                    tensor[task_idx, agent_idx, priority_idx] = 1.0
                else:
                    # Task not assigned - distribute probability across capable agents
                    capable_agents = []
                    for agent_id, capability in self._registered_agents.items():
                        if capability.can_handle_task(subtask):
                            capable_agents.append(agent_to_idx[agent_id])
                    
                    if capable_agents:
                        prob = 1.0 / len(capable_agents)
                        for agent_idx in capable_agents:
                            tensor[task_idx, agent_idx, priority_idx] = prob
            
            return tensor
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current status of the orchestration system"""
        with self._lock:
            total_tasks = len(self._subtasks)
            completed_tasks = sum(1 for task in self._subtasks.values() if task.completed_at is not None)
            running_tasks = sum(1 for task in self._subtasks.values() if task.state == TaskState.RUNNING)
            pending_tasks = total_tasks - completed_tasks - running_tasks
            
            agent_status = {}
            for agent_id, capability in self._registered_agents.items():
                agent_status[agent_id] = {
                    "current_load": capability.current_load,
                    "max_load": capability.max_load,
                    "skills": list(capability.skills),
                    "last_heartbeat": capability.last_heartbeat.isoformat()
                }
            
            return {
                "total_subtasks": total_tasks,
                "completed_subtasks": completed_tasks,
                "running_subtasks": running_tasks,
                "pending_subtasks": pending_tasks,
                "registered_agents": len(self._registered_agents),
                "agent_status": agent_status,
                "task_tensor_shape": self.get_task_tensor().shape
            }


# Global orchestrator instance
_orchestrator_instance: Optional[DistributedOrchestrator] = None


def get_orchestrator() -> DistributedOrchestrator:
    """Get the global orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = DistributedOrchestrator()
    return _orchestrator_instance