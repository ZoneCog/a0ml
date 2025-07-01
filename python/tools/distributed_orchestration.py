import json
from typing import Any, Dict, List
from python.helpers.tool import Tool, Response
from python.helpers.distributed_orchestrator import (
    get_orchestrator, TaskPriority, AtomicSubtask
)
from agent import AgentContext


class DistributedOrchestrationTool(Tool):
    """
    Tool for interacting with the distributed orchestration system
    """
    
    async def execute(self, **kwargs) -> Response:
        """Execute orchestration commands"""
        
        if self.method == "register_agent":
            return await self._register_agent(**kwargs)
        elif self.method == "decompose_goal":
            return await self._decompose_goal(**kwargs)
        elif self.method == "assign_tasks":
            return await self._assign_tasks(**kwargs)
        elif self.method == "get_status":
            return await self._get_status(**kwargs)
        elif self.method == "get_task_tensor":
            return await self._get_task_tensor(**kwargs)
        elif self.method == "mark_completed":
            return await self._mark_completed(**kwargs)
        elif self.method == "update_heartbeat":
            return await self._update_heartbeat(**kwargs)
        else:
            return Response(
                message=f"Unknown orchestration method: {self.method}",
                break_loop=False
            )
    
    async def _register_agent(self, **kwargs) -> Response:
        """Register current agent with the orchestrator"""
        skills = kwargs.get("skills", [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(",")]
        
        orchestrator = get_orchestrator()
        agent_id = orchestrator.register_agent(self.agent.context, skills)
        
        return Response(
            message=f"Agent registered successfully with ID: {agent_id}\nSkills: {skills}",
            break_loop=False
        )
    
    async def _decompose_goal(self, **kwargs) -> Response:
        """Decompose a goal into atomic subtasks"""
        goal = kwargs.get("goal", "")
        context = kwargs.get("context", "")
        
        if not goal:
            return Response(
                message="Goal parameter is required for decomposition",
                break_loop=False
            )
        
        orchestrator = get_orchestrator()
        subtasks = orchestrator.decompose_goal(goal, context)
        
        # Convert subtasks to serializable format
        subtask_data = []
        for subtask in subtasks:
            subtask_data.append({
                "uuid": subtask.uuid,
                "name": subtask.name,
                "description": subtask.description,
                "priority": subtask.priority.name,
                "estimated_duration": subtask.estimated_duration,
                "required_skills": subtask.required_skills,
                "dependencies": subtask.dependencies,
                "state": subtask.state.value
            })
        
        return Response(
            message=f"Goal decomposed into {len(subtasks)} subtasks:\n\n" + 
                   json.dumps(subtask_data, indent=2),
            break_loop=False
        )
    
    async def _assign_tasks(self, **kwargs) -> Response:
        """Assign available subtasks to capable agents"""
        orchestrator = get_orchestrator()
        assignments = orchestrator.assign_subtasks()
        
        if not assignments:
            return Response(
                message="No tasks could be assigned at this time. Check agent availability and dependencies.",
                break_loop=False
            )
        
        # Execute assignments
        results = []
        for subtask, agent_id in assignments:
            try:
                result = await orchestrator.execute_subtask_on_agent(subtask, agent_id)
                results.append(f"✓ {subtask.name} -> Agent {agent_id}")
            except Exception as e:
                results.append(f"✗ {subtask.name} -> Error: {str(e)}")
        
        return Response(
            message=f"Task assignment results:\n\n" + "\n".join(results),
            break_loop=False
        )
    
    async def _get_status(self, **kwargs) -> Response:
        """Get orchestration system status"""
        orchestrator = get_orchestrator()
        status = orchestrator.get_orchestration_status()
        
        return Response(
            message="Distributed Orchestration Status:\n\n" + 
                   json.dumps(status, indent=2, default=str),
            break_loop=False
        )
    
    async def _get_task_tensor(self, **kwargs) -> Response:
        """Get the task tensor representation"""
        orchestrator = get_orchestrator()
        tensor = orchestrator.get_task_tensor()
        
        return Response(
            message=f"Task Tensor T_task[n_tasks, n_agents, p_levels]:\n" +
                   f"Shape: {tensor.shape}\n" +
                   f"Tensor data:\n{tensor}",
            break_loop=False
        )
    
    async def _mark_completed(self, **kwargs) -> Response:
        """Mark a subtask as completed"""
        subtask_uuid = kwargs.get("subtask_uuid", "")
        result = kwargs.get("result", "Task completed successfully")
        
        if not subtask_uuid:
            return Response(
                message="subtask_uuid parameter is required",
                break_loop=False
            )
        
        orchestrator = get_orchestrator()
        orchestrator.mark_subtask_completed(subtask_uuid, result)
        
        return Response(
            message=f"Subtask {subtask_uuid} marked as completed with result: {result}",
            break_loop=False
        )
    
    async def _update_heartbeat(self, **kwargs) -> Response:
        """Update agent heartbeat"""
        orchestrator = get_orchestrator()
        orchestrator.update_agent_heartbeat(self.agent.context.id)
        
        return Response(
            message=f"Heartbeat updated for agent {self.agent.context.id}",
            break_loop=False
        )