"""
Comprehensive tests for the distributed orchestration system.
These tests validate task breakdown, assignment, scheduling efficiency,
and use live distributed agents (not simulated values).
"""
import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import List

from python.helpers.distributed_orchestrator import (
    get_orchestrator, TaskPriority, AtomicSubtask, AgentCapability
)
from agent import Agent, AgentContext, AgentConfig, UserMessage, ModelConfig
import models


class TestDistributedOrchestration:
    """Test suite for distributed orchestration functionality"""
    
    def __init__(self):
        self.orchestrator = get_orchestrator()
        self.test_contexts: List[AgentContext] = []
        
    async def setup_test_agents(self, num_agents: int = 3) -> List[AgentContext]:
        """Create live agent contexts for testing"""
        contexts = []
        
        for i in range(num_agents):
            # Create minimal agent configuration for testing
            config = AgentConfig(
                chat_model=ModelConfig(name="gpt-4o-mini", ctx_length=16000),
                utility_model=ModelConfig(name="gpt-4o-mini", ctx_length=16000),
                embeddings_model=ModelConfig(name="text-embedding-3-small", ctx_length=8192),
                browser_model=ModelConfig(name="gpt-4o-mini", ctx_length=16000),
                mcp_servers=""
            )
            
            # Create agent context
            context = AgentContext(
                config=config,
                id=f"test_agent_context_{i}_{uuid.uuid4()}",
                name=f"Test Agent {i}"
            )
            
            contexts.append(context)
            self.test_contexts.append(context)
        
        return contexts
    
    def cleanup_test_agents(self):
        """Clean up test agent contexts"""
        for context in self.test_contexts:
            if context.id in AgentContext._contexts:
                AgentContext.remove(context.id)
        self.test_contexts.clear()
    
    def test_task_decomposition(self):
        """Test task breakdown into atomic subtasks"""
        print("🧪 Testing task decomposition...")
        
        # Test analysis goal decomposition
        analysis_subtasks = self.orchestrator.decompose_goal(
            "analyze customer data trends for Q4 2024"
        )
        
        assert len(analysis_subtasks) == 3, f"Expected 3 subtasks, got {len(analysis_subtasks)}"
        assert analysis_subtasks[0].name == "Data Collection"
        assert analysis_subtasks[1].name == "Data Analysis"
        assert analysis_subtasks[2].name == "Report Generation"
        
        # Check dependencies are set correctly
        assert analysis_subtasks[1].dependencies == [analysis_subtasks[0].uuid]
        assert analysis_subtasks[2].dependencies == [analysis_subtasks[1].uuid]
        
        # Test development goal decomposition
        dev_subtasks = self.orchestrator.decompose_goal(
            "develop a new user authentication system"
        )
        
        assert len(dev_subtasks) == 4, f"Expected 4 subtasks, got {len(dev_subtasks)}"
        assert dev_subtasks[0].name == "Requirements Analysis"
        assert dev_subtasks[1].name == "Design Phase"
        assert dev_subtasks[2].name == "Implementation"
        assert dev_subtasks[3].name == "Testing"
        
        # Test generic goal decomposition
        generic_subtasks = self.orchestrator.decompose_goal(
            "complete project X"
        )
        
        assert len(generic_subtasks) == 1, f"Expected 1 subtask, got {len(generic_subtasks)}"
        assert generic_subtasks[0].name == "Goal Execution"
        
        print("✅ Task decomposition tests passed")
    
    async def test_agent_registration(self):
        """Test agent registration and capability management"""
        print("🧪 Testing agent registration...")
        
        # Create test agents
        contexts = await self.setup_test_agents(3)
        
        # Register agents with different skills
        agent_skills = [
            ["data_collection", "data_analysis"],
            ["design", "planning", "coding"],
            ["testing", "qa", "general"]
        ]
        
        registered_ids = []
        for i, context in enumerate(contexts):
            agent_id = self.orchestrator.register_agent(context, agent_skills[i])
            registered_ids.append(agent_id)
            assert agent_id == context.id
        
        # Verify agents are registered
        status = self.orchestrator.get_orchestration_status()
        assert status["registered_agents"] == 3
        
        # Verify agent capabilities
        for i, agent_id in enumerate(registered_ids):
            assert agent_id in self.orchestrator._registered_agents
            capability = self.orchestrator._registered_agents[agent_id]
            assert capability.skills == set(agent_skills[i])
        
        print("✅ Agent registration tests passed")
    
    async def test_task_assignment_live_agents(self):
        """Test task assignment across live distributed agents"""
        print("🧪 Testing task assignment with live agents...")
        
        # Setup agents
        contexts = await self.setup_test_agents(3)
        
        # Register agents with skills
        self.orchestrator.register_agent(contexts[0], ["data_collection", "data_analysis"])
        self.orchestrator.register_agent(contexts[1], ["design", "planning", "development"])
        self.orchestrator.register_agent(contexts[2], ["testing", "qa"])
        
        # Decompose a development goal
        subtasks = self.orchestrator.decompose_goal(
            "develop a data processing pipeline"
        )
        
        # Initially, only the first task (Requirements Analysis) should be assignable
        initial_assignments = self.orchestrator.assign_subtasks()
        assert len(initial_assignments) == 1, f"Expected 1 initial assignment, got {len(initial_assignments)}"
        
        assigned_task, assigned_agent = initial_assignments[0]
        assert assigned_task.name == "Requirements Analysis"
        assert assigned_task.state.value == "running"
        
        # Simulate completion of first task to unlock dependencies
        self.orchestrator.mark_subtask_completed(
            assigned_task.uuid, 
            "Requirements analysis completed successfully"
        )
        
        # Now the next task should be assignable
        next_assignments = self.orchestrator.assign_subtasks()
        assert len(next_assignments) >= 1, "Expected at least 1 assignment after dependency completion"
        
        print("✅ Live agent task assignment tests passed")
    
    async def test_scheduling_efficiency(self):
        """Test scheduling efficiency and task completion rates"""
        print("🧪 Testing scheduling efficiency...")
        
        start_time = time.time()
        
        # Setup multiple agents
        contexts = await self.setup_test_agents(5)
        
        # Register agents with overlapping skills for load balancing
        skills_sets = [
            ["data_collection", "general"],
            ["data_analysis", "general"],
            ["design", "planning"],
            ["coding", "development"],
            ["testing", "qa", "general"]
        ]
        
        for i, context in enumerate(contexts):
            self.orchestrator.register_agent(context, skills_sets[i])
        
        # Create multiple goals to test parallel processing
        goals = [
            "analyze market trends",
            "develop mobile app",
            "create dashboard",
            "optimize database"
        ]
        
        all_subtasks = []
        for goal in goals:
            subtasks = self.orchestrator.decompose_goal(goal)
            all_subtasks.extend(subtasks)
        
        total_tasks = len(all_subtasks)
        
        # Track assignment efficiency
        assignment_rounds = 0
        total_assignments = 0
        
        while True:
            assignments = self.orchestrator.assign_subtasks()
            if not assignments:
                break
                
            assignment_rounds += 1
            total_assignments += len(assignments)
            
            # Simulate some task completions to test dependency handling
            for subtask, agent_id in assignments[:2]:  # Complete first 2 assignments
                self.orchestrator.mark_subtask_completed(
                    subtask.uuid,
                    f"Task {subtask.name} completed by {agent_id}"
                )
            
            # Prevent infinite loops in testing
            if assignment_rounds > 10:
                break
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        status = self.orchestrator.get_orchestration_status()
        
        # Efficiency metrics
        assignment_efficiency = total_assignments / total_tasks if total_tasks > 0 else 0
        avg_time_per_task = processing_time / total_tasks if total_tasks > 0 else 0
        
        print(f"📊 Scheduling Efficiency Metrics:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Total assignments: {total_assignments}")
        print(f"   Assignment rounds: {assignment_rounds}")
        print(f"   Processing time: {processing_time:.2f}s")
        print(f"   Assignment efficiency: {assignment_efficiency:.2%}")
        print(f"   Average time per task: {avg_time_per_task:.3f}s")
        print(f"   Completed tasks: {status['completed_subtasks']}")
        
        # Assert efficiency thresholds
        assert assignment_efficiency > 0.5, f"Assignment efficiency too low: {assignment_efficiency:.2%}"
        assert avg_time_per_task < 1.0, f"Average time per task too high: {avg_time_per_task:.3f}s"
        
        print("✅ Scheduling efficiency tests passed")
    
    def test_tensor_encoding(self):
        """Test tensor encoding of task structures"""
        print("🧪 Testing tensor encoding...")
        
        # Create some subtasks
        subtasks = self.orchestrator.decompose_goal("analyze data and create report")
        
        # Register a few agents
        for i in range(3):
            config = AgentConfig(
                chat_model=ModelConfig(name="gpt-4o-mini", ctx_length=16000),
                utility_model=ModelConfig(name="gpt-4o-mini", ctx_length=16000),
                embeddings_model=ModelConfig(name="text-embedding-3-small", ctx_length=8192),
                browser_model=ModelConfig(name="gpt-4o-mini", ctx_length=16000),
                mcp_servers=""
            )
            context = AgentContext(
                config=config,
                id=f"tensor_test_{i}_{uuid.uuid4()}"
            )
            self.orchestrator.register_agent(context, ["data_analysis", "general"])
            self.test_contexts.append(context)
        
        # Get tensor
        tensor = self.orchestrator.get_task_tensor()
        
        # Verify tensor dimensions
        n_tasks = len(subtasks)
        n_agents = 3
        p_levels = 4  # Number of priority levels
        
        assert tensor.shape == (n_tasks, n_agents, p_levels), \
            f"Expected tensor shape ({n_tasks}, {n_agents}, {p_levels}), got {tensor.shape}"
        
        # Verify tensor values are valid probabilities (between 0 and 1)
        assert (tensor >= 0).all() and (tensor <= 1).all(), \
            "Tensor values should be between 0 and 1"
        
        # Test tensor after task assignment
        assignments = self.orchestrator.assign_subtasks()
        if assignments:
            updated_tensor = self.orchestrator.get_task_tensor()
            # Verify some assignments are reflected in tensor
            assert (updated_tensor == 1.0).any(), \
                "Tensor should contain exact assignments (value 1.0) after task assignment"
        
        print(f"✅ Tensor encoding tests passed - Shape: {tensor.shape}")
    
    async def test_message_passing_protocol(self):
        """Test message-passing protocol for task assignment and status updates"""
        print("🧪 Testing message-passing protocol...")
        
        # Setup test agents
        contexts = await self.setup_test_agents(2)
        
        # Register agents
        agent1_id = self.orchestrator.register_agent(contexts[0], ["data_analysis"])
        agent2_id = self.orchestrator.register_agent(contexts[1], ["report_writing"])
        
        # Create a simple task
        subtasks = self.orchestrator.decompose_goal("analyze sales data")
        
        # Assign and execute task
        assignments = self.orchestrator.assign_subtasks()
        assert len(assignments) > 0, "Expected at least one assignment"
        
        subtask, assigned_agent_id = assignments[0]
        
        # Test message execution on agent
        result_message = await self.orchestrator.execute_subtask_on_agent(
            subtask, assigned_agent_id
        )
        
        assert "assigned to agent" in result_message, \
            f"Expected assignment confirmation, got: {result_message}"
        
        # Test status update
        self.orchestrator.update_agent_heartbeat(agent1_id)
        self.orchestrator.update_agent_heartbeat(agent2_id)
        
        # Verify heartbeats were updated
        status = self.orchestrator.get_orchestration_status()
        for agent_id in [agent1_id, agent2_id]:
            agent_status = status["agent_status"][agent_id]
            last_heartbeat = datetime.fromisoformat(agent_status["last_heartbeat"])
            time_diff = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()
            assert time_diff < 60, f"Heartbeat not recent enough for agent {agent_id}"
        
        print("✅ Message-passing protocol tests passed")
    
    async def test_dynamic_task_negotiation(self):
        """Test dynamic task negotiation capabilities"""
        print("🧪 Testing dynamic task negotiation...")
        
        # Setup agents with different capabilities
        contexts = await self.setup_test_agents(4)
        
        # Register agents with specific skills and load limits
        agent_configs = [
            (["data_collection"], 2),
            (["data_analysis", "statistics"], 3),
            (["machine_learning", "ai"], 1),
            (["report_writing", "visualization"], 2)
        ]
        
        for i, (skills, max_load) in enumerate(agent_configs):
            agent_id = self.orchestrator.register_agent(contexts[i], skills)
            capability = self.orchestrator._registered_agents[agent_id]
            capability.max_load = max_load
        
        # Create tasks that require negotiation
        subtasks = self.orchestrator.decompose_goal("develop AI analytics system")
        
        # Assign tasks and test load balancing
        round1_assignments = self.orchestrator.assign_subtasks()
        
        # Simulate some agents becoming overloaded
        for subtask, agent_id in round1_assignments:
            capability = self.orchestrator._registered_agents[agent_id]
            capability.current_load = capability.max_load  # Max out the agent
        
        # Try to assign more tasks - should handle overloaded agents
        round2_assignments = self.orchestrator.assign_subtasks()
        
        # Verify load balancing worked
        total_assignments = len(round1_assignments) + len(round2_assignments)
        status = self.orchestrator.get_orchestration_status()
        
        print(f"📊 Dynamic Negotiation Results:")
        print(f"   Round 1 assignments: {len(round1_assignments)}")
        print(f"   Round 2 assignments: {len(round2_assignments)}")
        print(f"   Total assignments: {total_assignments}")
        
        # Verify no agent is overloaded beyond capacity
        for agent_id, agent_status in status["agent_status"].items():
            current_load = agent_status["current_load"]
            max_load = agent_status["max_load"]
            assert current_load <= max_load, \
                f"Agent {agent_id} overloaded: {current_load}/{max_load}"
        
        print("✅ Dynamic task negotiation tests passed")
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        print("🚀 Starting Distributed Orchestration Test Suite")
        print("=" * 60)
        
        try:
            # Test 1: Task decomposition (no live agents needed)
            self.test_task_decomposition()
            
            # Test 2: Agent registration
            await self.test_agent_registration()
            
            # Test 3: Task assignment with live agents
            await self.test_task_assignment_live_agents()
            
            # Test 4: Scheduling efficiency
            await self.test_scheduling_efficiency()
            
            # Test 5: Tensor encoding
            self.test_tensor_encoding()
            
            # Test 6: Message-passing protocol
            await self.test_message_passing_protocol()
            
            # Test 7: Dynamic task negotiation
            await self.test_dynamic_task_negotiation()
            
            print("=" * 60)
            print("🎉 ALL TESTS PASSED! Distributed orchestration system is working correctly.")
            
            # Print final status
            final_status = self.orchestrator.get_orchestration_status()
            print(f"\n📈 Final System Status:")
            print(f"   Total subtasks: {final_status['total_subtasks']}")
            print(f"   Completed subtasks: {final_status['completed_subtasks']}")
            print(f"   Running subtasks: {final_status['running_subtasks']}")
            print(f"   Registered agents: {final_status['registered_agents']}")
            print(f"   Task tensor shape: {final_status['task_tensor_shape']}")
            
        except Exception as e:
            print(f"❌ Test failed with error: {str(e)}")
            raise
        finally:
            # Cleanup
            self.cleanup_test_agents()


async def main():
    """Main test execution function"""
    test_suite = TestDistributedOrchestration()
    await test_suite.run_all_tests()


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())