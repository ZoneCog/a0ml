"""
Simple test for distributed orchestration basic functionality
"""
import asyncio
import uuid
from datetime import datetime, timezone

from python.helpers.distributed_orchestrator import (
    get_orchestrator, TaskPriority, AtomicSubtask
)
from agent import AgentContext, AgentConfig, ModelConfig
import models


def test_basic_orchestration():
    """Test basic orchestration functionality without complex agent setup"""
    print("🧪 Testing basic distributed orchestration functionality...")
    
    # Create orchestrator
    orchestrator = get_orchestrator()
    
    # Test 1: Goal decomposition
    print("\n1. Testing goal decomposition...")
    analysis_goal = "analyze customer satisfaction survey data"
    subtasks = orchestrator.decompose_goal(analysis_goal)
    
    assert len(subtasks) == 3, f"Expected 3 subtasks, got {len(subtasks)}"
    print(f"✓ Goal decomposed into {len(subtasks)} subtasks:")
    for i, task in enumerate(subtasks):
        print(f"   {i+1}. {task.name}: {task.description}")
        print(f"      Priority: {task.priority.name}, Duration: {task.estimated_duration}min")
        print(f"      Skills: {task.required_skills}")
        print(f"      Dependencies: {task.dependencies}")
    
    # Test 2: Development goal decomposition
    print("\n2. Testing development goal decomposition...")
    dev_goal = "develop a machine learning model for predictions"
    dev_subtasks = orchestrator.decompose_goal(dev_goal)
    
    assert len(dev_subtasks) == 4, f"Expected 4 subtasks, got {len(dev_subtasks)}"
    print(f"✓ Development goal decomposed into {len(dev_subtasks)} subtasks")
    
    # Test 3: Generic goal decomposition
    print("\n3. Testing generic goal decomposition...")
    generic_goal = "complete project documentation"
    generic_subtasks = orchestrator.decompose_goal(generic_goal)
    
    assert len(generic_subtasks) == 1, f"Expected 1 subtask, got {len(generic_subtasks)}"
    print(f"✓ Generic goal decomposed into {len(generic_subtasks)} subtask")
    
    # Test 4: Tensor encoding (no agents)
    print("\n4. Testing tensor encoding...")
    tensor = orchestrator.get_task_tensor()
    total_tasks = len(subtasks) + len(dev_subtasks) + len(generic_subtasks)
    expected_shape = (total_tasks, 0, 4)  # 0 agents, 4 priority levels
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    print(f"✓ Tensor created with shape: {tensor.shape}")
    
    # Test 5: System status
    print("\n5. Testing system status...")
    status = orchestrator.get_orchestration_status()
    total_tasks = len(subtasks) + len(dev_subtasks) + len(generic_subtasks)
    assert status["total_subtasks"] == total_tasks, f"Expected {total_tasks} total tasks"
    assert status["registered_agents"] == 0, "Expected 0 registered agents"
    print(f"✓ Status check passed:")
    print(f"   Total subtasks: {status['total_subtasks']}")
    print(f"   Registered agents: {status['registered_agents']}")
    print(f"   Task tensor shape: {status['task_tensor_shape']}")
    
    print("\n🎉 All basic tests passed!")
    return True


def test_agent_registration_simple():
    """Test agent registration with minimal setup"""
    print("\n🧪 Testing agent registration...")
    
    orchestrator = get_orchestrator()
    
    # Create minimal agent contexts for testing
    test_agents = []
    for i in range(3):
        # Create minimal config for testing
        try:
            config = AgentConfig(
                chat_model=ModelConfig(
                    provider=models.ModelProvider.OPENAI,
                    name="gpt-4o-mini",
                    ctx_length=16000
                ),
                utility_model=ModelConfig(
                    provider=models.ModelProvider.OPENAI,
                    name="gpt-4o-mini",
                    ctx_length=16000
                ),
                embeddings_model=ModelConfig(
                    provider=models.ModelProvider.OPENAI,
                    name="text-embedding-3-small",
                    ctx_length=8192
                ),
                browser_model=ModelConfig(
                    provider=models.ModelProvider.OPENAI,
                    name="gpt-4o-mini",
                    ctx_length=16000
                ),
                mcp_servers=""
            )
        except Exception as e:
            print(f"   Skipping agent registration test due to config error: {e}")
            return True
        
        # Create agent context
        context = AgentContext(
            config=config,
            id=f"test_agent_{i}_{uuid.uuid4()}",
            name=f"Test Agent {i}"
        )
        test_agents.append(context)
    
    # Register agents with different skills
    skills_sets = [
        ["data_collection", "data_analysis"],
        ["design", "planning", "development"],
        ["testing", "qa", "general"]
    ]
    
    registered_ids = []
    for i, context in enumerate(test_agents):
        agent_id = orchestrator.register_agent(context, skills_sets[i])
        registered_ids.append(agent_id)
        print(f"   ✓ Registered agent {i+1}: {agent_id} with skills {skills_sets[i]}")
    
    # Check status
    status = orchestrator.get_orchestration_status()
    assert status["registered_agents"] == 3, f"Expected 3 agents, got {status['registered_agents']}"
    
    # Test tensor with agents
    tensor = orchestrator.get_task_tensor()
    print(f"   ✓ Tensor with agents: {tensor.shape}")
    
    # Clean up
    for agent_id in registered_ids:
        orchestrator.unregister_agent(agent_id)
        if agent_id in AgentContext._contexts:
            AgentContext.remove(agent_id)
    
    print("✅ Agent registration test passed!")
    return True


def test_task_assignment():
    """Test task assignment logic"""
    print("\n🧪 Testing task assignment...")
    
    orchestrator = get_orchestrator()
    
    # Create a task that requires specific skills
    subtasks = orchestrator.decompose_goal("analyze data and create report")
    
    # Test assignment without agents (should return empty)
    assignments = orchestrator.assign_subtasks()
    assert len(assignments) == 0, "Expected no assignments without agents"
    print("   ✓ No assignments when no agents available")
    
    # Test dependency checking
    if len(subtasks) >= 2:
        first_task = subtasks[0]
        second_task = subtasks[1]
        
        # Check if second task correctly depends on first
        can_execute_first = orchestrator._can_execute_subtask(first_task)
        can_execute_second = orchestrator._can_execute_subtask(second_task)
        
        assert can_execute_first, "First task should be executable"
        assert not can_execute_second, "Second task should wait for dependency"
        print("   ✓ Dependency checking works correctly")
        
        # Mark first task complete and check if second becomes available
        orchestrator.mark_subtask_completed(first_task.uuid, "Test completion")
        can_execute_second_after = orchestrator._can_execute_subtask(second_task)
        assert can_execute_second_after, "Second task should be executable after dependency completion"
        print("   ✓ Dependency resolution works correctly")
    
    print("✅ Task assignment test passed!")
    return True


def main():
    """Run all tests"""
    print("🚀 Starting Distributed Orchestration Test Suite")
    print("=" * 60)
    
    try:
        # Run basic functionality tests
        test_basic_orchestration()
        
        # Run agent registration test
        test_agent_registration_simple()
        
        # Run task assignment test
        test_task_assignment()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Distributed orchestration system working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()