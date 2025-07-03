"""
Comprehensive demonstration of the distributed orchestration system
showing end-to-end workflow with multiple live agents
"""
import asyncio
import uuid
from datetime import datetime, timezone

from python.helpers.distributed_orchestrator import get_orchestrator, TaskPriority
from agent import AgentContext, AgentConfig, ModelConfig
import models


async def demo_distributed_workflow():
    """Demonstrate a complete distributed workflow"""
    print("🚀 Distributed Orchestration System Demo")
    print("=" * 60)
    
    # Get orchestrator instance
    orchestrator = get_orchestrator()
    
    # Create specialized agents for different capabilities
    print("\n1. Setting up specialized agents...")
    
    agent_specs = [
        ("Data Analyst", ["data_collection", "data_analysis", "statistics"]),
        ("ML Engineer", ["machine_learning", "model_training", "ai"]),
        ("Report Writer", ["report_writing", "visualization", "communication"]),
        ("QA Tester", ["testing", "qa", "validation"]),
        ("Project Manager", ["planning", "coordination", "general"])
    ]
    
    agents = []
    for name, skills in agent_specs:
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
            
            context = AgentContext(
                config=config,
                id=f"demo_agent_{uuid.uuid4()}",
                name=name
            )
            
            agent_id = orchestrator.register_agent(context, skills)
            agents.append((agent_id, name, skills, context))
            print(f"   ✓ {name} registered with skills: {skills}")
            
        except Exception as e:
            print(f"   ⚠️  Skipping {name} due to config error: {e}")
    
    if not agents:
        print("   ⚠️  No agents registered, using simplified demo")
        return await demo_without_agents()
    
    # Demonstrate complex goal decomposition
    print(f"\n2. Decomposing complex goals...")
    
    complex_goals = [
        "Develop a machine learning model to predict customer churn using historical data",
        "Analyze quarterly sales performance and create executive dashboard",
        "Build automated testing framework for web application"
    ]
    
    all_subtasks = []
    for i, goal in enumerate(complex_goals):
        print(f"\n   Goal {i+1}: {goal}")
        subtasks = orchestrator.decompose_goal(goal)
        all_subtasks.extend(subtasks)
        
        for j, task in enumerate(subtasks):
            print(f"      {j+1}. {task.name} (Priority: {task.priority.name})")
            print(f"         Skills needed: {task.required_skills}")
            print(f"         Duration: {task.estimated_duration}min")
            if task.dependencies:
                print(f"         Dependencies: {len(task.dependencies)} task(s)")
    
    print(f"\n   Total subtasks created: {len(all_subtasks)}")
    
    # Show tensor representation
    print(f"\n3. Task tensor representation...")
    tensor = orchestrator.get_task_tensor()
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   [n_tasks={tensor.shape[0]}, n_agents={tensor.shape[1]}, p_levels={tensor.shape[2]}]")
    
    # Demonstrate task assignment rounds
    print(f"\n4. Task assignment simulation...")
    round_num = 1
    total_assigned = 0
    completed_tasks = 0
    
    while round_num <= 5:  # Limit to 5 rounds for demo
        print(f"\n   Round {round_num}:")
        
        # Assign available tasks
        assignments = orchestrator.assign_subtasks()
        
        if not assignments:
            print("      No tasks available for assignment")
            break
        
        total_assigned += len(assignments)
        print(f"      Assigned {len(assignments)} tasks:")
        
        for task, agent_id in assignments:
            agent_name = next((name for aid, name, _, _ in agents if aid == agent_id), "Unknown")
            print(f"         • {task.name} → {agent_name}")
            
            # Simulate task execution and completion
            result = await simulate_task_execution(orchestrator, task, agent_id, agent_name)
            if result:
                completed_tasks += 1
        
        # Update heartbeats
        for agent_id, name, _, _ in agents:
            orchestrator.update_agent_heartbeat(agent_id)
        
        round_num += 1
        
        # Add small delay to simulate real work
        await asyncio.sleep(0.1)
    
    # Show final status
    print(f"\n5. Final orchestration status...")
    status = orchestrator.get_orchestration_status()
    
    print(f"   📊 Performance Metrics:")
    print(f"      Total subtasks: {status['total_subtasks']}")
    print(f"      Completed subtasks: {status['completed_subtasks']}")
    print(f"      Running subtasks: {status['running_subtasks']}")
    print(f"      Pending subtasks: {status['pending_subtasks']}")
    print(f"      Total assigned: {total_assigned}")
    print(f"      Registered agents: {status['registered_agents']}")
    
    efficiency = (status['completed_subtasks'] / status['total_subtasks'] * 100) if status['total_subtasks'] > 0 else 0
    print(f"      Completion rate: {efficiency:.1f}%")
    
    # Show agent utilization
    print(f"\n   🤖 Agent Utilization:")
    for agent_info in status['agent_status'].values():
        utilization = (agent_info['current_load'] / agent_info['max_load'] * 100) if agent_info['max_load'] > 0 else 0
        print(f"      {utilization:.1f}% load, Skills: {', '.join(agent_info['skills'])}")
    
    # Cleanup
    print(f"\n6. Cleanup...")
    for agent_id, name, _, context in agents:
        orchestrator.unregister_agent(agent_id)
        if agent_id in AgentContext._contexts:
            AgentContext.remove(agent_id)
        print(f"   ✓ {name} unregistered")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed! Distributed orchestration system working effectively.")
    
    return True


async def simulate_task_execution(orchestrator, task, agent_id, agent_name):
    """Simulate task execution by an agent"""
    try:
        # Simulate some task execution time
        await asyncio.sleep(0.05)
        
        # Create realistic completion results based on task type
        if "data" in task.name.lower():
            result = f"Data processed successfully by {agent_name}. Found key insights and patterns."
        elif "analysis" in task.name.lower():
            result = f"Analysis completed by {agent_name}. Generated comprehensive findings and recommendations."
        elif "design" in task.name.lower():
            result = f"Design phase completed by {agent_name}. Created detailed specifications and architecture."
        elif "implementation" in task.name.lower():
            result = f"Implementation finished by {agent_name}. Code developed and unit tested."
        elif "testing" in task.name.lower():
            result = f"Testing completed by {agent_name}. All test cases passed with 98% coverage."
        elif "report" in task.name.lower():
            result = f"Report generated by {agent_name}. Executive summary and detailed findings documented."
        else:
            result = f"Task '{task.name}' completed successfully by {agent_name}."
        
        # Mark task as completed
        orchestrator.mark_subtask_completed(task.uuid, result)
        return True
        
    except Exception as e:
        print(f"         ❌ Error executing {task.name}: {e}")
        return False


async def demo_without_agents():
    """Simplified demo without agent registration"""
    print("   Running simplified demo without agent registration...")
    
    orchestrator = get_orchestrator()
    
    # Just show decomposition capabilities
    goal = "Analyze customer data and generate insights"
    subtasks = orchestrator.decompose_goal(goal)
    
    print(f"   ✓ Decomposed goal into {len(subtasks)} subtasks:")
    for i, task in enumerate(subtasks):
        print(f"      {i+1}. {task.name}")
    
    return True


if __name__ == "__main__":
    asyncio.run(demo_distributed_workflow())