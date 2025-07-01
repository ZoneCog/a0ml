"""
Example integration of AtomSpace Hypergraph Memory with Agent Zero

This demonstrates how to use the new hypergraph memory system alongside
the existing memory infrastructure.
"""

import asyncio
import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.helpers.memory_atomspace import HypergraphMemoryAgent


async def agent_with_hypergraph_memory_example():
    """
    Example showing how an agent can use hypergraph memory for enhanced
    cognitive capabilities.
    """
    
    print("🤖 Agent Zero with Hypergraph Memory Integration")
    print("=" * 55)
    
    # Create an agent with hypergraph memory capabilities
    agent = HypergraphMemoryAgent("agent_zero_demo")
    memory = await agent.initialize_hypergraph_memory()
    
    print("\n1. 🧠 Enhanced Memory Storage")
    print("-" * 30)
    
    # Store agent experiences with rich context
    experience_id = await agent.remember_with_context(
        content="Successfully completed user task: web scraping with rate limiting",
        context_concepts=["web_scraping", "rate_limiting", "task_completion", "user_satisfaction"],
        memory_type="experience"
    )
    print(f"✓ Stored experience with rich context: {experience_id[:8]}...")
    
    # Store learned solutions
    solution_id = await agent.remember_with_context(
        content="When rate limited, implement exponential backoff with jitter",
        context_concepts=["rate_limiting", "exponential_backoff", "error_handling", "resilience"],
        memory_type="solution"
    )
    print(f"✓ Stored solution pattern: {solution_id[:8]}...")
    
    # Store domain knowledge
    knowledge_id = await agent.remember_with_context(
        content="Python requests library supports session objects for connection pooling",
        context_concepts=["python", "requests", "connection_pooling", "performance"],
        memory_type="knowledge"
    )
    print(f"✓ Stored domain knowledge: {knowledge_id[:8]}...")
    
    print("\n2. 🔍 Intelligent Memory Recall")
    print("-" * 30)
    
    # Agent faces a new web scraping task and recalls relevant experience
    recall_result = await agent.recall_by_association(
        "web scraping performance optimization",
        include_related=True
    )
    
    print("When facing new web scraping task, agent recalls:")
    for i, result in enumerate(recall_result['concept_results'][:3]):
        atom = result['atom']
        if isinstance(atom, dict):
            metadata = atom.get('metadata', {})
            if isinstance(metadata, str):
                import json
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            content = metadata.get('content', atom['name'])
        else:
            content = str(atom)
        print(f"  • {content[:60]}...")
    
    if recall_result['graph_results']:
        print("Related concepts discovered:")
        for result in recall_result['graph_results'][:3]:
            atom = result['atom']
            print(f"  • {atom['name']} (via {result['relationship']})")
    
    print("\n3. 📊 Memory Evolution Tracking")
    print("-" * 30)
    
    # Create learning checkpoint
    checkpoint = await agent.create_memory_checkpoint("Post web-scraping learning")
    print(f"✓ Created learning checkpoint: {checkpoint[:8]}...")
    
    # Show memory growth
    tensor_state = await memory.get_memory_tensor_state()
    shape = tensor_state['tensor_shape']
    print(f"📈 Agent's memory contains:")
    print(f"   • {shape[0] if len(shape) > 0 else 0} knowledge nodes")
    print(f"   • {shape[1] if len(shape) > 1 else 0} conceptual relationships")
    print(f"   • {shape[2] if len(shape) > 2 else 0} temporal snapshots")
    
    print("\n4. 🌐 Agent Collaboration Capability")
    print("-" * 30)
    
    # Demonstrate how agents can share knowledge
    specialist_agent = HypergraphMemoryAgent("web_scraping_specialist")
    specialist_memory = await specialist_agent.initialize_hypergraph_memory()
    
    # Specialist shares domain expertise
    await specialist_agent.remember_with_context(
        content="BeautifulSoup is excellent for HTML parsing but lxml is faster for large documents",
        context_concepts=["html_parsing", "beautifulsoup", "lxml", "performance_optimization"],
        memory_type="expertise"
    )
    
    # Main agent discovers specialist knowledge through pattern matching
    html_knowledge = await memory.read_hypergraph_fragment({
        "concept_type": "knowledge",
        "metadata": "*html_parsing*"
    })
    
    print(f"🤝 Agent discovered {len(html_knowledge)} pieces of specialist knowledge")
    
    print("\n5. ⚡ Real-time Pattern Recognition")
    print("-" * 30)
    
    # Agent encounters error pattern and finds related solutions
    error_pattern = await memory.read_hypergraph_fragment({
        "atom_type": "node",
        "concept_type": "solution",
        "truth_value_min": 0.8
    })
    
    print(f"🎯 Found {len(error_pattern)} high-confidence solution patterns")
    
    # Show relationship discovery
    relationships = await memory.read_hypergraph_fragment({
        "atom_type": "link",
        "link_type": "relates_to"
    })
    
    print(f"🔗 Agent has learned {len(relationships)} conceptual relationships")
    
    print("\n" + "=" * 55)
    print("✨ Agent Zero Enhanced with Hypergraph Memory!")
    print("\n🚀 Benefits for Agent Zero:")
    print("   • Rich contextual memory beyond simple vector similarity")
    print("   • Associative recall mimicking human-like memory")
    print("   • Temporal tracking of learning and experience")
    print("   • Cross-agent knowledge sharing and collaboration")
    print("   • Pattern recognition for solution reuse")
    print("   • Persistent memory across sessions and restarts")


# Example of how this could integrate with existing Agent class
class AgentZeroWithHypergraphMemory:
    """
    Extended Agent Zero with hypergraph memory capabilities
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.hypergraph_agent = None
    
    async def initialize_enhanced_memory(self):
        """Initialize enhanced memory alongside existing memory"""
        self.hypergraph_agent = HypergraphMemoryAgent(self.agent_id)
        await self.hypergraph_agent.initialize_hypergraph_memory()
    
    async def remember_experience(self, experience: str, context: list = None):
        """Remember an experience with rich context"""
        if self.hypergraph_agent:
            return await self.hypergraph_agent.remember_with_context(
                content=experience,
                context_concepts=context or [],
                memory_type="experience"
            )
    
    async def recall_similar_experiences(self, situation: str):
        """Recall similar experiences using associative memory"""
        if self.hypergraph_agent:
            return await self.hypergraph_agent.recall_by_association(
                query=situation,
                include_related=True
            )
        return {"concept_results": [], "graph_results": []}
    
    async def create_learning_checkpoint(self, description: str = None):
        """Create a checkpoint of current learning state"""
        if self.hypergraph_agent:
            return await self.hypergraph_agent.create_memory_checkpoint(description)


if __name__ == "__main__":
    asyncio.run(agent_with_hypergraph_memory_example())