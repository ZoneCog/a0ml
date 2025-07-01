# AtomSpace Hypergraph Memory System

This implementation provides a distributed memory agent with hypergraph AtomSpace integration for Agent Zero, enabling sophisticated cognitive representation and pattern matching capabilities.

## Overview

The AtomSpace Hypergraph Memory System extends Agent Zero with:

- **Hypergraph representation** of knowledge as nodes and links
- **Persistent storage** using SQLite for time-indexed memories
- **Pattern matching and retrieval** capabilities
- **Tensor-shaped memory state** T_memory[n_nodes, n_links, t_snapshots]
- **Distributed API** for cross-agent collaboration
- **Temporal tracking** of memory evolution

## Architecture

### Core Components

1. **AtomSpace** (`python/helpers/atomspace.py`)
   - Node and Link atoms with unique identifiers
   - SQLite-based persistent storage
   - Pattern matching and retrieval
   - Snapshot management and tensor representation

2. **Memory Integration** (`python/helpers/memory_atomspace.py`)
   - MemoryAtomSpaceWrapper for existing system integration
   - HypergraphMemoryAgent for agent-specific capabilities
   - Hybrid knowledge storage and retrieval

3. **Distributed API** (`python/api/atomspace_server.py`)
   - REST API for distributed agent access
   - Fragment read/write operations
   - Memory state queries and snapshots

## Quick Start

### Basic Usage

```python
from python.helpers.memory_atomspace import HypergraphMemoryAgent

# Create an agent with hypergraph memory
agent = HypergraphMemoryAgent("my_agent")
await agent.initialize_hypergraph_memory()

# Store knowledge with context
memory_id = await agent.remember_with_context(
    content="Machine learning algorithms learn from data",
    context_concepts=["ai", "algorithms", "data_science"],
    memory_type="knowledge"
)

# Recall through associations
results = await agent.recall_by_association(
    "machine learning data",
    include_related=True
)
```

### Advanced Features

```python
# Create memory checkpoints
checkpoint_id = await agent.create_memory_checkpoint("Learning milestone")

# Get tensor representation
tensor_state = await agent.memory_wrapper.get_memory_tensor_state()
print(f"Memory shape: {tensor_state['tensor_shape']}")

# Pattern-based retrieval
pattern = {"concept_type": "knowledge", "truth_value_min": 0.8}
results = await agent.memory_wrapper.read_hypergraph_fragment(pattern)
```

## API Endpoints

Start the distributed API server:

```bash
python python/api/atomspace_server.py --host 0.0.0.0 --port 5001
```

### Available Endpoints

- `POST /atomspace/read_fragment` - Read atoms matching pattern
- `POST /atomspace/write_fragment` - Write atom fragments
- `GET /atomspace/get_atom/<id>` - Get specific atom
- `POST /atomspace/create_snapshot` - Create memory snapshot
- `GET /atomspace/memory_state` - Get tensor memory state
- `POST /atomspace/add_node` - Add knowledge node
- `POST /atomspace/add_link` - Add relationship link

### Example API Usage

```python
import requests

# Read knowledge fragments
response = requests.post('http://localhost:5001/atomspace/read_fragment', json={
    'pattern': {'concept_type': 'knowledge'},
    'limit': 10
})

# Write new knowledge
response = requests.post('http://localhost:5001/atomspace/write_fragment', json={
    'atoms': [{
        'atom_type': 'node',
        'name': 'new_concept',
        'concept_type': 'knowledge',
        'truth_value': 1.0,
        'confidence': 0.9,
        'metadata': '{\"source\": \"api\"}'
    }]
})
```

## Memory Model

### Atoms

**Nodes** represent concepts, entities, or values:
```python
Node(
    id="uuid",
    name="concept_name",
    concept_type="knowledge",
    truth_value=1.0,
    confidence=0.9,
    metadata={"domain": "AI"}
)
```

**Links** represent relationships:
```python
Link(
    id="uuid",
    name="relationship_name",
    outgoing=["node_id_1", "node_id_2"],
    link_type="relates_to",
    truth_value=0.8,
    confidence=0.7
)
```

### Tensor Representation

Memory state is represented as a 3D tensor:
- **Dimension 0**: Number of nodes
- **Dimension 1**: Number of links  
- **Dimension 2**: Number of temporal snapshots

This enables tracking memory evolution over time and efficient pattern analysis.

## Integration with Agent Zero

The hypergraph memory system is designed to complement the existing vector-based memory:

```python
# Enhanced Agent Zero class
class AgentZeroWithHypergraphMemory:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.hypergraph_agent = None
    
    async def initialize_enhanced_memory(self):
        self.hypergraph_agent = HypergraphMemoryAgent(self.agent_id)
        await self.hypergraph_agent.initialize_hypergraph_memory()
    
    async def remember_experience(self, experience: str, context: list = None):
        return await self.hypergraph_agent.remember_with_context(
            content=experience,
            context_concepts=context or [],
            memory_type="experience"
        )
```

## Testing

Run the comprehensive test suite:

```bash
# Core AtomSpace tests
python /tmp/test_atomspace_core.py

# Integration tests
python /tmp/test_comprehensive.py

# Demo
python /tmp/demo_atomspace.py
```

## Storage

Data is stored in SQLite databases under `memory/{agent_id}/atomspace/`:
- `hypergraph.db` - Main atom storage
- Automatic schema creation and management
- ACID compliance for distributed environments

## Performance Considerations

- **Indexing**: Automatic indexes on atom types, names, and timestamps
- **Caching**: Local atom cache for frequently accessed items
- **Batching**: Support for bulk operations
- **Rate Limiting**: Built-in API rate limiting

## Distributed Architecture

The system supports true distributed operation:

1. **Persistence**: All data persists across agent restarts
2. **Network Boundaries**: Agents can operate across different processes/machines
3. **Collaboration**: Agents can discover and share knowledge fragments
4. **Consistency**: ACID transactions ensure data integrity

## Future Enhancements

- Scheme-based cognitive representation (lisp-like syntax)
- Advanced reasoning capabilities
- Distributed consensus for multi-agent environments
- Integration with OpenCog AtomSpace protocol
- Graph neural network pattern matching

## Examples

See `examples/hypergraph_memory_integration.py` for a complete integration example with Agent Zero.

## License

This implementation follows the same license as the Agent Zero project.