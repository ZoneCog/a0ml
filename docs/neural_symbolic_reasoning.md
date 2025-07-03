# Neural-Symbolic Reasoning Engine

A comprehensive neural-symbolic reasoning system that integrates PLN (Probabilistic Logic Networks), MOSES (Meta-Optimizing Semantic Evolutionary Search), and advanced pattern matching for the a0ml AtomSpace hypergraph memory system.

## Overview

The Neural-Symbolic Reasoning Engine bridges neural and symbolic AI approaches by providing:

- **PLN (Probabilistic Logic Networks)** for logical inference and uncertain reasoning
- **MOSES (Meta-Optimizing Semantic Evolutionary Search)** for evolutionary program optimization
- **Advanced Pattern Matcher** for hypergraph traversal and semantic relation extraction
- **Cognitive Kernels** encoded as tensors: `T_ai[n_patterns, n_reasoning, l_stages]`
- **Integrated Reasoning Pipeline** with multiple cognitive stages
- **RESTful API** for distributed neural-symbolic reasoning

## Architecture

### Core Components

1. **PLN Inference Engine** (`python/helpers/pln_reasoning.py`)
   - Truth value inference with strength and confidence measures
   - Forward chaining inference with logical operators (AND, OR, NOT, IMPLIES, EQUIVALENT)
   - Explanation generation for inference results
   - Uncertain reasoning over hypergraph structures

2. **MOSES Optimizer** (`python/helpers/moses_optimizer.py`)
   - Evolutionary optimization of cognitive programs
   - Population-based program evolution with mutation and crossover
   - Fitness evaluation for inference rules, pattern matchers, and cognitive kernels
   - Program serialization and persistence

3. **Pattern Matcher** (`python/helpers/pattern_matcher.py`)
   - Multiple matching strategies: exact, structural, semantic, fuzzy, temporal
   - Hypergraph traversal with BFS, DFS, and best-first strategies
   - Semantic relation extraction between atoms
   - Pattern registration and caching

4. **Neural-Symbolic Reasoning Engine** (`python/helpers/neural_symbolic_reasoning.py`)
   - Integrated reasoning pipeline with cognitive stages
   - Tensor-based cognitive kernel encoding
   - Multi-stage reasoning: perception → pattern recognition → inference → optimization → decision → action
   - API for neural-symbolic reasoning requests

### Cognitive Stages

The reasoning engine processes queries through six cognitive stages:

1. **Perception** - Gather and process input data
2. **Pattern Recognition** - Identify patterns and semantic relations
3. **Inference** - Perform logical reasoning with PLN
4. **Optimization** - Evolve and adapt programs with MOSES
5. **Decision** - Make decisions based on reasoning results
6. **Action** - Execute actions based on decisions

### Tensor Representation

Cognitive kernels are encoded as 3D tensors:
```
T_ai[n_patterns, n_reasoning, l_stages]
```
Where:
- `n_patterns`: Number of patterns (max 100)
- `n_reasoning`: Number of reasoning programs (max 50)  
- `l_stages`: Number of cognitive stages (6)

## Quick Start

### Basic Usage

```python
from python.helpers.atomspace import AtomSpace
from python.helpers.neural_symbolic_reasoning import NeuralSymbolicReasoningEngine

# Initialize system
atomspace = AtomSpace("path/to/database.db")
reasoning_engine = NeuralSymbolicReasoningEngine(atomspace)
await reasoning_engine.initialize_system()

# Perform reasoning
query = {
    "type": "infer",
    "concepts": ["dog", "mammal"],
    "include_details": True
}

result = await reasoning_engine.reason(query)
print(f"Reasoning confidence: {result['result']['confidence']}")
```

### REST API Usage

Start the API server:
```bash
python python/api/neural_symbolic_server.py
```

Perform reasoning via HTTP:
```bash
curl -X POST http://localhost:5002/neural-symbolic/reason \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "type": "infer", 
      "concepts": ["dog", "mammal"]
    }
  }'
```

## API Endpoints

### System Management

- `GET /neural-symbolic/status` - Get system status and statistics
- `POST /neural-symbolic/initialize` - Initialize the reasoning system
- `GET /neural-symbolic/help` - Get API documentation

### Reasoning Operations

- `POST /neural-symbolic/reason` - Perform integrated neural-symbolic reasoning
- `POST /neural-symbolic/pln/infer` - PLN inference operations
- `POST /neural-symbolic/moses/optimize` - MOSES program optimization
- `POST /neural-symbolic/pattern/match` - Pattern matching
- `POST /neural-symbolic/pattern/traverse` - Hypergraph traversal
- `POST /neural-symbolic/pattern/relations` - Semantic relation extraction

### Data Operations

- `GET /neural-symbolic/tensor/cognitive` - Get cognitive tensor representation
- `POST /neural-symbolic/atomspace/add_knowledge` - Add knowledge to AtomSpace

## Examples

### PLN Inference

```python
# Infer truth values
truth_value = await reasoning_engine.pln_engine.infer_truth_value(atom_id)
print(f"Strength: {truth_value.strength}, Confidence: {truth_value.confidence}")

# Forward chaining
inference_result = await reasoning_engine.pln_engine.forward_chaining(
    premises=["premise1", "premise2"], 
    max_iterations=10
)
print(f"Derived facts: {len(inference_result['derived_facts'])}")

# Get explanation
explanation = await reasoning_engine.pln_engine.get_inference_explanation(atom_id)
```

### MOSES Optimization

```python
# Optimize inference rules
optimization_result = await reasoning_engine.moses_optimizer.optimize_program(
    ProgramType.INFERENCE_RULE,
    generations=20
)

best_program = optimization_result["best_program"]
print(f"Best fitness: {best_program['fitness']}")
```

### Pattern Matching

```python
# Create and register pattern
pattern = Pattern(
    id="test_pattern",
    pattern_type=MatchType.SEMANTIC,
    template={"atom_type": "node", "concept_type": "concept"},
    weights={"truth_value": 0.8}
)

await reasoning_engine.pattern_matcher.register_pattern(pattern)

# Find matches
matches = await reasoning_engine.pattern_matcher.match_pattern("test_pattern")
print(f"Found {len(matches)} matches")

# Traverse hypergraph
traversal = await reasoning_engine.pattern_matcher.traverse_hypergraph(
    start_atom_id,
    TraversalStrategy.BREADTH_FIRST,
    max_depth=5
)
```

### Cognitive Tensor Operations

```python
# Get system tensor
system_tensor = await reasoning_engine.get_system_tensor()
print(f"Tensor shape: {system_tensor.shape}")

# Get specific kernel tensor
kernel_tensor = await reasoning_engine.get_cognitive_tensor("inference_kernel")
```

## Testing

Run the comprehensive test suite:

```bash
python test_neural_symbolic_reasoning.py
```

The test suite validates:
- PLN inference on real AtomSpace data
- MOSES program evolution and optimization
- Pattern matching with multiple strategies
- Integrated neural-symbolic reasoning
- Performance and scalability

## Performance Considerations

- **Caching**: Truth values, pattern matches, and traversals are cached
- **Indexing**: Automatic database indexes on atom types and timestamps
- **Batching**: Support for bulk operations
- **Concurrency**: Async operations support concurrent reasoning
- **Scalability**: Tested with large knowledge bases (1000+ atoms)

## Integration with Agent Zero

The neural-symbolic reasoning engine integrates seamlessly with the existing Agent Zero architecture:

- **AtomSpace Integration**: Uses existing hypergraph memory system
- **Tool Interface**: Can be called as a tool from agents
- **API Framework**: Compatible with existing Flask API structure
- **Context Management**: Works with AgentContext system
- **Distributed Operation**: Supports multi-agent environments

## Future Enhancements

- Advanced PLN inference rules and operators
- Deep learning integration for neural components
- Distributed consensus for multi-agent reasoning
- Graph neural networks for pattern matching
- Causal reasoning and counterfactual inference
- Natural language interface for reasoning queries

## Configuration

Key configuration parameters:

```python
# Tensor dimensions
max_patterns = 100
max_reasoning_programs = 50
max_stages = 6

# MOSES parameters
population_size = 50
mutation_rate = 0.1
crossover_rate = 0.7
max_complexity = 20

# Pattern matching
similarity_threshold = 0.7
max_traversal_depth = 5
```

## Error Handling

The system includes comprehensive error handling:
- Graceful degradation for missing data
- Timeout protection for long operations
- Exception logging and traceback capture
- API error responses with detailed messages

## License

This implementation follows the same license as the Agent Zero project.

---

For detailed API documentation, visit `http://localhost:5002/neural-symbolic/help` when the server is running.