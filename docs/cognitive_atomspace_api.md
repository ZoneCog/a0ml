# Scheme-based Cognitive AtomSpace API Usage Guide

This guide demonstrates how to use the enhanced AtomSpace API with Scheme-based cognitive representation capabilities.

## API Endpoints

### Cognitive Pattern Management

#### Store a Cognitive Pattern
```bash
curl -X POST http://localhost:5001/atomspace/cognitive/store_pattern \
  -H "Content-Type: application/json" \
  -d '{
    "name": "problem_solving_pattern",
    "scheme_expression": "(solve (problem ?p) (method ?m) (result ?r))",
    "metadata": {"domain": "general", "complexity": "medium"}
  }'
```

#### Evaluate a Cognitive Pattern
```bash
curl -X POST http://localhost:5001/atomspace/cognitive/evaluate_pattern \
  -H "Content-Type: application/json" \
  -d '{
    "pattern_id": "pattern_id_here",
    "bindings": {
      "p": "pathfinding",
      "m": "a_star", 
      "r": "optimal_path"
    }
  }'
```

#### Find Cognitive Patterns
```bash
curl -X POST http://localhost:5001/atomspace/cognitive/find_patterns \
  -H "Content-Type: application/json" \
  -d '{
    "query": "learning algorithms",
    "max_results": 10
  }'
```

#### Perform Cognitive Reasoning
```bash
curl -X POST http://localhost:5001/atomspace/cognitive/reason \
  -H "Content-Type: application/json" \
  -d '{
    "pattern_ids": ["pattern1_id", "pattern2_id"],
    "reasoning_type": "forward_chaining"
  }'
```

### Grammar Management

#### List Cognitive Grammars
```bash
curl http://localhost:5001/atomspace/cognitive/grammars
```

#### Get Specific Grammar
```bash
curl http://localhost:5001/atomspace/cognitive/grammars/perception_grammar
```

### System Statistics and Export/Import

#### Get Cognitive Statistics
```bash
curl http://localhost:5001/atomspace/cognitive/statistics
```

#### Export Cognitive Knowledge
```bash
curl http://localhost:5001/atomspace/cognitive/export
```

#### Import Cognitive Knowledge
```bash
curl -X POST http://localhost:5001/atomspace/cognitive/import \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "cognitive_patterns": {},
      "grammars": {}
    }
  }'
```

## Example Integration with Agent Zero

```python
import requests
import asyncio
from python.helpers.cognitive_atomspace_integration import EnhancedMemoryAtomSpaceWrapper

class CognitiveAgent:
    def __init__(self, agent_id: str, api_base_url: str = "http://localhost:5001"):
        self.agent_id = agent_id
        self.api_base_url = api_base_url
        self.enhanced_memory = EnhancedMemoryAtomSpaceWrapper(agent_id)
    
    async def store_learning_pattern(self, learning_content: str, scheme_pattern: str):
        """Store a learning experience as a cognitive pattern"""
        # Local storage with cognitive integration
        memory_result = await self.enhanced_memory.store_cognitive_memory(
            content=learning_content,
            scheme_pattern=scheme_pattern,
            memory_type="learning"
        )
        
        # Also store via distributed API for sharing
        response = requests.post(f"{self.api_base_url}/atomspace/cognitive/store_pattern", 
            json={
                "name": f"learning_{memory_result['node_id'][:8]}",
                "scheme_expression": scheme_pattern,
                "metadata": {
                    "content": learning_content,
                    "agent_id": self.agent_id,
                    "local_node_id": memory_result["node_id"]
                }
            }
        )
        
        return response.json() if response.status_code == 200 else None
    
    async def cognitive_query(self, query: str):
        """Query cognitive patterns both locally and via API"""
        # Local cognitive recall
        local_results = await self.enhanced_memory.cognitive_recall(query, use_scheme_reasoning=True)
        
        # API-based pattern search
        api_response = requests.post(f"{self.api_base_url}/atomspace/cognitive/find_patterns",
            json={"query": query, "max_results": 5}
        )
        
        api_results = api_response.json() if api_response.status_code == 200 else {"patterns": []}
        
        return {
            "local_results": local_results,
            "distributed_results": api_results["patterns"]
        }

# Usage example
async def main():
    agent = CognitiveAgent("example_agent")
    
    # Store a learning pattern
    await agent.store_learning_pattern(
        "Dynamic programming optimizes recursive algorithms by caching results",
        "(learn (technique dynamic_programming) (applies_to recursive_algorithms) (optimization caching))"
    )
    
    # Query for related patterns
    results = await agent.cognitive_query("optimization algorithms")
    print(f"Found {len(results['local_results']['cognitive_results'])} local patterns")
    print(f"Found {len(results['distributed_results'])} distributed patterns")
```

## Built-in Cognitive Operators

The system provides several built-in cognitive operators:

- `perceive` - For perceptual processes
- `reason` - For logical reasoning
- `decide` - For decision-making
- `act` - For action execution
- `remember` - For memory operations
- `learn` - For learning processes
- `compose` - For composing complex patterns
- `reflect` - For meta-cognitive reflection
- `anticipate` - For predictive processes
- `evaluate` - For assessment processes

## Example Scheme Expressions

### Simple Learning Pattern
```scheme
(learn (concept ?c) (from_experience ?e) (confidence ?conf))
```

### Complex Problem Solving
```scheme
(compose 
  (perceive (problem ?p) (context ?ctx))
  (reason (about ?p) (using ?knowledge))
  (decide (strategy ?s) (based_on ?reasoning))
  (act (execute ?s) (monitor ?progress)))
```

### Meta-Cognitive Reflection
```scheme
(reflect 
  (on_performance ?perf) 
  (identify_patterns ?patterns)
  (adjust_strategy ?new_strategy))
```

This enhanced AtomSpace provides a powerful foundation for Agent Zero's cognitive capabilities, enabling sophisticated reasoning, learning, and collaboration through Scheme-based cognitive representation.