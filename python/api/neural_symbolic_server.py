"""
Neural-Symbolic Reasoning API Server

REST API endpoints for PLN inference, MOSES optimization, and pattern matching
with neural-symbolic reasoning capabilities.
"""

import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify
from pathlib import Path
import os
import sys

# Add the project root to path  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from python.helpers.atomspace import AtomSpace
from python.helpers.neural_symbolic_reasoning import NeuralSymbolicReasoningEngine, ReasoningStage
from python.helpers.pln_reasoning import TruthValue, LogicalOperator
from python.helpers.moses_optimizer import ProgramType
from python.helpers.pattern_matcher import Pattern, MatchType

app = Flask(__name__)
app.logger.setLevel("ERROR")

# Global reasoning engine instance
reasoning_engine: Optional[NeuralSymbolicReasoningEngine] = None


def initialize_reasoning_engine(agent_id: str = "neural_symbolic_api"):
    """Initialize the neural-symbolic reasoning engine"""
    global reasoning_engine
    
    # Create AtomSpace
    memory_path = Path("memory") / agent_id / "neural_symbolic"
    memory_path.mkdir(parents=True, exist_ok=True)
    
    db_path = memory_path / "reasoning.db"
    atomspace = AtomSpace(str(db_path))
    
    # Create reasoning engine
    reasoning_engine = NeuralSymbolicReasoningEngine(atomspace)
    
    return reasoning_engine


async def ensure_reasoning_engine():
    """Ensure reasoning engine is initialized"""
    global reasoning_engine
    if reasoning_engine is None:
        reasoning_engine = initialize_reasoning_engine()
        await reasoning_engine.initialize_system()
    return reasoning_engine


@app.route('/neural-symbolic/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        if reasoning_engine is None:
            return jsonify({
                "status": "not_initialized",
                "message": "Neural-symbolic reasoning engine not initialized"
            }), 503
        
        stats = reasoning_engine.get_statistics()
        
        return jsonify({
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "statistics": stats
        })
    
    except Exception as e:
        app.logger.error("An error occurred in get_status: %s", traceback.format_exc())
        return jsonify({
            "error": "An internal error has occurred. Please contact support if the issue persists."
        }), 500


@app.route('/neural-symbolic/initialize', methods=['POST'])
def initialize_system():
    """Initialize the neural-symbolic reasoning system"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        init_result = loop.run_until_complete(engine.initialize_system())
        
        return jsonify({
            "success": True,
            "message": "Neural-symbolic reasoning engine initialized",
            "initialization_report": init_result
        })
    
    except Exception as e:
        app.logger.error("An error occurred in initialize_system: %s", traceback.format_exc())
        return jsonify({
            "error": "An internal error has occurred. Please contact support if the issue persists."
        }), 500


@app.route('/neural-symbolic/reason', methods=['POST'])
def perform_reasoning():
    """Perform neural-symbolic reasoning"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        query = data.get('query', {})
        kernel_id = data.get('kernel_id')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        result = loop.run_until_complete(engine.reason(query, kernel_id))
        
        return jsonify({
            "success": True,
            "reasoning_result": result
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/pln/infer', methods=['POST'])
def pln_inference():
    """Perform PLN inference"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        atom_ids = data.get('atom_ids', [])
        operation = data.get('operation', 'infer_truth_value')
        
        if not atom_ids:
            return jsonify({"error": "No atom IDs provided"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        
        results = {}
        
        if operation == 'infer_truth_value':
            # Infer truth values for provided atoms
            for atom_id in atom_ids:
                tv = loop.run_until_complete(engine.pln_engine.infer_truth_value(atom_id))
                results[atom_id] = tv.to_dict()
        
        elif operation == 'forward_chaining':
            # Perform forward chaining
            premises = atom_ids[:10]  # Limit premises
            max_iterations = data.get('max_iterations', 10)
            
            inference_result = loop.run_until_complete(
                engine.pln_engine.forward_chaining(premises, max_iterations)
            )
            results = inference_result
        
        elif operation == 'explain':
            # Generate explanation for first atom
            if atom_ids:
                explanation = loop.run_until_complete(
                    engine.pln_engine.get_inference_explanation(atom_ids[0])
                )
                results = explanation
        
        return jsonify({
            "success": True,
            "operation": operation,
            "results": results
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/moses/optimize', methods=['POST'])
def moses_optimization():
    """Perform MOSES program optimization"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        program_type = data.get('program_type', 'inference_rule')
        generations = data.get('generations', 10)
        population_size = data.get('population_size', 50)
        
        # Validate program type
        try:
            prog_type = ProgramType(program_type)
        except ValueError:
            return jsonify({"error": f"Invalid program type: {program_type}"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        
        # Set population size if provided
        if population_size != engine.moses_optimizer.population_size:
            engine.moses_optimizer.population_size = population_size
        
        # Run optimization
        optimization_result = loop.run_until_complete(
            engine.moses_optimizer.optimize_program(prog_type, generations)
        )
        
        return jsonify({
            "success": True,
            "optimization_result": optimization_result
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/pattern/match', methods=['POST'])
def pattern_matching():
    """Perform pattern matching"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        pattern_def = data.get('pattern')
        match_type = data.get('match_type', 'exact')
        max_matches = data.get('max_matches', 100)
        
        if not pattern_def:
            return jsonify({"error": "No pattern definition provided"}), 400
        
        # Validate match type
        try:
            m_type = MatchType(match_type)
        except ValueError:
            return jsonify({"error": f"Invalid match type: {match_type}"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        
        # Create pattern object
        pattern = Pattern(
            id=f"api_pattern_{datetime.now().timestamp()}",
            pattern_type=m_type,
            template=pattern_def.get('template', {}),
            constraints=pattern_def.get('constraints', {}),
            variables=set(pattern_def.get('variables', [])),
            weights=pattern_def.get('weights', {})
        )
        
        # Register and match pattern
        loop.run_until_complete(engine.pattern_matcher.register_pattern(pattern))
        matches = loop.run_until_complete(
            engine.pattern_matcher.match_pattern(pattern.id, max_matches=max_matches)
        )
        
        return jsonify({
            "success": True,
            "pattern_id": pattern.id,
            "matches": [match.to_dict() for match in matches]
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/pattern/traverse', methods=['POST'])
def hypergraph_traversal():
    """Perform hypergraph traversal"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        start_atom_id = data.get('start_atom_id')
        strategy = data.get('strategy', 'breadth_first')
        max_depth = data.get('max_depth', 5)
        max_nodes = data.get('max_nodes', 100)
        
        if not start_atom_id:
            return jsonify({"error": "No start atom ID provided"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        
        from python.helpers.pattern_matcher import TraversalStrategy
        
        # Convert strategy string to enum
        try:
            traversal_strategy = TraversalStrategy(strategy)
        except ValueError:
            return jsonify({"error": f"Invalid traversal strategy: {strategy}"}), 400
        
        # Perform traversal
        traversal_result = loop.run_until_complete(
            engine.pattern_matcher.traverse_hypergraph(
                start_atom_id, traversal_strategy, max_depth, max_nodes
            )
        )
        
        return jsonify({
            "success": True,
            "start_atom_id": start_atom_id,
            "strategy": strategy,
            "traversal_result": traversal_result
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/pattern/relations', methods=['POST'])
def semantic_relations():
    """Extract semantic relations"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        atom_ids = data.get('atom_ids', [])
        
        if not atom_ids:
            return jsonify({"error": "No atom IDs provided"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        
        # Extract semantic relations
        relations = loop.run_until_complete(
            engine.pattern_matcher.extract_semantic_relations(atom_ids)
        )
        
        return jsonify({
            "success": True,
            "atom_ids": atom_ids,
            "semantic_relations": relations
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/tensor/cognitive', methods=['GET'])
def get_cognitive_tensor():
    """Get cognitive tensor representation"""
    try:
        kernel_id = request.args.get('kernel_id')
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        
        if kernel_id:
            tensor = loop.run_until_complete(engine.get_cognitive_tensor(kernel_id))
            if tensor is None:
                return jsonify({"error": f"Kernel {kernel_id} not found"}), 404
        else:
            tensor = loop.run_until_complete(engine.get_system_tensor())
        
        return jsonify({
            "success": True,
            "kernel_id": kernel_id,
            "tensor_shape": tensor.shape,
            "tensor_stats": {
                "mean": float(tensor.mean()),
                "std": float(tensor.std()),
                "min": float(tensor.min()),
                "max": float(tensor.max())
            }
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/atomspace/add_knowledge', methods=['POST'])
def add_knowledge():
    """Add knowledge to the AtomSpace"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        knowledge_items = data.get('knowledge_items', [])
        
        if not knowledge_items:
            return jsonify({"error": "No knowledge items provided"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = loop.run_until_complete(ensure_reasoning_engine())
        
        added_atoms = []
        
        for item in knowledge_items:
            if item.get('type') == 'node':
                node = loop.run_until_complete(
                    engine.atomspace.add_node(
                        name=item.get('name', ''),
                        concept_type=item.get('concept_type', 'concept'),
                        truth_value=item.get('truth_value', 1.0),
                        confidence=item.get('confidence', 1.0),
                        metadata=item.get('metadata', {})
                    )
                )
                added_atoms.append(node.to_dict())
            
            elif item.get('type') == 'link':
                link = loop.run_until_complete(
                    engine.atomspace.add_link(
                        name=item.get('name', ''),
                        outgoing=item.get('outgoing', []),
                        link_type=item.get('link_type', 'inheritance'),
                        truth_value=item.get('truth_value', 1.0),
                        confidence=item.get('confidence', 1.0),
                        metadata=item.get('metadata', {})
                    )
                )
                added_atoms.append(link.to_dict())
        
        return jsonify({
            "success": True,
            "added_atoms": added_atoms
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/neural-symbolic/help', methods=['GET'])
def get_help():
    """Get API help and documentation"""
    return jsonify({
        "neural_symbolic_reasoning_api": {
            "version": "1.0.0",
            "description": "Neural-Symbolic Reasoning Engine API with PLN, MOSES, and Pattern Matching",
            "endpoints": {
                "GET /neural-symbolic/status": "Get system status and statistics",
                "POST /neural-symbolic/initialize": "Initialize the reasoning system",
                "POST /neural-symbolic/reason": "Perform integrated neural-symbolic reasoning",
                "POST /neural-symbolic/pln/infer": "Perform PLN inference operations",
                "POST /neural-symbolic/moses/optimize": "Run MOSES program optimization",
                "POST /neural-symbolic/pattern/match": "Perform pattern matching",
                "POST /neural-symbolic/pattern/traverse": "Traverse hypergraph structure",
                "POST /neural-symbolic/pattern/relations": "Extract semantic relations",
                "GET /neural-symbolic/tensor/cognitive": "Get cognitive tensor representation",
                "POST /neural-symbolic/atomspace/add_knowledge": "Add knowledge to AtomSpace",
                "GET /neural-symbolic/help": "Get this help information"
            },
            "examples": {
                "reasoning": {
                    "url": "/neural-symbolic/reason",
                    "method": "POST",
                    "body": {
                        "query": {
                            "type": "infer",
                            "concepts": ["dog", "mammal"],
                            "include_details": True
                        },
                        "kernel_id": "inference_kernel"
                    }
                },
                "pln_inference": {
                    "url": "/neural-symbolic/pln/infer",
                    "method": "POST",
                    "body": {
                        "atom_ids": ["atom_id_1", "atom_id_2"],
                        "operation": "infer_truth_value"
                    }
                },
                "pattern_matching": {
                    "url": "/neural-symbolic/pattern/match",
                    "method": "POST",
                    "body": {
                        "pattern": {
                            "template": {"atom_type": "node", "concept_type": "concept"},
                            "constraints": {"min_truth_value": 0.8}
                        },
                        "match_type": "exact",
                        "max_matches": 50
                    }
                }
            }
        }
    })


if __name__ == '__main__':
    print("🚀 Starting Neural-Symbolic Reasoning API Server...")
    print("📚 Initializing reasoning engine...")
    
    # Initialize the reasoning engine
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        engine = initialize_reasoning_engine()
        loop.run_until_complete(engine.initialize_system())
        
        print("✅ Neural-Symbolic Reasoning Engine initialized successfully!")
        print("🌐 API Server starting on http://localhost:5002")
        print("📖 Visit http://localhost:5002/neural-symbolic/help for API documentation")
        
        app.run(host='0.0.0.0', port=5002, debug=False)
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        traceback.print_exc()