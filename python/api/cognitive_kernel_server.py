"""
Cognitive Kernel API Server

REST API endpoints for the unified cognitive kernel system providing
meta-recursive attention allocation, cognitive grammar management,
and integrated subsystem orchestration.
"""

from flask import Flask, request, jsonify
import asyncio
import traceback
import json
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timezone

from python.helpers.cognitive_kernel import UnifiedCognitiveKernel
from python.helpers.scheme_grammar import SchemeCognitiveGrammarRegistry
from python.helpers.atomspace import AtomSpace

# Global variables
cognitive_kernel: Optional[UnifiedCognitiveKernel] = None
grammar_registry: Optional[SchemeCognitiveGrammarRegistry] = None
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def ensure_cognitive_kernel():
    """Ensure cognitive kernel is initialized"""
    global cognitive_kernel, grammar_registry
    
    if cognitive_kernel is None:
        # Initialize with temporary database
        atomspace = AtomSpace("/tmp/cognitive_kernel.db")
        cognitive_kernel = UnifiedCognitiveKernel(atomspace)
        await cognitive_kernel.initialize()
        
        # Initialize grammar registry
        grammar_registry = SchemeCognitiveGrammarRegistry(atomspace)
        
        logger.info("Cognitive kernel initialized")
    
    return cognitive_kernel


@app.route('/cognitive-kernel/initialize', methods=['POST'])
def initialize_system():
    """Initialize the cognitive kernel system"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        return jsonify({
            "success": True,
            "message": "Cognitive kernel initialized successfully",
            "kernel_id": kernel.kernel_id,
            "state": kernel.state.value,
            "subsystems": {
                "neural_symbolic": kernel.neural_symbolic_engine is not None,
                "ecan": kernel.ecan_system is not None,
                "task_orchestrator": kernel.task_orchestrator is not None
            }
        })
        
    except Exception as e:
        logger.error(f"Error initializing cognitive kernel: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to initialize cognitive kernel",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/invoke', methods=['POST'])
def invoke_kernel():
    """Invoke the cognitive kernel with a query"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', {})
        context = data.get('context', {})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        result = loop.run_until_complete(kernel.recursive_invoke(query, context))
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error invoking kernel: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to invoke kernel",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/tensor', methods=['GET'])
def get_kernel_tensor():
    """Get the current kernel tensor"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        tensor = kernel.get_kernel_tensor()
        
        if tensor is not None:
            return jsonify({
                "success": True,
                "tensor_shape": tensor.shape,
                "tensor_data": tensor.tolist(),
                "dimensions": {
                    "n_atoms": kernel.n_atoms,
                    "n_tasks": kernel.n_tasks,
                    "n_reasoning": kernel.n_reasoning,
                    "a_levels": kernel.a_levels,
                    "t_steps": kernel.t_steps
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": "Kernel tensor not available"
            })
            
    except Exception as e:
        logger.error(f"Error getting kernel tensor: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to get kernel tensor",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/membranes', methods=['GET'])
def get_attention_membranes():
    """Get attention membrane states"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        membranes = kernel.get_attention_membranes()
        
        return jsonify({
            "success": True,
            "membranes": membranes,
            "membrane_count": len(membranes)
        })
        
    except Exception as e:
        logger.error(f"Error getting attention membranes: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to get attention membranes",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/statistics', methods=['GET'])
def get_kernel_statistics():
    """Get kernel statistics"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        stats = kernel.get_kernel_statistics()
        
        return jsonify({
            "success": True,
            "statistics": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting kernel statistics: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to get kernel statistics",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/grammars', methods=['GET'])
def list_cognitive_grammars():
    """List all cognitive grammars"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        grammars = kernel.get_cognitive_grammars()
        
        # Also get registry grammars
        if grammar_registry:
            registry_grammars = grammar_registry.list_grammars()
            registry_data = {g.id: g.to_dict() for g in registry_grammars}
        else:
            registry_data = {}
        
        return jsonify({
            "success": True,
            "kernel_grammars": grammars,
            "registry_grammars": registry_data,
            "total_grammars": len(grammars) + len(registry_data)
        })
        
    except Exception as e:
        logger.error(f"Error listing grammars: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to list grammars",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/grammars', methods=['POST'])
def register_cognitive_grammar():
    """Register a new cognitive grammar"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        grammar_id = data.get('grammar_id')
        name = data.get('name')
        description = data.get('description')
        scheme_expression = data.get('scheme_expression')
        
        if not all([grammar_id, name, description, scheme_expression]):
            return jsonify({
                "error": "Missing required fields: grammar_id, name, description, scheme_expression"
            }), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        # Register in kernel
        kernel.register_cognitive_grammar(grammar_id, {
            "name": name,
            "description": description,
            "scheme_expression": scheme_expression
        })
        
        # Register in registry if available
        if grammar_registry:
            from python.helpers.scheme_grammar import CognitiveOperator
            
            # Parse cognitive operators from data
            cognitive_operators = []
            if 'cognitive_operators' in data:
                for op_str in data['cognitive_operators']:
                    try:
                        cognitive_operators.append(CognitiveOperator(op_str))
                    except ValueError:
                        pass
                        
            grammar_registry.register_grammar(
                grammar_id,
                name,
                description,
                scheme_expression,
                cognitive_operators,
                data.get('pattern_templates', [])
            )
        
        return jsonify({
            "success": True,
            "message": f"Grammar '{grammar_id}' registered successfully"
        })
        
    except Exception as e:
        logger.error(f"Error registering grammar: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to register grammar",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/grammars/<grammar_id>', methods=['GET'])
def get_cognitive_grammar(grammar_id):
    """Get a specific cognitive grammar"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        # Check kernel grammars
        kernel_grammars = kernel.get_cognitive_grammars()
        if grammar_id in kernel_grammars:
            grammar_data = kernel_grammars[grammar_id]
        elif grammar_registry:
            # Check registry
            grammar = grammar_registry.get_grammar(grammar_id)
            grammar_data = grammar.to_dict() if grammar else None
        else:
            grammar_data = None
            
        if grammar_data:
            return jsonify({
                "success": True,
                "grammar": grammar_data
            })
        else:
            return jsonify({
                "success": False,
                "error": f"Grammar '{grammar_id}' not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting grammar: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to get grammar",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/grammars/<grammar_id>/evaluate', methods=['POST'])
def evaluate_grammar(grammar_id):
    """Evaluate a cognitive grammar with bindings"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        bindings = data.get('bindings', {})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        if grammar_registry:
            result = grammar_registry.evaluate_grammar_expression(grammar_id, bindings)
            return jsonify({
                "success": True,
                "evaluation": result
            })
        else:
            return jsonify({
                "success": False,
                "error": "Grammar registry not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error evaluating grammar: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to evaluate grammar",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/grammars/<grammar_id>/extend', methods=['POST'])
def extend_grammar(grammar_id):
    """Extend a cognitive grammar"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        extension_expression = data.get('extension_expression')
        if not extension_expression:
            return jsonify({"error": "Missing extension_expression"}), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        if grammar_registry:
            success = grammar_registry.extend_grammar(grammar_id, extension_expression)
            if success:
                return jsonify({
                    "success": True,
                    "message": f"Grammar '{grammar_id}' extended successfully"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": f"Failed to extend grammar '{grammar_id}'"
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": "Grammar registry not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error extending grammar: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to extend grammar",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/grammars/<grammar_id>/specialize', methods=['POST'])
def specialize_grammar(grammar_id):
    """Create a specialized version of a grammar"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        specialization_name = data.get('specialization_name')
        specialization_expression = data.get('specialization_expression')
        
        if not all([specialization_name, specialization_expression]):
            return jsonify({
                "error": "Missing required fields: specialization_name, specialization_expression"
            }), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        if grammar_registry:
            specialized_id = grammar_registry.specialize_grammar(
                grammar_id, specialization_name, specialization_expression
            )
            
            if specialized_id:
                return jsonify({
                    "success": True,
                    "specialized_id": specialized_id,
                    "message": f"Created specialized grammar '{specialized_id}'"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": f"Failed to specialize grammar '{grammar_id}'"
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": "Grammar registry not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error specializing grammar: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to specialize grammar",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/grammars/compose', methods=['POST'])
def compose_grammars():
    """Compose multiple grammars into a new one"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        grammar_ids = data.get('grammar_ids', [])
        composition_name = data.get('composition_name')
        
        if not grammar_ids or not composition_name:
            return jsonify({
                "error": "Missing required fields: grammar_ids, composition_name"
            }), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        if grammar_registry:
            composition_id = grammar_registry.compose_grammars(grammar_ids, composition_name)
            
            if composition_id:
                return jsonify({
                    "success": True,
                    "composition_id": composition_id,
                    "message": f"Created composition grammar '{composition_id}'"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to compose grammars"
                }), 400
        else:
            return jsonify({
                "success": False,
                "error": "Grammar registry not available"
            }), 503
            
    except Exception as e:
        logger.error(f"Error composing grammars: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to compose grammars",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/meta-events', methods=['GET'])
def get_meta_events():
    """Get meta-cognitive events"""
    try:
        limit = request.args.get('limit', 100, type=int)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        # Get recent meta events
        recent_events = kernel.meta_events[-limit:] if kernel.meta_events else []
        events_data = [event.to_dict() for event in recent_events]
        
        return jsonify({
            "success": True,
            "events": events_data,
            "total_events": len(kernel.meta_events),
            "returned_events": len(events_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting meta events: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to get meta events",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/self-modify', methods=['POST'])
def trigger_self_modification():
    """Trigger a self-modification protocol"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        protocol_id = data.get('protocol_id')
        if not protocol_id:
            return jsonify({"error": "Missing protocol_id"}), 400
            
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        # Trigger self-modification
        await kernel._trigger_self_modification(protocol_id)
        
        return jsonify({
            "success": True,
            "message": f"Self-modification protocol '{protocol_id}' triggered"
        })
        
    except Exception as e:
        logger.error(f"Error triggering self-modification: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to trigger self-modification",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/status', methods=['GET'])
def get_kernel_status():
    """Get kernel status"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        kernel = loop.run_until_complete(ensure_cognitive_kernel())
        
        status = {
            "kernel_id": kernel.kernel_id,
            "state": kernel.state.value,
            "running": kernel.running,
            "statistics": kernel.get_kernel_statistics(),
            "attention_membranes": len(kernel.attention_membranes),
            "meta_events": len(kernel.meta_events),
            "cognitive_grammars": len(kernel.cognitive_grammars),
            "subsystems": {
                "neural_symbolic": kernel.neural_symbolic_engine is not None,
                "ecan": kernel.ecan_system is not None,
                "task_orchestrator": kernel.task_orchestrator is not None
            }
        }
        
        return jsonify({
            "success": True,
            "status": status
        })
        
    except Exception as e:
        logger.error(f"Error getting kernel status: {traceback.format_exc()}")
        return jsonify({
            "error": "Failed to get kernel status",
            "details": str(e)
        }), 500


@app.route('/cognitive-kernel/help', methods=['GET'])
def get_help():
    """Get API help and documentation"""
    return jsonify({
        "success": True,
        "cognitive_kernel_api": {
            "version": "1.0.0",
            "description": "Unified Cognitive Kernel API for meta-recursive attention and cognitive grammar management",
            "endpoints": {
                "POST /cognitive-kernel/initialize": "Initialize the cognitive kernel system",
                "POST /cognitive-kernel/invoke": "Invoke the kernel with a query",
                "GET /cognitive-kernel/tensor": "Get the current kernel tensor",
                "GET /cognitive-kernel/membranes": "Get attention membrane states",
                "GET /cognitive-kernel/statistics": "Get kernel statistics",
                "GET /cognitive-kernel/grammars": "List all cognitive grammars",
                "POST /cognitive-kernel/grammars": "Register a new cognitive grammar",
                "GET /cognitive-kernel/grammars/<id>": "Get a specific cognitive grammar",
                "POST /cognitive-kernel/grammars/<id>/evaluate": "Evaluate a grammar with bindings",
                "POST /cognitive-kernel/grammars/<id>/extend": "Extend a grammar",
                "POST /cognitive-kernel/grammars/<id>/specialize": "Create a specialized grammar",
                "POST /cognitive-kernel/grammars/compose": "Compose multiple grammars",
                "GET /cognitive-kernel/meta-events": "Get meta-cognitive events",
                "POST /cognitive-kernel/self-modify": "Trigger self-modification protocol",
                "GET /cognitive-kernel/status": "Get kernel status",
                "GET /cognitive-kernel/help": "Get this help information"
            },
            "examples": {
                "invoke_kernel": {
                    "url": "/cognitive-kernel/invoke",
                    "method": "POST",
                    "body": {
                        "query": {
                            "type": "reasoning",
                            "content": {
                                "concepts": ["intelligence", "cognition"],
                                "include_details": True
                            }
                        },
                        "context": {
                            "session_id": "user_session_123"
                        }
                    }
                },
                "register_grammar": {
                    "url": "/cognitive-kernel/grammars",
                    "method": "POST",
                    "body": {
                        "grammar_id": "my_grammar",
                        "name": "My Custom Grammar",
                        "description": "A custom cognitive grammar for specific reasoning",
                        "scheme_expression": "(reason (premise ?p1) (premise ?p2) (conclusion ?c))",
                        "cognitive_operators": ["reason"],
                        "pattern_templates": [
                            {
                                "template": {"reason": True, "premise": "?p", "conclusion": "?c"},
                                "match_type": "logical"
                            }
                        ]
                    }
                },
                "evaluate_grammar": {
                    "url": "/cognitive-kernel/grammars/reasoning_grammar/evaluate",
                    "method": "POST",
                    "body": {
                        "bindings": {
                            "p1": "All humans are mortal",
                            "p2": "Socrates is human",
                            "c": "Socrates is mortal"
                        }
                    }
                }
            }
        }
    })


if __name__ == '__main__':
    app.run(debug=True, port=5001)