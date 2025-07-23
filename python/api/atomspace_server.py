"""
AtomSpace API endpoint for distributed agent access with Scheme-based cognitive representation
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, request, jsonify
from python.helpers.atomspace import AtomSpace, DistributedAtomSpaceAPI
from python.helpers.cognitive_atomspace_integration import CognitiveAtomSpaceIntegration


class AtomSpaceServer:
    """
    Flask server for distributed AtomSpace access
    """
    
    def __init__(self, storage_path: str = "memory/atomspace"):
        self.storage_path = storage_path
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        
        db_path = Path(storage_path) / "distributed.db"
        self.atomspace = AtomSpace(str(db_path))
        self.api = DistributedAtomSpaceAPI(self.atomspace)
        
        # Initialize cognitive integration
        self.cognitive_integration = CognitiveAtomSpaceIntegration(self.atomspace)
        
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for AtomSpace operations"""
        
        @self.app.route('/atomspace/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'atomspace': 'ready'})
        
        @self.app.route('/atomspace/read_fragment', methods=['POST'])
        async def read_fragment():
            """Read hypergraph fragment matching pattern"""
            try:
                data = request.get_json()
                pattern = data.get('pattern', {})
                limit = data.get('limit', 100)
                
                # Limit to prevent abuse
                if limit > 1000:
                    limit = 1000
                
                # Add limit to pattern
                atoms = await self.atomspace.pattern_match(pattern, limit)
                return jsonify({
                    'success': True,
                    'atoms': [atom.to_dict() for atom in atoms],
                    'count': len(atoms)
                })
            except Exception as e:
                import logging
                logging.error("An error occurred: %s", str(e), exc_info=True)
                return jsonify({'success': False, 'error': 'An internal error has occurred.'}), 500
        
        @self.app.route('/atomspace/write_fragment', methods=['POST'])
        async def write_fragment():
            """Write hypergraph fragment"""
            try:
                data = request.get_json()
                atoms_data = data.get('atoms', [])
                
                if not atoms_data:
                    return jsonify({'success': False, 'error': 'No atoms provided'}), 400
                
                # Limit number of atoms to prevent abuse
                if len(atoms_data) > 100:
                    return jsonify({'success': False, 'error': 'Too many atoms (max 100)'}), 400
                
                atom_ids = await self.api.write_fragment(atoms_data)
                return jsonify({
                    'success': True,
                    'atom_ids': atom_ids,
                    'count': len(atom_ids)
                })
            except Exception as e:
                import logging
                logging.error("An error occurred: %s", str(e), exc_info=True)
                return jsonify({'success': False, 'error': 'An internal error has occurred.'}), 500
        
        @self.app.route('/atomspace/get_atom/<atom_id>', methods=['GET'])
        async def get_atom(atom_id):
            """Get specific atom by ID"""
            try:
                atom = await self.atomspace.get_atom(atom_id)
                if atom:
                    return jsonify({
                        'success': True,
                        'atom': atom.to_dict()
                    })
                else:
                    return jsonify({'success': False, 'error': 'Atom not found'}), 404
            except Exception as e:
                import logging
                logging.error("An error occurred: %s", str(e), exc_info=True)
                return jsonify({'success': False, 'error': 'An internal error has occurred.'}), 500
        
        @self.app.route('/atomspace/create_snapshot', methods=['POST'])
        async def create_snapshot():
            """Create memory snapshot"""
            try:
                data = request.get_json() or {}
                description = data.get('description', 'API snapshot')
                
                snapshot_id = await self.atomspace.create_snapshot(description)
                return jsonify({
                    'success': True,
                    'snapshot_id': snapshot_id
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': "An internal error has occurred."}), 500
        
        @self.app.route('/atomspace/memory_state', methods=['GET'])
        @self.app.route('/atomspace/memory_state/<snapshot_id>', methods=['GET'])
        async def get_memory_state(snapshot_id=None):
            """Get memory tensor state"""
            try:
                state = await self.api.get_memory_state(snapshot_id)
                return jsonify({
                    'success': True,
                    'memory_state': state
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': "An internal error has occurred."}), 500
        
        @self.app.route('/atomspace/add_node', methods=['POST'])
        async def add_node():
            """Add a new node to the hypergraph"""
            try:
                data = request.get_json()
                name = data.get('name')
                concept_type = data.get('concept_type', 'concept')
                truth_value = data.get('truth_value', 1.0)
                confidence = data.get('confidence', 1.0)
                metadata = data.get('metadata', {})
                
                if not name:
                    return jsonify({'success': False, 'error': 'Name is required'}), 400
                
                node = await self.atomspace.add_node(
                    name=name,
                    concept_type=concept_type,
                    truth_value=truth_value,
                    confidence=confidence,
                    metadata=metadata
                )
                
                return jsonify({
                    'success': True,
                    'node': node.to_dict()
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.app.route('/atomspace/add_link', methods=['POST'])
        async def add_link():
            """Add a new link to the hypergraph"""
            try:
                data = request.get_json()
                name = data.get('name')
                outgoing = data.get('outgoing', [])
                link_type = data.get('link_type', 'inheritance')
                truth_value = data.get('truth_value', 1.0)
                confidence = data.get('confidence', 1.0)
                metadata = data.get('metadata', {})
                
                if not name:
                    return jsonify({'success': False, 'error': 'Name is required'}), 400
                if not outgoing:
                    return jsonify({'success': False, 'error': 'Outgoing atoms required'}), 400
                
                link = await self.atomspace.add_link(
                    name=name,
                    outgoing=outgoing,
                    link_type=link_type,
                    truth_value=truth_value,
                    confidence=confidence,
                    metadata=metadata
                )
                
                return jsonify({
                    'success': True,
                    'link': link.to_dict()
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': "An internal error has occurred."}), 500

        # Scheme-based cognitive representation endpoints
        @self.app.route('/atomspace/cognitive/store_pattern', methods=['POST'])
        async def store_cognitive_pattern():
            """Store a cognitive pattern with Scheme expression"""
            try:
                data = request.get_json()
                name = data.get('name')
                scheme_expression = data.get('scheme_expression')
                grammar_id = data.get('grammar_id')
                metadata = data.get('metadata', {})
                
                if not name or not scheme_expression:
                    return jsonify({
                        'success': False, 
                        'error': 'Name and scheme_expression are required'
                    }), 400
                
                pattern = await self.cognitive_integration.store_cognitive_pattern(
                    name=name,
                    scheme_expression=scheme_expression,
                    grammar_id=grammar_id,
                    metadata=metadata
                )
                
                return jsonify({
                    'success': True,
                    'pattern': pattern.to_dict()
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/evaluate_pattern', methods=['POST'])
        async def evaluate_cognitive_pattern():
            """Evaluate a cognitive pattern with variable bindings"""
            try:
                data = request.get_json()
                pattern_id = data.get('pattern_id')
                bindings = data.get('bindings', {})
                
                if not pattern_id:
                    return jsonify({
                        'success': False, 
                        'error': 'Pattern ID is required'
                    }), 400
                
                result = await self.cognitive_integration.evaluate_cognitive_pattern(
                    pattern_id=pattern_id,
                    bindings=bindings
                )
                
                return jsonify({
                    'success': True,
                    'result': result
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/find_patterns', methods=['POST'])
        async def find_cognitive_patterns():
            """Find cognitive patterns matching a query"""
            try:
                data = request.get_json()
                query = data.get('query')
                max_results = data.get('max_results', 10)
                
                if not query:
                    return jsonify({
                        'success': False, 
                        'error': 'Query is required'
                    }), 400
                
                results = await self.cognitive_integration.find_cognitive_patterns(
                    query=query,
                    max_results=max_results
                )
                
                return jsonify({
                    'success': True,
                    'patterns': results,
                    'count': len(results)
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/reason', methods=['POST'])
        async def reason_with_patterns():
            """Perform reasoning using cognitive patterns"""
            try:
                data = request.get_json()
                pattern_ids = data.get('pattern_ids', [])
                reasoning_type = data.get('reasoning_type', 'forward_chaining')
                
                if not pattern_ids:
                    return jsonify({
                        'success': False, 
                        'error': 'Pattern IDs are required'
                    }), 400
                
                result = await self.cognitive_integration.reason_with_patterns(
                    pattern_ids=pattern_ids,
                    reasoning_type=reasoning_type
                )
                
                return jsonify({
                    'success': True,
                    'reasoning_result': result
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/statistics', methods=['GET'])
        async def get_cognitive_statistics():
            """Get cognitive system statistics"""
            try:
                stats = await self.cognitive_integration.get_cognitive_statistics()
                return jsonify({
                    'success': True,
                    'statistics': stats
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/export', methods=['GET'])
        async def export_cognitive_knowledge():
            """Export all cognitive knowledge"""
            try:
                data = await self.cognitive_integration.export_cognitive_knowledge()
                return jsonify({
                    'success': True,
                    'data': data
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/import', methods=['POST'])
        async def import_cognitive_knowledge():
            """Import cognitive knowledge"""
            try:
                data = request.get_json()
                knowledge_data = data.get('data')
                
                if not knowledge_data:
                    return jsonify({
                        'success': False, 
                        'error': 'Knowledge data is required'
                    }), 400
                
                success = await self.cognitive_integration.import_cognitive_knowledge(knowledge_data)
                
                return jsonify({
                    'success': success,
                    'message': 'Knowledge imported successfully' if success else 'Import failed'
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/grammars', methods=['GET'])
        async def list_cognitive_grammars():
            """List all cognitive grammars"""
            try:
                grammars = self.cognitive_integration.scheme_registry.list_grammars()
                return jsonify({
                    'success': True,
                    'grammars': [grammar.to_dict() for grammar in grammars],
                    'count': len(grammars)
                })
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500

        @self.app.route('/atomspace/cognitive/grammars/<grammar_id>', methods=['GET'])
        async def get_cognitive_grammar(grammar_id):
            """Get a specific cognitive grammar"""
            try:
                grammar = self.cognitive_integration.scheme_registry.get_grammar(grammar_id)
                if grammar:
                    return jsonify({
                        'success': True,
                        'grammar': grammar.to_dict()
                    })
                else:
                    return jsonify({'success': False, 'error': 'Grammar not found'}), 404
            except Exception as e:
                import logging
                logging.error("Exception occurred", exc_info=True)
                return jsonify({'success': False, 'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5001, debug=False):
        """Run the AtomSpace server"""
        self.app.run(host=host, port=port, debug=debug)


# Standalone script functionality
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='AtomSpace Hypergraph Memory Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to')
    parser.add_argument('--storage', default='memory/atomspace', help='Storage path for AtomSpace database')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    server = AtomSpaceServer(storage_path=args.storage)
    print(f"Starting AtomSpace server on {args.host}:{args.port}")
    print(f"Storage path: {args.storage}")
    
    server.run(host=args.host, port=args.port, debug=args.debug)