"""
AtomSpace API endpoint for distributed agent access
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, request, jsonify
from python.helpers.atomspace import AtomSpace, DistributedAtomSpaceAPI


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