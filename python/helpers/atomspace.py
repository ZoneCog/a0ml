"""
AtomSpace Hypergraph Memory System

Implements a distributed memory agent with hypergraph AtomSpace integration
for cognitive representation and pattern matching.
"""

import uuid
import json
import sqlite3
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from pathlib import Path


class AtomType(Enum):
    """Types of atoms in the hypergraph"""
    NODE = "node"
    LINK = "link"


@dataclass
class Atom:
    """Base class for all atoms in the hypergraph"""
    id: str
    atom_type: AtomType
    name: str
    truth_value: float = 1.0
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert atom to dictionary for storage"""
        return {
            'id': self.id,
            'atom_type': self.atom_type.value,
            'name': self.name,
            'truth_value': self.truth_value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'metadata': json.dumps(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Atom':
        """Create atom from dictionary"""
        return cls(
            id=data['id'],
            atom_type=AtomType(data['atom_type']),
            name=data['name'],
            truth_value=data['truth_value'],
            confidence=data['confidence'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=json.loads(data['metadata']) if data['metadata'] else {}
        )


@dataclass
class Node(Atom):
    """Node atom representing concepts, entities, or values"""
    concept_type: str = "concept"
    
    def __post_init__(self):
        self.atom_type = AtomType.NODE


@dataclass
class Link(Atom):
    """Link atom representing relationships between nodes"""
    outgoing: List[str] = field(default_factory=list)  # List of atom IDs
    link_type: str = "inheritance"
    
    def __post_init__(self):
        self.atom_type = AtomType.LINK
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert link to dictionary with outgoing connections"""
        data = super().to_dict()
        data['outgoing'] = json.dumps(self.outgoing)
        data['link_type'] = self.link_type
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Link':
        """Create link from dictionary"""
        link = super().from_dict(data)
        link.outgoing = json.loads(data['outgoing']) if data['outgoing'] else []
        link.link_type = data.get('link_type', 'inheritance')
        return link


class HypergraphStorage:
    """Persistent storage backend for hypergraph using SQLite"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS atoms (
                    id TEXT PRIMARY KEY,
                    atom_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    truth_value REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    timestamp TEXT NOT NULL,
                    metadata TEXT,
                    concept_type TEXT,
                    link_type TEXT,
                    outgoing TEXT
                );
                
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    n_nodes INTEGER NOT NULL,
                    n_links INTEGER NOT NULL,
                    description TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(atom_type);
                CREATE INDEX IF NOT EXISTS idx_atoms_name ON atoms(name);
                CREATE INDEX IF NOT EXISTS idx_atoms_timestamp ON atoms(timestamp);
                CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON snapshots(timestamp);
            ''')
    
    async def store_atom(self, atom: Atom) -> bool:
        """Store an atom in the database"""
        try:
            data = atom.to_dict()
            with sqlite3.connect(self.db_path) as conn:
                if isinstance(atom, Link):
                    conn.execute('''
                        INSERT OR REPLACE INTO atoms 
                        (id, atom_type, name, truth_value, confidence, timestamp, metadata, link_type, outgoing)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data['id'], data['atom_type'], data['name'], data['truth_value'], 
                         data['confidence'], data['timestamp'], data['metadata'], 
                         data['link_type'], data['outgoing']))
                else:
                    conn.execute('''
                        INSERT OR REPLACE INTO atoms 
                        (id, atom_type, name, truth_value, confidence, timestamp, metadata, concept_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (data['id'], data['atom_type'], data['name'], data['truth_value'], 
                         data['confidence'], data['timestamp'], data['metadata'], 
                         getattr(atom, 'concept_type', 'concept')))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error storing atom {atom.id}: {e}")
            return False
    
    async def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Retrieve an atom by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('SELECT * FROM atoms WHERE id = ?', (atom_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                data = dict(row)
                if data['atom_type'] == AtomType.LINK.value:
                    return Link.from_dict(data)
                else:
                    node = Node.from_dict(data)
                    node.concept_type = data.get('concept_type', 'concept')
                    return node
        except Exception as e:
            print(f"Error retrieving atom {atom_id}: {e}")
            return None
    
    async def get_atoms_by_pattern(self, pattern: Dict[str, Any], limit: int = 100) -> List[Atom]:
        """Retrieve atoms matching a pattern"""
        try:
            conditions = []
            params = []
            
            for key, value in pattern.items():
                if key in ['atom_type', 'name', 'concept_type', 'link_type']:
                    conditions.append(f"{key} = ?")
                    params.append(value)
                elif key == 'truth_value_min':
                    conditions.append("truth_value >= ?")
                    params.append(value)
                elif key == 'timestamp_after':
                    conditions.append("timestamp > ?")
                    params.append(value)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            query = f"SELECT * FROM atoms WHERE {where_clause} LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                atoms = []
                for row in rows:
                    data = dict(row)
                    if data['atom_type'] == AtomType.LINK.value:
                        atoms.append(Link.from_dict(data))
                    else:
                        node = Node.from_dict(data)
                        node.concept_type = data.get('concept_type', 'concept')
                        atoms.append(node)
                
                return atoms
        except Exception as e:
            print(f"Error retrieving atoms by pattern: {e}")
            return []
    
    async def create_snapshot(self, snapshot_id: str, description: str = "") -> bool:
        """Create a snapshot of the current hypergraph state"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count nodes and links
                cursor = conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_type = ?", (AtomType.NODE.value,))
                n_nodes = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM atoms WHERE atom_type = ?", (AtomType.LINK.value,))
                n_links = cursor.fetchone()[0]
                
                # Store snapshot metadata
                conn.execute('''
                    INSERT INTO snapshots (snapshot_id, timestamp, n_nodes, n_links, description)
                    VALUES (?, ?, ?, ?, ?)
                ''', (snapshot_id, datetime.now(timezone.utc).isoformat(), n_nodes, n_links, description))
                conn.commit()
            return True
        except Exception as e:
            print(f"Error creating snapshot {snapshot_id}: {e}")
            return False
    
    async def get_tensor_state(self, snapshot_id: Optional[str] = None) -> np.ndarray:
        """Get tensor representation T_memory[n_nodes, n_links, t_snapshots]"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Get snapshot info
                if snapshot_id:
                    cursor = conn.execute('SELECT * FROM snapshots WHERE snapshot_id = ?', (snapshot_id,))
                    snapshot = cursor.fetchone()
                    if not snapshot:
                        return np.array([])
                    
                    snapshots = [dict(snapshot)]
                else:
                    cursor = conn.execute('SELECT * FROM snapshots ORDER BY timestamp')
                    snapshots = [dict(row) for row in cursor.fetchall()]
                
                if not snapshots:
                    return np.array([])
                
                # Create tensor dimensions
                max_nodes = max(s['n_nodes'] for s in snapshots)
                max_links = max(s['n_links'] for s in snapshots) 
                n_snapshots = len(snapshots)
                
                # Initialize tensor
                tensor = np.zeros((max_nodes, max_links, n_snapshots))
                
                # Fill tensor with truth values (simplified representation)
                for i, snapshot in enumerate(snapshots):
                    tensor[:snapshot['n_nodes'], :snapshot['n_links'], i] = 1.0
                
                return tensor
                
        except Exception as e:
            print(f"Error getting tensor state: {e}")
            return np.array([])


class AtomSpace:
    """Main AtomSpace class for hypergraph operations"""
    
    def __init__(self, storage_path: str):
        self.storage = HypergraphStorage(storage_path)
        self.local_atoms: Dict[str, Atom] = {}
        
    async def add_node(self, name: str, concept_type: str = "concept", 
                      truth_value: float = 1.0, confidence: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None) -> Node:
        """Add a new node to the hypergraph"""
        node = Node(
            id=str(uuid.uuid4()),
            name=name,
            concept_type=concept_type,
            truth_value=truth_value,
            confidence=confidence,
            metadata=metadata or {},
            atom_type=AtomType.NODE
        )
        
        self.local_atoms[node.id] = node
        await self.storage.store_atom(node)
        return node
    
    async def add_link(self, name: str, outgoing: List[str], link_type: str = "inheritance",
                      truth_value: float = 1.0, confidence: float = 1.0,
                      metadata: Optional[Dict[str, Any]] = None) -> Link:
        """Add a new link to the hypergraph"""
        link = Link(
            id=str(uuid.uuid4()),
            name=name,
            outgoing=outgoing,
            link_type=link_type,
            truth_value=truth_value,
            confidence=confidence,
            metadata=metadata or {},
            atom_type=AtomType.LINK
        )
        
        self.local_atoms[link.id] = link
        await self.storage.store_atom(link)
        return link
    
    async def get_atom(self, atom_id: str) -> Optional[Atom]:
        """Get an atom by ID (checks local cache first)"""
        if atom_id in self.local_atoms:
            return self.local_atoms[atom_id]
        
        atom = await self.storage.get_atom(atom_id)
        if atom:
            self.local_atoms[atom_id] = atom
        return atom
    
    async def pattern_match(self, pattern: Dict[str, Any], limit: int = 100) -> List[Atom]:
        """Find atoms matching a pattern"""
        return await self.storage.get_atoms_by_pattern(pattern, limit)
    
    async def create_snapshot(self, description: str = "") -> str:
        """Create a timestamped snapshot"""
        snapshot_id = str(uuid.uuid4())
        await self.storage.create_snapshot(snapshot_id, description)
        return snapshot_id
    
    async def get_memory_tensor(self, snapshot_id: Optional[str] = None) -> np.ndarray:
        """Get tensor representation of memory state"""
        return await self.storage.get_tensor_state(snapshot_id)


class DistributedAtomSpaceAPI:
    """API for distributed agents to interact with AtomSpace"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        
    async def read_fragment(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read hypergraph fragment matching pattern"""
        atoms = await self.atomspace.pattern_match(pattern)
        return [atom.to_dict() for atom in atoms]
    
    async def write_fragment(self, atoms_data: List[Dict[str, Any]]) -> List[str]:
        """Write hypergraph fragment from atom data"""
        atom_ids = []
        
        for atom_data in atoms_data:
            if atom_data['atom_type'] == AtomType.NODE.value:
                node = await self.atomspace.add_node(
                    name=atom_data['name'],
                    concept_type=atom_data.get('concept_type', 'concept'),
                    truth_value=atom_data.get('truth_value', 1.0),
                    confidence=atom_data.get('confidence', 1.0),
                    metadata=json.loads(atom_data.get('metadata', '{}'))
                )
                atom_ids.append(node.id)
            elif atom_data['atom_type'] == AtomType.LINK.value:
                link = await self.atomspace.add_link(
                    name=atom_data['name'],
                    outgoing=json.loads(atom_data.get('outgoing', '[]')),
                    link_type=atom_data.get('link_type', 'inheritance'),
                    truth_value=atom_data.get('truth_value', 1.0),
                    confidence=atom_data.get('confidence', 1.0),
                    metadata=json.loads(atom_data.get('metadata', '{}'))
                )
                atom_ids.append(link.id)
        
        return atom_ids
    
    async def get_memory_state(self, snapshot_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current memory state as tensor"""
        tensor = await self.atomspace.get_memory_tensor(snapshot_id)
        return {
            'tensor_shape': tensor.shape,
            'tensor_data': tensor.tolist() if tensor.size > 0 else [],
            'snapshot_id': snapshot_id
        }