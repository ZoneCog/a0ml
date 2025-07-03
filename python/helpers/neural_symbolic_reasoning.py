"""
Neural-Symbolic Reasoning Engine

Main engine that integrates PLN, MOSES, and Pattern Matcher for comprehensive
neural-symbolic reasoning with tensor-based cognitive representations.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime, timezone
import json
import uuid

from .atomspace import AtomSpace
from .pln_reasoning import PLNInferenceEngine, TruthValue
from .moses_optimizer import MOSESOptimizer, Program, ProgramType
from .pattern_matcher import HypergraphPatternMatcher, Pattern, Match, MatchType


class ReasoningStage(Enum):
    """Stages of neural-symbolic reasoning"""
    PERCEPTION = "perception"
    PATTERN_RECOGNITION = "pattern_recognition"
    INFERENCE = "inference"
    OPTIMIZATION = "optimization"
    DECISION = "decision"
    ACTION = "action"


@dataclass
class CognitiveKernel:
    """Represents a cognitive kernel with tensor encoding"""
    id: str
    name: str
    patterns: List[str]  # Pattern IDs
    reasoning_programs: List[str]  # Program IDs
    stages: List[ReasoningStage]
    tensor_representation: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "patterns": self.patterns,
            "reasoning_programs": self.reasoning_programs,
            "stages": [stage.value for stage in self.stages],
            "tensor_shape": self.tensor_representation.shape if self.tensor_representation is not None else None,
            "metadata": self.metadata
        }


class NeuralSymbolicReasoningEngine:
    """
    Main neural-symbolic reasoning engine integrating PLN, MOSES, and Pattern Matcher
    """
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.pln_engine = PLNInferenceEngine(atomspace)
        self.moses_optimizer = MOSESOptimizer(atomspace, self.pln_engine)
        self.pattern_matcher = HypergraphPatternMatcher(atomspace, self.pln_engine)
        
        self.cognitive_kernels: Dict[str, CognitiveKernel] = {}
        self.reasoning_sessions: List[Dict[str, Any]] = []
        self.tensor_cache: Dict[str, np.ndarray] = {}
        
        # Tensor dimensions
        self.max_patterns = 100
        self.max_reasoning_programs = 50
        self.max_stages = len(ReasoningStage)
    
    async def initialize_system(self) -> Dict[str, Any]:
        """Initialize the neural-symbolic reasoning system"""
        initialization_report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "pln_engine": "initialized",
                "moses_optimizer": "initialized", 
                "pattern_matcher": "initialized"
            },
            "tensor_dimensions": {
                "max_patterns": self.max_patterns,
                "max_reasoning_programs": self.max_reasoning_programs,
                "max_stages": self.max_stages
            }
        }
        
        # Initialize basic cognitive kernels
        await self._create_default_cognitive_kernels()
        
        initialization_report["default_kernels"] = len(self.cognitive_kernels)
        return initialization_report
    
    async def _create_default_cognitive_kernels(self):
        """Create default cognitive kernels"""
        # Perception kernel
        perception_kernel = CognitiveKernel(
            id="perception_kernel",
            name="Perception and Pattern Recognition",
            patterns=[],
            reasoning_programs=[],
            stages=[ReasoningStage.PERCEPTION, ReasoningStage.PATTERN_RECOGNITION],
            metadata={"type": "perception", "priority": "high"}
        )
        
        # Inference kernel
        inference_kernel = CognitiveKernel(
            id="inference_kernel",
            name="Logical Inference and Reasoning",
            patterns=[],
            reasoning_programs=[],
            stages=[ReasoningStage.INFERENCE, ReasoningStage.DECISION],
            metadata={"type": "inference", "priority": "high"}
        )
        
        # Optimization kernel
        optimization_kernel = CognitiveKernel(
            id="optimization_kernel",
            name="Program Optimization and Learning",
            patterns=[],
            reasoning_programs=[],
            stages=[ReasoningStage.OPTIMIZATION, ReasoningStage.ACTION],
            metadata={"type": "optimization", "priority": "medium"}
        )
        
        self.cognitive_kernels["perception_kernel"] = perception_kernel
        self.cognitive_kernels["inference_kernel"] = inference_kernel
        self.cognitive_kernels["optimization_kernel"] = optimization_kernel
    
    async def create_cognitive_kernel(self, name: str, patterns: List[str], 
                                   reasoning_programs: List[str], 
                                   stages: List[ReasoningStage]) -> CognitiveKernel:
        """Create a new cognitive kernel"""
        kernel_id = f"kernel_{uuid.uuid4().hex[:8]}"
        
        kernel = CognitiveKernel(
            id=kernel_id,
            name=name,
            patterns=patterns,
            reasoning_programs=reasoning_programs,
            stages=stages
        )
        
        # Generate tensor representation
        kernel.tensor_representation = await self._encode_cognitive_kernel_tensor(kernel)
        
        self.cognitive_kernels[kernel_id] = kernel
        return kernel
    
    async def _encode_cognitive_kernel_tensor(self, kernel: CognitiveKernel) -> np.ndarray:
        """
        Encode cognitive kernel as tensor T_ai[n_patterns, n_reasoning, l_stages]
        """
        # Initialize tensor
        tensor = np.zeros((self.max_patterns, self.max_reasoning_programs, self.max_stages))
        
        # Encode patterns
        for i, pattern_id in enumerate(kernel.patterns[:self.max_patterns]):
            # Get pattern match statistics
            try:
                matches = await self.pattern_matcher.match_pattern(pattern_id, max_matches=10)
                pattern_strength = len(matches) / 10.0  # Normalize
                
                # Encode across reasoning and stages
                for j in range(min(len(kernel.reasoning_programs), self.max_reasoning_programs)):
                    for k, stage in enumerate(kernel.stages):
                        if k < self.max_stages:
                            tensor[i, j, k] = pattern_strength
            except Exception:
                pass
        
        # Encode reasoning programs
        for j, program_id in enumerate(kernel.reasoning_programs[:self.max_reasoning_programs]):
            try:
                program = await self.moses_optimizer.load_program(program_id)
                if program:
                    program_fitness = program.fitness
                    
                    # Encode across patterns and stages
                    for i in range(min(len(kernel.patterns), self.max_patterns)):
                        for k, stage in enumerate(kernel.stages):
                            if k < self.max_stages:
                                tensor[i, j, k] = max(tensor[i, j, k], program_fitness)
            except Exception:
                pass
        
        # Encode stages
        for k, stage in enumerate(kernel.stages):
            if k < self.max_stages:
                stage_weight = 1.0  # Could be learned
                tensor[:, :, k] *= stage_weight
        
        return tensor
    
    async def reason(self, query: Dict[str, Any], 
                   kernel_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform neural-symbolic reasoning on a query
        
        Args:
            query: Query containing reasoning request
            kernel_id: Optional specific kernel to use
            
        Returns:
            Reasoning result with explanations
        """
        session_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        reasoning_session = {
            "session_id": session_id,
            "timestamp": start_time.isoformat(),
            "query": query,
            "kernel_id": kernel_id,
            "stages": [],
            "result": None,
            "explanations": []
        }
        
        # Select cognitive kernel
        kernel = self._select_kernel(query, kernel_id)
        if not kernel:
            reasoning_session["result"] = {"error": "No suitable kernel found"}
            return reasoning_session
        
        # Execute reasoning stages
        context = {"query": query, "kernel": kernel, "session_id": session_id}
        
        for stage in kernel.stages:
            stage_result = await self._execute_reasoning_stage(stage, context)
            reasoning_session["stages"].append(stage_result)
            
            # Update context with stage results
            context.update(stage_result.get("outputs", {}))
        
        # Compile final result
        reasoning_session["result"] = await self._compile_reasoning_result(reasoning_session)
        reasoning_session["duration"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        self.reasoning_sessions.append(reasoning_session)
        return reasoning_session
    
    def _select_kernel(self, query: Dict[str, Any], kernel_id: Optional[str]) -> Optional[CognitiveKernel]:
        """Select appropriate cognitive kernel for query"""
        if kernel_id and kernel_id in self.cognitive_kernels:
            return self.cognitive_kernels[kernel_id]
        
        # Simple selection based on query type
        query_type = query.get("type", "general")
        
        if query_type in ["pattern", "match", "find"]:
            return self.cognitive_kernels.get("perception_kernel")
        elif query_type in ["infer", "deduce", "reason"]:
            return self.cognitive_kernels.get("inference_kernel")
        elif query_type in ["optimize", "learn", "adapt"]:
            return self.cognitive_kernels.get("optimization_kernel")
        
        # Default to inference kernel
        return self.cognitive_kernels.get("inference_kernel")
    
    async def _execute_reasoning_stage(self, stage: ReasoningStage, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific reasoning stage"""
        stage_result = {
            "stage": stage.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": dict(context),
            "outputs": {},
            "duration": 0.0,
            "success": False
        }
        
        start_time = datetime.now(timezone.utc)
        
        try:
            if stage == ReasoningStage.PERCEPTION:
                stage_result["outputs"] = await self._perception_stage(context)
            elif stage == ReasoningStage.PATTERN_RECOGNITION:
                stage_result["outputs"] = await self._pattern_recognition_stage(context)
            elif stage == ReasoningStage.INFERENCE:
                stage_result["outputs"] = await self._inference_stage(context)
            elif stage == ReasoningStage.OPTIMIZATION:
                stage_result["outputs"] = await self._optimization_stage(context)
            elif stage == ReasoningStage.DECISION:
                stage_result["outputs"] = await self._decision_stage(context)
            elif stage == ReasoningStage.ACTION:
                stage_result["outputs"] = await self._action_stage(context)
            
            stage_result["success"] = True
            
        except Exception as e:
            stage_result["outputs"] = {"error": str(e)}
            stage_result["success"] = False
        
        stage_result["duration"] = (datetime.now(timezone.utc) - start_time).total_seconds()
        return stage_result
    
    async def _perception_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perception stage - gather and process input"""
        query = context["query"]
        
        # Extract relevant atoms from query
        relevant_atoms = []
        
        if "atom_ids" in query:
            relevant_atoms = query["atom_ids"]
        elif "concepts" in query:
            # Find atoms related to concepts
            for concept in query["concepts"]:
                pattern = {"concept_type": "knowledge", "name": concept}
                atoms = await self.atomspace.storage.get_atoms_by_pattern(pattern, limit=10)
                relevant_atoms.extend([atom.id for atom in atoms])
        
        return {
            "relevant_atoms": relevant_atoms,
            "perception_complete": True
        }
    
    async def _pattern_recognition_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pattern recognition stage - identify patterns in data"""
        relevant_atoms = context.get("relevant_atoms", [])
        
        # Apply pattern matching
        all_matches = []
        
        # Use all registered patterns
        for pattern_id in self.pattern_matcher.patterns.keys():
            matches = await self.pattern_matcher.match_pattern(pattern_id, max_matches=5)
            all_matches.extend(matches)
        
        # Filter matches related to relevant atoms
        filtered_matches = []
        for match in all_matches:
            if any(atom_id in relevant_atoms for atom_id in match.matched_atoms):
                filtered_matches.append(match)
        
        # Extract semantic relations
        if relevant_atoms:
            semantic_relations = await self.pattern_matcher.extract_semantic_relations(relevant_atoms)
        else:
            semantic_relations = {}
        
        return {
            "pattern_matches": [match.to_dict() for match in filtered_matches],
            "semantic_relations": semantic_relations,
            "pattern_recognition_complete": True
        }
    
    async def _inference_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inference stage - perform logical reasoning"""
        relevant_atoms = context.get("relevant_atoms", [])
        
        # Perform forward chaining inference
        inference_results = []
        
        if relevant_atoms:
            # Use relevant atoms as premises
            premises = relevant_atoms[:5]  # Limit premises
            inference_result = await self.pln_engine.forward_chaining(premises, max_iterations=5)
            inference_results.append(inference_result)
        
        # Infer truth values for relevant atoms
        truth_values = {}
        for atom_id in relevant_atoms[:10]:  # Limit for performance
            try:
                tv = await self.pln_engine.infer_truth_value(atom_id)
                truth_values[atom_id] = tv.to_dict()
            except Exception:
                pass
        
        return {
            "inference_results": inference_results,
            "truth_values": truth_values,
            "inference_complete": True
        }
    
    async def _optimization_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimization stage - optimize programs and patterns"""
        # Run a short optimization cycle
        optimization_results = []
        
        # Optimize inference rules
        try:
            optimization_result = await self.moses_optimizer.optimize_program(
                ProgramType.INFERENCE_RULE,
                generations=3,
                seed_programs=None
            )
            optimization_results.append(optimization_result)
        except Exception as e:
            optimization_results.append({"error": str(e)})
        
        return {
            "optimization_results": optimization_results,
            "optimization_complete": True
        }
    
    async def _decision_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Decision stage - make decisions based on reasoning"""
        # Analyze all previous stage outputs
        truth_values = context.get("truth_values", {})
        pattern_matches = context.get("pattern_matches", [])
        
        # Simple decision making based on confidence
        decisions = []
        
        # Decision based on truth values
        for atom_id, tv in truth_values.items():
            if tv.get("confidence", 0) > 0.7:
                decisions.append({
                    "type": "high_confidence_belief",
                    "atom_id": atom_id,
                    "confidence": tv.get("confidence", 0),
                    "strength": tv.get("strength", 0)
                })
        
        # Decision based on pattern matches
        high_confidence_matches = [m for m in pattern_matches if m.get("confidence", 0) > 0.8]
        if high_confidence_matches:
            decisions.append({
                "type": "strong_pattern_detected",
                "matches": len(high_confidence_matches),
                "patterns": [m.get("pattern_id") for m in high_confidence_matches]
            })
        
        return {
            "decisions": decisions,
            "decision_complete": True
        }
    
    async def _action_stage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Action stage - execute actions based on decisions"""
        decisions = context.get("decisions", [])
        
        actions_taken = []
        
        for decision in decisions:
            if decision["type"] == "high_confidence_belief":
                # Could create new knowledge nodes
                actions_taken.append({
                    "action": "reinforce_belief",
                    "atom_id": decision["atom_id"]
                })
            elif decision["type"] == "strong_pattern_detected":
                # Could create new patterns
                actions_taken.append({
                    "action": "strengthen_patterns",
                    "patterns": decision["patterns"]
                })
        
        return {
            "actions_taken": actions_taken,
            "action_complete": True
        }
    
    async def _compile_reasoning_result(self, reasoning_session: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final reasoning result from all stages"""
        result = {
            "session_id": reasoning_session["session_id"],
            "success": True,
            "summary": {},
            "detailed_results": {},
            "confidence": 0.0
        }
        
        # Extract key results from each stage
        for stage_result in reasoning_session["stages"]:
            stage_name = stage_result["stage"]
            stage_outputs = stage_result.get("outputs", {})
            
            result["detailed_results"][stage_name] = stage_outputs
            
            # Update summary
            if stage_name == "inference":
                result["summary"]["inference_results"] = len(stage_outputs.get("inference_results", []))
                result["summary"]["truth_values"] = len(stage_outputs.get("truth_values", {}))
            elif stage_name == "pattern_recognition":
                result["summary"]["pattern_matches"] = len(stage_outputs.get("pattern_matches", []))
                result["summary"]["semantic_relations"] = len(stage_outputs.get("semantic_relations", {}))
            elif stage_name == "decision":
                result["summary"]["decisions"] = len(stage_outputs.get("decisions", []))
            elif stage_name == "action":
                result["summary"]["actions"] = len(stage_outputs.get("actions_taken", []))
        
        # Calculate overall confidence
        confidences = []
        for stage_result in reasoning_session["stages"]:
            if stage_result["success"]:
                confidences.append(1.0)
            else:
                confidences.append(0.0)
        
        result["confidence"] = np.mean(confidences) if confidences else 0.0
        
        return result
    
    async def get_cognitive_tensor(self, kernel_id: str) -> Optional[np.ndarray]:
        """Get tensor representation of cognitive kernel"""
        if kernel_id in self.cognitive_kernels:
            kernel = self.cognitive_kernels[kernel_id]
            if kernel.tensor_representation is not None:
                return kernel.tensor_representation
            else:
                # Generate tensor if not exists
                tensor = await self._encode_cognitive_kernel_tensor(kernel)
                kernel.tensor_representation = tensor
                return tensor
        
        return None
    
    async def get_system_tensor(self) -> np.ndarray:
        """
        Get system-wide tensor representation T_ai[n_patterns, n_reasoning, l_stages]
        """
        cache_key = "system_tensor"
        
        if cache_key in self.tensor_cache:
            return self.tensor_cache[cache_key]
        
        # Initialize system tensor
        system_tensor = np.zeros((self.max_patterns, self.max_reasoning_programs, self.max_stages))
        
        # Aggregate from all cognitive kernels
        for kernel in self.cognitive_kernels.values():
            kernel_tensor = await self.get_cognitive_tensor(kernel.id)
            if kernel_tensor is not None:
                # Add kernel tensor to system tensor
                system_tensor += kernel_tensor
        
        # Normalize
        max_val = np.max(system_tensor)
        if max_val > 0:
            system_tensor /= max_val
        
        self.tensor_cache[cache_key] = system_tensor
        return system_tensor
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            "cognitive_kernels": len(self.cognitive_kernels),
            "reasoning_sessions": len(self.reasoning_sessions),
            "pln_statistics": self.pln_engine.get_statistics(),
            "moses_statistics": self.moses_optimizer.get_statistics(),
            "pattern_matcher_statistics": self.pattern_matcher.get_statistics(),
            "tensor_dimensions": {
                "max_patterns": self.max_patterns,
                "max_reasoning_programs": self.max_reasoning_programs,
                "max_stages": self.max_stages
            }
        }
    
    def clear_caches(self):
        """Clear all caches"""
        self.pln_engine.clear_cache()
        self.pattern_matcher.clear_cache()
        self.tensor_cache.clear()
    
    async def create_reasoning_api_request(self, query_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create API request for neural-symbolic reasoning"""
        api_request = {
            "type": query_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data
        }
        
        # Process the request
        reasoning_result = await self.reason(api_request)
        
        # Format API response
        api_response = {
            "success": reasoning_result["result"]["success"],
            "session_id": reasoning_result["session_id"],
            "query_type": query_type,
            "results": reasoning_result["result"]["summary"],
            "confidence": reasoning_result["result"]["confidence"],
            "duration": reasoning_result["duration"],
            "detailed_results": reasoning_result["result"]["detailed_results"] if data.get("include_details", False) else None
        }
        
        return api_response