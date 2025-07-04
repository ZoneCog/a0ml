"""
Integration Adapters for Agent Zero and Bolt.diy

Modular adapters that plug the unified cognitive kernel into external systems
for distributed, real-world agent deployment and development environments.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import logging
from abc import ABC, abstractmethod

from ..helpers.cognitive_kernel import UnifiedCognitiveKernel
from ..helpers.scheme_grammar import SchemeCognitiveGrammarRegistry


class BaseAdapter(ABC):
    """Base class for cognitive kernel adapters"""
    
    def __init__(self, kernel: UnifiedCognitiveKernel):
        self.kernel = kernel
        self.logger = logging.getLogger(self.__class__.__name__)
        self.adapter_id = f"adapter_{id(self)}"
        
    @abstractmethod
    async def initialize(self) -> Dict[str, Any]:
        """Initialize the adapter"""
        pass
        
    @abstractmethod
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the cognitive kernel"""
        pass
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get adapter and kernel capabilities"""
        return {
            "adapter_id": self.adapter_id,
            "adapter_type": self.__class__.__name__,
            "kernel_id": self.kernel.kernel_id,
            "kernel_state": self.kernel.state.value,
            "subsystems": {
                "neural_symbolic": self.kernel.neural_symbolic_engine is not None,
                "ecan": self.kernel.ecan_system is not None,
                "task_orchestrator": self.kernel.task_orchestrator is not None
            },
            "grammars": len(self.kernel.cognitive_grammars),
            "protocols": len(self.kernel.modification_protocols)
        }


class AgentZeroAdapter(BaseAdapter):
    """
    Adapter for Agent Zero integration
    
    Provides seamless integration with the Agent Zero framework,
    enabling cognitive kernel-powered agents with advanced reasoning,
    memory management, and self-modification capabilities.
    """
    
    def __init__(self, kernel: UnifiedCognitiveKernel):
        super().__init__(kernel)
        self.agent_contexts: Dict[str, Dict[str, Any]] = {}
        self.tool_registry: Dict[str, Any] = {}
        
    async def initialize(self) -> Dict[str, Any]:
        """Initialize Agent Zero adapter"""
        self.logger.info("Initializing Agent Zero adapter")
        
        # Register cognitive kernel tools
        await self._register_cognitive_tools()
        
        # Setup agent context templates
        self._setup_agent_templates()
        
        return {
            "adapter_initialized": True,
            "tools_registered": len(self.tool_registry),
            "kernel_capabilities": self.get_capabilities()
        }
        
    async def _register_cognitive_tools(self):
        """Register cognitive kernel tools for Agent Zero"""
        
        # Memory tool
        self.tool_registry["cognitive_memory"] = {
            "name": "Cognitive Memory Access",
            "description": "Access hypergraph memory with pattern matching and semantic queries",
            "parameters": {
                "concepts": {"type": "array", "description": "Concepts to query"},
                "patterns": {"type": "array", "description": "Patterns to match"},
                "depth": {"type": "integer", "description": "Query depth"}
            },
            "function": self._memory_tool
        }
        
        # Reasoning tool
        self.tool_registry["cognitive_reasoning"] = {
            "name": "Neural-Symbolic Reasoning",
            "description": "Perform advanced reasoning with PLN and pattern inference",
            "parameters": {
                "reasoning_type": {"type": "string", "description": "Type of reasoning"},
                "premises": {"type": "array", "description": "Reasoning premises"},
                "confidence_threshold": {"type": "number", "description": "Minimum confidence"}
            },
            "function": self._reasoning_tool
        }
        
        # Grammar tool
        self.tool_registry["cognitive_grammar"] = {
            "name": "Cognitive Grammar Processing",
            "description": "Execute Scheme-based cognitive grammars for complex thinking",
            "parameters": {
                "grammar_id": {"type": "string", "description": "Grammar identifier"},
                "bindings": {"type": "object", "description": "Variable bindings"},
                "operation": {"type": "string", "description": "Grammar operation"}
            },
            "function": self._grammar_tool
        }
        
        # Meta-cognitive tool
        self.tool_registry["meta_cognition"] = {
            "name": "Meta-Cognitive Control",
            "description": "Monitor and control cognitive processes, trigger self-modification",
            "parameters": {
                "operation": {"type": "string", "description": "Meta-cognitive operation"},
                "parameters": {"type": "object", "description": "Operation parameters"}
            },
            "function": self._meta_cognitive_tool
        }
        
        self.logger.info(f"Registered {len(self.tool_registry)} cognitive tools")
        
    def _setup_agent_templates(self):
        """Setup agent context templates"""
        
        # Cognitive agent template
        self.cognitive_agent_template = {
            "name": "Cognitive Agent",
            "description": "Agent with full cognitive kernel capabilities",
            "tools": list(self.tool_registry.keys()),
            "memory_enabled": True,
            "reasoning_enabled": True,
            "meta_cognition_enabled": True,
            "self_modification_enabled": True
        }
        
        # Specialized agent templates
        self.specialized_templates = {
            "memory_agent": {
                "name": "Memory Specialist",
                "tools": ["cognitive_memory"],
                "focus": "memory and knowledge management"
            },
            "reasoning_agent": {
                "name": "Reasoning Specialist", 
                "tools": ["cognitive_reasoning", "cognitive_grammar"],
                "focus": "logical reasoning and inference"
            },
            "meta_agent": {
                "name": "Meta-Cognitive Agent",
                "tools": ["meta_cognition", "cognitive_grammar"],
                "focus": "self-monitoring and adaptation"
            }
        }
        
    async def create_agent_context(self, agent_config: Dict[str, Any]) -> str:
        """Create a new agent context"""
        agent_id = f"agent_{len(self.agent_contexts)}"
        
        # Use template if specified
        template_name = agent_config.get("template")
        if template_name in self.specialized_templates:
            template = self.specialized_templates[template_name]
            agent_config.update(template)
        elif template_name == "cognitive":
            agent_config.update(self.cognitive_agent_template)
            
        # Create context
        context = {
            "agent_id": agent_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "config": agent_config,
            "sessions": [],
            "tool_usage": {},
            "performance_metrics": {
                "queries_processed": 0,
                "avg_response_time": 0.0,
                "success_rate": 1.0
            }
        }
        
        self.agent_contexts[agent_id] = context
        self.logger.info(f"Created agent context: {agent_id}")
        
        return agent_id
        
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process Agent Zero request through cognitive kernel"""
        start_time = datetime.now()
        
        try:
            # Extract request components
            agent_id = request.get("agent_id")
            tool_name = request.get("tool")
            parameters = request.get("parameters", {})
            
            # Get agent context
            if agent_id and agent_id in self.agent_contexts:
                context = self.agent_contexts[agent_id]
            else:
                context = None
                
            # Route to appropriate tool
            if tool_name in self.tool_registry:
                tool_func = self.tool_registry[tool_name]["function"]
                result = await tool_func(parameters, context)
            else:
                # Direct kernel invocation
                result = await self.kernel.recursive_invoke(request)
                
            # Update context if available
            if context:
                self._update_agent_context(context, tool_name, result, start_time)
                
            return {
                "success": True,
                "result": result,
                "agent_id": agent_id,
                "tool": tool_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing Agent Zero request: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    def _update_agent_context(self, context: Dict[str, Any], tool: str, 
                            result: Dict[str, Any], start_time: datetime):
        """Update agent context with usage metrics"""
        duration = (datetime.now() - start_time).total_seconds()
        
        # Update tool usage
        if tool not in context["tool_usage"]:
            context["tool_usage"][tool] = {"count": 0, "total_time": 0.0}
            
        context["tool_usage"][tool]["count"] += 1
        context["tool_usage"][tool]["total_time"] += duration
        
        # Update performance metrics
        metrics = context["performance_metrics"]
        metrics["queries_processed"] += 1
        
        # Update average response time
        total_time = sum(usage["total_time"] for usage in context["tool_usage"].values())
        metrics["avg_response_time"] = total_time / metrics["queries_processed"]
        
        # Update success rate
        if result.get("success", False):
            success_count = metrics["queries_processed"] * metrics["success_rate"] + 1
            metrics["success_rate"] = success_count / (metrics["queries_processed"] + 1)
        else:
            success_count = metrics["queries_processed"] * metrics["success_rate"]
            metrics["success_rate"] = success_count / (metrics["queries_processed"] + 1)
            
    async def _memory_tool(self, parameters: Dict[str, Any], 
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Cognitive memory tool implementation"""
        query = {
            "type": "memory",
            "content": {
                "concepts": parameters.get("concepts", []),
                "patterns": parameters.get("patterns", []),
                "semantic_depth": parameters.get("depth", 2)
            }
        }
        
        return await self.kernel.recursive_invoke(query)
        
    async def _reasoning_tool(self, parameters: Dict[str, Any],
                            context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Cognitive reasoning tool implementation"""
        query = {
            "type": "reasoning",
            "content": {
                "reasoning_type": parameters.get("reasoning_type", "deductive"),
                "premises": parameters.get("premises", []),
                "confidence_threshold": parameters.get("confidence_threshold", 0.7)
            }
        }
        
        return await self.kernel.recursive_invoke(query)
        
    async def _grammar_tool(self, parameters: Dict[str, Any],
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Cognitive grammar tool implementation"""
        grammar_id = parameters.get("grammar_id")
        bindings = parameters.get("bindings", {})
        operation = parameters.get("operation", "evaluate")
        
        if operation == "evaluate" and grammar_id:
            # Use grammar registry for evaluation
            if hasattr(self.kernel, 'grammar_registry'):
                result = self.kernel.grammar_registry.evaluate_grammar_expression(
                    grammar_id, bindings
                )
                return {"grammar_result": result}
                
        # Fallback to general query
        query = {
            "type": "reasoning",
            "content": {
                "grammar_processing": True,
                "grammar_id": grammar_id,
                "bindings": bindings
            }
        }
        
        return await self.kernel.recursive_invoke(query)
        
    async def _meta_cognitive_tool(self, parameters: Dict[str, Any],
                                 context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Meta-cognitive tool implementation"""
        operation = parameters.get("operation", "status")
        
        if operation == "status":
            return {
                "kernel_statistics": self.kernel.get_kernel_statistics(),
                "attention_membranes": self.kernel.get_attention_membranes(),
                "meta_events": len(self.kernel.meta_events)
            }
        elif operation == "self_modify":
            protocol_id = parameters.get("protocol_id", "attention_reallocation")
            await self.kernel._trigger_self_modification(protocol_id)
            return {"self_modification": "triggered", "protocol": protocol_id}
        else:
            query = {
                "type": "autonomy",
                "content": parameters
            }
            return await self.kernel.recursive_invoke(query)


# Factory function for creating adapters
def create_adapter(adapter_type: str, kernel: UnifiedCognitiveKernel) -> BaseAdapter:
    """Create an adapter of the specified type"""
    
    if adapter_type.lower() == "agent_zero":
        return AgentZeroAdapter(kernel)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")


# Example usage functions
async def example_agent_zero_integration():
    """Example of Agent Zero integration"""
    from ..helpers.atomspace import AtomSpace
    
    # Setup
    atomspace = AtomSpace("/tmp/agent_zero_demo.db")
    kernel = UnifiedCognitiveKernel(atomspace)
    await kernel.initialize()
    
    # Create adapter
    adapter = AgentZeroAdapter(kernel)
    await adapter.initialize()
    
    # Create cognitive agent
    agent_id = await adapter.create_agent_context({
        "template": "cognitive",
        "name": "Advanced Cognitive Agent",
        "capabilities": ["memory", "reasoning", "meta_cognition"]
    })
    
    # Process requests
    memory_request = {
        "agent_id": agent_id,
        "tool": "cognitive_memory",
        "parameters": {
            "concepts": ["intelligence", "learning"],
            "depth": 3
        }
    }
    
    result = await adapter.process_request(memory_request)
    print(f"Memory query result: {result['success']}")
    
    await kernel.shutdown()


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_agent_zero_integration())