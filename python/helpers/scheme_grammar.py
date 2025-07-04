"""
Scheme-Based Cognitive Grammar Extension

Extends hypergraph pattern encoding to natively support Scheme cognitive grammars,
enabling expressive, composable agentic thought processes with dynamic vocabulary
registry for runtime pattern extension.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from datetime import datetime, timezone
import logging

from .atomspace import AtomSpace, AtomType
from .pattern_matcher import Pattern, MatchType


class SchemeNodeType(Enum):
    """Types of Scheme nodes in cognitive grammar"""
    ATOM = "atom"
    LIST = "list"
    SYMBOL = "symbol"
    LAMBDA = "lambda"
    PREDICATE = "predicate"
    FUNCTION = "function"
    COGNITIVE_OPERATOR = "cognitive_operator"


class CognitiveOperator(Enum):
    """Cognitive operators for agentic thought processes"""
    PERCEIVE = "perceive"
    REASON = "reason"
    DECIDE = "decide"
    ACT = "act"
    REMEMBER = "remember"
    LEARN = "learn"
    COMPOSE = "compose"
    REFLECT = "reflect"
    ANTICIPATE = "anticipate"
    EVALUATE = "evaluate"


@dataclass
class SchemeNode:
    """A node in a Scheme cognitive grammar expression"""
    node_type: SchemeNodeType
    value: Any
    children: List['SchemeNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "node_type": self.node_type.value,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SchemeNode':
        """Create from dictionary representation"""
        node = cls(
            node_type=SchemeNodeType(data["node_type"]),
            value=data["value"],
            metadata=data.get("metadata", {})
        )
        node.children = [cls.from_dict(child) for child in data.get("children", [])]
        return node


@dataclass
class CognitiveGrammar:
    """A cognitive grammar definition in Scheme"""
    id: str
    name: str
    description: str
    scheme_expression: str
    parsed_tree: Optional[SchemeNode] = None
    cognitive_operators: List[CognitiveOperator] = field(default_factory=list)
    pattern_templates: List[Dict[str, Any]] = field(default_factory=list)
    active: bool = True
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "scheme_expression": self.scheme_expression,
            "parsed_tree": self.parsed_tree.to_dict() if self.parsed_tree else None,
            "cognitive_operators": [op.value for op in self.cognitive_operators],
            "pattern_templates": self.pattern_templates,
            "active": self.active,
            "created": self.created.isoformat(),
            "usage_count": self.usage_count
        }


class SchemeGrammarParser:
    """Parser for Scheme cognitive grammar expressions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def parse(self, expression: str) -> SchemeNode:
        """Parse a Scheme expression into a syntax tree"""
        tokens = self._tokenize(expression)
        return self._parse_tokens(tokens)
        
    def _tokenize(self, expression: str) -> List[str]:
        """Tokenize a Scheme expression"""
        # Simple tokenizer for Scheme-like syntax
        expression = expression.strip()
        tokens = []
        i = 0
        
        while i < len(expression):
            char = expression[i]
            
            if char.isspace():
                i += 1
                continue
                
            elif char == '(':
                tokens.append('(')
                i += 1
                
            elif char == ')':
                tokens.append(')')
                i += 1
                
            elif char == '"':
                # String literal
                i += 1
                string_content = ""
                while i < len(expression) and expression[i] != '"':
                    if expression[i] == '\\' and i + 1 < len(expression):
                        i += 1
                        string_content += expression[i]
                    else:
                        string_content += expression[i]
                    i += 1
                if i < len(expression):
                    i += 1  # Skip closing quote
                tokens.append(f'"{string_content}"')
                
            else:
                # Symbol, number, or cognitive operator
                token = ""
                while i < len(expression) and not expression[i].isspace() and expression[i] not in '()':
                    token += expression[i]
                    i += 1
                tokens.append(token)
                
        return tokens
        
    def _parse_tokens(self, tokens: List[str]) -> SchemeNode:
        """Parse tokens into syntax tree"""
        if not tokens:
            raise ValueError("Empty token list")
            
        return self._parse_expression(tokens, 0)[0]
        
    def _parse_expression(self, tokens: List[str], index: int) -> Tuple[SchemeNode, int]:
        """Parse a single expression starting at index"""
        if index >= len(tokens):
            raise ValueError("Unexpected end of tokens")
            
        token = tokens[index]
        
        if token == '(':
            # Parse list
            return self._parse_list(tokens, index)
        elif token.startswith('"') and token.endswith('"'):
            # String literal
            return SchemeNode(SchemeNodeType.ATOM, token[1:-1]), index + 1
        elif token.isdigit() or (token.startswith('-') and token[1:].isdigit()):
            # Number
            return SchemeNode(SchemeNodeType.ATOM, int(token)), index + 1
        elif '.' in token and token.replace('.', '').replace('-', '').isdigit():
            # Float
            return SchemeNode(SchemeNodeType.ATOM, float(token)), index + 1
        elif token in [op.value for op in CognitiveOperator]:
            # Cognitive operator
            return SchemeNode(SchemeNodeType.COGNITIVE_OPERATOR, token), index + 1
        elif token.startswith('lambda'):
            # Lambda expression
            return SchemeNode(SchemeNodeType.LAMBDA, token), index + 1
        else:
            # Symbol
            return SchemeNode(SchemeNodeType.SYMBOL, token), index + 1
            
    def _parse_list(self, tokens: List[str], index: int) -> Tuple[SchemeNode, int]:
        """Parse a list expression"""
        if tokens[index] != '(':
            raise ValueError(f"Expected '(' but got '{tokens[index]}'")
            
        index += 1  # Skip opening paren
        children = []
        
        while index < len(tokens) and tokens[index] != ')':
            child, index = self._parse_expression(tokens, index)
            children.append(child)
            
        if index >= len(tokens):
            raise ValueError("Unexpected end of tokens in list")
            
        index += 1  # Skip closing paren
        
        # Determine list type
        if children and children[0].node_type == SchemeNodeType.COGNITIVE_OPERATOR:
            node_type = SchemeNodeType.COGNITIVE_OPERATOR
        elif children and children[0].node_type == SchemeNodeType.LAMBDA:
            node_type = SchemeNodeType.LAMBDA
        else:
            node_type = SchemeNodeType.LIST
            
        return SchemeNode(node_type, None, children), index


class SchemeCognitiveGrammarRegistry:
    """Registry for Scheme-based cognitive grammars"""
    
    def __init__(self, atomspace: AtomSpace):
        self.atomspace = atomspace
        self.logger = logging.getLogger(__name__)
        self.parser = SchemeGrammarParser()
        
        # Grammar storage
        self.grammars: Dict[str, CognitiveGrammar] = {}
        self.grammar_patterns: Dict[str, List[Pattern]] = {}
        
        # Built-in cognitive grammars
        self._register_builtin_grammars()
        
    def _register_builtin_grammars(self):
        """Register built-in cognitive grammars"""
        builtin_grammars = [
            {
                "id": "perception_grammar",
                "name": "Perception Grammar",
                "description": "Grammar for perceptual cognitive processes",
                "scheme_expression": "(perceive (object ?x) (property ?p) (context ?c))",
                "cognitive_operators": [CognitiveOperator.PERCEIVE],
                "pattern_templates": [
                    {
                        "template": {"perceive": "?x", "property": "?p", "context": "?c"},
                        "match_type": "semantic"
                    }
                ]
            },
            {
                "id": "reasoning_grammar",
                "name": "Reasoning Grammar", 
                "description": "Grammar for logical reasoning processes",
                "scheme_expression": "(reason (premise ?p1) (premise ?p2) (conclusion ?c))",
                "cognitive_operators": [CognitiveOperator.REASON],
                "pattern_templates": [
                    {
                        "template": {"reason": True, "premise": "?p", "conclusion": "?c"},
                        "match_type": "logical"
                    }
                ]
            },
            {
                "id": "decision_grammar",
                "name": "Decision Grammar",
                "description": "Grammar for decision-making processes",
                "scheme_expression": "(decide (options ?opts) (criteria ?crit) (choice ?choice))",
                "cognitive_operators": [CognitiveOperator.DECIDE],
                "pattern_templates": [
                    {
                        "template": {"decide": True, "options": "?opts", "criteria": "?crit"},
                        "match_type": "semantic"
                    }
                ]
            },
            {
                "id": "learning_grammar",
                "name": "Learning Grammar",
                "description": "Grammar for learning and adaptation processes",
                "scheme_expression": "(learn (experience ?exp) (pattern ?pat) (update ?model))",
                "cognitive_operators": [CognitiveOperator.LEARN],
                "pattern_templates": [
                    {
                        "template": {"learn": True, "experience": "?exp", "pattern": "?pat"},
                        "match_type": "semantic"
                    }
                ]
            },
            {
                "id": "composition_grammar",
                "name": "Composition Grammar",
                "description": "Grammar for composing complex cognitive processes",
                "scheme_expression": "(compose (process ?p1) (process ?p2) (result ?r))",
                "cognitive_operators": [CognitiveOperator.COMPOSE],
                "pattern_templates": [
                    {
                        "template": {"compose": True, "process": "?p", "result": "?r"},
                        "match_type": "structural"
                    }
                ]
            },
            {
                "id": "reflection_grammar",
                "name": "Reflection Grammar",
                "description": "Grammar for meta-cognitive reflection",
                "scheme_expression": "(reflect (state ?s) (evaluation ?e) (adjustment ?a))",
                "cognitive_operators": [CognitiveOperator.REFLECT],
                "pattern_templates": [
                    {
                        "template": {"reflect": True, "state": "?s", "evaluation": "?e"},
                        "match_type": "meta"
                    }
                ]
            }
        ]
        
        for grammar_def in builtin_grammars:
            self.register_grammar(
                grammar_def["id"],
                grammar_def["name"],
                grammar_def["description"],
                grammar_def["scheme_expression"],
                grammar_def["cognitive_operators"],
                grammar_def["pattern_templates"]
            )
            
        self.logger.info(f"Registered {len(builtin_grammars)} built-in cognitive grammars")
        
    def register_grammar(self, grammar_id: str, name: str, description: str, 
                        scheme_expression: str, cognitive_operators: List[CognitiveOperator] = None,
                        pattern_templates: List[Dict[str, Any]] = None) -> CognitiveGrammar:
        """Register a new cognitive grammar"""
        try:
            # Parse the Scheme expression
            parsed_tree = self.parser.parse(scheme_expression)
            
            # Create grammar
            grammar = CognitiveGrammar(
                id=grammar_id,
                name=name,
                description=description,
                scheme_expression=scheme_expression,
                parsed_tree=parsed_tree,
                cognitive_operators=cognitive_operators or [],
                pattern_templates=pattern_templates or []
            )
            
            # Store grammar
            self.grammars[grammar_id] = grammar
            
            # Generate patterns from grammar
            patterns = self._generate_patterns_from_grammar(grammar)
            self.grammar_patterns[grammar_id] = patterns
            
            sanitized_grammar_id = grammar_id.replace('\r\n', '').replace('\n', '')
            self.logger.info(f"Registered cognitive grammar: {sanitized_grammar_id}")
            return grammar
            
        except Exception as e:
            sanitized_grammar_id = grammar_id.replace('\r\n', '').replace('\n', '')
            sanitized_grammar_id = grammar_id.replace('\r\n', '').replace('\n', '')
            self.logger.error(f"Failed to register grammar {sanitized_grammar_id}: {e}")
            raise
            
    def _generate_patterns_from_grammar(self, grammar: CognitiveGrammar) -> List[Pattern]:
        """Generate pattern matcher patterns from cognitive grammar"""
        patterns = []
        
        for template in grammar.pattern_templates:
            pattern_id = f"{grammar.id}_pattern_{uuid.uuid4().hex[:8]}"
            
            # Determine match type
            match_type_str = template.get("match_type", "semantic")
            if match_type_str == "semantic":
                match_type = MatchType.SEMANTIC
            elif match_type_str == "structural":
                match_type = MatchType.STRUCTURAL
            elif match_type_str == "logical":
                match_type = MatchType.SEMANTIC  # Use semantic for logical
            elif match_type_str == "exact":
                match_type = MatchType.EXACT
            elif match_type_str == "fuzzy":
                match_type = MatchType.FUZZY
            elif match_type_str == "temporal":
                match_type = MatchType.TEMPORAL
            elif match_type_str == "meta":
                match_type = MatchType.SEMANTIC  # Use semantic for meta
            else:
                match_type = MatchType.SEMANTIC
                
            pattern = Pattern(
                id=pattern_id,
                pattern_type=match_type,
                template=template["template"],
                weights=template.get("weights", {"truth_value": 0.8}),
                metadata={
                    "grammar_id": grammar.id,
                    "cognitive_operators": [op.value for op in grammar.cognitive_operators],
                    "scheme_expression": grammar.scheme_expression
                }
            )
            
            patterns.append(pattern)
            
        return patterns
        
    def get_grammar(self, grammar_id: str) -> Optional[CognitiveGrammar]:
        """Get a cognitive grammar by ID"""
        return self.grammars.get(grammar_id)
        
    def list_grammars(self) -> List[CognitiveGrammar]:
        """List all registered grammars"""
        return list(self.grammars.values())
        
    def get_patterns_for_grammar(self, grammar_id: str) -> List[Pattern]:
        """Get patterns for a specific grammar"""
        return self.grammar_patterns.get(grammar_id, [])
        
    def extend_grammar(self, grammar_id: str, extension_expression: str) -> bool:
        """Extend an existing grammar with new expressions"""
        if grammar_id not in self.grammars:
            return False
            
        try:
            grammar = self.grammars[grammar_id]
            
            # Parse extension
            extension_tree = self.parser.parse(extension_expression)
            
            # Add to grammar's parsed tree as a composition
            if grammar.parsed_tree:
                # Create composition node
                composition_node = SchemeNode(
                    SchemeNodeType.COGNITIVE_OPERATOR,
                    "compose",
                    [grammar.parsed_tree, extension_tree]
                )
                grammar.parsed_tree = composition_node
            else:
                grammar.parsed_tree = extension_tree
                
            # Update scheme expression
            grammar.scheme_expression = f"(compose {grammar.scheme_expression} {extension_expression})"
            
            # Regenerate patterns
            patterns = self._generate_patterns_from_grammar(grammar)
            self.grammar_patterns[grammar_id] = patterns
            
            sanitized_grammar_id = grammar_id.replace('\r\n', '').replace('\n', '')
            self.logger.info(f"Extended grammar {sanitized_grammar_id}")
            return True
            
        except Exception as e:
            sanitized_grammar_id = grammar_id.replace('\r\n', '').replace('\n', '')
            sanitized_grammar_id = grammar_id.replace('\r\n', '').replace('\n', '')
            self.logger.error(f"Failed to extend grammar {sanitized_grammar_id}: {e}")
            return False
            
    def specialize_grammar(self, base_grammar_id: str, specialization_name: str,
                          specialization_expression: str) -> Optional[str]:
        """Create a specialized version of a grammar"""
        base_grammar = self.grammars.get(base_grammar_id)
        if not base_grammar:
            return None
            
        try:
            # Create specialized grammar ID
            specialized_id = f"{base_grammar_id}_{specialization_name}_{uuid.uuid4().hex[:8]}"
            
            # Create specialized expression
            specialized_expression = f"(specialize {base_grammar.scheme_expression} {specialization_expression})"
            
            # Register specialized grammar
            specialized_grammar = self.register_grammar(
                specialized_id,
                f"{base_grammar.name} - {specialization_name}",
                f"Specialized version of {base_grammar.name}: {specialization_name}",
                specialized_expression,
                base_grammar.cognitive_operators.copy(),
                base_grammar.pattern_templates.copy()
            )
            
            return specialized_id
            
        except Exception as e:
            sanitized_base_grammar_id = base_grammar_id.replace('\r\n', '').replace('\n', '')
            sanitized_base_grammar_id = base_grammar_id.replace('\r\n', '').replace('\n', '')
            self.logger.error(f"Failed to specialize grammar {sanitized_base_grammar_id}: {e}")
            return None
            
    def compose_grammars(self, grammar_ids: List[str], composition_name: str) -> Optional[str]:
        """Compose multiple grammars into a new one"""
        if not grammar_ids:
            return None
            
        try:
            # Get all grammars
            grammars = [self.grammars[gid] for gid in grammar_ids if gid in self.grammars]
            if not grammars:
                return None
                
            # Create composition ID
            composition_id = f"composition_{composition_name}_{uuid.uuid4().hex[:8]}"
            
            # Create composition expression
            expressions = [g.scheme_expression for g in grammars]
            composition_expression = f"(compose {' '.join(expressions)})"
            
            # Combine cognitive operators
            all_operators = set()
            for grammar in grammars:
                all_operators.update(grammar.cognitive_operators)
                
            # Combine pattern templates
            all_templates = []
            for grammar in grammars:
                all_templates.extend(grammar.pattern_templates)
                
            # Register composition
            composition_grammar = self.register_grammar(
                composition_id,
                f"Composition: {composition_name}",
                f"Composition of grammars: {', '.join(grammar_ids)}",
                composition_expression,
                list(all_operators),
                all_templates
            )
            
            return composition_id
            
        except Exception as e:
            self.logger.error(f"Failed to compose grammars: {e}")
            return None
            
    def evaluate_grammar_expression(self, grammar_id: str, bindings: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a grammar expression with variable bindings"""
        grammar = self.grammars.get(grammar_id)
        if not grammar or not grammar.parsed_tree:
            return {"error": "Grammar not found or not parsed"}
            
        try:
            result = self._evaluate_scheme_node(grammar.parsed_tree, bindings)
            
            # Update usage count
            grammar.usage_count += 1
            
            return {
                "success": True,
                "result": result,
                "grammar_id": grammar_id,
                "bindings": bindings
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {e}"}
            
    def _evaluate_scheme_node(self, node: SchemeNode, bindings: Dict[str, Any]) -> Any:
        """Evaluate a Scheme node with variable bindings"""
        if node.node_type == SchemeNodeType.SYMBOL:
            # Variable substitution
            if isinstance(node.value, str) and node.value.startswith('?'):
                var_name = node.value[1:]  # Remove '?' prefix
                return bindings.get(var_name, node.value)
            else:
                return node.value
                
        elif node.node_type == SchemeNodeType.ATOM:
            return node.value
            
        elif node.node_type == SchemeNodeType.COGNITIVE_OPERATOR:
            # Evaluate cognitive operator
            operator_name = node.value
            if node.children:
                args = [self._evaluate_scheme_node(child, bindings) for child in node.children]
                return self._apply_cognitive_operator(operator_name, args)
            else:
                return operator_name
                
        elif node.node_type == SchemeNodeType.LIST:
            # Evaluate list elements
            return [self._evaluate_scheme_node(child, bindings) for child in node.children]
            
        else:
            return node.value
            
    def _apply_cognitive_operator(self, operator_name: str, args: List[Any]) -> Any:
        """Apply a cognitive operator with arguments"""
        if operator_name == "perceive":
            return {"operator": "perceive", "args": args, "type": "perception"}
        elif operator_name == "reason":
            return {"operator": "reason", "args": args, "type": "reasoning"}
        elif operator_name == "decide":
            return {"operator": "decide", "args": args, "type": "decision"}
        elif operator_name == "learn":
            return {"operator": "learn", "args": args, "type": "learning"}
        elif operator_name == "compose":
            return {"operator": "compose", "args": args, "type": "composition"}
        elif operator_name == "reflect":
            return {"operator": "reflect", "args": args, "type": "reflection"}
        else:
            return {"operator": operator_name, "args": args, "type": "unknown"}
            
    def get_grammar_statistics(self) -> Dict[str, Any]:
        """Get statistics about registered grammars"""
        total_grammars = len(self.grammars)
        total_patterns = sum(len(patterns) for patterns in self.grammar_patterns.values())
        
        operator_counts = {}
        for grammar in self.grammars.values():
            for op in grammar.cognitive_operators:
                operator_counts[op.value] = operator_counts.get(op.value, 0) + 1
                
        active_grammars = sum(1 for g in self.grammars.values() if g.active)
        
        return {
            "total_grammars": total_grammars,
            "active_grammars": active_grammars,
            "total_patterns": total_patterns,
            "cognitive_operator_usage": operator_counts,
            "average_usage": sum(g.usage_count for g in self.grammars.values()) / max(total_grammars, 1)
        }
        
    def export_grammars(self) -> Dict[str, Any]:
        """Export all grammars to dictionary format"""
        return {
            "grammars": {gid: grammar.to_dict() for gid, grammar in self.grammars.items()},
            "statistics": self.get_grammar_statistics(),
            "exported_at": datetime.now(timezone.utc).isoformat()
        }
        
    def import_grammars(self, grammar_data: Dict[str, Any]) -> bool:
        """Import grammars from dictionary format"""
        try:
            for gid, grammar_dict in grammar_data.get("grammars", {}).items():
                grammar = CognitiveGrammar(
                    id=grammar_dict["id"],
                    name=grammar_dict["name"],
                    description=grammar_dict["description"],
                    scheme_expression=grammar_dict["scheme_expression"],
                    cognitive_operators=[CognitiveOperator(op) for op in grammar_dict["cognitive_operators"]],
                    pattern_templates=grammar_dict["pattern_templates"],
                    active=grammar_dict.get("active", True),
                    usage_count=grammar_dict.get("usage_count", 0)
                )
                
                # Parse the expression
                if grammar.scheme_expression:
                    grammar.parsed_tree = self.parser.parse(grammar.scheme_expression)
                    
                # Store grammar
                self.grammars[gid] = grammar
                
                # Generate patterns
                patterns = self._generate_patterns_from_grammar(grammar)
                self.grammar_patterns[gid] = patterns
                
            self.logger.info(f"Imported {len(grammar_data.get('grammars', {}))} grammars")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import grammars: {e}")
            return False