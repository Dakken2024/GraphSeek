"""
Base tool definitions for Agent framework.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import time


@dataclass
class ToolResult:
    """Result of a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        return self.success


@dataclass
class ToolDefinition:
    """Definition of a tool's interface."""
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: str = "Any"


class Tool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def get_definition(self) -> ToolDefinition:
        """Get the tool's definition for LLM prompting."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self._get_parameters(),
            returns=self._get_return_type(),
        )
    
    def _get_parameters(self) -> Dict[str, Any]:
        """Get parameter schema (override in subclasses)."""
        return {}
    
    def _get_return_type(self) -> str:
        """Get return type description (override in subclasses)."""
        return "Any"


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_all_definitions(self) -> List[ToolDefinition]:
        """Get definitions for all registered tools."""
        return [tool.get_definition() for tool in self._tools.values()]
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with parameters."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
            )
        
        start_time = time.time()
        try:
            result = tool.execute(**kwargs)
            result.execution_time = time.time() - start_time
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()
