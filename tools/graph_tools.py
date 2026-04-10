"""
Graph-related tools for Agent framework.
"""
from typing import Dict, List, Optional, Any
import json

from .base import Tool, ToolResult


class GraphQueryTool(Tool):
    """Tool for querying the knowledge graph."""
    
    def __init__(self, graph_service) -> None:
        super().__init__(
            name="graph_query",
            description="Query the knowledge graph to find related entities and concepts",
        )
        self.graph_service = graph_service
    
    def execute(self, query: str, top_k: int = 5) -> ToolResult:
        """Execute graph query."""
        try:
            if not self.graph_service:
                return ToolResult(
                    success=False,
                    error="Graph service not available",
                )
            
            results = self.graph_service.query_graph(query, top_k=top_k)
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "related_nodes": results,
                    "count": len(results),
                },
                metadata={"top_k": top_k},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Graph query failed: {str(e)}",
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "Search query to find related entities"},
            "top_k": {"type": "integer", "description": "Maximum number of results", "default": 5},
        }
    
    def _get_return_type(self) -> str:
        return "List of related entity names"


class GraphStatsTool(Tool):
    """Tool for getting knowledge graph statistics."""
    
    def __init__(self, graph_service) -> None:
        super().__init__(
            name="graph_stats",
            description="Get statistics about the knowledge graph structure",
        )
        self.graph_service = graph_service
    
    def execute(self) -> ToolResult:
        """Get graph statistics."""
        try:
            if not self.graph_service:
                return ToolResult(
                    success=False,
                    error="Graph service not available",
                )
            
            stats = self.graph_service.get_stats()
            
            return ToolResult(
                success=True,
                data=stats,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to get graph stats: {str(e)}",
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {}
    
    def _get_return_type(self) -> str:
        return "Dictionary with graph statistics (nodes, edges, samples)"


class EntitySearchTool(Tool):
    """Tool for searching specific entities in the graph."""
    
    def __init__(self, graph_service) -> None:
        super().__init__(
            name="entity_search",
            description="Search for specific entities in the knowledge graph and get their neighbors",
        )
        self.graph_service = graph_service
    
    def execute(self, entity_name: str, include_neighbors: bool = True) -> ToolResult:
        """Search for an entity in the graph."""
        try:
            if not self.graph_service:
                return ToolResult(
                    success=False,
                    error="Graph service not available",
                )
            
            graph = self.graph_service.graph
            matching_nodes = [
                node for node in graph.nodes 
                if entity_name.lower() in node.lower()
            ]
            
            result_data = {
                "search_term": entity_name,
                "matching_nodes": matching_nodes,
                "count": len(matching_nodes),
            }
            
            if include_neighbors and matching_nodes:
                neighbors = {}
                for node in matching_nodes[:10]:  # Limit to first 10
                    neighbors[node] = list(graph.neighbors(node))
                result_data["neighbors"] = neighbors
            
            return ToolResult(
                success=True,
                data=result_data,
                metadata={"include_neighbors": include_neighbors},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Entity search failed: {str(e)}",
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "entity_name": {"type": "string", "description": "Name or partial name of entity to search"},
            "include_neighbors": {"type": "boolean", "description": "Whether to include neighboring nodes", "default": True},
        }
    
    def _get_return_type(self) -> str:
        return "Dictionary with matching entities and optionally their neighbors"
