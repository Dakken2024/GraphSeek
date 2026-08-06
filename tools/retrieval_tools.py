"""
Retrieval-related tools for Agent framework.
"""
from typing import Dict, List, Any, Optional
from langchain_core.documents import Document

from .base import Tool, ToolResult


class DocumentSearchTool(Tool):
    """Tool for searching documents using ensemble retrieval."""
    
    def __init__(self, ensemble_retriever, max_results: int = 5) -> None:
        super().__init__(
            name="document_search",
            description="Search documents using hybrid retrieval (BM25 + embeddings)",
        )
        self.ensemble_retriever = ensemble_retriever
        self.max_results = max_results
    
    def execute(self, query: str, k: Optional[int] = None) -> ToolResult:
        """Execute document search."""
        try:
            if not self.ensemble_retriever:
                return ToolResult(
                    success=False,
                    error="Ensemble retriever not available",
                )
            
            top_k = k or self.max_results
            # langchain 0.3 不支持 invoke 传 k，通过 search_kwargs 控制
            self.ensemble_retriever.search_kwargs = {"k": top_k}
            docs = self.ensemble_retriever.invoke(query)
            
            results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in docs
            ]
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "documents": results,
                    "count": len(results),
                },
                metadata={"k": top_k},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Document search failed: {str(e)}",
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "Search query"},
            "k": {"type": "integer", "description": "Number of results to return", "default": self.max_results},
        }
    
    def _get_return_type(self) -> str:
        return "List of document contents with metadata"


class HybridSearchTool(Tool):
    """Tool for advanced hybrid search with optional reranking."""
    
    def __init__(
        self, 
        ensemble_retriever, 
        reranker=None,
        graph_service=None,
    ) -> None:
        super().__init__(
            name="hybrid_search",
            description="Advanced hybrid search with reranking and GraphRAG integration",
        )
        self.ensemble_retriever = ensemble_retriever
        self.reranker = reranker
        self.graph_service = graph_service
    
    def execute(
        self, 
        query: str, 
        use_reranking: bool = True,
        use_graph: bool = True,
        top_k: int = 5,
    ) -> ToolResult:
        """Execute hybrid search with enhancements."""
        try:
            # Base retrieval（langchain 0.3 通过 search_kwargs 控制 top_k）
            self.ensemble_retriever.search_kwargs = {"k": top_k * 2}
            docs = self.ensemble_retriever.invoke(query)
            
            # GraphRAG enhancement
            graph_docs = []
            if use_graph and self.graph_service:
                related_nodes = self.graph_service.query_graph(query, top_k=top_k)
                graph_docs = [Document(page_content=node) for node in related_nodes]
                docs = graph_docs + docs
            
            # Reranking
            if use_reranking and self.reranker and docs:
                pairs = [[query, doc.page_content] for doc in docs]
                scores = self.reranker.predict(pairs)
                docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
            
            # Limit results
            docs = docs[:top_k]
            
            results = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source_type": "graph" if doc in graph_docs else "vector",
                }
                for i, doc in enumerate(docs)
            ]
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "documents": results,
                    "count": len(results),
                    "used_reranking": use_reranking and self.reranker is not None,
                    "used_graph": use_graph and self.graph_service is not None,
                },
                metadata={"top_k": top_k},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Hybrid search failed: {str(e)}",
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "Search query"},
            "use_reranking": {"type": "boolean", "description": "Whether to apply neural reranking", "default": True},
            "use_graph": {"type": "boolean", "description": "Whether to include GraphRAG results", "default": True},
            "top_k": {"type": "integer", "description": "Maximum number of results", "default": 5},
        }
    
    def _get_return_type(self) -> str:
        return "List of ranked documents with source type indicators"
