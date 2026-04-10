"""
Retrieval service for GraphSeek application.
Handles document retrieval with hybrid search, reranking, and GraphRAG.
"""
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document

from core.cache import RetrievalCache
from utils.monitoring import Monitor
from utils.logger import get_logger


logger = get_logger(__name__)


class RetrievalService:
    """Service for retrieving documents using multiple strategies."""
    
    def __init__(
        self,
        ensemble_retriever,
        reranker=None,
        knowledge_graph=None,
        graph_service=None,
        cache_enabled: bool = True,
        cache_ttl: float = 3600.0,
    ) -> None:
        self.ensemble_retriever = ensemble_retriever
        self.reranker = reranker
        self.knowledge_graph = knowledge_graph
        self.graph_service = graph_service
        
        # Initialize cache
        self.cache_enabled = cache_enabled
        self.cache = RetrievalCache(max_size=1000, default_ttl=cache_ttl) if cache_enabled else None
        
        # Initialize monitoring
        self.monitor = Monitor()
    
    def retrieve(
        self,
        query: str,
        chat_history: str = "",
        enable_hyde: bool = True,
        enable_graph_rag: bool = True,
        enable_reranking: bool = True,
        max_contexts: int = 3,
        llm_service=None,
        use_cache: bool = True,
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search and optional enhancements.
        
        Args:
            query: User's query string
            chat_history: Previous conversation history
            enable_hyde: Whether to use HyDE query expansion
            enable_graph_rag: Whether to include GraphRAG results
            enable_reranking: Whether to apply neural reranking
            max_contexts: Maximum number of documents to return
            llm_service: LLM service for HyDE expansion
            use_cache: Whether to use cached results
            
        Returns:
            List of retrieved documents
        """
        # Check cache first
        cache_key_params = {
            "chat_history": chat_history,
            "enable_hyde": enable_hyde,
            "enable_graph_rag": enable_graph_rag,
            "enable_reranking": enable_reranking,
            "max_contexts": max_contexts,
        }
        
        if use_cache and self.cache_enabled and self.cache:
            cached_result = self.cache.get(query, **cache_key_params)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
        
        with self.monitor.measure("retrieval.retrieve"):
            expanded_query = self._expand_query(
                query,
                chat_history,
                enable_hyde,
                llm_service,
            )
            
            docs = self.ensemble_retriever.invoke(expanded_query)
            
            if enable_graph_rag and self.graph_service:
                graph_docs = self._retrieve_from_graph(query)
                if graph_docs:
                    docs = graph_docs + docs
            
            if enable_reranking and self.reranker:
                docs = self._rerank_documents(query, docs)
            
            result_docs = docs[:max_contexts]
            
            # Cache the result
            if use_cache and self.cache_enabled and self.cache:
                self.cache.set(
                    query,
                    result_docs,
                    metadata={"doc_count": len(result_docs)},
                    **cache_key_params
                )
            
            return result_docs
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return None
    
    def clear_cache(self) -> bool:
        """Clear the retrieval cache."""
        if self.cache:
            self.cache.clear()
            return True
        return False
    
    def _expand_query(
        self,
        query: str,
        chat_history: str,
        enable_hyde: bool,
        llm_service,
    ) -> str:
        """
        Expand query using HyDE if enabled.
        
        Args:
            query: Original query
            chat_history: Conversation history
            enable_hyde: Whether to use HyDE
            llm_service: LLM service for generation
            
        Returns:
            Expanded or original query
        """
        if not enable_hyde or not llm_service:
            return f"{chat_history}\n{query}" if chat_history else query
        
        combined_query = f"{chat_history}\n{query}" if chat_history else query
        hypothetical = llm_service.generate_hypothetical_answer(combined_query)
        return f"{combined_query}\n{hypothetical}"
    
    def _retrieve_from_graph(self, query: str) -> List[Document]:
        """
        Retrieve documents from knowledge graph.
        
        Args:
            query: Search query
            
        Returns:
            List of documents from graph retrieval
        """
        if not self.graph_service:
            return []
        
        related_nodes = self.graph_service.query_graph(query)
        return [Document(page_content=node) for node in related_nodes]
    
    def _rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        Rerank documents using cross-encoder.
        
        Args:
            query: Original query
            docs: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        ranked_docs = [
            doc for _, doc in sorted(zip(scores, docs), reverse=True)
        ]
        return ranked_docs
