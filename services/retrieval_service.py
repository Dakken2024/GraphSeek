"""
Retrieval service for GraphSeek application.
Handles document retrieval with hybrid search, reranking, and GraphRAG.
"""
from typing import List, Optional
from langchain_core.documents import Document


class RetrievalService:
    """Service for retrieving documents using multiple strategies."""
    
    def __init__(
        self,
        ensemble_retriever,
        reranker=None,
        knowledge_graph=None,
        graph_service=None,
    ) -> None:
        self.ensemble_retriever = ensemble_retriever
        self.reranker = reranker
        self.knowledge_graph = knowledge_graph
        self.graph_service = graph_service
    
    def retrieve(
        self,
        query: str,
        chat_history: str = "",
        enable_hyde: bool = True,
        enable_graph_rag: bool = True,
        enable_reranking: bool = True,
        max_contexts: int = 3,
        llm_service=None,
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
            
        Returns:
            List of retrieved documents
        """
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
        
        return docs[:max_contexts]
    
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
