"""
Knowledge graph service for GraphSeek application.
Handles graph construction and retrieval operations.
"""
import re
from typing import List, Set, Tuple
import networkx as nx


class KnowledgeGraphService:
    """Service for building and querying knowledge graphs."""
    
    def __init__(self) -> None:
        self.graph = nx.Graph()
    
    def build_graph(self, documents: List) -> nx.Graph:
        """
        Build a knowledge graph from documents by extracting entities.
        
        Args:
            documents: List of document objects with page_content attribute
            
        Returns:
            NetworkX graph with entities as nodes and relationships as edges
        """
        self.graph = nx.Graph()
        
        for doc in documents:
            entities = self._extract_entities(doc.page_content)
            self._add_relationships(entities)
        
        return self.graph
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text using regex pattern.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entity names
        """
        pattern = r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b'
        return re.findall(pattern, text)
    
    def _add_relationships(self, entities: List[str]) -> None:
        """
        Add relationships between consecutive entities in the list.
        
        Args:
            entities: List of entity names
        """
        if len(entities) > 1:
            for i in range(len(entities) - 1):
                self.graph.add_edge(entities[i], entities[i + 1])
    
    def query_graph(self, query: str, top_k: int = 5) -> List[str]:
        """
        Query the knowledge graph for related nodes.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of related node names
        """
        query_words = query.lower().split()
        matched_nodes = self._find_matching_nodes(query_words)
        
        if not matched_nodes:
            return []
        
        related_nodes = self._get_related_nodes(matched_nodes)
        return related_nodes[:top_k]
    
    def _find_matching_nodes(self, query_words: List[str]) -> List[str]:
        """
        Find nodes that match query words.
        
        Args:
            query_words: List of words from the query
            
        Returns:
            List of matching node names
        """
        return [
            node for node in self.graph.nodes 
            if any(word in node.lower() for word in query_words)
        ]
    
    def _get_related_nodes(self, nodes: List[str]) -> List[str]:
        """
        Get neighbors of the given nodes.
        
        Args:
            nodes: List of node names
            
        Returns:
            List of related node names (neighbors)
        """
        related = []
        for node in nodes:
            related.extend(list(self.graph.neighbors(node)))
        return related
    
    def get_stats(self) -> dict:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with graph statistics
        """
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "sample_nodes": list(self.graph.nodes)[:10],
            "sample_edges": list(self.graph.edges)[:10],
        }
