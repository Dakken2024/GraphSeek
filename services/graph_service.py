"""
Knowledge graph service for GraphSeek application.
Handles graph construction, retrieval, persistence, and advanced query operations.
Enhanced with better entity extraction, graph persistence, and complex query algorithms.
"""
import re
import json
import hashlib
from typing import List, Set, Tuple, Dict, Any, Optional
from pathlib import Path
import networkx as nx
from collections import Counter

from utils.logger import get_logger
from utils.monitoring import Monitor


logger = get_logger(__name__)


class EnhancedEntityExtractor:
    """Enhanced entity extraction with multiple strategies."""
    
    def __init__(self) -> None:
        # Patterns for different entity types
        self.patterns = {
            'capitalized': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'technical_terms': r'\b[A-Za-z]+-[A-Za-z]+|\b[A-Za-z]+\d+\b',
            'acronyms': r'\b[A-Z]{2,}\b',
        }
    
    def extract(self, text: str, min_length: int = 2) -> List[str]:
        """
        Extract entities using multiple strategies.
        
        Args:
            text: Input text
            min_length: Minimum number of words in entity
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract capitalized phrases
        capitalized = re.findall(self.patterns['capitalized'], text)
        entities.extend([e for e in capitalized if len(e.split()) >= min_length or len(e) > 3])
        
        # Extract technical terms
        technical = re.findall(self.patterns['technical_terms'], text)
        entities.extend(technical)
        
        # Filter and deduplicate
        filtered = self._filter_entities(entities)
        return list(dict.fromkeys(filtered))
    
    def _filter_entities(self, entities: List[str]) -> List[str]:
        """Filter out common false positives."""
        false_positives = {
            'The', 'This', 'That', 'These', 'Those',
            'What', 'When', 'Where', 'Which', 'Who', 'Why', 'How',
            'Introduction', 'Conclusion', 'References', 'Abstract',
            'Figure', 'Table', 'Chapter', 'Section',
        }
        
        return [
            entity for entity in entities
            if entity not in false_positives
            and not entity.startswith(('The ', 'A ', 'An '))
        ]


class KnowledgeGraphService:
    """Enhanced service for building and querying knowledge graphs with persistence."""
    
    def __init__(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
    ) -> None:
        self.graph = nx.Graph()
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.auto_save = auto_save
        self.entity_extractor = EnhancedEntityExtractor()
        self.monitor = Monitor()
        
        # Load existing graph if available
        if self.persistence_path and self.persistence_path.exists():
            self.load_graph()
    
    def build_graph(self, documents: List) -> nx.Graph:
        """
        Build a knowledge graph from documents by extracting entities.
        
        Args:
            documents: List of document objects with page_content attribute
            
        Returns:
            NetworkX graph with entities as nodes and relationships as edges
        """
        with self.monitor.measure("graph.build"):
            self.graph = nx.Graph()
            
            for doc in documents:
                entities = self.entity_extractor.extract(doc.page_content)
                self._add_relationships(entities, source=doc.metadata.get('source', 'unknown'))
            
            if self.auto_save:
                self.save_graph()
            
            logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            return self.graph
    
    def _add_relationships(
        self, 
        entities: List[str], 
        source: str = "unknown",
        window_size: int = 3,
    ) -> None:
        """
        Add relationships between entities within a sliding window.
        
        Args:
            entities: List of entity names
            source: Source document identifier
            window_size: Size of sliding window for relationship detection
        """
        if len(entities) < 2:
            return
        
        # Add nodes with metadata
        for entity in entities:
            if not self.graph.has_node(entity):
                self.graph.add_node(
                    entity,
                    first_seen=source,
                    mention_count=1,
                )
            else:
                self.graph.nodes[entity]['mention_count'] += 1
        
        # Add edges with weighted connections based on proximity
        for i in range(len(entities)):
            for j in range(i + 1, min(i + window_size, len(entities))):
                if entities[i] != entities[j]:
                    weight = 1.0 / (j - i)  # Closer entities have stronger weights
                    
                    if self.graph.has_edge(entities[i], entities[j]):
                        self.graph[entities[i]][entities[j]]['weight'] += weight
                        self.graph[entities[i]][entities[j]]['sources'].append(source)
                    else:
                        self.graph.add_edge(
                            entities[i],
                            entities[j],
                            weight=weight,
                            sources=[source],
                        )
    
    def save_graph(self, path: Optional[str] = None) -> bool:
        """
        Save graph to disk.
        
        Args:
            path: Optional custom path (uses default if not provided)
            
        Returns:
            Success status
        """
        try:
            save_path = Path(path) if path else self.persistence_path
            if not save_path:
                return False
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert graph to serializable format
            graph_data = {
                'nodes': [
                    {'id': node, **attrs}
                    for node, attrs in self.graph.nodes(data=True)
                ],
                'edges': [
                    {'source': u, 'target': v, **attrs}
                    for u, v, attrs in self.graph.edges(data=True)
                ],
            }
            
            with open(save_path, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            logger.info(f"Saved graph to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save graph: {str(e)}")
            return False
    
    def load_graph(self, path: Optional[str] = None) -> bool:
        """
        Load graph from disk.
        
        Args:
            path: Optional custom path (uses default if not provided)
            
        Returns:
            Success status
        """
        try:
            load_path = Path(path) if path else self.persistence_path
            if not load_path or not load_path.exists():
                return False
            
            with open(load_path, 'r') as f:
                graph_data = json.load(f)
            
            self.graph = nx.Graph()
            
            # Reconstruct nodes
            for node_data in graph_data['nodes']:
                node_id = node_data.pop('id')
                self.graph.add_node(node_id, **node_data)
            
            # Reconstruct edges
            for edge_data in graph_data['edges']:
                source = edge_data.pop('source')
                target = edge_data.pop('target')
                self.graph.add_edge(source, target, **edge_data)
            
            logger.info(f"Loaded graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            return True
        except Exception as e:
            logger.error(f"Failed to load graph: {str(e)}")
            return False
    
    def query_graph(
        self, 
        query: str, 
        top_k: int = 5,
        use_page_rank: bool = True,
        min_weight: float = 0.5,
    ) -> List[str]:
        """
        Enhanced query with PageRank and weighted traversal.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            use_page_rank: Whether to use PageRank for ranking
            min_weight: Minimum edge weight for traversal
            
        Returns:
            List of related node names ranked by relevance
        """
        with self.monitor.measure("graph.query"):
            query_words = query.lower().split()
            matched_nodes = self._find_matching_nodes(query_words)
            
            if not matched_nodes:
                # Fuzzy matching fallback
                matched_nodes = self._fuzzy_match(query, threshold=0.6)
            
            if not matched_nodes:
                return []
            
            # Rank and expand
            if use_page_rank:
                related_nodes = self._get_related_nodes_pagerank(
                    matched_nodes, 
                    top_k=top_k,
                    min_weight=min_weight,
                )
            else:
                related_nodes = self._get_related_nodes_weighted(
                    matched_nodes,
                    top_k=top_k,
                    min_weight=min_weight,
                )
            
            return related_nodes
    
    def _find_matching_nodes(self, query_words: List[str]) -> List[str]:
        """Find nodes that match query words exactly or partially."""
        matches = []
        for node in self.graph.nodes:
            node_lower = node.lower()
            score = sum(1 for word in query_words if word in node_lower)
            if score > 0:
                matches.append((node, score))
        
        # Sort by match score
        matches.sort(key=lambda x: x[1], reverse=True)
        return [node for node, score in matches if score > 0]
    
    def _fuzzy_match(self, query: str, threshold: float = 0.6) -> List[str]:
        """Fuzzy matching for queries with no exact matches."""
        from difflib import SequenceMatcher
        
        query_lower = query.lower()
        matches = []
        
        for node in self.graph.nodes:
            ratio = SequenceMatcher(None, query_lower, node.lower()).ratio()
            if ratio >= threshold:
                matches.append((node, ratio))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return [node for node, score in matches[:5]]
    
    def _get_related_nodes_pagerank(
        self, 
        seed_nodes: List[str], 
        top_k: int = 5,
        min_weight: float = 0.5,
    ) -> List[str]:
        """Get related nodes using Personalized PageRank."""
        try:
            # Create subgraph with weighted edges
            subgraph = self._create_weighted_subgraph(seed_nodes, min_weight)
            
            if len(subgraph) == 0:
                return []
            
            # Calculate personalized PageRank
            pagerank = nx.pagerank(
                subgraph,
                weight='weight',
                personalization={node: 1.0/len(seed_nodes) for node in seed_nodes if node in subgraph},
                max_iter=100,
            )
            
            # Remove seed nodes from results and sort by PageRank
            results = [
                node for node, score in sorted(
                    pagerank.items(), key=lambda x: x[1], reverse=True
                )
                if node not in seed_nodes
            ]
            
            return results[:top_k]
        except Exception as e:
            logger.warning(f"PageRank failed, falling back to weighted traversal: {str(e)}")
            return self._get_related_nodes_weighted(seed_nodes, top_k, min_weight)
    
    def _get_related_nodes_weighted(
        self,
        seed_nodes: List[str],
        top_k: int = 5,
        min_weight: float = 0.5,
    ) -> List[str]:
        """Get related nodes using weighted BFS."""
        related_scores = Counter()
        
        for seed in seed_nodes:
            if seed not in self.graph:
                continue
            
            # BFS with weighted scoring
            visited = {seed}
            queue = [(seed, 1.0)]  # (node, cumulative_weight)
            
            while queue and len(related_scores) < top_k * 2:
                current, cum_weight = queue.pop(0)
                
                for neighbor in self.graph.neighbors(current):
                    if neighbor in visited:
                        continue
                    
                    edge_weight = self.graph[current][neighbor].get('weight', 1.0)
                    if edge_weight < min_weight:
                        continue
                    
                    visited.add(neighbor)
                    new_weight = cum_weight * edge_weight
                    related_scores[neighbor] += new_weight
                    
                    queue.append((neighbor, new_weight))
        
        # Return top-k by score
        return [node for node, score in related_scores.most_common(top_k)]
    
    def _create_weighted_subgraph(
        self, 
        seed_nodes: List[str], 
        min_weight: float = 0.5,
        max_hops: int = 2,
    ) -> nx.Graph:
        """Create a weighted subgraph around seed nodes."""
        subgraph = nx.Graph()
        
        # Add seed nodes
        for node in seed_nodes:
            if node in self.graph:
                subgraph.add_node(node, **self.graph.nodes[node])
        
        # Add neighbors up to max_hops
        current_layer = set(seed_nodes)
        visited = set(seed_nodes)
        
        for _ in range(max_hops):
            next_layer = set()
            for node in current_layer:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    
                    edge_weight = self.graph[node][neighbor].get('weight', 1.0)
                    if edge_weight >= min_weight:
                        if neighbor not in subgraph:
                            subgraph.add_node(neighbor, **self.graph.nodes[neighbor])
                        subgraph.add_edge(node, neighbor, weight=edge_weight)
                        next_layer.add(neighbor)
                        visited.add(neighbor)
            
            current_layer = next_layer
        
        return subgraph
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics about the knowledge graph."""
        stats = {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "density": nx.density(self.graph) if len(self.graph) > 0 else 0,
            "avg_degree": sum(dict(self.graph.degree()).values()) / len(self.graph) if len(self.graph) > 0 else 0,
            "connected_components": nx.number_connected_components(self.graph) if len(self.graph) > 0 else 0,
        }
        
        # Top nodes by degree
        if len(self.graph) > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            stats["top_nodes_by_centrality"] = [
                {"node": node, "centrality": round(score, 4)}
                for node, score in top_nodes
            ]
        
        # Sample data
        stats["sample_nodes"] = list(self.graph.nodes)[:10]
        stats["sample_edges"] = [
            {"source": u, "target": v, "weight": round(attrs.get('weight', 1.0), 2)}
            for u, v, attrs in list(self.graph.edges(data=True))[:10]
        ]
        
        return stats
    
    def find_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        try:
            return nx.shortest_path(self.graph, source, target, weight='weight')
        except nx.NetworkXNoPath:
            return None
    
    def get_entity_neighbors(self, entity: str, max_depth: int = 1) -> Dict[str, Any]:
        """Get detailed information about an entity and its neighborhood."""
        if entity not in self.graph:
            return {"error": "Entity not found"}
        
        result = {
            "entity": entity,
            "metadata": self.graph.nodes[entity],
            "neighbors": [],
        }
        
        visited = {entity}
        current_layer = [entity]
        
        for depth in range(max_depth):
            next_layer = []
            for node in current_layer:
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    
                    edge_data = self.graph[node][neighbor]
                    neighbor_data = {
                        "name": neighbor,
                        "depth": depth + 1,
                        "edge_weight": edge_data.get('weight', 1.0),
                        "metadata": self.graph.nodes[neighbor],
                    }
                    result["neighbors"].append(neighbor_data)
                    next_layer.append(neighbor)
                    visited.add(neighbor)
            
            current_layer = next_layer
        
        return result
