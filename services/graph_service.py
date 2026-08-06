"""
Knowledge graph service for GraphSeek application.
Handles graph construction, retrieval, persistence, and advanced query operations.
Enhanced with LLM-driven NER/RE extraction, incremental merge (Merger Agent),
and community summary dual-layer index (LightRAG-style High-level index).
"""
import re
import json
import hashlib
from typing import List, Set, Tuple, Dict, Any, Optional
from pathlib import Path
import networkx as nx
from collections import Counter
from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.monitoring import Monitor


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LLM 结构化抽取 Schema（白皮书附录 A：Pydantic 约束 JSON 输出）
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """图谱实体。"""
    name: str = Field(description="实体名称（标准名，如人名/机构/技术/地点）")
    type: str = Field(description="实体类型，如 person/organization/technology/product/location/concept")
    description: str = Field(default="", description="一句话描述")


class Relationship(BaseModel):
    """实体关系。"""
    source: str = Field(description="源实体名称（必须与 entities 中的 name 一致）")
    target: str = Field(description="目标实体名称（必须与 entities 中的 name 一致）")
    relation_type: str = Field(description="关系类型，如 founded_by/part_of/uses/located_in/related_to")
    weight: float = Field(default=1.0, ge=0.1, le=5.0, description="关系强度")


class SubGraph(BaseModel):
    """一次抽取得到的局部子图。"""
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)


class CommunitySummary(BaseModel):
    """社区摘要（High-level 索引节点）。"""
    community_id: str = Field(description="社区标识（如 c0）")
    summary: str = Field(description="社区主题摘要（3-5 句）")
    key_entities: List[str] = Field(default_factory=list, description="社区内核心实体")
    member_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# 实体提取器
# ---------------------------------------------------------------------------

class EnhancedEntityExtractor:
    """正则实体提取器（无 LLM 时的降级方案）。"""

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


class LLMEntityExtractor:
    """
    LLM 驱动实体/关系抽取器（白皮书：Agentic Graph Construction 的 Extractor Agent）。

    使用多后端网关的 Pydantic 结构化输出能力，强制 JSON Schema 输出；
    当 LLM 不可用或输出校验失败时，自动降级为 EnhancedEntityExtractor。
    """

    def __init__(self, llm_service=None, max_chunk_chars: int = 3000) -> None:
        self.llm_service = llm_service
        self.max_chunk_chars = max_chunk_chars
        self.fallback = EnhancedEntityExtractor()

    def extract(self, text: str) -> SubGraph:
        """
        从文本中抽取实体与关系。

        Args:
            text: 文档片段

        Returns:
            SubGraph（LLM 失败时返回基于正则的近似子图）
        """
        with Monitor().measure("graph.llm_extract"):
            if self.llm_service is None:
                return self._regex_fallback(text)

            prompt = (
                "你是知识图谱构建专家。请从文本片段中抽取核心实体（人名/机构/技术/产品/地点/概念）"
                "以及它们之间的显式关系。\n"
                "要求：\n"
                "1. 实体名称使用标准名，去除多余修饰；\n"
                "2. 关系必须连接已列出的实体名称（严格一致）；\n"
                "3. 忽略泛化背景描述，只抽取明确提到的信息；\n"
                "4. 中文文本优先使用中文实体名。\n\n"
                f"文本片段（前 {self.max_chunk_chars} 字符）：\n{text[:self.max_chunk_chars]}"
            )
            try:
                return self.llm_service.generate_structured(
                    prompt=prompt,
                    schema=SubGraph,
                    temperature=0.1,
                    max_tokens=4096,
                    max_attempts=2,
                )
            except Exception as e:
                logger.warning(f"LLM 抽取失败，降级为正则抽取: {e}")
                return self._regex_fallback(text)

    def _regex_fallback(self, text: str) -> SubGraph:
        entities = self.fallback.extract(text)
        subgraph = SubGraph(
            entities=[Entity(name=e, type="concept", description="") for e in entities],
            relationships=[],
        )
        # 滑动窗口补关系
        for i in range(len(entities)):
            for j in range(i + 1, min(i + 3, len(entities))):
                if entities[i] != entities[j]:
                    subgraph.relationships.append(
                        Relationship(source=entities[i], target=entities[j],
                                     relation_type="related_to", weight=1.0 / (j - i))
                    )
        return subgraph

    @staticmethod
    def _normalize(name: str) -> str:
        """实体名归一化（用于消歧比较）：小写并去除空白/下划线/连字符。"""
        return re.sub(r"[\s_\-]+", "", name.strip()).lower()


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
        self.community_summaries: Dict[str, Dict[str, Any]] = {}  # community_id -> summary dict
        self._node_index: Dict[str, str] = {}  # normalized name -> canonical node
        self.monitor = Monitor()
        
        # Load existing graph if available
        if self.persistence_path and self.persistence_path.exists():
            self.load_graph()
    
    def build_graph(self, documents: List, llm_extractor: Optional[LLMEntityExtractor] = None) -> nx.Graph:
        """
        Build a knowledge graph from documents by extracting entities.

        Args:
            documents: List of document objects with page_content attribute
            llm_extractor: LLM 抽取器（None 时使用正则，兼容旧行为）

        Returns:
            NetworkX graph with entities as nodes and relationships as edges
        """
        with self.monitor.measure("graph.build"):
            self.graph = nx.Graph()
            self._node_index = {}
            self.community_summaries = {}

            if llm_extractor is not None:
                # LLM 驱动抽取 + 增量合并（白皮书：Extractor + Merger）
                self.add_documents_incremental(documents, llm_extractor)
            else:
                for doc in documents:
                    entities = self.entity_extractor.extract(doc.page_content)
                    self._add_relationships(entities, source=doc.metadata.get('source', 'unknown'))
                self._rebuild_node_index()
            
            if self.auto_save:
                self.save_graph()
            
            logger.info(f"Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
            return self.graph

    def _rebuild_node_index(self) -> None:
        """从现有节点重建归一化索引。"""
        self._node_index = {
            LLMEntityExtractor._normalize(node): node for node in self.graph.nodes
        }
    
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
    
    # ------------------------------------------------------------------
    # 增量建图（Merger Agent）：merge_subgraph / add_documents_incremental
    # ------------------------------------------------------------------

    def merge_subgraph(self, subgraph: SubGraph, source: str = "unknown") -> int:
        """
        将 LLM 抽取的局部子图合并进全局图谱（白皮书附录 A：Merger Agent）。

        实体消歧：归一化名称匹配（大小写/空白不敏感），已存在实体复用节点并合并属性；
        边合并：同源同目标关系权重累加、sources 追加。

        Args:
            subgraph: 待合并的子图（Entity/Relationship）
            source: 来源文档标识

        Returns:
            新增节点数
        """
        with self.monitor.measure("graph.merge_subgraph"):
            added_nodes = 0
            # 实体归一化 -> 规范名映射
            canonical: Dict[str, str] = {}
            for entity in subgraph.entities:
                norm = LLMEntityExtractor._normalize(entity.name)
                existing = self._node_index.get(norm)
                if existing is not None and self.graph.has_node(existing):
                    canonical[norm] = existing
                    # 合并属性
                    node_attrs = self.graph.nodes[existing]
                    node_attrs['mention_count'] = node_attrs.get('mention_count', 0) + 1
                    if entity.description and not node_attrs.get('description'):
                        node_attrs['description'] = entity.description
                    if entity.type and entity.type != 'concept':
                        node_attrs['type'] = entity.type
                    if source not in node_attrs.get('sources', []):
                        node_attrs.setdefault('sources', []).append(source)
                else:
                    canonical[norm] = entity.name
                    self.graph.add_node(
                        entity.name,
                        type=entity.type,
                        description=entity.description,
                        first_seen=source,
                        mention_count=1,
                        sources=[source],
                    )
                    self._node_index[norm] = entity.name
                    added_nodes += 1

            # 合并边
            for rel in subgraph.relationships:
                src = canonical.get(LLMEntityExtractor._normalize(rel.source), rel.source)
                tgt = canonical.get(LLMEntityExtractor._normalize(rel.target), rel.target)
                if src == tgt or src not in self.graph or tgt not in self.graph:
                    continue
                if self.graph.has_edge(src, tgt):
                    edge = self.graph[src][tgt]
                    edge['weight'] = edge.get('weight', 1.0) + rel.weight
                    if source not in edge.get('sources', []):
                        edge.setdefault('sources', []).append(source)
                    if rel.relation_type != 'related_to':
                        edge.setdefault('relation_types', []).append(rel.relation_type)
                else:
                    self.graph.add_edge(
                        src, tgt,
                        weight=rel.weight,
                        sources=[source],
                        relation_types=[rel.relation_type],
                    )

            if self.auto_save:
                self.save_graph()
            return added_nodes

    def add_documents_incremental(
        self,
        documents: List,
        llm_extractor: Optional[LLMEntityExtractor] = None,
    ) -> int:
        """
        增量处理文档：按文档逐个抽取子图并 merge，无需全局重建（白皮书：图随需而建）。

        Args:
            documents: 文档对象列表（page_content / metadata.source）
            llm_extractor: LLM 抽取器（None 时使用正则）

        Returns:
            累计新增节点数
        """
        extractor = llm_extractor or LLMEntityExtractor(llm_service=None)
        added = 0
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            subgraph = extractor.extract(doc.page_content)
            added += self.merge_subgraph(subgraph, source=source)
        logger.info(
            f"Incremental build: {len(documents)} docs, {added} new nodes, "
            f"total {len(self.graph.nodes)} nodes / {len(self.graph.edges)} edges"
        )
        return added

    # ------------------------------------------------------------------
    # 社区检测与摘要（LightRAG High-level 索引）
    # ------------------------------------------------------------------

    def detect_communities(self, resolution: float = 1.0) -> Dict[str, List[str]]:
        """
        使用 Louvain 算法（python-louvain）检测图社区。

        Args:
            resolution: 分辨率参数（越大社区越细）

        Returns:
            community_id -> [node, ...] 映射
        """
        with self.monitor.measure("graph.detect_communities"):
            if len(self.graph) == 0:
                return {}
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(
                    self.graph, weight='weight', resolution=resolution
                )
            except ImportError:
                logger.warning("python-louvain 未安装，降级为连通分量作为社区")
                partition = {}
                for i, comp in enumerate(nx.connected_components(self.graph)):
                    for node in comp:
                        partition[node] = i
            communities: Dict[str, List[str]] = {}
            for node, cid in partition.items():
                communities.setdefault(f"c{cid}", []).append(node)
            return communities

    def build_community_summaries(
        self,
        llm_service=None,
        resolution: float = 1.0,
        top_entities: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        """
        生成社区摘要（High-level 摘要图，白皮书附录 B）。

        Args:
            llm_service: LLM 服务（None 时使用规则摘要）
            resolution: 社区检测分辨率
            top_entities: 每个社区保留的核心实体数

        Returns:
            community_id -> {summary, key_entities, member_count, member_entities}
        """
        communities = self.detect_communities(resolution)
        self.community_summaries = {}

        for cid, members in communities.items():
            # 社区内核心实体：按度中心性排序
            centrality = nx.degree_centrality(self.graph)
            key_entities = sorted(
                members, key=lambda n: centrality.get(n, 0), reverse=True
            )[:top_entities]

            if llm_service is not None:
                try:
                    summary = self._llm_community_summary(llm_service, cid, key_entities, members)
                except Exception as e:
                    logger.warning(f"社区摘要生成失败，使用规则摘要: {e}")
                    summary = self._rule_community_summary(key_entities)
            else:
                summary = self._rule_community_summary(key_entities)

            self.community_summaries[cid] = {
                "summary": summary,
                "key_entities": key_entities,
                "member_count": len(members),
                "member_entities": members,
            }

        if self.auto_save:
            self.save_graph()
        return self.community_summaries

    def _llm_community_summary(self, llm_service, cid: str, key_entities: List[str], members: List[str]) -> str:
        prompt = (
            "你是知识图谱分析师。以下是一个知识图谱社区的成员实体列表，"
            "请用 3-5 句话总结该社区的主题与实体之间的核心关系。\n"
            f"社区 ID: {cid}\n"
            f"核心实体: {', '.join(key_entities)}\n"
            f"全部成员({len(members)}): {', '.join(members[:40])}"
        )
        return llm_service.generate_non_streaming(
            prompt, temperature=0.2, max_context=1024
        ).strip()

    def _rule_community_summary(self, key_entities: List[str]) -> str:
        return (
            "该社区围绕以下核心实体聚集: " + "、".join(key_entities) + "。"
            "（规则摘要，配置 LLM 后自动升级为语义摘要）"
        )

    def query_community(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        宏观查询：检索 High-level 社区摘要（白皮书附录 B 分层路由）。

        Args:
            query: 宏观问题（如"总结投资趋势"）
            top_k: 返回社区数

        Returns:
            匹配社区摘要列表 [{community_id, summary, key_entities}]
        """
        with self.monitor.measure("graph.query_community"):
            if not self.community_summaries:
                return []
            query_words = set(query.lower().split())
            scored: List[Tuple[float, str]] = []
            for cid, info in self.community_summaries.items():
                haystack = (info.get("summary", "") + " " + " ".join(info.get("key_entities", []))).lower()
                score = sum(1 for w in query_words if w in haystack)
                if score > 0:
                    scored.append((score, cid))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for _, cid in scored[:top_k]:
                info = self.community_summaries[cid]
                results.append({
                    "community_id": cid,
                    "summary": info["summary"],
                    "key_entities": info["key_entities"],
                    "member_count": info["member_count"],
                })
            return results
    
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
                'community_summaries': self.community_summaries,
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
                self._node_index[LLMEntityExtractor._normalize(node_id)] = node_id
            
            # Reconstruct edges
            for edge_data in graph_data['edges']:
                source = edge_data.pop('source')
                target = edge_data.pop('target')
                self.graph.add_edge(source, target, **edge_data)
            
            self.community_summaries = graph_data.get('community_summaries', {})
            
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
