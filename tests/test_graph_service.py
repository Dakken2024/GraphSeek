"""图谱服务单元测试：增量合并、实体消歧、社区摘要、持久化。"""
import os

import pytest

from services.graph_service import (
    KnowledgeGraphService,
    LLMEntityExtractor,
    Entity,
    Relationship,
    SubGraph,
)


@pytest.fixture
def graph_service():
    return KnowledgeGraphService(auto_save=False)


def _subgraph1():
    return SubGraph(
        entities=[
            Entity(name="OpenAI", type="organization", description="AI 公司"),
            Entity(name="ChatGPT", type="product"),
            Entity(name="GPT-4", type="technology"),
        ],
        relationships=[
            Relationship(source="OpenAI", target="ChatGPT", relation_type="develops", weight=2.0),
            Relationship(source="ChatGPT", target="GPT-4", relation_type="based_on", weight=1.5),
        ],
    )


class TestMergeSubgraph:
    def test_incremental_add(self, graph_service):
        added = graph_service.merge_subgraph(_subgraph1(), source="doc_a")
        assert added == 3
        assert len(graph_service.graph.nodes) == 3
        assert len(graph_service.graph.edges) == 2

    def test_entity_resolution(self, graph_service):
        graph_service.merge_subgraph(_subgraph1(), source="doc_a")
        # 别名 "open ai" / 大小写差异应消歧合并到 "OpenAI"
        sg2 = SubGraph(
            entities=[
                Entity(name="open ai", type="organization"),
                Entity(name="DALL-E", type="product"),
            ],
            relationships=[
                Relationship(source="open ai", target="DALL-E", relation_type="develops", weight=1.0),
            ],
        )
        added = graph_service.merge_subgraph(sg2, source="doc_b")
        assert added == 1  # 仅 DALL-E 新增
        assert "OpenAI" in graph_service.graph.nodes
        assert "open ai" not in graph_service.graph.nodes
        # OpenAI - DALL-E 边已建立
        assert graph_service.graph.has_edge("OpenAI", "DALL-E")

    def test_edge_weight_accumulate(self, graph_service):
        graph_service.merge_subgraph(_subgraph1(), source="doc_a")
        graph_service.merge_subgraph(_subgraph1(), source="doc_c")
        w = graph_service.graph["OpenAI"]["ChatGPT"]["weight"]
        assert w == pytest.approx(4.0)  # 2.0 + 2.0
        assert len(graph_service.graph["OpenAI"]["ChatGPT"]["sources"]) == 2


class TestLLMExtractorFallback:
    def test_regex_fallback(self):
        extractor = LLMEntityExtractor(llm_service=None)
        subgraph = extractor.extract("Stanford and Google released AI models in 2023.")
        assert len(subgraph.entities) >= 2
        assert any(e.name == "Stanford" for e in subgraph.entities)
        assert any(e.name == "Google" for e in subgraph.entities)

    def test_normalize(self):
        assert LLMEntityExtractor._normalize("  OpenAI ") == "openai"
        assert LLMEntityExtractor._normalize("Open-AI") == "openai"
        assert LLMEntityExtractor._normalize("open ai") == "openai"


class TestCommunities:
    def test_detect_and_summarize(self, graph_service):
        graph_service.merge_subgraph(_subgraph1(), source="a")
        communities = graph_service.detect_communities()
        assert communities, "图非空时应检测出社区"

        summaries = graph_service.build_community_summaries(llm_service=None)
        assert len(summaries) >= 1
        first = list(summaries.values())[0]
        assert "summary" in first
        assert "key_entities" in first

    def test_query_community(self, graph_service):
        graph_service.merge_subgraph(_subgraph1(), source="a")
        graph_service.build_community_summaries(llm_service=None)
        results = graph_service.query_community("OpenAI GPT")
        assert isinstance(results, list)
        # 至少能命中含 OpenAI 的社区
        assert any("OpenAI" in r["key_entities"] or "OpenAI" in r["summary"] for r in results)


class TestPersistence:
    def test_save_load_roundtrip(self, graph_service, tmp_path):
        graph_service.merge_subgraph(_subgraph1(), source="a")
        graph_service.build_community_summaries(llm_service=None)
        path = tmp_path / "graph.json"
        graph_service.save_graph(str(path))

        svc2 = KnowledgeGraphService(persistence_path=str(path))
        assert len(svc2.graph.nodes) == 3
        assert svc2.community_summaries, "社区摘要应持久化"
        # 消歧索引重建
        assert svc2._node_index["openai"] == "OpenAI"


class TestQueryGraph:
    def test_pagerank_query(self, graph_service):
        graph_service.merge_subgraph(_subgraph1(), source="a")
        results = graph_service.query_graph("OpenAI")
        assert "GPT-4" in results or "ChatGPT" in results

    def test_shortest_path(self, graph_service):
        graph_service.merge_subgraph(_subgraph1(), source="a")
        path = graph_service.find_shortest_path("OpenAI", "GPT-4")
        assert path == ["OpenAI", "ChatGPT", "GPT-4"]
