"""检索链路单元测试：RRF、多目标重排、查询规划、RetrievalService 降级。"""
from langchain_core.documents import Document

from services.retrieval_service import rrf_fusion, RetrievalService
from services.reranker import MultiObjectiveReranker
from services.query_planner import QueryPlanner
from services.llm_gateway import MockBackend, LLMGateway
from services.graph_service import (
    KnowledgeGraphService, Entity, Relationship, SubGraph,
)
from services.colbert_retriever import ColbertRetriever


class FakeEnsemble:
    """基于关键词的简单检索器（模拟 ensemble）。"""

    def __init__(self, docs):
        self.docs = docs
        self.search_kwargs = {"k": 3}

    def invoke(self, query):
        k = self.search_kwargs.get("k", 3)
        qw = set(query.lower().split())
        scored = []
        for d in self.docs:
            dw = set(d.page_content.lower().split())
            scored.append((len(qw & dw), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]


class FakeLLM:
    """Mock LLM 服务的测试替身。"""

    def __init__(self, gateway):
        self.gateway = gateway

    def generate_structured(self, prompt, schema, temperature=0.1, max_tokens=1024, max_attempts=2):
        return self.gateway.generate_structured(prompt, schema, temperature, max_tokens, max_attempts)

    def generate_hypothetical_answer(self, query):
        return "[hypothetical] " + query


class TestRRFFusion:
    def test_rank_boost(self):
        d1 = Document(page_content="doc1", metadata={"source": "s1"})
        d2 = Document(page_content="doc2", metadata={"source": "s2"})
        d3 = Document(page_content="doc3", metadata={"source": "s3"})
        fused = rrf_fusion([[d1, d2], [d3, d1]])
        assert fused[0].page_content == "doc1"

    def test_dedupe(self):
        d1 = Document(page_content="same", metadata={"source": "a"})
        d2 = Document(page_content="same", metadata={"source": "b"})
        fused = rrf_fusion([[d1, d2]])
        assert len(fused) == 1

    def test_empty(self):
        assert rrf_fusion([[], []]) == []


class TestMultiObjectiveReranker:
    def setup_method(self):
        self.graph = KnowledgeGraphService(auto_save=False)
        self.graph.merge_subgraph(SubGraph(
            entities=[
                Entity(name="OpenAI", type="org"),
                Entity(name="GPT-4", type="tech"),
            ],
            relationships=[
                Relationship(source="OpenAI", target="GPT-4", relation_type="develops"),
            ],
        ), source="a")

    def test_graph_centrality_boost(self):
        docs = [
            Document(page_content="OpenAI GPT-4 发布人工智能模型", metadata={"source": "a"}),
            Document(page_content="DeepSeek 发布开源大模型", metadata={"source": "c"}),
        ]
        reranker = MultiObjectiveReranker(cross_encoder=None, graph_service=self.graph)
        ranked, details = reranker.rerank("OpenAI GPT-4", docs)
        assert ranked[0].page_content.startswith("OpenAI")
        assert details[0]["s_graph"] > details[1]["s_graph"]

    def test_mmr_diversity(self):
        docs = [
            Document(page_content="OpenAI GPT-4 发布人工智能模型 训练", metadata={"source": "a"}),
            Document(page_content="OpenAI GPT-4 模型发布 人工智能训练", metadata={"source": "b"}),
            Document(page_content="量子计算使用量子比特", metadata={"source": "c"}),
        ]
        reranker = MultiObjectiveReranker(cross_encoder=None, graph_service=None, mmr_lambda=0.5)
        ranked, _ = reranker.rerank("OpenAI", docs, enable_mmr=True)
        # MMR 不应把两条高度重复的文档排在最前
        assert ranked[0].page_content != ranked[1].page_content

    def test_time_decay(self):
        import datetime
        old = Document(page_content="旧文档", metadata={"source": "x", "created": datetime.datetime.now().timestamp() - 365 * 86400})
        new = Document(page_content="新文档", metadata={"source": "y", "created": datetime.datetime.now().timestamp()})
        reranker = MultiObjectiveReranker(cross_encoder=None, graph_service=None)
        scores = reranker._score_time([old, new])
        assert scores[0] < scores[1]

    def test_empty_docs(self):
        reranker = MultiObjectiveReranker(cross_encoder=None)
        ranked, details = reranker.rerank("q", [])
        assert ranked == [] and details == []


class TestQueryPlanner:
    def test_complex_query_decomposed(self):
        plan_json = '{"sub_queries": ["OpenAI 发布了哪些产品", "GPT-4 的技术特点"]}'
        gw = LLMGateway(backend=MockBackend(responses={"复杂问题": plan_json}))
        planner = QueryPlanner(llm_service=FakeLLM(gw))
        sub = planner.plan("OpenAI 和 DeepSeek 有什么区别，各自发布了什么产品？")
        assert len(sub) == 2

    def test_simple_query_not_decomposed(self):
        planner = QueryPlanner(llm_service=None)
        assert planner.plan("什么是量子计算？") == ["什么是量子计算？"]

    def test_llm_failure_fallback(self):
        gw = LLMGateway(backend=MockBackend(responses={}))  # 无匹配 → 非法 JSON
        planner = QueryPlanner(llm_service=FakeLLM(gw))
        sub = planner.plan("A 与 B 以及 C 的区别是什么？")
        assert sub == ["A 与 B 以及 C 的区别是什么？"]


class TestRetrievalService:
    def setup_method(self):
        self.corpus = [
            Document(page_content="OpenAI 发布了 ChatGPT 产品", metadata={"source": "x"}),
            Document(page_content="DeepSeek 发布了 DeepSeek-R1 推理模型", metadata={"source": "y"}),
            Document(page_content="量子计算使用量子比特", metadata={"source": "z"}),
        ]
        self.graph = KnowledgeGraphService(auto_save=False)
        self.graph.merge_subgraph(SubGraph(
            entities=[Entity(name="OpenAI", type="org"), Entity(name="ChatGPT", type="product")],
            relationships=[Relationship(source="OpenAI", target="ChatGPT", relation_type="develops")],
        ), source="x")

    def test_retrieve_with_colbert_fallback(self):
        """ColBERT 模型不可用时自动降级，检索仍工作。"""
        svc = RetrievalService(
            ensemble_retriever=FakeEnsemble(self.corpus),
            graph_service=self.graph,
            colbert_retriever=ColbertRetriever(model_name="__nonexistent__"),
            candidate_documents=self.corpus,
            llm_service=None,
        )
        assert not svc.colbert_retriever.available
        result = svc.retrieve(
            "OpenAI 发布了什么",
            enable_hyde=False, enable_colbert=True,
            enable_query_planning=False, use_cache=False,
        )
        assert len(result) >= 1
        assert result[0].page_content.startswith("OpenAI")

    def test_retrieve_graph_docs(self):
        svc = RetrievalService(
            ensemble_retriever=FakeEnsemble(self.corpus),
            graph_service=self.graph,
            llm_service=None,
        )
        result = svc.retrieve(
            "OpenAI ChatGPT 关系", enable_hyde=False,
            enable_colbert=False, enable_query_planning=False, use_cache=False,
        )
        # 图谱节点应被召回
        contents = " ".join(d.page_content for d in result)
        assert "ChatGPT" in contents

    def test_retrieve_without_reranker(self):
        svc = RetrievalService(
            ensemble_retriever=FakeEnsemble(self.corpus),
            graph_service=None,
            llm_service=None,
        )
        result = svc.retrieve(
            "量子计算", enable_hyde=False, enable_reranking=False,
            enable_colbert=False, enable_query_planning=False, use_cache=False,
        )
        assert any("量子" in d.page_content for d in result)

    def test_rerank_details_exposed(self):
        svc = RetrievalService(
            ensemble_retriever=FakeEnsemble(self.corpus),
            graph_service=None, llm_service=None,
        )
        svc.retrieve(
            "OpenAI", enable_hyde=False, enable_colbert=False,
            enable_query_planning=False, use_cache=False,
        )
        assert isinstance(svc.last_rerank_details, list)
