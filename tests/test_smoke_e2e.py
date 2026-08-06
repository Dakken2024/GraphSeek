"""端到端冒烟测试：查询规划 -> 多路召回 -> RRF -> 多目标重排 -> 生成 -> Harness -> 评估。

全程使用 Mock 后端与伪检索器，不依赖 Ollama / FAISS / 外部网络。
"""
from langchain_core.documents import Document

from services.retrieval_service import RetrievalService
from services.llm_gateway import MockBackend, LLMGateway
from services.llm_service import LLMService
from services.graph_service import (
    KnowledgeGraphService, Entity, Relationship, SubGraph,
)
from services.harness import HarnessValidator
from services.context_compressor import ContextCompressor
from evaluation.evaluator import RAGEvaluator


class FakeEnsemble:
    def __init__(self, docs):
        self.docs = docs
        self.search_kwargs = {"k": 5}

    def invoke(self, query):
        k = self.search_kwargs.get("k", 5)
        qw = set(query.lower().split())
        scored = []
        for d in self.docs:
            dw = set(d.page_content.lower().split())
            scored.append((len(qw & dw), d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]


def build_mock_llm() -> LLMService:
    """带全场景响应的 Mock LLM。"""
    responses = {
        "复杂问题": '{"sub_queries": ["OpenAI 发布了哪些产品", "DeepSeek 发布了哪些产品"]}',
        "事实抽取": '{"claims": ["OpenAI 发布了 ChatGPT"]}',
        "事实核查": '{"claim": "OpenAI 发布了 ChatGPT", "supported": true, "reason": "证据支持"}',
        "句子": '{"relevant_sentences": ["OpenAI 发布了 ChatGPT 产品。"]}',
        "回答质量": '{"score": 0.9, "reason": "直接回答"}',
        "关键证据": '{"key_sentences": ["OpenAI 发布了 ChatGPT 产品。"]}',
    }
    gateway = LLMGateway(backend=MockBackend(responses=responses))
    return LLMService(api_url="mock", model="mock-1b", gateway=gateway)


def build_pipeline():
    corpus = [
        Document(page_content="OpenAI 发布了 ChatGPT 对话产品", metadata={"source": "a.txt"}),
        Document(page_content="DeepSeek 发布了 DeepSeek-R1 推理模型", metadata={"source": "b.txt"}),
        Document(page_content="量子计算使用量子比特进行运算", metadata={"source": "c.txt"}),
    ]
    graph = KnowledgeGraphService(auto_save=False)
    graph.merge_subgraph(SubGraph(
        entities=[
            Entity(name="OpenAI", type="organization"),
            Entity(name="ChatGPT", type="product"),
            Entity(name="DeepSeek", type="organization"),
        ],
        relationships=[
            Relationship(source="OpenAI", target="ChatGPT", relation_type="develops", weight=2.0),
        ],
    ), source="a.txt")
    graph.build_community_summaries(llm_service=None)
    return corpus, graph


def test_full_chain_smoke():
    corpus, graph = build_pipeline()
    llm = build_mock_llm()

    svc = RetrievalService(
        ensemble_retriever=FakeEnsemble(corpus),
        graph_service=graph,
        candidate_documents=corpus,
        llm_service=llm,
    )
    docs = svc.retrieve(
        "OpenAI 和 DeepSeek 都发布了什么产品？它们有何区别？",
        enable_hyde=False, enable_colbert=False,
        use_cache=False, enable_mmr=True,
    )
    assert len(docs) >= 1

    # 生成（Mock）
    answer = llm.generate_non_streaming(
        f"基于资料回答问题。资料: {docs[0].page_content}。问题: OpenAI 发布了什么？"
    )
    assert answer.strip() != ""

    # Harness 校验
    harness = HarnessValidator(llm_service=llm, max_rounds=1)
    h_result = harness.run(answer, docs)
    assert h_result.support_rate >= 0
    assert h_result.to_dict()["original_answer"] == answer

    # 上下文压缩
    compressor = ContextCompressor(llm_service=llm)
    compressed = compressor.compress("OpenAI 发布了什么", [d.page_content for d in docs])
    assert len(compressed) == len(docs)
    assert all(c.strip() for c in compressed)

    # 评估
    evaluator = RAGEvaluator(llm_service=llm)
    results = evaluator.evaluate([{
        "question": "OpenAI 发布了什么产品？",
        "answer": answer,
        "contexts": [d.page_content for d in docs],
    }])
    summary = evaluator.summarize(results)
    assert summary["samples"] == 1
    assert 0 <= summary["faithfulness_avg"] <= 1
    assert 0 <= summary["answer_relevance_avg"] <= 1

    # 溯源明细
    assert isinstance(svc.last_rerank_details, list)
    assert all({"s_rel", "s_graph", "s_time", "s_div", "final"} <= set(d) for d in svc.last_rerank_details)
