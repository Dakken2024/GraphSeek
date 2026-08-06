"""RAGAS 风格评估器单元测试。"""
from evaluation.evaluator import RAGEvaluator


class TestHeuristicEvaluation:
    """无 LLM 的启发式评估路径。"""

    def setup_method(self):
        self.evaluator = RAGEvaluator(llm_service=None)

    def test_faithfulness_supported(self):
        f = self.evaluator.faithfulness(
            "OpenAI 发布了什么",
            "OpenAI 发布了 ChatGPT。",
            ["OpenAI 于 2022 年发布了 ChatGPT 产品。"],
        )
        assert f > 0.5

    def test_faithfulness_zero_without_context(self):
        f = self.evaluator.faithfulness("q", "答案", [])
        assert f == 0.0

    def test_context_relevance(self):
        r = self.evaluator.context_relevance(
            "OpenAI ChatGPT",
            ["OpenAI 发布了 ChatGPT。", "量子计算使用量子比特。"],
        )
        assert 0 < r < 1

    def test_answer_relevance(self):
        r = self.evaluator.answer_relevance("OpenAI 发布了什么产品", "OpenAI 发布了 ChatGPT 产品。")
        assert r > 0.5

    def test_batch_evaluate(self):
        samples = [
            {
                "question": "OpenAI 发布了什么",
                "answer": "OpenAI 发布了 ChatGPT。",
                "contexts": ["OpenAI 于 2022 年发布了 ChatGPT。"],
            },
            {
                "question": "什么是量子计算",
                "answer": "量子计算使用量子比特。",
                "contexts": ["量子计算使用量子比特作为信息单元。"],
            },
        ]
        results = self.evaluator.evaluate(samples)
        summary = self.evaluator.summarize(results)
        assert summary["samples"] == 2
        assert 0 <= summary["faithfulness_avg"] <= 1


class TestLLMPath:
    def test_llm_metrics(self):
        class FakeLLM:
            def generate_structured(self, prompt, schema, temperature=0.1, max_tokens=1024, max_attempts=2):
                if "事实抽取" in prompt:
                    return schema.model_validate({"claims": ["OpenAI 发布了 ChatGPT"]})
                if "事实核查" in prompt:
                    return schema.model_validate(
                        {"claim": "OpenAI 发布了 ChatGPT", "supported": True, "reason": "有证据"}
                    )
                if "句子" in prompt:
                    return schema.model_validate(
                        {"relevant_sentences": ["OpenAI 发布了 ChatGPT。"]}
                    )
                return schema.model_validate({"score": 0.9, "reason": "直接回答"})

        evaluator = RAGEvaluator(llm_service=FakeLLM())
        f = evaluator.faithfulness(
            "OpenAI 发布了什么",
            "OpenAI 发布了 ChatGPT。",
            ["OpenAI 发布了 ChatGPT 产品。"],
        )
        assert f == 1.0

        ar = evaluator.answer_relevance("OpenAI 发布了什么", "OpenAI 发布了 ChatGPT。")
        assert ar == 0.9

        cr = evaluator.context_relevance(
            "OpenAI 发布了什么",
            ["OpenAI 发布了 ChatGPT。", "无关句子。"],
        )
        # 1 句相关 / 2 句 -> 0.5
        assert cr == 0.5


class TestReport:
    def test_report_generation(self):
        from evaluation import report as report_mod
        from evaluation.evaluator import SampleResult

        results = [SampleResult(
            question="q1", answer="a1", contexts=["c1"],
            faithfulness=0.8, context_relevance=0.7, answer_relevance=0.9,
        )]
        summary = {"samples": 1, "faithfulness_avg": 0.8,
                   "context_relevance_avg": 0.7, "answer_relevance_avg": 0.9}
        md = report_mod.to_markdown(results, summary)
        assert "# GraphSeek RAG 评估报告" in md
        assert "0.8" in md
        js = report_mod.to_json(results, summary)
        import json
        assert json.loads(js)["samples"][0]["faithfulness"] == 0.8
