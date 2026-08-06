"""
自研 RAGAS 风格评估器（白皮书阶段三：RAGAS / DeepEval 自动化评估）。

三大核心指标（与 RAGAS 定义对齐，基于多后端 LLM 网关）：
- faithfulness（忠实度）      : 答案断言被检索上下文支持的比例
- context_relevance（上下文相关性）: 检索上下文中与问题相关句子的比例
- answer_relevance（答案相关性）: 答案是否真正回答了问题

LLM 不可用时自动降级为启发式近似，保证评估流水线始终可运行。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from services.harness import HarnessValidator, ClaimVerdict
from utils.logger import get_logger
from utils.monitoring import Monitor
from utils.text_utils import term_tokens, split_sentences


logger = get_logger(__name__)


class RelevantSentences(BaseModel):
    """问题相关的上下文句子。"""
    relevant_sentences: List[str] = Field(default_factory=list)


class RelevanceScore(BaseModel):
    """答案相关性评分。"""
    score: float = Field(ge=0.0, le=1.0, description="0-1 相关性得分")
    reason: str = Field(default="", description="简短理由")


@dataclass
class SampleResult:
    """单条评估样本的结果。"""
    question: str
    answer: str
    contexts: List[str]
    faithfulness: float = 0.0
    context_relevance: float = 0.0
    answer_relevance: float = 0.0
    verdicts: List[ClaimVerdict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "faithfulness": round(self.faithfulness, 4),
            "context_relevance": round(self.context_relevance, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "answer": self.answer[:200],
            "contexts_count": len(self.contexts),
        }


class RAGEvaluator:
    """RAG 质量评估器（RAGAS 风格）。"""

    def __init__(self, llm_service=None, max_claims: int = 8) -> None:
        self.llm_service = llm_service
        self.harness = HarnessValidator(llm_service=llm_service, max_claims=max_claims)
        self.monitor = Monitor()

    # -- 指标计算 ----------------------------------------------------------

    def faithfulness(self, question: str, answer: str, contexts: Sequence[str]) -> float:
        """忠实度：答案断言被上下文支持的比例。"""
        with self.monitor.measure("eval.faithfulness"):
            if not answer or not contexts:
                return 0.0
            evidence = "\n\n".join(contexts)
            claims = self.harness.extract_claims(answer)
            if not claims:
                return 0.0
            verdicts = [self.harness.verify_claim(c, evidence) for c in claims]
            return sum(1 for v in verdicts if v.supported) / len(verdicts)

    def context_relevance(self, question: str, contexts: Sequence[str]) -> float:
        """上下文相关性：相关句子占全部句子的比例。"""
        with self.monitor.measure("eval.context_relevance"):
            if not contexts:
                return 0.0
            all_sentences = []
            for ctx in contexts:
                all_sentences.extend(split_sentences(ctx))
            if not all_sentences:
                return 0.0
            if self.llm_service is not None:
                try:
                    prompt = (
                        "你是检索质量评估员。判断下面每个句子是否与问题相关，"
                        "输出所有相关句子的原句。\n\n"
                        f"问题: {question}\n\n句子列表:\n" +
                        "\n".join(f"- {i}. {s}" for i, s in enumerate(all_sentences))
                    )
                    rel = self.llm_service.generate_structured(
                        prompt=prompt, schema=RelevantSentences,
                        temperature=0.1, max_tokens=2048, max_attempts=2,
                    ).relevant_sentences
                    if rel:
                        return min(1.0, len(rel) / len(all_sentences))
                except Exception as e:
                    logger.warning(f"LLM 上下文相关性评估失败，使用启发式: {e}")
            # 启发式：含问题关键词的句子比例
            q_terms = self._terms(question)
            if not q_terms:
                return 0.5
            relevant = sum(1 for s in all_sentences if self._terms(s) & q_terms)
            return relevant / len(all_sentences)

    def answer_relevance(self, question: str, answer: str) -> float:
        """答案相关性：答案是否直接回答/覆盖问题。"""
        with self.monitor.measure("eval.answer_relevance"):
            if not answer:
                return 0.0
            if self.llm_service is not None:
                try:
                    prompt = (
                        "你是回答质量评估员。评分（0.0-1.0）表示该回答在多大程度上直接、"
                        "完整地回答了问题，而不是无关内容。\n\n"
                        f"问题: {question}\n\n回答: {answer}"
                    )
                    return max(0.0, min(1.0, self.llm_service.generate_structured(
                        prompt=prompt, schema=RelevanceScore,
                        temperature=0.1, max_tokens=256, max_attempts=2,
                    ).score))
                except Exception as e:
                    logger.warning(f"LLM 答案相关性评估失败，使用启发式: {e}")
            # 启发式：答案与问题关键词重合率
            q_terms = self._terms(question)
            if not q_terms:
                return 0.5
            a_terms = self._terms(answer)
            overlap = len(q_terms & a_terms) / len(q_terms)
            return max(0.0, min(1.0, overlap))

    # -- 批量评估 ----------------------------------------------------------

    def evaluate(
        self,
        samples: Sequence[Dict[str, Any]],
        generate_fn=None,
    ) -> List[SampleResult]:
        """
        批量评估。

        Args:
            samples: [{"question": ..., "answer": ..., "contexts": [...]}]
            generate_fn: 可选；若提供，对没有 answer 的样本调用 generate_fn(question, contexts) -> str

        Returns:
            List[SampleResult]
        """
        results = []
        for sample in samples:
            question = sample["question"]
            contexts = list(sample.get("contexts", []))
            answer = sample.get("answer")
            if not answer and generate_fn is not None:
                answer = generate_fn(question, contexts)
            if not answer:
                answer = ""
            results.append(SampleResult(
                question=question,
                answer=answer,
                contexts=contexts,
                faithfulness=self.faithfulness(question, answer, contexts),
                context_relevance=self.context_relevance(question, contexts),
                answer_relevance=self.answer_relevance(question, answer),
                verdicts=[],
            ))
        return results

    def summarize(self, results: List[SampleResult]) -> Dict[str, Any]:
        """汇总指标。"""
        if not results:
            return {"samples": 0}
        n = len(results)
        return {
            "samples": n,
            "faithfulness_avg": round(sum(r.faithfulness for r in results) / n, 4),
            "context_relevance_avg": round(sum(r.context_relevance for r in results) / n, 4),
            "answer_relevance_avg": round(sum(r.answer_relevance for r in results) / n, 4),
        }

    @staticmethod
    def _terms(text: str) -> set:
        return term_tokens(text)
