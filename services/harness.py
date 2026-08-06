"""
Harness Loop 事实校验护栏（白皮书：引入 Harness Loop 验证与护栏回路）。

流程（Step 1 生成 -> Step 2 事实校验 -> Step 3 自我修正）：
1. 从生成答案中提取事实断言（Claim）
2. 将断言与检索证据交叉校验（LLM-as-a-Judge / NLI 风格）
3. 无证据支持的断言反馈给 LLM 重新作答（最多 max_rounds 轮）
4. 仍无证据的断言以 [UNKNOWN] 标注，避免幻觉

降级策略：LLM 不可用时使用启发式（证据关键词包含度）判定，保守标记。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.monitoring import Monitor
from utils.text_utils import term_tokens


logger = get_logger(__name__)


class ClaimSet(BaseModel):
    """答案中的事实断言集合。"""
    claims: List[str] = Field(default_factory=list, description="提取的事实断言列表（每条独立完整）")


class ClaimVerdict(BaseModel):
    """单条断言的校验结论。"""
    claim: str = Field(description="被校验的断言")
    supported: bool = Field(description="断言是否被证据支持")
    reason: str = Field(default="", description="简短判定理由")


@dataclass
class HarnessResult:
    """Harness 校验结果。"""
    original_answer: str
    corrected_answer: str
    claims: List[str] = field(default_factory=list)
    verdicts: List[ClaimVerdict] = field(default_factory=list)
    rounds: int = 0
    unsupported_claims: List[str] = field(default_factory=list)
    corrected: bool = False

    @property
    def support_rate(self) -> float:
        """证据支持率。"""
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.supported) / len(self.verdicts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_answer": self.original_answer,
            "corrected_answer": self.corrected_answer,
            "claims": self.claims,
            "verdicts": [{"claim": v.claim, "supported": v.supported, "reason": v.reason} for v in self.verdicts],
            "rounds": self.rounds,
            "unsupported_claims": self.unsupported_claims,
            "corrected": self.corrected,
            "support_rate": round(self.support_rate, 4),
        }


class HarnessValidator:
    """Harness Loop 事实校验器。"""

    UNKNOWN_MARKER = "[UNKNOWN]"

    def __init__(self, llm_service=None, max_rounds: int = 2, max_claims: int = 8) -> None:
        self.llm_service = llm_service
        self.max_rounds = max_rounds
        self.max_claims = max_claims
        self.monitor = Monitor()

    # -- Step 2: 断言提取 --------------------------------------------------

    def extract_claims(self, answer: str) -> List[str]:
        """从答案中提取事实断言。"""
        if not answer or not answer.strip():
            return []
        if self.llm_service is not None:
            try:
                prompt = (
                    "你是事实抽取器。请从下面的回答中提取独立、可被证据验证的事实断言。\n"
                    "要求：\n"
                    "1. 每条断言必须独立完整、可被查证；\n"
                    "2. 忽略提问、建议、无信息量的话（如'总之'）；\n"
                    "3. 最多提取 8 条。\n\n"
                    f"回答: {answer}"
                )
                claims = self.llm_service.generate_structured(
                    prompt=prompt, schema=ClaimSet, temperature=0.1,
                    max_tokens=1024, max_attempts=2,
                ).claims
                return [c.strip() for c in claims if c.strip()][: self.max_claims]
            except Exception as e:
                logger.warning(f"LLM 断言提取失败，使用启发式: {e}")
        return self._heuristic_claims(answer)

    def _heuristic_claims(self, answer: str) -> List[str]:
        """启发式断言提取：按句拆分为断言。"""
        claims = []
        for line in re.split(r"[。．.!?？；;！\n]", answer):
            line = line.strip()
            if len(line) >= 5:  # 中文短句（如 7 字断言）也保留
                claims.append(line)
        return claims[: self.max_claims]

    # -- Step 2: 单条断言校验 ---------------------------------------------

    def verify_claim(self, claim: str, evidence: str) -> ClaimVerdict:
        """校验单条断言是否有证据支持。"""
        if self.llm_service is not None:
            try:
                prompt = (
                    "你是事实核查员。判断下面的断言能否被给出的证据直接支持。\n"
                    "要求：只依据证据，不额外引入知识；证据不足时 supported=false。\n\n"
                    f"断言: {claim}\n\n证据:\n{evidence[:4000]}"
                )
                return self.llm_service.generate_structured(
                    prompt=prompt, schema=ClaimVerdict, temperature=0.1,
                    max_tokens=512, max_attempts=2,
                )
            except Exception as e:
                logger.warning(f"LLM 校验失败，使用启发式: {e}")
        return self._heuristic_verify(claim, evidence)

    def _heuristic_verify(self, claim: str, evidence: str) -> ClaimVerdict:
        """启发式校验：断言术语在证据中的覆盖率。"""
        claim_terms = term_tokens(claim)
        evidence_lower = evidence.lower()
        if not claim_terms:
            return ClaimVerdict(claim=claim, supported=False, reason="断言无可比较术语")
        hit = sum(1 for t in claim_terms if t in evidence_lower)
        rate = hit / len(claim_terms)
        supported = rate >= 0.6
        return ClaimVerdict(
            claim=claim, supported=supported,
            reason=f"术语覆盖率 {rate:.0%}（阈值 60%）",
        )

    # -- Step 3: 自我修正 --------------------------------------------------

    def _revise(
        self, answer: str, unsupported: List[str], evidence: str, round_no: int
    ) -> str:
        if self.llm_service is None:
            return answer
        prompt = (
            "你的上一个回答中存在缺乏证据支持的断言。请基于提供的证据重新组织回答：\n"
            "- 保留有证据支持的内容；\n"
            f"- 删除或改写无法被证据支持的断言；\n"
            f"- 确属证据不足的内容，以 {self.UNKNOWN_MARKER} 标注。\n\n"
            f"无证据断言: {'; '.join(unsupported)}\n\n"
            f"证据:\n{evidence[:5000]}\n\n"
            f"原回答:\n{answer}"
        )
        return self.llm_service.generate_non_streaming(
            prompt, temperature=0.2, max_context=4096
        ).strip()

    # -- 主入口 ------------------------------------------------------------

    def run(
        self,
        answer: str,
        evidence_docs: Sequence[Any],
        llm_service=None,
        max_rounds: Optional[int] = None,
    ) -> HarnessResult:
        """
        执行完整 Harness 校验回路。

        Args:
            answer: LLM 生成的初始答案
            evidence_docs: 检索到的证据文档（Document 对象或字符串）
            llm_service: 覆盖 LLM 服务（None 使用构造时的）
            max_rounds: 最大修正轮数

        Returns:
            HarnessResult
        """
        with self.monitor.measure("harness.run"):
            llm = llm_service or self.llm_service
            rounds = max_rounds if max_rounds is not None else self.max_rounds
            evidence = "\n\n".join(
                d.page_content if hasattr(d, "page_content") else str(d)
                for d in evidence_docs
            )

            claims = self.extract_claims(answer)
            current = answer
            all_unsupported: List[str] = []
            all_verdicts: List[ClaimVerdict] = []
            performed = 0

            for i in range(rounds + 1):
                performed = i
                verdicts = [self.verify_claim(c, evidence) for c in claims]
                all_verdicts = verdicts
                unsupported = [v.claim for v in verdicts if not v.supported]
                all_unsupported = unsupported

                if not unsupported or i >= rounds:
                    break
                revised = self._revise(current, unsupported, evidence, i)
                if not revised or revised == current:
                    break
                current = revised
                # 重新提取断言进行下一轮校验
                claims = self.extract_claims(current)

            result = HarnessResult(
                original_answer=answer,
                corrected_answer=current,
                claims=claims,
                verdicts=all_verdicts,
                rounds=performed,
                unsupported_claims=all_unsupported,
                corrected=current != answer,
            )
            logger.info(
                f"Harness: rounds={result.rounds}, support_rate={result.support_rate:.2f}, "
                f"corrected={result.corrected}"
            )
            return result
