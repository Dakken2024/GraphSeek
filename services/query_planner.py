"""
Agentic Query Planner（白皮书演进 4：Agentic Query Planning 替代 HyDE）。

Plan-and-Solve 机制：将复杂问题拆分为 2-3 个可并行检索的子查询，
各子查询独立检索后由生成阶段聚合，比 HyDE 单向生成更可控。

- LLM 结构化输出拆解（Pydantic Schema）；
- LLM 不可用或问题简单时自动降级为单查询；
- 支持自定义复杂问题启发式规则。
"""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.monitoring import Monitor


logger = get_logger(__name__)

# 简单问题直接检索的字符阈值
SIMPLE_MAX_LEN = 40

# 复杂问题信号词（中文/英文）
_COMPLEX_MARKERS = [
    "并且", "以及", "和", "与", "及", "或", "对比", "比较", "区别", "差异",
    "总结", "趋势", "关系", "影响", "原因", "为什么", "如何",
    " and ", " or ", " vs ", " compare ", " difference ", " summarize ",
    " relationship ", " why ", " how ",
]


class QueryPlan(BaseModel):
    """查询分解计划。"""
    sub_queries: List[str] = Field(
        default_factory=list,
        description="2-3 个可独立检索的子查询（每个都是完整可检索的自然语言问题）",
    )


class QueryPlanner:
    """查询规划器：判断问题复杂度并分解子查询。"""

    def __init__(self, llm_service=None, max_sub_queries: int = 3) -> None:
        self.llm_service = llm_service
        self.max_sub_queries = max_sub_queries
        self.monitor = Monitor()

    def is_complex(self, query: str) -> bool:
        """启发式判断是否复杂问题。"""
        if len(query) > SIMPLE_MAX_LEN:
            return True
        q_lower = query.lower()
        return any(m in q_lower for m in _COMPLEX_MARKERS)

    def plan(self, query: str) -> List[str]:
        """
        将查询分解为子查询列表。

        Returns:
            子查询列表；简单问题或 LLM 不可用/失败时返回 [query]
        """
        if not self.is_complex(query):
            return [query]

        if self.llm_service is None:
            return [query]

        with self.monitor.measure("planner.decompose"):
            prompt = (
                "你是检索规划专家。请将下面的复杂问题拆解为 "
                f"{self.max_sub_queries} 个以内的、可独立检索的子问题。\n"
                "要求：\n"
                "1. 每个子问题必须独立完整、可直接用于文档/图谱检索；\n"
                "2. 子问题应尽量正交（避免重复覆盖同一信息）；\n"
                "3. 如果问题本身很简单，只返回 1 个子问题（即原问题）。\n\n"
                f"问题: {query}"
            )
            try:
                plan = self.llm_service.generate_structured(
                    prompt=prompt,
                    schema=QueryPlan,
                    temperature=0.1,
                    max_tokens=1024,
                    max_attempts=2,
                )
                sub_queries = [q.strip() for q in plan.sub_queries if q.strip()]
                if sub_queries:
                    logger.info(f"查询分解: {query[:40]} -> {len(sub_queries)} 个子查询")
                    return sub_queries[: self.max_sub_queries]
            except Exception as e:
                logger.warning(f"查询分解失败，回退单查询: {e}")

        return [query]
