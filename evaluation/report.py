"""评估报告生成（Markdown / JSON）。"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

from evaluation.evaluator import SampleResult


def to_json(results: List[SampleResult], summary: Dict[str, any]) -> str:
    payload = {
        "generated_at": datetime.now().isoformat(),
        "summary": summary,
        "samples": [r.to_dict() for r in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_markdown(results: List[SampleResult], summary: Dict[str, any]) -> str:
    lines = [
        "# GraphSeek RAG 评估报告",
        "",
        f"生成时间: {datetime.now().isoformat()}",
        f"样本数: {summary.get('samples', 0)}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 平均值 |",
        "| :--- | :--- |",
        f"| Faithfulness（忠实度） | {summary.get('faithfulness_avg', '-')} |",
        f"| Context Relevance（上下文相关性） | {summary.get('context_relevance_avg', '-')} |",
        f"| Answer Relevance（答案相关性） | {summary.get('answer_relevance_avg', '-')} |",
        "",
        "## 逐样本明细",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"### 样本 {i}: {r.question[:60]}",
            "",
            f"- **Faithfulness**: {r.faithfulness:.2f}",
            f"- **Context Relevance**: {r.context_relevance:.2f}",
            f"- **Answer Relevance**: {r.answer_relevance:.2f}",
            f"- **上下文数**: {len(r.contexts)}",
            "",
        ]
    return "\n".join(lines)
