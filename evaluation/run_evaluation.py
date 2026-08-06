"""RAG 评估流水线 CLI 入口。

用法:
    python -m evaluation.run_evaluation [--data samples.json] [--output report.md] [--backend ollama|openai|mock]

data 文件格式: [{"question": "...", "contexts": ["..."], "answer": "..."}, ...]
answer 缺省时可通过 --generate 调用 LLM 生成。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.evaluator import RAGEvaluator
from evaluation import report as report_mod
from services.llm_gateway import LLMGateway
from services.llm_service import LLMService


def build_llm(args) -> LLMService:
    if args.backend == "mock":
        from services.llm_gateway import MockBackend
        # 注入评估流水线常见提示词的确定性响应，演示 LLM 评估路径
        responses = {
            "事实抽取": '{"claims": ["OpenAI 于 2022 年发布了 ChatGPT", "DeepSeek-R1 是推理模型"]}',
            "事实核查": '{"claim": "断言", "supported": true, "reason": "证据支持"}',
            "句子": '{"relevant_sentences": ["问题相关的句子。"]}',
            "回答质量": '{"score": 0.85, "reason": "直接回答"}',
        }
        return LLMService(api_url="mock", model="mock-1b",
                          gateway=LLMGateway(backend=MockBackend(responses=responses)))
    if args.backend == "openai":
        from services.llm_gateway import OpenAICompatBackend
        import os
        backend = OpenAICompatBackend(
            model=os.getenv("LLM_MODEL", "qwen3.6-27b"),
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_API_BASE", "") or None,
        )
        return LLMService(api_url="openai", model=backend.model,
                          gateway=LLMGateway(backend=backend))
    # ollama 默认
    from config import AppConfig
    config = AppConfig.from_environment()
    return LLMService(
        api_url=config.models.ollama_api_url, model=config.models.llm_model
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="GraphSeek RAG 评估流水线")
    parser.add_argument("--data", default="evaluation/sample_data.json",
                        help="评估样本 JSON 路径")
    parser.add_argument("--output", default="evaluation/report.md",
                        help="报告输出路径（.md 或 .json）")
    parser.add_argument("--backend", default="ollama",
                        choices=["ollama", "openai", "mock"],
                        help="LLM 后端")
    parser.add_argument("--generate", action="store_true",
                        help="对缺少 answer 的样本调用 LLM 生成答案")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"样本文件不存在: {data_path}", file=sys.stderr)
        return 1
    samples = json.loads(data_path.read_text(encoding="utf-8"))

    llm = build_llm(args)
    evaluator = RAGEvaluator(llm_service=llm)

    def generate_fn(question, contexts):
        context_block = "\n\n".join(contexts)
        return llm.generate_non_streaming(
            f"请基于以下资料回答问题。\n资料:\n{context_block}\n问题: {question}",
            temperature=0.2, max_context=4096,
        )

    results = evaluator.evaluate(
        samples, generate_fn=generate_fn if args.generate else None
    )
    summary = evaluator.summarize(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".json":
        out.write_text(report_mod.to_json(results, summary), encoding="utf-8")
    else:
        out.write_text(report_mod.to_markdown(results, summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
