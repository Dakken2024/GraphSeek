"""Harness Loop 事实校验护栏单元测试。"""
from langchain_core.documents import Document

from services.harness import HarnessValidator, ClaimSet, ClaimVerdict


class FakeLLM:
    """可编程 Mock LLM（按 prompt 关键字返回不同结构化结果）。"""

    def __init__(self, claims, verdicts, revised="修正后的答案"):
        self.claims = claims
        self.verdicts = verdicts
        self.revised = revised

    def generate_structured(self, prompt, schema, temperature=0.1, max_tokens=1024, max_attempts=2):
        if "事实抽取" in prompt:
            return schema.model_validate({"claims": self.claims})
        if "事实核查" in prompt:
            return schema.model_validate({
                "claim": self.verdicts.get("claim", "断言"),
                "supported": self.verdicts.get("supported", True),
                "reason": self.verdicts.get("reason", ""),
            })
        return schema()

    def generate_non_streaming(self, prompt, temperature=0.2, max_context=4096):
        return self.revised


class TestHeuristicFallback:
    """无 LLM 时的启发式路径。"""

    def setup_method(self):
        self.harness = HarnessValidator(llm_service=None)

    def test_extract_claims_splits_sentences(self):
        claims = self.harness.extract_claims("OpenAI 发布了 ChatGPT。DeepSeek 发布了 R1。")
        assert len(claims) >= 2

    def test_verify_supported(self):
        verdict = self.harness.verify_claim(
            "OpenAI 发布了 ChatGPT",
            "OpenAI 于 2022 年发布了 ChatGPT 产品。",
        )
        assert verdict.supported is True

    def test_verify_unsupported(self):
        verdict = self.harness.verify_claim(
            "火星上有外星人",
            "火星是一颗红色的行星。",
        )
        assert verdict.supported is False

    def test_run_no_revision_without_llm(self):
        result = self.harness.run(
            "OpenAI 发布了 ChatGPT。火星上有外星人。",
            [Document(page_content="OpenAI 发布了 ChatGPT 产品。", metadata={})],
        )
        assert result.corrected is False
        assert 0 < result.support_rate < 1


class TestLLMPath:
    """LLM 校验 + 自我修正回路。"""

    def test_correction_round(self):
        class TwoPhaseLLM:
            """第一轮：两条断言，火星断言不支持 -> 触发修正；第二轮只剩支持的断言。"""

            def __init__(self):
                self.phase = 0

            def generate_structured(self, prompt, schema, temperature=0.1, max_tokens=1024, max_attempts=2):
                self.phase += 1
                if "事实抽取" in prompt:
                    if self.phase <= 2:
                        return schema.model_validate(
                            {"claims": ["OpenAI 发布了 ChatGPT", "火星有外星人"]}
                        )
                    return schema.model_validate({"claims": ["OpenAI 发布了 ChatGPT"]})
                if "事实核查" in prompt:
                    supported = "火星" not in prompt
                    return schema.model_validate(
                        {"claim": "断言", "supported": supported, "reason": "有证据" if supported else "无证据"}
                    )
                return schema()

            def generate_non_streaming(self, prompt, temperature=0.2, max_context=4096):
                return "修正后的答案"

        harness = HarnessValidator(llm_service=TwoPhaseLLM(), max_rounds=2)
        result = harness.run(
            "OpenAI 发布了 ChatGPT。火星有外星人。",
            [Document(page_content="OpenAI 发布了 ChatGPT 产品。", metadata={})],
        )
        assert result.corrected is True
        assert result.corrected_answer == "修正后的答案"
        assert result.support_rate > 0

    def test_unknown_marker_prompt(self):
        """修正 prompt 应包含 [UNKNOWN] 标注指令。"""
        class Recorder:
            def __init__(self):
                self.prompts = []

            def generate_non_streaming(self, prompt, temperature=0.2, max_context=4096):
                self.prompts.append(prompt)
                return "修正后的答案"

        recorder = Recorder()
        harness = HarnessValidator(llm_service=recorder)
        harness._revise("原答案", ["无证据断言"], "证据文本", 0)
        assert "[UNKNOWN]" in recorder.prompts[-1]

    def test_empty_answer(self):
        harness = HarnessValidator(llm_service=None)
        result = harness.run("", [Document(page_content="证据", metadata={})])
        assert result.claims == []

    def test_llm_extract_and_verify(self):
        llm = FakeLLM(
            claims=["OpenAI 发布了 ChatGPT"],
            verdicts={"claim": "OpenAI 发布了 ChatGPT", "supported": True, "reason": "证据支持"},
        )
        harness = HarnessValidator(llm_service=llm)
        claims = harness.extract_claims("OpenAI 发布了 ChatGPT。")
        assert claims == ["OpenAI 发布了 ChatGPT"]
        verdict = harness.verify_claim("OpenAI 发布了 ChatGPT", "OpenAI 发布了 ChatGPT 产品。")
        assert verdict.supported is True
