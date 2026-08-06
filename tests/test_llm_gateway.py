"""LLM 多后端网关单元测试。"""
import json

import pytest
from pydantic import BaseModel, Field

from services.llm_gateway import (
    MockBackend,
    LLMGateway,
    OllamaBackend,
    OpenAICompatBackend,
    BackendConfigError,
)


class Entity(BaseModel):
    name: str = Field(description="名称")
    type: str = Field(description="类型")


class TestJSONExtraction:
    def test_plain_json(self):
        assert LLMGateway._extract_json('{"a": 1}') == '{"a": 1}'

    def test_markdown_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert json.loads(LLMGateway._extract_json(raw)) == {"a": 1}

    def test_noise_around(self):
        raw = '好的，结果如下： {"a": [1, 2]} 完毕。'
        assert json.loads(LLMGateway._extract_json(raw)) == {"a": [1, 2]}

    def test_invalid(self):
        assert LLMGateway._extract_json("不是 JSON") is None


class TestMockBackend:
    def setup_method(self):
        self.mock = MockBackend(responses={
            "实体抽取": '{"name": "量子计算", "type": "技术"}',
        })
        self.gw = LLMGateway(backend=self.mock)

    def test_generate(self):
        assert "mock" in self.gw.generate("测试")

    def test_structured_output(self):
        ent = self.gw.generate_structured("请实体抽取", Entity)
        assert ent.name == "量子计算"
        assert ent.type == "技术"

    def test_structured_retry_on_invalid(self):
        """首次输出非法 JSON，二次重试得到合法结果。"""
        class _FlakyBackend(MockBackend):
            def __init__(self):
                super().__init__(responses={"x": "not-json"})
                self.count = 0

            def generate_structured_raw(self, prompt, json_schema, temperature=0.1, max_tokens=4096):
                self.count += 1
                if self.count == 1:
                    return "这不是 JSON"
                return '{"name": "OK", "type": "type"}'

        gw = LLMGateway(backend=_FlakyBackend())
        ent = gw.generate_structured("x", Entity)
        assert ent.name == "OK"

    def test_token_stats(self):
        self.gw.generate("hello world")
        stats = self.gw.get_token_stats()
        assert stats["request_count"] >= 1
        assert stats["total_tokens"] > 0

    def test_backend_name(self):
        assert self.gw.backend_name == "mock"


class TestBackendConfig:
    def test_openai_requires_key(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "openai")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        with pytest.raises(BackendConfigError):
            LLMGateway.from_environment()

    def test_unknown_backend(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "unknown")
        with pytest.raises(BackendConfigError):
            LLMGateway.from_environment()

    def test_mock_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "mock")
        gw = LLMGateway.from_environment()
        assert gw.backend_name == "mock"

    def test_ollama_backend_instantiation(self):
        backend = OllamaBackend(model="qwen3:27b", base_url="http://localhost:11434")
        assert backend.name == "ollama"
        assert backend.model == "qwen3:27b"

    def test_openai_backend_requires_sdk(self):
        # openai SDK 已安装，验证能实例化（不发起网络请求）
        backend = OpenAICompatBackend(
            model="qwen3.6-27b", api_key="test-key",
            base_url="https://api.siliconflow.cn/v1",
        )
        assert backend.name == "openai"
        assert backend.supports_schema_mode()
