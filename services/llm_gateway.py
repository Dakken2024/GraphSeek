"""
LLM Gateway - 多后端统一接入层（白皮书：基座模型换擎）

统一管理不同推理后端的差异，支持通过环境变量在以下后端间无缝切换：
- ollama       : 本地 Ollama（默认，兼容 deepseek-r1:7b / qwen3 系列）
- openai       : OpenAI 兼容接口（vLLM / SGLang / 硅基流动 / ModelScope / DashScope）

核心能力：
1. 流式 / 非流式生成（chat 与 completion 两种形态）
2. Pydantic 结构化输出（response_format json_schema / json_object + 自动重试修复）
3. 指数退避重试 + Token 统计 + 性能监控
4. 自动降级：结构化输出失败时回退到 Prompt 约束 + Pydantic 校验
"""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Generator, List, Optional, Sequence, Type, TypeVar, Union,
)

import requests
from pydantic import BaseModel, ValidationError

from utils.logger import get_logger
from utils.monitoring import Monitor


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class TokenStats:
    """Token 用量统计。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def add_prompt(self, count: int) -> None:
        self.prompt_tokens += count
        self.total_tokens += count

    def add_completion(self, count: int) -> None:
        self.completion_tokens += count
        self.total_tokens += count
        self.request_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "avg_completion_tokens": round(self.completion_tokens / self.request_count, 2)
            if self.request_count else 0,
        }


class BackendConfigError(RuntimeError):
    """后端配置错误（缺少 API Key / 非法后端名等）。"""


# ---------------------------------------------------------------------------
# 后端抽象
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """LLM 推理后端抽象基类。"""

    name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    # -- 基础能力 ----------------------------------------------------------

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """流式生成（返回 token 迭代器）。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """非流式生成。"""

    @abstractmethod
    def chat_stream(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """多轮对话流式生成。"""

    @abstractmethod
    def check_models(self, required: Optional[List[str]] = None) -> Dict[str, Any]:
        """探测可用模型。"""

    # -- 结构化输出 --------------------------------------------------------

    def generate_structured_raw(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """
        生成符合 JSON Schema 的原始文本（子类按各自能力实现）。
        默认实现：仅通过 Prompt 约束 + format=json 提示。
        """
        raise NotImplementedError

    def supports_schema_mode(self) -> bool:
        """是否原生支持 json_schema response_format。"""
        return False


# ---------------------------------------------------------------------------
# Ollama 后端
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """Ollama 本地后端（/api/chat）。"""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _chat_payload(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        format_: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if format_:
            payload["format"] = format_
        return payload

    def _iter_ndjson(self, response: requests.Response) -> Generator[str, None, None]:
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode())
            except json.JSONDecodeError:
                continue
            delta = data.get("message", {}).get("content", "")
            if delta:
                yield delta
            if data.get("done"):
                break

    def generate_stream(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 4096
    ) -> Generator[str, None, None]:
        messages = [{"role": "user", "content": prompt}]
        yield from self.chat_stream(messages, temperature, max_tokens)

    def generate(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 4096
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self._chat_complete(messages, temperature, max_tokens)

    def _chat_complete(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        format_: Optional[str] = None,
    ) -> str:
        resp = self.session.post(
            f"{self.base_url}/api/chat",
            json=self._chat_payload(messages, temperature, max_tokens, stream=False, format_=format_),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        resp = self.session.post(
            f"{self.base_url}/api/chat",
            json=self._chat_payload(messages, temperature, max_tokens, stream=True),
            stream=True,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        yield from self._iter_ndjson(resp)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """非流式多轮对话。"""
        return self._chat_complete(messages, temperature, max_tokens)

    def generate_structured_raw(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You must output ONLY valid JSON conforming to the provided schema. "
                    "Do not include markdown fences or any other text."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            # Ollama 原生支持 format=json 强制 JSON 输出
            return self._chat_complete(
                messages, temperature, max_tokens, format_="json"
            )
        except Exception as e:
            logger.warning(f"Ollama structured raw failed, fallback: {e}")
            return self._chat_complete(messages, temperature, max_tokens)

    def check_models(self, required: Optional[List[str]] = None) -> Dict[str, Any]:
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            available = [m["model"] for m in resp.json().get("models", [])]
        except Exception as e:
            return {"available": False, "error": str(e), "all_models": []}
        missing = [m for m in (required or []) if m not in available]
        return {
            "available": not missing,
            "missing_models": missing,
            "all_models": available,
            "backend": self.name,
        }


# ---------------------------------------------------------------------------
# OpenAI 兼容后端（vLLM / SGLang / 硅基流动 / ModelScope / DashScope）
# ---------------------------------------------------------------------------

class OpenAICompatBackend(LLMBackend):
    """OpenAI 兼容接口后端，覆盖 vLLM、SGLang、硅基流动、ModelScope、DashScope 等。"""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model)
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise BackendConfigError(
                "openai SDK 未安装，请执行 pip install openai"
            ) from e
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.base_url = base_url

    def _run_chat(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        stream: bool,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        return self.client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            response_format=response_format,
        )

    def generate_stream(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 4096
    ) -> Generator[str, None, None]:
        yield from self.chat_stream(
            [{"role": "user", "content": prompt}], temperature, max_tokens
        )

    def chat_stream(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        stream = self._run_chat(messages, temperature, max_tokens, stream=True)
        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """非流式多轮对话。"""
        resp = self._run_chat(messages, temperature, max_tokens, stream=False)
        return resp.choices[0].message.content or ""

    def generate(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 4096
    ) -> str:
        resp = self._run_chat(
            [{"role": "user", "content": prompt}], temperature, max_tokens, stream=False
        )
        return resp.choices[0].message.content or ""

    def generate_structured_raw(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You must output ONLY valid JSON conforming to the provided schema. "
                    "Do not include markdown fences or any other text."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        # 策略：优先 json_schema（OpenAI 原生 / 新版 vLLM），失败回退 json_object，再回退纯 Prompt
        for fmt in (self._schema_format(json_schema), {"type": "json_object"}):
            try:
                resp = self._run_chat(
                    messages, temperature, max_tokens,
                    stream=False, response_format=fmt,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"structured output with {fmt.get('type')} failed: {e}")
        resp = self._run_chat(messages, temperature, max_tokens, stream=False)
        return resp.choices[0].message.content or ""

    def _schema_format(self, json_schema: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "json_schema", "json_schema": {"name": "result", "schema": json_schema}}

    def supports_schema_mode(self) -> bool:
        return True

    def check_models(self, required: Optional[List[str]] = None) -> Dict[str, Any]:
        # OpenAI 兼容接口通常不提供模型列表，直接按配置模型视为可用
        return {
            "available": True,
            "all_models": [self.model],
            "missing_models": [],
            "backend": self.name,
            "note": "OpenAI 兼容后端按配置模型直接可用，不做预探测",
        }


# ---------------------------------------------------------------------------
# Mock 后端（离线测试 / 演示）
# ---------------------------------------------------------------------------

class MockBackend(LLMBackend):
    """确定性 Mock 后端：离线开发、单元测试与演示使用。"""

    name = "mock"

    def __init__(
        self,
        model: str = "mock-1b",
        responses: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(model)
        self.responses = responses or {}
        self.calls: List[Dict[str, Any]] = []

    def _answer(self, prompt: str) -> str:
        self.calls.append({"prompt": prompt})
        for key, value in self.responses.items():
            if key.lower() in prompt.lower():
                return value
        return f"[mock] 已收到问题，未配置该场景答案。问题摘要: {prompt[:80]}"

    def generate_stream(self, prompt: str, temperature=0.3, max_tokens=4096):
        for token in self._answer(prompt).split(" "):
            yield token + " "

    def generate(self, prompt: str, temperature=0.3, max_tokens=4096) -> str:
        return self._answer(prompt)

    def chat_stream(self, messages, temperature=0.3, max_tokens=4096):
        yield from self.generate_stream(str(messages))

    def chat(self, messages, temperature=0.3, max_tokens=4096) -> str:
        return self._answer(str(messages))

    def generate_structured_raw(self, prompt, json_schema, temperature=0.1, max_tokens=4096) -> str:
        return self._answer(prompt)

    def check_models(self, required=None) -> Dict[str, Any]:
        return {
            "available": True,
            "all_models": [self.model],
            "missing_models": [],
            "backend": self.name,
        }


# ---------------------------------------------------------------------------
# 网关
# ---------------------------------------------------------------------------

class LLMGateway:
    """统一 LLM 网关：负责后端选择、重试、Token 统计与结构化输出。"""

    def __init__(
        self,
        backend: LLMBackend,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        enable_token_stats: bool = True,
    ) -> None:
        self.backend = backend
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.token_stats = TokenStats() if enable_token_stats else None
        self.monitor = Monitor()

    # -- 工厂 --------------------------------------------------------------

    @classmethod
    def from_environment(cls) -> "LLMGateway":
        """从环境变量构建网关。

        环境变量:
            LLM_BACKEND     : ollama | openai | mock（默认 ollama）
            LLM_MODEL       : 模型名（默认继承 config 中的 llm_model）
            LLM_API_KEY     : OpenAI 兼容后端的 API Key
            LLM_API_BASE    : OpenAI 兼容后端的 base_url
            OLLAMA_API_URL  : Ollama 地址（默认 http://localhost:11434）
        """
        backend_name = os.getenv("LLM_BACKEND", "ollama").strip().lower()
        model = os.getenv("LLM_MODEL", "").strip()
        ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        api_key = os.getenv("LLM_API_KEY", "").strip()
        api_base = os.getenv("LLM_API_BASE", "").strip()

        if not model:
            # 从 config 兜底
            from config import AppConfig
            model = AppConfig.from_environment().models.llm_model

        if backend_name in ("ollama", "ollama-chat"):
            backend: LLMBackend = OllamaBackend(model=model, base_url=ollama_url)
        elif backend_name == "openai":
            if not api_key:
                raise BackendConfigError(
                    "使用 openai 后端必须设置 LLM_API_KEY 环境变量"
                )
            backend = OpenAICompatBackend(
                model=model, api_key=api_key, base_url=api_base or None
            )
        elif backend_name == "mock":
            backend = MockBackend(model=model or "mock-1b")
        else:
            raise BackendConfigError(
                f"未知 LLM_BACKEND: {backend_name}（可选: ollama | openai | mock）"
            )
        return cls(backend=backend)

    @property
    def model(self) -> str:
        return self.backend.model

    @property
    def backend_name(self) -> str:
        return self.backend.name

    # -- 基础生成 ----------------------------------------------------------

    def _retry(self, func, *args, **kwargs):
        """指数退避重试包装（针对可重试的异常）。"""
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except (requests.exceptions.RequestException, TimeoutError, ConnectionError) as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = self.backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"[{self.backend_name}] 请求失败(第{attempt + 1}次)，"
                        f"{wait:.1f}s 后重试: {e}"
                    )
                    time.sleep(wait)
        raise last_exc

    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        with self.monitor.measure("llm.generate"):
            result = self._retry(
                self.backend.generate, prompt, temperature, max_tokens
            )
            if self.token_stats:
                self.token_stats.add_prompt(self._estimate_tokens(prompt))
                self.token_stats.add_completion(self._estimate_tokens(result))
            return result

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        with self.monitor.measure("llm.generate_stream"):
            token_count = 0
            try:
                for token in self.backend.generate_stream(prompt, temperature, max_tokens):
                    token_count += 1
                    yield token
            except Exception as e:
                logger.error(f"[{self.backend_name}] 流式生成失败: {e}")
                yield f"[Error: {e}]"
            finally:
                if self.token_stats and token_count:
                    self.token_stats.add_prompt(self._estimate_tokens(prompt))
                    self.token_stats.add_completion(token_count)

    def generate_chat(
        self,
        messages: Sequence[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        with self.monitor.measure("llm.chat"):
            chat_fn = getattr(self.backend, "chat", None)
            if callable(chat_fn):
                return self._retry(chat_fn, messages, temperature, max_tokens)
            # 后端不支持 chat：拼接为单 prompt
            parts = [f"{m['role']}: {m['content']}" for m in messages]
            return self.generate("\n".join(parts), temperature, max_tokens)

    # -- 结构化输出 --------------------------------------------------------

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_attempts: int = 2,
    ) -> T:
        """
        生成并 Pydantic 校验的结构化输出。

        Args:
            prompt: 抽取/生成指令
            schema: Pydantic 模型类（BaseModel 子类）
            temperature: 采样温度
            max_tokens: 最大生成长度
            max_attempts: 解析失败后的重试次数（重新以"校验错误信息"为上下文追问）

        Returns:
            校验通过的 Pydantic 模型实例

        Raises:
            ValueError: 多次尝试仍无法得到合法 JSON
        """
        json_schema = schema.model_json_schema()
        last_error: Optional[str] = None
        current_prompt = prompt

        for attempt in range(max_attempts):
            with self.monitor.measure("llm.generate_structured"):
                try:
                    raw = self._retry(
                        self.backend.generate_structured_raw,
                        current_prompt,
                        json_schema,
                        temperature,
                        max_tokens,
                    )
                except Exception as e:
                    last_error = f"后端调用失败: {e}"
                    logger.error(last_error)
                    break

            cleaned = self._extract_json(raw)
            if cleaned is None:
                last_error = "响应中未找到合法 JSON"
                logger.warning(f"结构化输出解析失败(第{attempt + 1}次): {raw[:200]}")
                current_prompt = (
                    f"{prompt}\n\n上次输出无效({last_error})，请重新输出严格符合 Schema 的 JSON。\n"
                    f"无效输出示例: {raw[:300]}"
                )
                continue

            try:
                return schema.model_validate_json(cleaned)
            except ValidationError as e:
                last_error = str(e)
                logger.warning(f"Schema 校验失败(第{attempt + 1}次): {last_error[:300]}")
                current_prompt = (
                    f"{prompt}\n\n上次输出未通过校验，错误如下：\n{last_error}\n"
                    f"请修正后重新输出严格符合 Schema 的 JSON。"
                )

        raise ValueError(f"结构化输出失败: {last_error}")

    # -- 工具方法 ----------------------------------------------------------

    def check_models(self, required: Optional[List[str]] = None) -> Dict[str, Any]:
        return self.backend.check_models(required)

    def get_token_stats(self) -> Optional[Dict[str, Any]]:
        return self.token_stats.to_dict() if self.token_stats else None

    def reset_token_stats(self) -> None:
        if self.token_stats:
            self.token_stats = TokenStats()

    def close(self) -> None:
        if hasattr(self.backend, "session"):
            self.backend.session.close()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数（英文 ~4 字符/token，中文 ~1.5 字/token）。"""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        return (len(text) - cjk) // 4 + cjk

    @staticmethod
    def _extract_json(raw: str) -> Optional[str]:
        """从模型输出中提取 JSON（容忍 markdown 代码块与前后杂讯）。"""
        text = raw.strip()
        # 去掉 markdown 代码块围栏
        if text.startswith("```"):
            lines = text.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        # 尝试整体解析
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        # 提取第一个 { ... } 或 [ ... ] 块
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    json.loads(candidate)
                    return candidate
                except json.JSONDecodeError:
                    continue
        return None
