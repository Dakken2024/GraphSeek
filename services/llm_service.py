"""
LLM Service for GraphSeek application.

基于 LLMGateway 的兼容层：保持原有公开 API（generate_response / generate_non_streaming /
generate_hypothetical_answer / check_models / get_token_stats）不变，底层统一走多后端网关。
新增 generate_structured 结构化输出能力（白皮书：原生结构化输出）。
"""
from typing import Generator, Optional, Dict, Any, List, Type, TypeVar
import asyncio

from services.llm_gateway import (
    LLMGateway,
    OllamaBackend,
    MockBackend,
    BackendConfigError,
)
from utils.logger import get_logger
from pydantic import BaseModel


logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """与后端网关解耦的 LLM 服务兼容层。"""

    def __init__(
        self,
        api_url: str,
        model: str,
        timeout: int = 120,
        max_retries: int = 3,
        enable_token_stats: bool = True,
        gateway: Optional[LLMGateway] = None,
    ) -> None:
        self.api_url = api_url
        self.model = model

        if gateway is not None:
            self.gateway = gateway
        else:
            # 环境变量优先（LLM_BACKEND/LLM_API_KEY...），默认 Ollama 保持旧行为
            import os
            backend_name = os.getenv("LLM_BACKEND", "ollama").strip().lower()
            if backend_name == "openai":
                try:
                    self.gateway = LLMGateway.from_environment()
                    model = self.gateway.model
                    self.model = model
                except BackendConfigError as e:
                    logger.warning(f"OpenAI 后端初始化失败，回退 Ollama: {e}")
                    self.gateway = LLMGateway(
                        backend=OllamaBackend(model=model, base_url=api_url.replace("/api/generate", ""),
                                              timeout=timeout),
                        max_retries=max_retries,
                        enable_token_stats=enable_token_stats,
                    )
            elif backend_name == "mock":
                self.gateway = LLMGateway(backend=MockBackend(model=model),
                                          max_retries=max_retries,
                                          enable_token_stats=enable_token_stats)
            else:
                self.gateway = LLMGateway(
                    backend=OllamaBackend(model=model, base_url=api_url.replace("/api/generate", ""),
                                          timeout=timeout),
                    max_retries=max_retries,
                    enable_token_stats=enable_token_stats,
                )

    # -- 兼容旧 API --------------------------------------------------------

    def check_models(self, required_models: list) -> dict:
        return self.gateway.check_models(required_models)

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        HyDE 假设性文档生成（保留供旧链路调用；新链路由 QueryPlanner 替代）。
        """
        prompt = (
            "Generate a hypothetical passage that would answer the question. "
            "It should be factual, concise and in the same style as a source document.\n"
            f"Question: {query}\nPassage:"
        )
        return self.gateway.generate(prompt, temperature=0.3, max_tokens=512)

    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_context: int = 4096,
    ) -> Generator[str, None, None]:
        yield from self.gateway.generate_stream(prompt, temperature, max_context)

    async def generate_response_async(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_context: int = 4096,
    ):
        """异步版本：通过线程池执行非流式生成后按 chunk 产出。"""
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, self.gateway.generate, prompt, temperature, max_context
        )
        for i in range(0, len(text), 64):
            yield text[i : i + 64]

    def generate_non_streaming(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_context: int = 4096,
    ) -> str:
        return self.gateway.generate(prompt, temperature, max_context)

    # -- 新增能力 ----------------------------------------------------------

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        max_attempts: int = 2,
    ) -> T:
        """Pydantic 结构化输出（白皮书：原生结构化输出，消灭正则解析）。"""
        return self.gateway.generate_structured(
            prompt, schema, temperature=temperature,
            max_tokens=max_tokens, max_attempts=max_attempts,
        )

    @property
    def backend_name(self) -> str:
        return self.gateway.backend_name

    # -- 统计 --------------------------------------------------------------

    def get_token_stats(self) -> Optional[Dict[str, Any]]:
        return self.gateway.get_token_stats()

    def reset_token_stats(self) -> None:
        self.gateway.reset_token_stats()

    def close(self) -> None:
        self.gateway.close()
