"""
LLM service for GraphSeek application.
Handles interactions with Ollama API for text generation and embeddings.
Enhanced with async support, retry mechanism, and token statistics.
"""
import json
import asyncio
from typing import Generator, Optional, Dict, Any, List
from functools import wraps
import time
import hashlib
from dataclasses import dataclass, field
import requests
from requests.adapters import HTTPAdapter, Retry

from utils.logger import get_logger
from utils.monitoring import Monitor


logger = get_logger(__name__)


@dataclass
class TokenStats:
    """Statistics for token usage."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    
    def add_completion(self, token_count: int) -> None:
        """Add completion tokens."""
        self.completion_tokens += token_count
        self.total_tokens += token_count
        self.request_count += 1
    
    def add_prompt(self, token_count: int) -> None:
        """Add prompt tokens."""
        self.prompt_tokens += token_count
        self.total_tokens += token_count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "avg_completion_tokens": round(self.completion_tokens / self.request_count, 2) if self.request_count > 0 else 0,
        }


def retry_with_backoff(max_retries: int = 3, backoff_factor: float = 0.5):
    """Decorator for retrying failed requests with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {wait_time:.2f}s: {str(e)}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Request failed after {max_retries + 1} attempts: {str(e)}")
            raise last_exception
        return wrapper
    return decorator


DEFAULT_BACKOFF_FACTOR = 0.5


class LLMService:
    """Service for interacting with Ollama LLM API with enhanced features."""
    
    def __init__(
        self, 
        api_url: str, 
        model: str,
        timeout: int = 120,
        max_retries: int = 3,
        enable_token_stats: bool = True,
    ) -> None:
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_token_stats = enable_token_stats
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=DEFAULT_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Token statistics
        self.token_stats = TokenStats() if enable_token_stats else None
        
        # Monitoring
        self.monitor = Monitor()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text (rough approximation)."""
        # Simple heuristic: ~4 characters per token for English
        return len(text) // 4
    
    def check_models(self, required_models: list) -> dict:
        """
        Check if required models are available in Ollama.
        
        Args:
            required_models: List of required model names
            
        Returns:
            Dictionary with availability status and missing models
        """
        try:
            base_url = self.api_url.replace("/api/generate", "")
            response = self.session.get(f"{base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            available_models = [model['model'] for model in response.json()['models']]
            
            missing_models = [model for model in required_models if model not in available_models]
            
            return {
                "available": len(missing_models) == 0,
                "missing_models": missing_models,
                "all_models": available_models,
            }
        except Exception as e:
            logger.error(f"Failed to check models: {str(e)}")
            return {
                "available": False,
                "error": str(e),
            }
    
    @retry_with_backoff()
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer for HyDE query expansion.
        
        Args:
            query: Original query string
            
        Returns:
            Generated hypothetical answer or original query if failed
        """
        with self.monitor.measure("llm.generate_hypothetical"):
            try:
                response = self.session.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": f"Generate a hypothetical answer to: {query}",
                        "stream": False,
                    },
                    timeout=self.timeout,
                ).json()
                
                result = response.get("response", query)
                
                # Update token stats
                if self.token_stats:
                    self.token_stats.add_prompt(self._estimate_tokens(query))
                    self.token_stats.add_completion(self._estimate_tokens(result))
                
                return result
            except Exception as e:
                logger.error(f"Hypothetical answer generation failed: {str(e)}")
                return query
    
    def generate_response(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_context: int = 4096,
    ) -> Generator[str, None, None]:
        """
        Stream a response from the LLM.
        
        Args:
            prompt: Input prompt for generation
            temperature: Temperature parameter for generation
            max_context: Maximum context window size
            
        Yields:
            Generated tokens as strings
        """
        with self.monitor.measure("llm.generate_stream"):
            token_count = 0
            try:
                response = self.session.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": temperature,
                            "num_ctx": max_context,
                        },
                    },
                    stream=True,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode())
                            token = data.get("response", "")
                            yield token
                            
                            if token:
                                token_count += 1
                            
                            if data.get("done", False):
                                # Update token stats if available from response
                                if self.token_stats and "prompt_eval_count" in data:
                                    self.token_stats.add_prompt(data.get("prompt_eval_count", 0))
                                    self.token_stats.add_completion(data.get("eval_count", token_count))
                                break
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Stream generation failed: {str(e)}")
                yield f"[Error: {str(e)}]"
            
            # Fallback token estimation
            if self.token_stats and token_count > 0:
                self.token_stats.add_prompt(self._estimate_tokens(prompt))
                self.token_stats.add_completion(token_count)
    
    async def generate_response_async(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_context: int = 4096,
    ):
        """
        Async version of generate_response for non-blocking operations.
        
        Args:
            prompt: Input prompt for generation
            temperature: Temperature parameter for generation
            max_context: Maximum context window size
            
        Yields:
            Generated tokens as strings
        """
        import aiohttp
        
        with self.monitor.measure("llm.generate_async"):
            token_count = 0
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.api_url,
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": True,
                            "options": {
                                "temperature": temperature,
                                "num_ctx": max_context,
                            },
                        },
                        timeout=aiohttp.ClientTimeout(total=self.timeout),
                    ) as response:
                        async for line in response.content:
                            if line:
                                try:
                                    data = json.loads(line.decode())
                                    token = data.get("response", "")
                                    yield token
                                    
                                    if token:
                                        token_count += 1
                                    
                                    if data.get("done", False):
                                        if self.token_stats and "prompt_eval_count" in data:
                                            self.token_stats.add_prompt(data.get("prompt_eval_count", 0))
                                            self.token_stats.add_completion(data.get("eval_count", token_count))
                                        break
                                except json.JSONDecodeError:
                                    continue
            except Exception as e:
                logger.error(f"Async generation failed: {str(e)}")
                yield f"[Error: {str(e)}]"
            
            if self.token_stats and token_count > 0:
                self.token_stats.add_prompt(self._estimate_tokens(prompt))
                self.token_stats.add_completion(token_count)
    
    def generate_non_streaming(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_context: int = 4096,
    ) -> str:
        """
        Generate a complete response without streaming.
        
        Args:
            prompt: Input prompt for generation
            temperature: Temperature parameter for generation
            max_context: Maximum context window size
            
        Returns:
            Complete generated response
        """
        with self.monitor.measure("llm.generate_non_streaming"):
            try:
                response = self.session.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_ctx": max_context,
                        },
                    },
                    timeout=self.timeout,
                ).json()
                
                result = response.get("response", "")
                
                # Update token stats
                if self.token_stats:
                    self.token_stats.add_prompt(response.get("prompt_eval_count", self._estimate_tokens(prompt)))
                    self.token_stats.add_completion(response.get("eval_count", self._estimate_tokens(result)))
                
                return result
            except Exception as e:
                logger.error(f"Non-streaming generation failed: {str(e)}")
                raise
    
    def get_token_stats(self) -> Optional[Dict[str, Any]]:
        """Get token usage statistics."""
        if self.token_stats:
            return self.token_stats.to_dict()
        return None
    
    def reset_token_stats(self) -> None:
        """Reset token statistics."""
        if self.token_stats:
            self.token_stats = TokenStats()
    
    def close(self) -> None:
        """Close the session."""
        self.session.close()
