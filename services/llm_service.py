"""
LLM service for GraphSeek application.
Handles interactions with Ollama API for text generation and embeddings.
"""
import json
from typing import Generator, Optional
import requests


class LLMService:
    """Service for interacting with Ollama LLM API."""
    
    def __init__(self, api_url: str, model: str) -> None:
        self.api_url = api_url
        self.model = model
    
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
            response = requests.get(f"{base_url}/api/tags")
            available_models = [model['model'] for model in response.json()['models']]
            
            missing_models = [model for model in required_models if model not in available_models]
            
            return {
                "available": len(missing_models) == 0,
                "missing_models": missing_models,
                "all_models": available_models,
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
            }
    
    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer for HyDE query expansion.
        
        Args:
            query: Original query string
            
        Returns:
            Generated hypothetical answer or original query if failed
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": f"Generate a hypothetical answer to: {query}",
                    "stream": False,
                },
            ).json()
            return response.get("response", query)
        except Exception:
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
        response = requests.post(
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
        )
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode())
                token = data.get("response", "")
                yield token
                
                if data.get("done", False):
                    break
