"""
Configuration management for GraphSeek application.
"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv, find_dotenv


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration settings."""
    llm_model: str
    embeddings_model: str
    cross_encoder_model: str
    ollama_base_url: str
    
    @property
    def ollama_api_url(self) -> str:
        return f"{self.ollama_base_url}/api/generate"


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    bm25_weight: float = 0.4
    faiss_weight: float = 0.6
    default_top_k: int = 5


@dataclass
class AppConfig:
    """Main application configuration."""
    models: ModelConfig
    retrieval: RetrievalConfig = RetrievalConfig()
    device: str = "cpu"
    
    @classmethod
    def from_environment(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        load_dotenv(find_dotenv())
        
        ollama_base_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        model = os.getenv("MODEL", "deepseek-r1:7b")
        
        # Determine device
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        
        return cls(
            models=ModelConfig(
                llm_model=model,
                embeddings_model="nomic-embed-text:latest",
                cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                ollama_base_url=ollama_base_url,
            ),
            device=device,
        )
