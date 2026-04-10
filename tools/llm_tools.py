"""
LLM-related tools for Agent framework.
"""
from typing import Dict, Any, Optional, List

from .base import Tool, ToolResult


class TextGenerationTool(Tool):
    """Tool for generating text using LLM."""
    
    def __init__(self, llm_service) -> None:
        super().__init__(
            name="text_generation",
            description="Generate text responses using the LLM model",
        )
        self.llm_service = llm_service
    
    def execute(
        self, 
        prompt: str, 
        temperature: float = 0.3,
        max_context: int = 4096,
        stream: bool = False,
    ) -> ToolResult:
        """Execute text generation."""
        try:
            if not self.llm_service:
                return ToolResult(
                    success=False,
                    error="LLM service not available",
                )
            
            if stream:
                # For streaming, collect all tokens
                tokens = []
                for token in self.llm_service.generate_response(
                    prompt=prompt,
                    temperature=temperature,
                    max_context=max_context,
                ):
                    tokens.append(token)
                response = "".join(tokens)
            else:
                # Non-streaming mode would need a different API call
                # For now, use streaming and collect
                tokens = []
                for token in self.llm_service.generate_response(
                    prompt=prompt,
                    temperature=temperature,
                    max_context=max_context,
                ):
                    tokens.append(token)
                response = "".join(tokens)
            
            return ToolResult(
                success=True,
                data={
                    "prompt": prompt,
                    "response": response,
                    "temperature": temperature,
                },
                metadata={
                    "token_count": len(response.split()),
                    "streamed": stream,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Text generation failed: {str(e)}",
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "prompt": {"type": "string", "description": "Input prompt for generation"},
            "temperature": {"type": "float", "description": "Temperature for randomness (0.0-1.0)", "default": 0.3},
            "max_context": {"type": "integer", "description": "Maximum context window size", "default": 4096},
            "stream": {"type": "boolean", "description": "Whether to stream the response", "default": False},
        }
    
    def _get_return_type(self) -> str:
        return "Generated text response"


class QueryExpansionTool(Tool):
    """Tool for expanding queries using HyDE (Hypothetical Document Embeddings)."""
    
    def __init__(self, llm_service) -> None:
        super().__init__(
            name="query_expansion",
            description="Expand a query by generating hypothetical answers for better retrieval",
        )
        self.llm_service = llm_service
    
    def execute(self, query: str, chat_history: str = "") -> ToolResult:
        """Execute query expansion using HyDE."""
        try:
            if not self.llm_service:
                return ToolResult(
                    success=False,
                    error="LLM service not available",
                )
            
            # Combine with chat history if provided
            combined_query = f"{chat_history}\n{query}" if chat_history else query
            
            # Generate hypothetical answer
            hypothetical = self.llm_service.generate_hypothetical_answer(combined_query)
            
            # Return expanded query
            expanded_query = f"{combined_query}\n{hypothetical}"
            
            return ToolResult(
                success=True,
                data={
                    "original_query": query,
                    "expanded_query": expanded_query,
                    "hypothetical_answer": hypothetical,
                    "has_chat_history": bool(chat_history),
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Query expansion failed: {str(e)}",
            )
    
    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {"type": "string", "description": "Original query to expand"},
            "chat_history": {"type": "string", "description": "Previous conversation context", "default": ""},
        }
    
    def _get_return_type(self) -> str:
        return "Expanded query with hypothetical answer"
