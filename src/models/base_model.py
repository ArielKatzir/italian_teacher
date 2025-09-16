"""
Base language model interface for Italian Teacher agents.

This module defines the abstract interface that all language models must implement,
ensuring consistent behavior across different model providers (OpenAI, open source, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelType(Enum):
    """Types of language models."""

    OPENAI = "openai"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    LOCAL = "local"


@dataclass
class ModelResponse:
    """Response from a language model."""

    text: str
    metadata: Dict[str, Any]
    model_used: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    confidence: Optional[float] = None


class ModelConfig(BaseModel):
    """Configuration for language models."""

    model_name: str = Field(..., description="Name/ID of the model")
    model_type: ModelType = Field(..., description="Type of model (openai, llama, etc.)")

    # Generation parameters
    max_tokens: int = Field(default=150, ge=1, le=4000, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")

    # Model-specific settings
    device: str = Field(default="auto", description="Device to run model on (cpu, cuda, auto)")
    quantization: Optional[str] = Field(default=None, description="Model quantization (4bit, 8bit)")
    cache_dir: Optional[str] = Field(default=None, description="Directory for model cache")

    # API settings (for hosted models)
    api_key: Optional[str] = Field(default=None, description="API key if required")
    base_url: Optional[str] = Field(default=None, description="Base URL for API")

    # Italian-specific settings
    italian_system_prompt: bool = Field(
        default=True, description="Use Italian-optimized system prompt"
    )
    cultural_context: bool = Field(default=True, description="Include Italian cultural context")


class BaseLLM(ABC):
    """Abstract base class for all language models used in Italian Teacher."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model_name = config.model_name
        self.model_type = config.model_type
        self._is_loaded = False

    @abstractmethod
    async def load_model(self) -> bool:
        """Load the language model into memory."""

    @abstractmethod
    async def generate_response(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> ModelResponse:
        """Generate a response from the model."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the model is available and ready."""

    @abstractmethod
    async def unload_model(self) -> bool:
        """Unload the model from memory."""

    def get_italian_system_prompt(self, agent_name: str = "Assistant") -> str:
        """Get Italian-optimized system prompt."""
        return f"""You are {agent_name}, a friendly Italian language teacher and conversation partner.

Key guidelines:
- Help users learn Italian through natural conversation
- Be encouraging and patient with mistakes
- Provide gentle corrections when needed
- Include cultural context when appropriate
- Mix Italian and English naturally based on user level
- Be enthusiastic about Italian language and culture

Always respond in a warm, encouraging manner that makes learning enjoyable."""

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._is_loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "name": self.model_name,
            "type": self.model_type.value,
            "loaded": self._is_loaded,
            "config": self.config.dict(),
        }
