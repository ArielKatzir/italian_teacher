"""
Language Models for Italian Teacher

This module provides interfaces and implementations for language models
used by the Italian Teacher agents.
"""

from .base_model import BaseLLM, ModelConfig, ModelResponse, ModelType
from .model_config import (
    ModelConfigLoader,
)
from .model_config import get_available_models as get_available_model_configs
from .model_config import (
    load_model_config,
)
from .open_source_models import (
    LlamaModel,
    MistralModel,
    MockLocalModel,
    OpenSourceModelManager,
    create_model,
    get_available_models,
)

__all__ = [
    "BaseLLM",
    "ModelConfig",
    "ModelResponse",
    "ModelType",
    "ModelConfigLoader",
    "load_model_config",
    "get_available_model_configs",
    "LlamaModel",
    "MistralModel",
    "MockLocalModel",
    "OpenSourceModelManager",
    "get_available_models",
    "create_model",
]
