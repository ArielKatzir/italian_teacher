"""
Open source language model implementations for Italian Teacher.

This module provides implementations for popular open source models:
- Llama 3.1 (Meta's latest model, excellent for Italian)
- Mistral 7B (Multilingual, good performance/cost ratio)
- Gemma 2B (Google's efficient model)

All models are free to use with no usage limits!
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .base_model import BaseLLM, ModelConfig, ModelResponse, ModelType
from .model_config import ModelConfigError, load_model_config

logger = logging.getLogger(__name__)


class OpenSourceModelManager:
    """Manager for open source model installations and availability."""

    RECOMMENDED_MODELS = {
        "llama3.1-8b": {
            "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "description": "Meta's latest Llama model, excellent for Italian conversations",
            "size": "8B parameters (~16GB RAM needed)",
            "italian_quality": "Excellent",
            "speed": "Medium",
            "best_for": "Natural conversations, cultural context",
        },
        "mistral-7b": {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "Mistral's multilingual model, great performance/efficiency",
            "size": "7B parameters (~14GB RAM needed)",
            "italian_quality": "Very Good",
            "speed": "Fast",
            "best_for": "Quick responses, grammar correction",
        },
        "llama3.1-3b": {
            "name": "meta-llama/Meta-Llama-3.1-3B-Instruct",
            "description": "Smaller Llama model, good for limited resources",
            "size": "3B parameters (~6GB RAM needed)",
            "italian_quality": "Good",
            "speed": "Very Fast",
            "best_for": "Resource-constrained environments",
        },
    }

    @classmethod
    def get_recommended_model(cls, resource_level: str = "medium") -> str:
        """Get recommended model based on available resources."""
        if resource_level == "high":
            return "llama3.1-8b"
        elif resource_level == "low":
            return "llama3.1-3b"
        else:
            return "mistral-7b"

    @classmethod
    def get_model_info(cls, model_key: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        return cls.RECOMMENDED_MODELS.get(model_key, {})


class LlamaModel(BaseLLM):
    """Llama model implementation using transformers library."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.generation_config = None

    async def load_model(self) -> bool:
        """Load Llama model using HuggingFace transformers."""
        try:
            logger.info(f"Loading Llama model: {self.config.model_name}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, cache_dir=self.config.cache_dir, trust_remote_code=True
            )

            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Determine device
            device = self._get_device()

            # Load model with appropriate settings
            model_kwargs = {
                "cache_dir": self.config.cache_dir,
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            }

            # Add quantization if specified
            if self.config.quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif self.config.quantization == "8bit":
                model_kwargs["load_in_8bit"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, **model_kwargs
            )

            # Move to device if not quantized
            if not self.config.quantization:
                self.model = self.model.to(device)

            # Set up generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            self._is_loaded = True
            logger.info(f"Successfully loaded Llama model on {device}")
            return True

        except ImportError as e:
            logger.error(f"Missing dependencies for Llama model: {e}")
            logger.info("Install with: pip install transformers torch accelerate")
            return False
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            return False

    async def generate_response(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> ModelResponse:
        """Generate response using loaded Llama model."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        try:
            # Prepare system prompt
            if system_prompt is None and self.config.italian_system_prompt:
                system_prompt = self.get_italian_system_prompt("Marco")

            # Format prompt for Llama 3.1 instruction format
            formatted_prompt = self._format_prompt(prompt, system_prompt)

            # Tokenize
            inputs = self.tokenizer(
                formatted_prompt, return_tensors="pt", truncation=True, max_length=2048
            )

            # Move to model device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, generation_config=self.generation_config, **kwargs
                )

            # Decode response
            generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            response_time = time.time() - start_time

            return ModelResponse(
                text=response_text.strip(),
                metadata={
                    "model": self.config.model_name,
                    "prompt_length": len(formatted_prompt),
                    "generation_config": self.generation_config.to_dict(),
                },
                model_used=self.config.model_name,
                tokens_used=len(generated_tokens),
                response_time=response_time,
            )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Llama 3.1 instruction format."""
        if system_prompt:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def _get_device(self) -> str:
        """Determine the best device for the model."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    async def is_available(self) -> bool:
        """Check if model is available."""
        return self._is_loaded and self.model is not None

    async def unload_model(self) -> bool:
        """Unload model from memory."""
        try:
            if self.model is not None:
                del self.model
                del self.tokenizer
                self.model = None
                self.tokenizer = None

                # Clear CUDA cache if available
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

            self._is_loaded = False
            logger.info("Model unloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False


class MistralModel(LlamaModel):
    """Mistral model implementation (inherits from Llama as they use similar architecture)."""

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt for Mistral instruction format."""
        if system_prompt:
            return f"""<s>[INST] {system_prompt}

{prompt} [/INST]"""
        else:
            return f"""<s>[INST] {prompt} [/INST]"""


class MockLocalModel(BaseLLM):
    """Mock model for testing without actual model loading."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    async def load_model(self) -> bool:
        """Mock load - always succeeds."""
        logger.info(f"Mock loading model: {self.config.model_name}")
        await asyncio.sleep(0.1)  # Simulate loading time
        self._is_loaded = True
        return True

    async def generate_response(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> ModelResponse:
        """Generate mock response for testing."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded")

        # Simple mock responses
        mock_responses = [
            "Ciao! Come stai? I'm happy to help you practice Italian!",
            "Molto bene! Let's continue our conversation. What would you like to talk about?",
            "That's a great question! In Italian, we would say...",
            "Fantastico! Your Italian is improving. Keep practicing!",
            "Interessante! Tell me more about that. (Mock response for testing)",
        ]

        import random

        response_text = random.choice(mock_responses)

        return ModelResponse(
            text=response_text,
            metadata={"mock": True, "system_prompt": system_prompt is not None},
            model_used=self.config.model_name,
            tokens_used=len(response_text.split()),
            response_time=0.1,
        )

    async def is_available(self) -> bool:
        """Mock availability check."""
        return self._is_loaded

    async def unload_model(self) -> bool:
        """Mock unload."""
        self._is_loaded = False
        return True


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get information about all available open source models."""
    return OpenSourceModelManager.RECOMMENDED_MODELS


def create_model(model_key: str, **config_overrides) -> BaseLLM:
    """
    Create a model instance by key, loading configuration from YAML files.

    Args:
        model_key: Model identifier (e.g., 'llama3.1-8b', 'mistral-7b', 'mock')
        **config_overrides: Override any configuration values

    Returns:
        BaseLLM instance configured from YAML file

    Raises:
        ValueError: If model_key is not found in configs
    """
    try:
        # Load configuration from YAML file
        config = load_model_config(model_key)

        # Apply any overrides
        if config_overrides:
            config_dict = config.model_dump()
            config_dict.update(config_overrides)
            config = ModelConfig(**config_dict)

        # Create appropriate model class based on model type
        if config.model_type == ModelType.MISTRAL or "mistral" in model_key:
            return MistralModel(config)
        elif (
            config.model_type == ModelType.LOCAL
            or "mock" in model_key
            or config_overrides.get("mock", False)
        ):
            return MockLocalModel(config)
        else:
            return LlamaModel(config)

    except ModelConfigError as e:
        # Fall back to hardcoded configs if YAML loading fails
        logger.warning(f"Failed to load config for {model_key}: {e}")
        logger.info("Falling back to hardcoded configuration...")

        if model_key not in OpenSourceModelManager.RECOMMENDED_MODELS:
            raise ValueError(f"Unknown model: {model_key} and no fallback available")

        model_info = OpenSourceModelManager.RECOMMENDED_MODELS[model_key]
        model_name = model_info["name"]

        # Default config for fallback
        config_dict = {
            "model_name": model_name,
            "model_type": ModelType.LLAMA if "llama" in model_key else ModelType.MISTRAL,
            **config_overrides,
        }
        config = ModelConfig(**config_dict)

        # Create appropriate model class
        if "mistral" in model_key:
            return MistralModel(config)
        elif "mock" in model_key or config_overrides.get("mock", False):
            return MockLocalModel(config)
        else:
            return LlamaModel(config)
