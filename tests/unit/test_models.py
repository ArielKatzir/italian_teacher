"""
Unit tests for language model implementations.

Tests verify that all model classes implement the BaseLLM interface correctly
and that the open source models work as expected.
"""

import pytest

from models import (
    LlamaModel,
    MistralModel,
    MockLocalModel,
    ModelConfig,
    ModelResponse,
    ModelType,
    OpenSourceModelManager,
    create_model,
    get_available_models,
)


class TestModelConfig:
    """Test model configuration validation."""

    def test_basic_config(self):
        """Test basic model configuration."""
        config = ModelConfig(model_name="test-model", model_type=ModelType.LLAMA)

        assert config.model_name == "test-model"
        assert config.model_type == ModelType.LLAMA
        assert config.max_tokens == 150  # default
        assert config.temperature == 0.7  # default

    def test_config_with_overrides(self):
        """Test model configuration with custom values."""
        config = ModelConfig(
            model_name="custom-model",
            model_type=ModelType.MISTRAL,
            max_tokens=500,
            temperature=0.9,
            device="cuda",
            quantization="4bit",
        )

        assert config.max_tokens == 500
        assert config.temperature == 0.9
        assert config.device == "cuda"
        assert config.quantization == "4bit"

    def test_config_validation(self):
        """Test that config validation works."""
        # Test temperature bounds
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", model_type=ModelType.LLAMA, temperature=3.0)  # Too high

        # Test max_tokens bounds
        with pytest.raises(ValueError):
            ModelConfig(model_name="test", model_type=ModelType.LLAMA, max_tokens=0)  # Too low


class TestMockLocalModel:
    """Test the mock model implementation."""

    @pytest.fixture
    def mock_config(self):
        """Create a test configuration for mock model."""
        return ModelConfig(model_name="mock-test-model", model_type=ModelType.LOCAL, max_tokens=100)

    @pytest.fixture
    async def loaded_mock_model(self, mock_config):
        """Create and load a mock model."""
        model = MockLocalModel(mock_config)
        await model.load_model()
        return model

    def test_mock_model_initialization(self, mock_config):
        """Test mock model initialization."""
        model = MockLocalModel(mock_config)

        assert model.model_name == "mock-test-model"
        assert model.model_type == ModelType.LOCAL
        assert not model.is_loaded

    async def test_mock_model_loading(self, mock_config):
        """Test mock model loading."""
        model = MockLocalModel(mock_config)

        # Initially not loaded
        assert not model.is_loaded
        assert not await model.is_available()

        # Load model
        result = await model.load_model()
        assert result is True
        assert model.is_loaded
        assert await model.is_available()

    async def test_mock_model_response_generation(self, loaded_mock_model):
        """Test mock model response generation."""
        response = await loaded_mock_model.generate_response("Hello!")

        assert isinstance(response, ModelResponse)
        assert len(response.text) > 0
        assert response.model_used == "mock-test-model"
        assert response.tokens_used is not None
        assert response.response_time is not None
        assert response.metadata["mock"] is True

    async def test_mock_model_with_system_prompt(self, loaded_mock_model):
        """Test mock model with system prompt."""
        response = await loaded_mock_model.generate_response(
            "Ciao!", system_prompt="You are Marco, an Italian teacher"
        )

        assert response.metadata["system_prompt"] is True

    async def test_mock_model_unloading(self, loaded_mock_model):
        """Test mock model unloading."""
        assert loaded_mock_model.is_loaded

        result = await loaded_mock_model.unload_model()
        assert result is True
        assert not loaded_mock_model.is_loaded

    async def test_mock_model_error_when_not_loaded(self, mock_config):
        """Test that mock model raises error when not loaded."""
        model = MockLocalModel(mock_config)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.generate_response("Hello!")


class TestBaseLLMInterface:
    """Test the base LLM interface."""

    def test_italian_system_prompt(self):
        """Test Italian system prompt generation."""
        config = ModelConfig(model_name="test", model_type=ModelType.LOCAL)
        model = MockLocalModel(config)

        prompt = model.get_italian_system_prompt("Marco")

        assert "Marco" in prompt
        assert "Italian" in prompt
        assert "encouraging" in prompt
        assert "cultural context" in prompt

    def test_model_info(self):
        """Test model info retrieval."""
        config = ModelConfig(model_name="test-model", model_type=ModelType.LLAMA, max_tokens=200)
        model = MockLocalModel(config)

        info = model.get_model_info()

        assert info["name"] == "test-model"
        assert info["type"] == "llama"
        assert info["loaded"] is False
        assert info["config"]["max_tokens"] == 200


class TestLlamaModel:
    """Test Llama model implementation."""

    @pytest.fixture
    def llama_config(self):
        """Create a test configuration for Llama model."""
        return ModelConfig(
            model_name="meta-llama/Meta-Llama-3.1-3B-Instruct",
            model_type=ModelType.LLAMA,
            device="cpu",  # Use CPU for testing
            max_tokens=50,
        )

    def test_llama_initialization(self, llama_config):
        """Test Llama model initialization."""
        model = LlamaModel(llama_config)

        assert model.model_name == "meta-llama/Meta-Llama-3.1-3B-Instruct"
        assert model.model_type == ModelType.LLAMA
        assert not model.is_loaded

    def test_llama_prompt_formatting(self, llama_config):
        """Test Llama prompt formatting."""
        model = LlamaModel(llama_config)

        # Test without system prompt
        formatted = model._format_prompt("Hello!")
        assert "<|begin_of_text|>" in formatted
        assert "<|start_header_id|>user<|end_header_id|>" in formatted
        assert "Hello!" in formatted

        # Test with system prompt
        formatted_with_system = model._format_prompt("Hello!", "You are Marco")
        assert "<|start_header_id|>system<|end_header_id|>" in formatted_with_system
        assert "You are Marco" in formatted_with_system

    def test_llama_device_detection(self, llama_config):
        """Test device detection logic."""
        model = LlamaModel(llama_config)

        # Test explicit device
        llama_config.device = "cpu"
        device = model._get_device()
        assert device == "cpu"

        # Test auto detection (will be cpu in test environment)
        llama_config.device = "auto"
        device = model._get_device()
        assert device in ["cpu", "cuda", "mps"]

    @pytest.mark.skipif(
        True,  # Skip by default as it requires large model download
        reason="Requires model download and significant resources",
    )
    async def test_llama_model_loading(self, llama_config):
        """Test actual Llama model loading (resource intensive)."""
        model = LlamaModel(llama_config)

        # This test requires transformers, torch, etc. to be installed
        # and will download a large model file
        try:
            result = await model.load_model()
            if result:
                assert model.is_loaded
                await model.unload_model()
        except ImportError:
            pytest.skip("Missing dependencies for Llama model")


class TestMistralModel:
    """Test Mistral model implementation."""

    @pytest.fixture
    def mistral_config(self):
        """Create a test configuration for Mistral model."""
        return ModelConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            model_type=ModelType.MISTRAL,
            device="cpu",
        )

    def test_mistral_initialization(self, mistral_config):
        """Test Mistral model initialization."""
        model = MistralModel(mistral_config)

        assert model.model_name == "mistralai/Mistral-7B-Instruct-v0.2"
        assert model.model_type == ModelType.MISTRAL

    def test_mistral_prompt_formatting(self, mistral_config):
        """Test Mistral prompt formatting."""
        model = MistralModel(mistral_config)

        # Test without system prompt
        formatted = model._format_prompt("Hello!")
        assert "<s>[INST]" in formatted
        assert "Hello!" in formatted
        assert "[/INST]" in formatted

        # Test with system prompt
        formatted_with_system = model._format_prompt("Hello!", "You are Marco")
        assert "You are Marco" in formatted_with_system
        assert "Hello!" in formatted_with_system


class TestOpenSourceModelManager:
    """Test open source model management utilities."""

    def test_get_available_models(self):
        """Test getting available model information."""
        models = get_available_models()

        assert isinstance(models, dict)
        assert len(models) > 0

        # Check that recommended models are included
        assert "llama3.1-8b" in models
        assert "mistral-7b" in models
        assert "llama3.1-3b" in models

        # Check model info structure
        for model_key, info in models.items():
            assert "name" in info
            assert "description" in info
            assert "italian_quality" in info

    def test_get_recommended_model(self):
        """Test model recommendation based on resources."""
        # Test different resource levels
        high_resource = OpenSourceModelManager.get_recommended_model("high")
        assert high_resource == "llama3.1-8b"

        medium_resource = OpenSourceModelManager.get_recommended_model("medium")
        assert medium_resource == "mistral-7b"

        low_resource = OpenSourceModelManager.get_recommended_model("low")
        assert low_resource == "llama3.1-3b"

    def test_get_model_info(self):
        """Test getting specific model information."""
        info = OpenSourceModelManager.get_model_info("mistral-7b")

        assert info["name"] == "mistralai/Mistral-7B-Instruct-v0.2"
        assert "italian_quality" in info
        assert "speed" in info

        # Test unknown model
        unknown_info = OpenSourceModelManager.get_model_info("unknown-model")
        assert unknown_info == {}

    def test_create_model(self):
        """Test model creation by key."""
        # Test Llama model creation
        llama_model = create_model("llama3.1-3b")
        assert isinstance(llama_model, LlamaModel)
        assert "llama" in llama_model.model_name.lower()

        # Test Mistral model creation
        mistral_model = create_model("mistral-7b")
        assert isinstance(mistral_model, MistralModel)
        assert "mistral" in mistral_model.model_name.lower()

        # Test mock model creation
        mock_model = create_model("llama3.1-3b", mock=True)
        assert isinstance(mock_model, MockLocalModel)

        # Test unknown model
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("unknown-model")

    def test_create_model_with_overrides(self):
        """Test model creation with configuration overrides."""
        model = create_model("mistral-7b", max_tokens=500, temperature=0.9, device="cuda")

        assert model.config.max_tokens == 500
        assert model.config.temperature == 0.9
        assert model.config.device == "cuda"


if __name__ == "__main__":
    pytest.main([__file__])
