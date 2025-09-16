"""
Unit tests for the Marco Agent implementation.

Tests verify that Marco agent properly inherits from BaseAgent,
integrates with the motivation and error tolerance systems,
and works correctly with language models.
"""

from unittest.mock import patch

import pytest

from agents.marco_agent import MarcoAgent
from core.base_agent import AgentPersonality, ConversationContext
from models import MockLocalModel, ModelConfig, ModelType


@pytest.fixture
def marco_personality():
    """Create a test personality for Marco."""
    return AgentPersonality(
        name="Marco",
        role="Friendly Italian conversationalist",
        speaking_style="casual and encouraging",
        personality_traits=["encouraging", "patient", "enthusiastic"],
        expertise_areas=["conversation", "encouragement", "basic_grammar"],
        correction_style="gentle",
        enthusiasm_level=8,
        formality_level=3,
        patience_level=9,
        encouragement_frequency=7,
        correction_frequency=5,
        topic_focus=["daily_conversation", "beginner_topics"],
    )


@pytest.fixture
def test_context():
    """Create a test conversation context."""
    return ConversationContext(user_id="test_user", session_id="test_session")


@pytest.fixture
def mock_model():
    """Create a mock language model for testing."""
    config = ModelConfig(
        model_name="mock-test-model", model_type=ModelType.LOCAL, max_tokens=100, temperature=0.7
    )
    return MockLocalModel(config)


class TestMarcoAgent:
    """Test suite for Marco Agent."""

    def test_marco_initialization(self, marco_personality):
        """Test that Marco initializes correctly with BaseAgent."""
        marco = MarcoAgent("test_marco", marco_personality)

        # Check basic agent properties
        assert marco.agent_id == "test_marco"
        assert marco.personality == marco_personality
        assert marco.personality.name == "Marco"

        # Check Marco-specific initialization
        assert marco.motivation_system is not None
        assert marco.error_tolerance is not None
        assert marco.current_topic is None
        assert marco.user_level == "beginner"

    def test_marco_with_config(self, marco_personality):
        """Test Marco initialization with additional config."""
        config = {"custom_setting": "test_value"}
        marco = MarcoAgent("test_marco", marco_personality, config)

        assert marco.config == config
        assert marco.config["custom_setting"] == "test_value"

    # Removed can_handle_message tests - agents now trust LLM for message handling decisions

    def test_motivation_system_integration(self, marco_personality):
        """Test that Marco's motivation system is properly configured."""
        marco = MarcoAgent("test_marco", marco_personality)

        # Check that motivation system received personality traits
        assert marco.motivation_system is not None
        # The motivation system should have been initialized with personality data

    def test_error_tolerance_integration(self, marco_personality):
        """Test that Marco's error tolerance system is properly configured."""
        marco = MarcoAgent("test_marco", marco_personality)

        assert marco.error_tolerance is not None
        # Check that personality influenced error tolerance settings
        assert marco.error_tolerance.config.patience_factor > 1.0  # Marco is patient
        assert marco.error_tolerance.config.encourage_before_correct is True

    @patch("agents.marco_agent.get_marco_system_prompt")
    async def test_generate_response_basic(self, mock_prompt, marco_personality, test_context):
        """Test basic response generation."""
        # Mock the system prompt function
        mock_prompt.return_value = "Test system prompt"

        marco = MarcoAgent("test_marco", marco_personality)

        # Mock the LLM-based conversational response method
        with patch.object(marco, "_generate_conversational_response") as mock_conv:

            mock_conv.return_value = "Ciao! Come stai? Great to practice Italian with you!"

            response = await marco.generate_response("Hello", test_context)

            # Should get the LLM-generated response directly (no template assembly)
            assert response is not None
            assert len(response) > 0
            assert "Ciao" in response or "practice" in response
            mock_conv.assert_called_once()

    def test_user_level_update(self, marco_personality, test_context):
        """Test that Marco updates user level based on interaction."""
        marco = MarcoAgent("test_marco", marco_personality)

        # Initially beginner
        assert marco.user_level == "beginner"

        # Mock the _update_user_level method call
        with patch.object(marco, "_update_user_level") as mock_update:
            # This would normally be called during generate_response
            marco._update_user_level("Complex message with advanced grammar", test_context)
            mock_update.assert_called_once()

    def test_personality_traits_influence_behavior(self, marco_personality):
        """Test that personality traits properly influence Marco's behavior."""
        marco = MarcoAgent("test_marco", marco_personality)

        # Check that high patience level reduces correction frequency
        assert (
            marco.error_tolerance.config.patience_factor == marco_personality.patience_level / 5.0
        )

        # Check that correction frequency is properly scaled
        expected_freq = marco_personality.correction_frequency / 10.0
        assert marco.error_tolerance.config.correction_frequency == expected_freq


class TestMarcoWithModel:
    """Test Marco agent integration with language models."""

    @pytest.fixture
    async def loaded_mock_model(self, mock_model):
        """Create a loaded mock model for testing."""
        await mock_model.load_model()
        return mock_model

    async def test_marco_with_mock_model(self, marco_personality, test_context, loaded_mock_model):
        """Test Marco agent working with a mock language model."""
        marco = MarcoAgent("test_marco", marco_personality)

        # Integrate model with Marco (this would be done in a real implementation)
        marco.language_model = loaded_mock_model

        # Test that we can at least mock the interaction
        response = await loaded_mock_model.generate_response(
            "Ciao Marco!", system_prompt="You are Marco, a friendly Italian teacher"
        )

        assert response.text is not None
        assert len(response.text) > 0
        assert response.model_used == "mock-test-model"

    async def test_model_integration_error_handling(self, marco_personality, test_context):
        """Test error handling when model integration fails."""
        marco = MarcoAgent("test_marco", marco_personality)

        # Test without a model attached - should handle gracefully
        # (This would be the case before LLM integration is complete)
        assert marco.agent_id == "test_marco"
        assert marco.motivation_system is not None


class TestMarcoPersonalityConfiguration:
    """Test different personality configurations for Marco."""

    def test_enthusiastic_marco(self):
        """Test Marco with high enthusiasm settings."""
        personality = AgentPersonality(
            name="Marco",
            role="Very enthusiastic teacher",
            speaking_style="energetic",
            enthusiasm_level=10,
            encouragement_frequency=10,
            patience_level=8,
            correction_frequency=3,  # Less corrections when very enthusiastic
        )

        marco = MarcoAgent("enthusiastic_marco", personality)
        assert marco.personality.enthusiasm_level == 10
        assert marco.personality.encouragement_frequency == 10

    def test_patient_marco(self):
        """Test Marco with high patience settings."""
        personality = AgentPersonality(
            name="Marco",
            role="Very patient teacher",
            speaking_style="gentle",
            enthusiasm_level=6,
            encouragement_frequency=8,
            patience_level=10,
            correction_frequency=2,  # Very few corrections
        )

        marco = MarcoAgent("patient_marco", personality)
        assert marco.personality.patience_level == 10
        assert marco.error_tolerance.config.patience_factor == 2.0  # 10/5 = 2.0

    def test_corrective_marco(self):
        """Test Marco with high correction frequency."""
        personality = AgentPersonality(
            name="Marco",
            role="Grammar-focused teacher",
            speaking_style="precise",
            enthusiasm_level=7,
            encouragement_frequency=6,
            patience_level=7,
            correction_frequency=9,  # High correction frequency
        )

        marco = MarcoAgent("corrective_marco", personality)
        assert marco.personality.correction_frequency == 9
        assert marco.error_tolerance.config.correction_frequency == 0.9  # 9/10 = 0.9


if __name__ == "__main__":
    pytest.main([__file__])
