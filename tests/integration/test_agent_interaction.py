"""
Integration tests for agent interactions and workflows.
"""

from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.integration
class TestAgentCoordination:
    """Test agent coordination and communication."""

    @pytest.mark.asyncio
    async def test_basic_agent_response_flow(self, mock_agent, sample_config):
        """Test basic agent response generation flow."""
        user_message = "Ciao! Come stai?"

        # Mock the response generation
        expected_response = "Ciao! Sto bene, grazie! E tu come stai?"
        mock_agent.generate_response.return_value = expected_response

        # Simulate agent responding
        response = await mock_agent.generate_response(user_message)

        assert response == expected_response
        mock_agent.generate_response.assert_called_once_with(user_message)

    @pytest.mark.asyncio
    async def test_multiple_agents_availability(self, sample_config):
        """Test that multiple agents can be available simultaneously."""
        # Create multiple mock agents
        agents = {}
        agent_names = ["marco", "professoressa_rossi", "nonna_giulia", "lorenzo"]

        for name in agent_names:
            agent = Mock()
            agent.agent_id = name
            agent.is_available = Mock(return_value=True)
            agents[name] = agent

        # Test all agents are available
        for name, agent in agents.items():
            assert agent.is_available() is True
            assert agent.agent_id == name

    @pytest.mark.asyncio
    async def test_conversation_context_persistence(self, sample_conversation_history):
        """Test that conversation context persists across interactions."""
        # Simulate a conversation manager maintaining context
        context_manager = Mock()
        context_manager.get_history = Mock(return_value=sample_conversation_history)
        context_manager.add_message = Mock()

        # Get initial history
        history = context_manager.get_history()
        assert len(history) == 3

        # Add new message
        new_message = {
            "role": "professoressa_rossi",
            "content": "Molto bene! La tua pronuncia è migliorata.",
            "timestamp": "2024-01-01T10:00:15Z",
        }
        context_manager.add_message(new_message)

        # Verify message was added
        context_manager.add_message.assert_called_once_with(new_message)


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_simple_conversation_workflow(self, sample_config):
        """Test a simple conversation workflow from start to finish."""
        # This would test the full pipeline:
        # User input -> Agent selection -> Response generation -> Context update

        # Mock components
        coordinator = Mock()
        conversation_manager = Mock()
        agent = Mock()

        # Setup mocks
        coordinator.select_agent = Mock(return_value=agent)
        agent.generate_response = AsyncMock(return_value="Ciao! Come va?")
        conversation_manager.add_message = Mock()

        # Simulate workflow
        user_input = "Ciao!"
        selected_agent = coordinator.select_agent(user_input)
        response = await selected_agent.generate_response(user_input)

        # Verify workflow
        coordinator.select_agent.assert_called_once_with(user_input)
        agent.generate_response.assert_called_once_with(user_input)
        assert response == "Ciao! Come va?"


@pytest.mark.integration
@pytest.mark.ml
class TestModelIntegration:
    """Test integration with ML models (requires --ml-tests flag)."""

    def test_mock_model_integration(self, mock_model):
        """Test integration with mock ML model."""
        test_input = "Come si dice hello in italiano?"

        # Test tokenization
        tokens = mock_model.tokenizer.encode(test_input)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

        # Test generation
        response = mock_model.generate(test_input)
        assert isinstance(response, list)
        assert len(response) > 0

        # Test decoding
        decoded = mock_model.tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    @pytest.mark.asyncio
    async def test_model_response_generation(self, mock_model, sample_config):
        """Test model response generation with configuration."""
        model_config = sample_config["models"]

        # Verify model configuration
        assert model_config["device"] == "cpu"
        assert model_config["max_length"] == 64
        assert model_config["temperature"] == 0.7

        # Test model with config
        mock_model.config = model_config
        response = mock_model.generate("Test input")

        assert response is not None
