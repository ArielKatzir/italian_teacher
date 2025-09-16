"""
Unit tests for the BaseAgent class and related components.

These tests verify that the core agent functionality works correctly,
including message handling, status management, and personality configuration.
"""

from datetime import datetime
from typing import Optional, Set
from unittest.mock import AsyncMock

import pytest

from core import (
    AgentMessage,
    AgentPersonality,
    AgentStatus,
    BaseAgent,
    ConversationContext,
    MessageType,
)
from core.agent_events import AgentEvent, AgentResponse, EventType
from core.conversation_state import ConversationStateManager, InMemoryConversationStore


# Test agent implementation for testing the abstract base class
class ConcreteTestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def __init__(
        self,
        agent_id: str,
        personality: AgentPersonality,
        config=None,
        event_bus=None,
        conversation_manager=None,
        agent_registry=None,
    ):
        super().__init__(
            agent_id,
            personality,
            config,
            event_bus=event_bus,
            conversation_manager=conversation_manager,
            agent_registry=agent_registry,
        )
        self.generate_response_mock = AsyncMock()

    async def generate_response(
        self, user_message: str, context: Optional[ConversationContext] = None
    ) -> AgentMessage:
        return await self.generate_response_mock(user_message, context)

    # Removed can_handle_message - agents now trust LLM for message handling

    async def handle_event(self, event: AgentEvent) -> Optional[AgentResponse]:
        """Mock event handler for testing."""
        return AgentResponse(
            original_event_id=event.id,
            responder_id=self.agent_id,
            success=True,
            payload={"test_response": "Mock response"},
        )

    def get_handled_event_types(self) -> Set[EventType]:
        """Return empty set for testing."""
        return set()


@pytest.fixture
def sample_personality():
    """Create a sample agent personality for testing."""
    return AgentPersonality(
        name="Test Agent",
        role="Testing Specialist",
        speaking_style="professional",
        personality_traits=["helpful", "precise"],
        expertise_areas=["testing", "debugging"],
        correction_style="direct",
        enthusiasm_level=7,
    )


@pytest.fixture
def sample_context():
    """Create a sample conversation context for testing."""
    return ConversationContext(
        user_id="test_user_123",
        session_id="test_session_456",
        user_language_level="intermediate",
        learning_goals=["grammar", "conversation"],
        current_topic="greetings",
    )


@pytest.fixture
def test_agent(sample_personality):
    """Create a test agent instance."""
    return ConcreteTestAgent("test_agent_1", sample_personality)


@pytest.fixture
def conversation_manager():
    """Create a conversation state manager for testing."""
    return ConversationStateManager(InMemoryConversationStore())


@pytest.fixture
def test_agent_with_conversation_manager(sample_personality, conversation_manager):
    """Create a test agent with conversation state management."""
    return ConcreteTestAgent(
        "test_agent_with_state", sample_personality, conversation_manager=conversation_manager
    )


class TestAgentMessage:
    """Test the AgentMessage dataclass."""

    def test_message_creation(self):
        """Test basic message creation with defaults."""
        message = AgentMessage(
            sender_id="agent_1", recipient_id="user_1", content="Ciao! Come stai?"
        )

        assert message.sender_id == "agent_1"
        assert message.recipient_id == "user_1"
        assert message.content == "Ciao! Come stai?"
        assert message.message_type == MessageType.CONVERSATION
        assert message.priority == 1
        assert isinstance(message.timestamp, datetime)
        assert len(message.id) > 0  # UUID should be generated

    def test_message_with_custom_values(self):
        """Test message creation with custom values."""
        custom_metadata = {"source": "test"}

        message = AgentMessage(
            sender_id="agent_2",
            message_type=MessageType.CORRECTION,
            content="La forma corretta è 'io sono'",
            priority=5,
            metadata=custom_metadata,
        )

        assert message.message_type == MessageType.CORRECTION
        assert message.priority == 5
        assert message.metadata == custom_metadata


class TestAgentPersonality:
    """Test the AgentPersonality dataclass."""

    def test_personality_creation(self, sample_personality):
        """Test personality creation with all fields."""
        assert sample_personality.name == "Test Agent"
        assert sample_personality.role == "Testing Specialist"
        assert sample_personality.speaking_style == "professional"
        assert "helpful" in sample_personality.personality_traits
        assert "testing" in sample_personality.expertise_areas
        assert sample_personality.correction_style == "direct"
        assert sample_personality.enthusiasm_level == 7

    def test_personality_defaults(self):
        """Test personality creation with minimal fields (using defaults)."""
        personality = AgentPersonality(
            name="Minimal Agent", role="Basic Role", speaking_style="casual"
        )

        assert personality.personality_traits == []
        assert personality.expertise_areas == []
        assert personality.response_patterns == {}
        assert personality.cultural_knowledge == {}
        assert personality.correction_style == "gentle"
        assert personality.enthusiasm_level == 5


class TestConversationContext:
    """Test the ConversationContext dataclass."""

    def test_context_creation(self, sample_context):
        """Test context creation with all fields."""
        assert sample_context.user_id == "test_user_123"
        assert sample_context.session_id == "test_session_456"
        assert sample_context.user_language_level == "intermediate"
        assert "grammar" in sample_context.learning_goals
        assert sample_context.current_topic == "greetings"
        assert sample_context.conversation_history == []
        assert sample_context.last_interaction is None


class TestBaseAgent:
    """Test the BaseAgent abstract class functionality."""

    def test_agent_initialization(self, test_agent, sample_personality):
        """Test agent initialization with proper defaults."""
        assert test_agent.agent_id == "test_agent_1"
        assert test_agent.personality == sample_personality
        assert test_agent.status == AgentStatus.INACTIVE
        assert test_agent.context is None
        assert test_agent.activity_log == []
        assert not test_agent._is_initialized
        assert test_agent._error_count == 0
        assert isinstance(test_agent.created_at, datetime)

    @pytest.mark.asyncio
    async def test_agent_initialization_success(self, test_agent):
        """Test successful agent initialization."""
        # Mock the protected methods
        test_agent._load_personality_config = AsyncMock()
        test_agent._prepare_response_templates = AsyncMock()

        result = await test_agent.initialize()

        assert result is True
        assert test_agent._is_initialized is True
        assert test_agent.status == AgentStatus.ACTIVE
        test_agent._load_personality_config.assert_called_once()
        test_agent._prepare_response_templates.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_initialization_failure(self, test_agent):
        """Test agent initialization failure handling."""
        # Mock a method to raise an exception
        test_agent._load_personality_config = AsyncMock(side_effect=Exception("Config error"))

        with pytest.raises(RuntimeError, match="initialization failed"):
            await test_agent.initialize()

        assert test_agent.status == AgentStatus.ERROR
        assert test_agent._error_count == 1
        assert not test_agent._is_initialized

    @pytest.mark.asyncio
    async def test_agent_activation(self, test_agent, sample_context):
        """Test agent activation with context."""
        # Mock initialization
        test_agent._is_initialized = True

        await test_agent.activate(sample_context)

        assert test_agent.context == sample_context
        assert test_agent.status == AgentStatus.ACTIVE
        assert isinstance(test_agent.last_active, datetime)

        # Check that activation was logged in activity log
        assert len(test_agent.activity_log) == 1
        activation_event = test_agent.activity_log[0]
        assert activation_event.event_type == "activated"
        assert "activated" in activation_event.description

    @pytest.mark.asyncio
    async def test_agent_activation_auto_init(self, test_agent, sample_context):
        """Test that activation automatically initializes agent."""
        test_agent._load_personality_config = AsyncMock()
        test_agent._prepare_response_templates = AsyncMock()

        await test_agent.activate(sample_context)

        assert test_agent._is_initialized is True
        assert test_agent.context == sample_context

    @pytest.mark.asyncio
    async def test_agent_deactivation(self, test_agent, sample_context):
        """Test agent deactivation."""
        # Set up agent as active
        test_agent.context = sample_context
        test_agent.status = AgentStatus.ACTIVE
        test_agent._save_conversation_state = AsyncMock()

        await test_agent.deactivate()

        assert test_agent.status == AgentStatus.INACTIVE
        test_agent._save_conversation_state.assert_called_once()

        # Check deactivation was logged in activity log
        deactivation_events = [
            event for event in test_agent.activity_log if event.event_type == "deactivated"
        ]
        assert len(deactivation_events) == 1

    def test_message_conversation_management(self, test_agent, sample_context):
        """Test adding messages to conversation history."""
        test_agent.context = sample_context
        message1 = AgentMessage(sender_id="user", content="Hello")
        message2 = AgentMessage(sender_id="test_agent_1", content="Ciao")

        test_agent.add_message_to_conversation(message1)
        test_agent.add_message_to_conversation(message2)

        assert len(test_agent.context.conversation_history) == 2
        assert test_agent.context.conversation_history[0] == message1
        assert test_agent.context.conversation_history[1] == message2

    def test_conversation_updates_context(self, test_agent, sample_context):
        """Test that adding messages updates context properly."""
        test_agent.context = sample_context
        message = AgentMessage(sender_id="user", content="Test message")

        test_agent.add_message_to_conversation(message)

        assert len(test_agent.context.conversation_history) == 1
        assert test_agent.context.conversation_history[0] == message
        assert test_agent.context.last_interaction is not None

    def test_recent_messages_retrieval(self, test_agent, sample_context):
        """Test getting recent messages from conversation history."""
        test_agent.context = sample_context

        # Add several messages
        for i in range(15):
            message = AgentMessage(sender_id="user", content=f"Message {i}")
            test_agent.add_message_to_conversation(message)

        recent = test_agent.get_recent_messages(5)
        assert len(recent) == 5
        assert recent[-1].content == "Message 14"  # Most recent
        assert recent[0].content == "Message 10"  # 5th most recent

        # Test with no context
        empty_agent = ConcreteTestAgent("empty", test_agent.personality)
        assert empty_agent.get_recent_messages(10) == []

    def test_agent_availability(self, test_agent):
        """Test agent availability checking."""
        # Initially not available (not initialized)
        assert not test_agent.is_available()

        # After initialization and activation
        test_agent._is_initialized = True
        test_agent.status = AgentStatus.ACTIVE
        assert test_agent.is_available()

        # Not available when busy
        test_agent.status = AgentStatus.BUSY
        assert not test_agent.is_available()

        # Not available with too many errors
        test_agent.status = AgentStatus.ACTIVE
        test_agent._error_count = 10
        assert not test_agent.is_available()

    def test_status_info(self, test_agent, sample_context):
        """Test getting agent status information."""
        test_agent.context = sample_context
        test_agent.add_message_to_conversation(AgentMessage(sender_id="user", content="test"))
        test_agent.add_message_to_conversation(
            AgentMessage(sender_id="test_agent_1", content="response")
        )

        status = test_agent.get_status_info()

        assert status["agent_id"] == "test_agent_1"
        assert status["name"] == "Test Agent"
        assert status["role"] == "Testing Specialist"
        assert status["status"] == "inactive"
        assert status["error_count"] == 0
        assert status["conversation_message_count"] == 2
        assert status["my_message_count"] == 1  # Only messages from this agent
        assert status["current_session"] == "test_session_456"
        assert "last_active" in status

    @pytest.mark.asyncio
    async def test_error_handling(self, test_agent):
        """Test error handling functionality."""
        test_error = ValueError("Test error")

        await test_agent.handle_error(test_error, "test context")

        assert test_agent._error_count == 1
        assert test_agent.status == AgentStatus.ACTIVE  # Still active with low error count

        # Check error was logged in activity log
        error_events = [event for event in test_agent.activity_log if event.event_type == "error"]
        assert len(error_events) == 1
        assert "Error in test context" in error_events[0].description

    @pytest.mark.asyncio
    async def test_error_handling_max_errors(self, test_agent):
        """Test that agent goes to error state after max errors."""
        test_error = ValueError("Test error")

        # Add errors up to max
        for i in range(test_agent._max_errors):
            await test_agent.handle_error(test_error, f"context {i}")

        assert test_agent._error_count == test_agent._max_errors
        assert test_agent.status == AgentStatus.ERROR
        assert not test_agent.is_available()

    def test_create_response_message(self, test_agent, sample_context):
        """Test helper method for creating response messages."""
        test_agent.context = sample_context

        message = test_agent._create_response_message(
            "Test response", MessageType.CORRECTION, priority=3, metadata={"test": "value"}
        )

        assert message.sender_id == "test_agent_1"
        assert message.content == "Test response"
        assert message.message_type == MessageType.CORRECTION
        assert message.priority == 3
        assert message.metadata["test"] == "value"

    def test_agent_string_representation(self, test_agent):
        """Test agent string representation."""
        repr_str = repr(test_agent)
        assert "ConcreteTestAgent" in repr_str
        assert "test_agent_1" in repr_str
        assert "Test Agent" in repr_str
        assert "inactive" in repr_str

    @pytest.mark.asyncio
    async def test_abstract_methods_implemented(self, test_agent):
        """Test that abstract methods are properly implemented in test agent."""
        # Only generate_response is required now - can_handle_message removed
        await test_agent.generate_response("Hello")
        test_agent.generate_response_mock.assert_called_once()

    def test_agent_with_custom_config(self, sample_personality):
        """Test agent creation with custom configuration."""
        custom_config = {"max_response_length": 200, "debug": True}
        agent = ConcreteTestAgent("custom_agent", sample_personality, custom_config)

        assert agent.config == custom_config
        assert agent.config["max_response_length"] == 200
        assert agent.config["debug"] is True


class TestBaseAgentConversationState:
    """Test BaseAgent integration with conversation state management."""

    def test_agent_with_conversation_manager(
        self, test_agent_with_conversation_manager, conversation_manager
    ):
        """Test agent initialization with conversation manager."""
        agent = test_agent_with_conversation_manager
        assert agent.conversation_manager is conversation_manager

    def test_agent_with_default_conversation_manager(self, sample_personality):
        """Test that agent gets default conversation manager if none provided."""
        agent = ConcreteTestAgent("test_agent", sample_personality)
        assert agent.conversation_manager is not None
        # Should get the default instance
        from core.conversation_state import default_conversation_manager

        assert agent.conversation_manager is default_conversation_manager

    @pytest.mark.asyncio
    async def test_activation_saves_context(
        self, test_agent_with_conversation_manager, sample_context
    ):
        """Test that agent activation saves conversation context."""
        agent = test_agent_with_conversation_manager
        agent._is_initialized = True

        await agent.activate(sample_context)

        # Verify context was saved to conversation manager
        loaded_context = await agent.conversation_manager.get_context(sample_context.session_id)
        assert loaded_context is not None
        assert loaded_context.session_id == sample_context.session_id
        assert loaded_context.user_id == sample_context.user_id

    @pytest.mark.asyncio
    async def test_deactivation_saves_state(
        self, test_agent_with_conversation_manager, sample_context
    ):
        """Test that agent deactivation saves conversation state."""
        agent = test_agent_with_conversation_manager
        agent.context = sample_context
        agent.status = AgentStatus.ACTIVE

        await agent.deactivate()

        # Verify final state was saved
        loaded_context = await agent.conversation_manager.get_context(sample_context.session_id)
        assert loaded_context is not None

    @pytest.mark.asyncio
    async def test_load_conversation_state(
        self, test_agent_with_conversation_manager, sample_context
    ):
        """Test loading conversation state from persistence."""
        agent = test_agent_with_conversation_manager

        # Save context first
        await agent.conversation_manager.save_context(sample_context)

        # Load it back
        loaded_context = await agent.load_conversation_state(sample_context.session_id)

        assert loaded_context is not None
        assert loaded_context.session_id == sample_context.session_id
        assert loaded_context.user_id == sample_context.user_id
