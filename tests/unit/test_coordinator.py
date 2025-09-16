"""
Tests for the Coordinator Service implementation.

The new coordinator works as a background service that monitors agent events
and facilitates handoffs, rather than directly handling user messages.
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from core.agent_discovery import AgentDiscoveryService
from core.agent_events import AgentEvent, EventType
from core.agent_registry import AgentCapabilities, AgentRegistration, AgentSpecialization
from core.coordinator import CoordinatorAgent  # Backward compatibility alias
from core.coordinator import (
    ConversationPhase,
    CoordinatorService,
    LearningGoal,
    SessionState,
)
from core.event_bus import AgentEventBus


@pytest.fixture
def mock_discovery_service():
    """Create a mock discovery service."""
    service = AsyncMock(spec=AgentDiscoveryService)

    # Create mock agent registrations
    marco_agent = AgentRegistration(
        agent_id="marco_001",
        agent_name="Marco",
        agent_type="marco",
        capabilities=AgentCapabilities(
            specializations={AgentSpecialization.CONVERSATION},
            confidence_scores={AgentSpecialization.CONVERSATION: 0.9},
        ),
    )

    grammar_agent = AgentRegistration(
        agent_id="professoressa_001",
        agent_name="Professoressa Rossi",
        agent_type="professoressa_rossi",
        capabilities=AgentCapabilities(
            specializations={AgentSpecialization.GRAMMAR},
            confidence_scores={AgentSpecialization.GRAMMAR: 0.95},
        ),
    )

    # Mock the discovery service methods with AgentMatch objects
    from core.agent_registry import AgentMatch

    grammar_match = AgentMatch(
        registration=grammar_agent,
        total_score=0.95,
        specialization_score=0.95,
        confidence_score=0.95,
        load_score=1.0,
        performance_score=1.0,
    )

    service.find_agent_for_handoff.return_value = grammar_match
    service.find_agent_for_help_request.return_value = grammar_match
    service.find_agent_for_correction_review.return_value = grammar_match

    return service


@pytest.fixture
def mock_event_bus():
    """Create a mock event bus."""
    bus = AsyncMock(spec=AgentEventBus)
    bus.subscribe = AsyncMock()
    bus.publish_event = AsyncMock()
    return bus


@pytest.fixture
def coordinator_service(mock_discovery_service, mock_event_bus):
    """Create a coordinator service for testing."""
    return CoordinatorService(
        coordinator_id="test_coordinator",
        discovery_service=mock_discovery_service,
        event_bus=mock_event_bus,
    )


@pytest.fixture
def sample_learning_goals():
    """Create sample learning goals."""
    return [
        LearningGoal(
            goal_type="conversation",
            target_level="beginner",
            specific_topics=["greetings", "introductions"],
            estimated_duration=20,
            priority=1,
        ),
        LearningGoal(
            goal_type="grammar",
            target_level="beginner",
            specific_topics=["present tense"],
            estimated_duration=30,
            priority=2,
        ),
    ]


class TestLearningGoal:
    """Test learning goal creation and validation."""

    def test_learning_goal_creation(self):
        """Test creating a basic learning goal."""
        goal = LearningGoal(goal_type="conversation", target_level="beginner")

        assert goal.goal_type == "conversation"
        assert goal.target_level == "beginner"
        assert goal.specific_topics == []
        assert goal.estimated_duration is None
        assert goal.priority == 1

    def test_learning_goal_with_all_fields(self):
        """Test creating a learning goal with all fields."""
        goal = LearningGoal(
            goal_type="grammar",
            target_level="intermediate",
            specific_topics=["past tense", "subjunctive"],
            estimated_duration=45,
            priority=2,
        )

        assert goal.goal_type == "grammar"
        assert goal.target_level == "intermediate"
        assert goal.specific_topics == ["past tense", "subjunctive"]
        assert goal.estimated_duration == 45
        assert goal.priority == 2


class TestSessionState:
    """Test session state management."""

    def test_session_state_creation(self):
        """Test creating a basic session state."""
        session = SessionState(
            session_id="test_session_123",
            user_id="user123",
            started_at=datetime.now(),
            current_phase=ConversationPhase.GREETING,
            current_agent_id="marco_001",
            learning_goals=[],
            completed_goals=[],
        )

        assert session.session_id == "test_session_123"
        assert session.user_id == "user123"
        assert session.current_phase == ConversationPhase.GREETING
        assert session.current_agent_id == "marco_001"
        assert session.messages_exchanged == 0
        assert session.engagement_score == 0.0
        assert len(session.conversation_history) == 0
        assert len(session.topics_covered) == 0

    def test_session_state_with_goals(self, sample_learning_goals):
        """Test creating session state with learning goals."""
        session = SessionState(
            session_id="test_session_456",
            user_id="user456",
            started_at=datetime.now(),
            current_phase=ConversationPhase.LEARNING,
            current_agent_id="marco_001",
            learning_goals=sample_learning_goals,
            completed_goals=[],
        )

        assert len(session.learning_goals) == 2
        assert session.learning_goals[0].goal_type == "conversation"
        assert session.learning_goals[1].goal_type == "grammar"


class TestCoordinatorService:
    """Test the background coordinator service."""

    def test_coordinator_initialization(self, coordinator_service):
        """Test coordinator service initialization."""
        assert coordinator_service.coordinator_id == "test_coordinator"
        assert coordinator_service.discovery_service is not None
        assert coordinator_service.event_bus is not None
        assert len(coordinator_service.active_sessions) == 0

    def test_coordinator_with_custom_id(self, mock_discovery_service, mock_event_bus):
        """Test coordinator with custom ID."""
        coordinator = CoordinatorService(
            coordinator_id="custom_coordinator",
            discovery_service=mock_discovery_service,
            event_bus=mock_event_bus,
        )

        assert coordinator.coordinator_id == "custom_coordinator"

    @pytest.mark.asyncio
    async def test_create_session(self, coordinator_service, sample_learning_goals):
        """Test creating a new session."""
        session_id = await coordinator_service.create_session(
            user_id="user123", initial_agent_id="marco_001", learning_goals=sample_learning_goals
        )

        assert session_id is not None
        assert session_id in coordinator_service.active_sessions

        session = coordinator_service.active_sessions[session_id]
        assert session.user_id == "user123"
        assert session.current_agent_id == "marco_001"
        assert len(session.learning_goals) == 2
        assert session.current_phase == ConversationPhase.GREETING

    @pytest.mark.asyncio
    async def test_create_session_without_goals(self, coordinator_service):
        """Test creating a session without predefined goals."""
        session_id = await coordinator_service.create_session(
            user_id="user456", initial_agent_id="professoressa_001"
        )

        session = coordinator_service.active_sessions[session_id]
        assert len(session.learning_goals) == 0
        assert session.current_agent_id == "professoressa_001"

    @pytest.mark.asyncio
    async def test_handle_handoff_request(self, coordinator_service, mock_discovery_service):
        """Test handling handoff requests from agents."""
        # Create a session first
        session_id = await coordinator_service.create_session(
            user_id="user123", initial_agent_id="marco_001"
        )

        # Create handoff request event
        handoff_event = AgentEvent(
            sender_id="marco_001",
            event_type=EventType.REQUEST_HANDOFF,
            session_id=session_id,
            payload={
                "required_specialization": AgentSpecialization.GRAMMAR,
                "user_message": "What's the past tense of 'essere'?",
                "reason": "Grammar question detected",
                "transition_message": "Let me get my grammar expert friend!",
            },
        )

        # Handle the event
        response = await coordinator_service._handle_handoff_request(handoff_event)

        assert response.success is True
        assert "new_agent_id" in response.payload

        # Check that session was updated
        session = coordinator_service.active_sessions[session_id]
        assert session.current_agent_id == "professoressa_001"  # Should be updated
        assert "handoff_reason" in session.conversation_context

    @pytest.mark.asyncio
    async def test_handle_help_request(self, coordinator_service, mock_discovery_service):
        """Test handling help requests (agent stays primary)."""
        help_event = AgentEvent(
            sender_id="marco_001",
            event_type=EventType.REQUEST_HELP,
            payload={
                "help_type": AgentSpecialization.GRAMMAR,
                "user_message": "Is this sentence correct?",
                "context": {"current_topic": "verb_conjugation"},
            },
        )

        response = await coordinator_service._handle_help_request(help_event)

        assert response.success is True
        assert response.payload["requesting_agent"] == "marco_001"
        assert response.payload["expert_agent"] == "professoressa_001"
        assert response.payload["help_type"] == AgentSpecialization.GRAMMAR

    @pytest.mark.asyncio
    async def test_update_session_progress(self, coordinator_service):
        """Test updating session progress."""
        # Create session
        session_id = await coordinator_service.create_session(
            user_id="user123", initial_agent_id="marco_001"
        )

        # Update progress
        await coordinator_service.update_session_progress(
            session_id=session_id,
            message_content="I want to learn about food and cooking",
            agent_id="marco_001",
        )

        session = coordinator_service.active_sessions[session_id]
        assert session.messages_exchanged == 1
        assert session.engagement_score > 0
        assert len(session.conversation_history) == 1
        assert (
            "conversation" in session.topics_covered
        )  # Generic topic classification instead of pattern matching

    @pytest.mark.asyncio
    async def test_extract_topics(self, coordinator_service):
        """Test topic extraction from messages."""
        topics = await coordinator_service._extract_topics(
            "I want to learn about Italian food and travel"
        )

        assert "conversation" in topics  # 9 words = conversation (3-10 words)
        assert len(topics) == 1  # Now returns single topic category instead of multiple keywords

    @pytest.mark.asyncio
    async def test_get_session_status(self, coordinator_service):
        """Test getting session status."""
        # Create and update session
        session_id = await coordinator_service.create_session(
            user_id="user123", initial_agent_id="marco_001"
        )

        await coordinator_service.update_session_progress(
            session_id=session_id, message_content="Ciao! Come stai?", agent_id="marco_001"
        )

        status = await coordinator_service.get_session_status(session_id)

        assert status is not None
        assert status["session_id"] == session_id
        assert status["current_agent"] == "marco_001"
        assert status["messages_exchanged"] == 1
        assert status["engagement_score"] > 0

    @pytest.mark.asyncio
    async def test_get_session_status_invalid(self, coordinator_service):
        """Test getting status for invalid session."""
        status = await coordinator_service.get_session_status("nonexistent")
        assert status is None

    @pytest.mark.asyncio
    async def test_end_session(self, coordinator_service):
        """Test ending a session."""
        # Create and populate session
        session_id = await coordinator_service.create_session(
            user_id="user123", initial_agent_id="marco_001"
        )

        await coordinator_service.update_session_progress(
            session_id=session_id, message_content="Great lesson!", agent_id="marco_001"
        )

        summary = await coordinator_service.end_session(session_id)

        assert "session_id" in summary
        assert "duration_minutes" in summary
        assert summary["messages_exchanged"] == 1
        assert session_id not in coordinator_service.active_sessions  # Should be removed

    @pytest.mark.asyncio
    async def test_end_session_invalid(self, coordinator_service):
        """Test ending invalid session."""
        summary = await coordinator_service.end_session("nonexistent")
        assert "error" in summary


class TestConversationPhase:
    """Test conversation phase enum."""

    def test_conversation_phases(self):
        """Test all conversation phases exist."""
        phases = list(ConversationPhase)
        expected_phases = [
            ConversationPhase.GREETING,
            ConversationPhase.GOAL_SETTING,
            ConversationPhase.LEARNING,
            ConversationPhase.PRACTICE,
            ConversationPhase.CORRECTION,
            ConversationPhase.REINFORCEMENT,
            ConversationPhase.WRAP_UP,
        ]

        for phase in expected_phases:
            assert phase in phases


class TestBackwardCompatibility:
    """Test backward compatibility alias."""

    def test_coordinator_agent_alias(self, mock_discovery_service, mock_event_bus):
        """Test that CoordinatorAgent is an alias for CoordinatorService."""
        coordinator = CoordinatorAgent(
            coordinator_id="test_coordinator",
            discovery_service=mock_discovery_service,
            event_bus=mock_event_bus,
        )

        assert isinstance(coordinator, CoordinatorService)
        assert coordinator.coordinator_id == "test_coordinator"


class TestIntegration:
    """Integration tests for coordinator workflow."""

    @pytest.mark.asyncio
    async def test_full_handoff_workflow(self, coordinator_service, mock_discovery_service):
        """Test complete handoff workflow from agent perspective."""
        # Agent (Marco) starts conversation with user
        session_id = await coordinator_service.create_session(
            user_id="user123", initial_agent_id="marco_001"
        )

        # User asks grammar question - Marco requests handoff
        await coordinator_service.update_session_progress(
            session_id=session_id,
            message_content="What's the difference between 'ho mangiato' and 'mangiavo'?",
            agent_id="marco_001",
        )

        # Marco creates handoff request
        handoff_event = AgentEvent(
            sender_id="marco_001",
            event_type=EventType.REQUEST_HANDOFF,
            session_id=session_id,
            payload={
                "required_specialization": AgentSpecialization.GRAMMAR,
                "user_message": "What's the difference between 'ho mangiato' and 'mangiavo'?",
                "reason": "Grammar question - need grammar expert",
                "transition_message": "Let me get Professoressa Rossi - she's the grammar expert!",
            },
        )

        # Coordinator handles handoff
        response = await coordinator_service._handle_handoff_request(handoff_event)
        assert response.success is True

        # Session should now be assigned to grammar expert
        session = coordinator_service.active_sessions[session_id]
        assert session.current_agent_id == "professoressa_001"

        # Grammar expert can now continue the conversation
        await coordinator_service.update_session_progress(
            session_id=session_id,
            message_content="Perfect question! 'Ho mangiato' is passato prossimo...",
            agent_id="professoressa_001",
        )

        # Verify session progress
        status = await coordinator_service.get_session_status(session_id)
        assert status["current_agent"] == "professoressa_001"
        assert status["messages_exchanged"] == 2

        # End session
        summary = await coordinator_service.end_session(session_id)
        assert summary["messages_exchanged"] == 2
