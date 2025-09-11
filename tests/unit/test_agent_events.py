"""
Tests for agent event system and communication protocols.
"""

from datetime import datetime
from typing import Set

import pytest

from core.agent_events import (
    AgentEvent,
    AgentResponse,
    ContextShare,
    CorrectionReview,
    EventHandler,
    EventSubscription,
    EventType,
    HandoffRequest,
    HelpRequest,
)


class MockEventHandler(EventHandler):
    """Mock event handler for testing."""

    def __init__(self, handled_types: Set[EventType]):
        self.handled_types = handled_types
        self.received_events = []

    async def handle_event(self, event: AgentEvent):
        """Handle an event and record it."""
        self.received_events.append(event)

        return AgentResponse(
            original_event_id=event.id,
            responder_id="mock_handler",
            success=True,
            payload={"message": "Event handled successfully"},
        )

    def get_handled_event_types(self) -> Set[EventType]:
        """Return the event types this handler supports."""
        return self.handled_types


class TestAgentEvent:
    """Test the base AgentEvent class."""

    def test_agent_event_creation(self):
        """Test creating a basic agent event."""
        event = AgentEvent(
            sender_id="marco",
            event_type=EventType.REQUEST_HELP,
            payload={"message": "Need help with grammar"},
            session_id="test_session_123",
        )

        assert event.sender_id == "marco"
        assert event.event_type == EventType.REQUEST_HELP
        assert event.payload["message"] == "Need help with grammar"
        assert event.session_id == "test_session_123"
        assert event.target_agent is None  # Should be None for broadcast
        assert event.priority == 1  # Default priority
        assert isinstance(event.timestamp, datetime)
        assert len(event.id) > 0  # Should have generated ID

    def test_agent_event_with_target(self):
        """Test creating an event with a specific target agent."""
        event = AgentEvent(
            sender_id="marco",
            event_type=EventType.REQUEST_HANDOFF,
            target_agent="professoressa_rossi",
            priority=5,
        )

        assert event.target_agent == "professoressa_rossi"
        assert event.priority == 5

    def test_agent_event_validation(self):
        """Test event validation."""
        with pytest.raises(ValueError):
            AgentEvent(
                sender_id="marco",
                event_type=EventType.REQUEST_HELP,
                priority=11,  # Invalid priority (max is 10)
            )

        with pytest.raises(ValueError):
            AgentEvent(
                sender_id="marco",
                event_type=EventType.REQUEST_HELP,
                priority=0,  # Invalid priority (min is 1)
            )


class TestAgentResponse:
    """Test the AgentResponse class."""

    def test_agent_response_success(self):
        """Test creating a successful response."""
        response = AgentResponse(
            original_event_id="event_123",
            responder_id="professoressa_rossi",
            success=True,
            payload={"correction": "Ciao invece di 'Cao'"},
        )

        assert response.original_event_id == "event_123"
        assert response.responder_id == "professoressa_rossi"
        assert response.success is True
        assert response.payload["correction"] == "Ciao invece di 'Cao'"
        assert response.error_message is None
        assert isinstance(response.timestamp, datetime)

    def test_agent_response_failure(self):
        """Test creating a failure response."""
        response = AgentResponse(
            original_event_id="event_456",
            responder_id="marco",
            success=False,
            error_message="Cannot handle this type of request",
        )

        assert response.success is False
        assert response.error_message == "Cannot handle this type of request"


class TestSpecificEventTypes:
    """Test specific event type classes."""

    def test_help_request_creation(self):
        """Test creating a help request without target agent."""
        event = HelpRequest(
            sender_id="marco",
            session_id="session_123",
            payload={
                "user_message": "Come si dice 'hello' in italiano?",
                "help_type": "translation",
            },
        )

        assert event.sender_id == "marco"
        assert event.event_type == EventType.REQUEST_HELP
        assert event.target_agent is None  # Should be None (no target specified)
        assert event.payload["help_type"] == "translation"

    def test_handoff_request_creation(self):
        """Test creating a handoff request without target agent."""
        event = HandoffRequest(
            sender_id="marco",
            session_id="session_456",
            payload={
                "reason": "grammar_correction_needed",
                "handoff_type": "temporary",
                "message_count": 3,
            },
        )

        assert event.sender_id == "marco"
        assert event.event_type == EventType.REQUEST_HANDOFF
        assert event.target_agent is None  # Should be None (orchestrator selects)
        assert event.payload["reason"] == "grammar_correction_needed"

    def test_context_share_creation(self):
        """Test creating a context share event."""
        event = ContextShare(
            sender_id="nonna_giulia",
            payload={
                "context_type": "cultural_background",
                "region": "Sicily",
                "traditions": ["pasta_making", "storytelling"],
            },
        )

        assert event.sender_id == "nonna_giulia"
        assert event.event_type == EventType.SHARE_CONTEXT
        assert event.payload["region"] == "Sicily"

    def test_correction_review_creation(self):
        """Test creating a correction review request."""
        # Test with default target (None)
        event = CorrectionReview(
            sender_id="marco",
            payload={
                "original_text": "Io va al cinema",
                "suggested_correction": "Io vado al cinema",
                "correction_type": "verb_conjugation",
            },
        )

        assert event.sender_id == "marco"
        assert event.event_type == EventType.REQUEST_CORRECTION_REVIEW
        assert event.target_agent is None  # Should be None (orchestrator selects)

    def test_correction_review_with_specific_target(self):
        """Test creating a correction review with specific target."""
        event = CorrectionReview(
            sender_id="marco",
            target_agent="professoressa_rossi",
            payload={
                "original_text": "Io va al cinema",
                "suggested_correction": "Io vado al cinema",
            },
        )

        assert event.target_agent == "professoressa_rossi"


class TestEventHandler:
    """Test the EventHandler interface and related classes."""

    def test_mock_event_handler(self):
        """Test the mock event handler implementation."""
        handled_types = {EventType.REQUEST_HELP, EventType.SHARE_CONTEXT}
        handler = MockEventHandler(handled_types)

        assert handler.get_handled_event_types() == handled_types
        assert len(handler.received_events) == 0

    @pytest.mark.asyncio
    async def test_event_handler_processing(self):
        """Test event handler processing."""
        handler = MockEventHandler({EventType.REQUEST_HELP})

        event = HelpRequest(sender_id="marco", payload={"help_type": "pronunciation"})

        response = await handler.handle_event(event)

        assert len(handler.received_events) == 1
        assert handler.received_events[0] is event
        assert response.original_event_id == event.id
        assert response.responder_id == "mock_handler"
        assert response.success is True


class TestEventSubscription:
    """Test the EventSubscription class."""

    def test_event_subscription_creation(self):
        """Test creating an event subscription."""
        handler = MockEventHandler({EventType.REQUEST_HELP})
        event_types = {EventType.REQUEST_HELP, EventType.SHARE_CONTEXT}

        subscription = EventSubscription(agent_id="marco", event_types=event_types, handler=handler)

        assert subscription.agent_id == "marco"
        assert subscription.event_types == event_types
        assert subscription.handler is handler
        assert isinstance(subscription.created_at, datetime)


class TestEventTypeEnum:
    """Test the EventType enumeration."""

    def test_all_event_types_present(self):
        """Test that all expected event types are defined."""
        expected_types = {
            "REQUEST_HANDOFF",
            "HANDOFF_ACCEPTED",
            "HANDOFF_COMPLETED",
            "REQUEST_HELP",
            "PROVIDE_HELP",
            "SHARE_CONTEXT",
            "REQUEST_CONTEXT",
            "REQUEST_CORRECTION_REVIEW",
            "CORRECTION_FEEDBACK",
        }

        actual_types = {event_type.name for event_type in EventType}
        assert actual_types == expected_types

    def test_event_type_values(self):
        """Test event type string values."""
        assert EventType.REQUEST_HELP.value == "request_help"
        assert EventType.REQUEST_HANDOFF.value == "request_handoff"
        assert EventType.SHARE_CONTEXT.value == "share_context"


class TestEventIntegration:
    """Integration tests for event system."""

    @pytest.mark.asyncio
    async def test_full_event_lifecycle(self):
        """Test complete event creation, handling, and response cycle."""
        # Create handler
        handler = MockEventHandler({EventType.REQUEST_HELP, EventType.REQUEST_HANDOFF})

        # Create help request
        help_event = HelpRequest(
            sender_id="marco",
            session_id="session_789",
            payload={"user_message": "Mi aiuti con la grammatica?", "help_type": "grammar"},
        )

        # Handle event
        response = await handler.handle_event(help_event)

        # Verify event was processed
        assert len(handler.received_events) == 1
        assert handler.received_events[0] is help_event

        # Verify response
        assert response.original_event_id == help_event.id
        assert response.success is True
        assert "handled successfully" in response.payload["message"]

    def test_event_serialization(self):
        """Test that events can be serialized/deserialized (for message queues)."""
        event = HelpRequest(sender_id="marco", payload={"help_type": "cultural"})

        # Convert to dict (simulates JSON serialization)
        event_dict = event.model_dump()

        # Verify structure
        assert event_dict["sender_id"] == "marco"
        assert event_dict["event_type"] == EventType.REQUEST_HELP
        assert event_dict["payload"]["help_type"] == "cultural"
        assert "timestamp" in event_dict
        assert "id" in event_dict

    def test_event_filtering_by_type(self):
        """Test filtering events by type (simulates event bus routing)."""
        events = [
            HelpRequest(sender_id="marco", payload={"help_type": "grammar"}),
            HandoffRequest(sender_id="marco", payload={"reason": "complexity"}),
            ContextShare(sender_id="nonna_giulia", payload={"region": "Tuscany"}),
            HelpRequest(sender_id="lorenzo", payload={"help_type": "slang"}),
        ]

        # Filter for help requests only
        help_requests = [e for e in events if e.event_type == EventType.REQUEST_HELP]
        assert len(help_requests) == 2
        assert all(isinstance(e, HelpRequest) for e in help_requests)

        # Filter for handoff requests
        handoff_requests = [e for e in events if e.event_type == EventType.REQUEST_HANDOFF]
        assert len(handoff_requests) == 1
        assert isinstance(handoff_requests[0], HandoffRequest)
