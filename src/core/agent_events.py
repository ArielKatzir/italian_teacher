"""
Agent communication events and event bus system.

This module provides the event-driven communication infrastructure for agents
to collaborate, request help, and coordinate conversations.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field
from typing_extensions import Literal


class EventType(Enum):
    """Types of events agents can send and receive."""

    # Handoff events
    REQUEST_HANDOFF = "request_handoff"
    HANDOFF_ACCEPTED = "handoff_accepted"
    HANDOFF_COMPLETED = "handoff_completed"

    # Collaboration events
    REQUEST_HELP = "request_help"
    PROVIDE_HELP = "provide_help"

    # Context sharing
    SHARE_CONTEXT = "share_context"
    REQUEST_CONTEXT = "request_context"

    # Corrections
    REQUEST_CORRECTION_REVIEW = "request_correction_review"
    CORRECTION_FEEDBACK = "correction_feedback"


class AgentEvent(BaseModel):
    """Base class for all agent communication events."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    sender_id: str = Field(..., description="Agent that sent this event")
    event_type: EventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now)
    target_agent: Optional[str] = Field(None, description="Target agent (None = broadcast)")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data")
    session_id: Optional[str] = Field(None, description="Related conversation session")
    priority: int = Field(1, ge=1, le=10, description="Event priority (1=low, 10=urgent)")


class AgentResponse(BaseModel):
    """Response to an agent event."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    original_event_id: str = Field(..., description="ID of event this responds to")
    responder_id: str = Field(..., description="Agent that sent this response")
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool = Field(..., description="Whether the request was successful")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Response data")
    error_message: Optional[str] = Field(None, description="Error details if success=False")


# Specific event types for common communication patterns


class HandoffRequest(AgentEvent):
    """Request to hand off conversation to another agent."""

    event_type: Literal[EventType.REQUEST_HANDOFF] = Field(EventType.REQUEST_HANDOFF)

    def __init__(self, sender_id: str, **kwargs):
        super().__init__(
            sender_id=sender_id,
            event_type=EventType.REQUEST_HANDOFF,
            **kwargs,
        )


class HelpRequest(AgentEvent):
    """Request help from another agent while staying primary."""

    event_type: Literal[EventType.REQUEST_HELP] = Field(EventType.REQUEST_HELP)

    def __init__(self, sender_id: str, **kwargs):
        super().__init__(
            sender_id=sender_id,
            event_type=EventType.REQUEST_HELP,
            **kwargs,
        )


class ContextShare(AgentEvent):
    """Share conversation context with other agents."""

    event_type: Literal[EventType.SHARE_CONTEXT] = Field(EventType.SHARE_CONTEXT)

    def __init__(self, sender_id: str, **kwargs):
        super().__init__(sender_id=sender_id, event_type=EventType.SHARE_CONTEXT, **kwargs)


class CorrectionReview(AgentEvent):
    """Request review of a language correction."""

    event_type: Literal[EventType.REQUEST_CORRECTION_REVIEW] = Field(
        EventType.REQUEST_CORRECTION_REVIEW
    )

    def __init__(self, sender_id: str, target_agent: Optional[str] = None, **kwargs):
        super().__init__(
            sender_id=sender_id,
            target_agent=target_agent,
            event_type=EventType.REQUEST_CORRECTION_REVIEW,
            **kwargs,
        )


# Event handler interface
class EventHandler(ABC):
    """Interface for objects that can handle agent events."""

    @abstractmethod
    async def handle_event(self, event: AgentEvent) -> Optional[AgentResponse]:
        """
        Handle an incoming agent event.

        Args:
            event: The event to handle

        Returns:
            Optional response to the event
        """

    @abstractmethod
    def get_handled_event_types(self) -> Set[EventType]:
        """
        Get the event types this handler can process.

        Returns:
            Set of event types this handler supports
        """


class EventSubscription:
    """Represents an agent's subscription to specific event types."""

    def __init__(self, agent_id: str, event_types: Set[EventType], handler: EventHandler):
        self.agent_id = agent_id
        self.event_types = event_types
        self.handler = handler
        self.created_at = datetime.now()
