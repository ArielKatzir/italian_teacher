"""
Event Handler Interface

Defines the contract for objects that can handle agent events.
This interface enables different event handling implementations
while maintaining consistent API for event processing.
"""

from abc import ABC, abstractmethod

# Import types using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Optional, Set

if TYPE_CHECKING:
    from ..agent_events import AgentEvent, AgentResponse, EventType


class EventHandler(ABC):
    """
    Abstract interface for objects that can handle agent events.

    This interface defines the contract that all event handlers must satisfy.
    It supports:
    - Event processing and response generation
    - Event type filtering and routing
    - Async event handling for scalability

    Implementations:
    - BaseAgent: Core agent event handling
    - EventBus: Central event routing and delivery
    - SpecializedHandlers: Custom event processors (future)
    """

    @abstractmethod
    async def handle_event(self, event: "AgentEvent") -> Optional["AgentResponse"]:
        """
        Handle an incoming agent event.

        Args:
            event: The event to handle

        Returns:
            Optional response to the event. None if no response is needed.
        """

    @abstractmethod
    def get_handled_event_types(self) -> Set["EventType"]:
        """
        Get the event types this handler can process.

        Returns:
            Set of event types this handler supports. Used for routing
            and filtering to ensure events are sent to appropriate handlers.
        """
