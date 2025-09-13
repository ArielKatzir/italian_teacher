"""
Agent Event Bus implementation for inter-agent communication.

This module provides the core event bus that enables agents to communicate
asynchronously through events and subscriptions.
"""

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Set

from .agent_events import AgentEvent, AgentResponse, EventHandler, EventSubscription, EventType
from .logging_config import get_agent_logger


class AgentEventBus:
    """
    Central event bus for agent communication.

    Handles event routing, subscription management, and delivery to interested agents.
    """

    def __init__(self, discovery_service=None):
        """Initialize the event bus."""
        self.subscriptions: Dict[EventType, List[EventSubscription]] = defaultdict(list)
        self.agents: Dict[str, EventHandler] = {}
        self.logger = get_agent_logger("event_bus", "EventBus")
        self._delivery_timeout = 5.0  # seconds (Config override available)

        # Discovery service for intelligent agent selection
        self.discovery_service = discovery_service
        if discovery_service is None:
            from .agent_discovery import default_discovery_service

            self.discovery_service = default_discovery_service

    async def subscribe(
        self, agent_id: str, event_types: Set[EventType], handler: EventHandler
    ) -> None:
        """
        Subscribe an agent to specific event types.

        Args:
            agent_id: Unique identifier for the agent
            event_types: Set of event types the agent wants to receive
            handler: Handler that will process the events
        """
        # Register agent handler
        self.agents[agent_id] = handler

        # Create subscription for each event type
        for event_type in event_types:
            subscription = EventSubscription(agent_id, {event_type}, handler)
            self.subscriptions[event_type].append(subscription)

        self.logger.info(
            "agent_subscribed",
            agent_id=agent_id,
            event_types=[et.value for et in event_types],
            total_subscriptions=len(self.agents),
        )

    async def unsubscribe(self, agent_id: str) -> None:
        """
        Remove all subscriptions for an agent.

        Args:
            agent_id: Agent to unsubscribe
        """
        # Remove from agents registry
        if agent_id in self.agents:
            del self.agents[agent_id]

        # Remove from all subscription lists
        for event_type in list(self.subscriptions.keys()):
            self.subscriptions[event_type] = [
                sub for sub in self.subscriptions[event_type] if sub.agent_id != agent_id
            ]

            # Clean up empty subscription lists
            if not self.subscriptions[event_type]:
                del self.subscriptions[event_type]

        self.logger.info("agent_unsubscribed", agent_id=agent_id)

    async def publish(self, event: AgentEvent) -> List[AgentResponse]:
        """
        Publish an event to interested agents.

        Args:
            event: Event to publish

        Returns:
            List of responses from agents who handled the event
        """
        self.logger.info(
            "event_published",
            event_id=event.id,
            event_type=event.event_type.value,
            sender_id=event.sender_id,
            target_agent=event.target_agent,
            priority=event.priority,
        )

        # Determine target handlers
        handlers = await self._get_target_handlers(event)

        if not handlers:
            self.logger.info(
                "no_handlers_found", event_id=event.id, event_type=event.event_type.value
            )
            return []

        # Deliver event to all handlers concurrently
        responses = await self._deliver_event(event, handlers)

        self.logger.info(
            "event_delivery_completed",
            event_id=event.id,
            handlers_count=len(handlers),
            responses_count=len(responses),
            successful_responses=sum(1 for r in responses if r.success),
        )

        return responses

    async def _get_target_handlers(self, event: AgentEvent) -> List[EventHandler]:
        """Get handlers that should receive this event."""
        handlers = []

        # If event has a specific target, deliver only to that agent
        if event.target_agent:
            subscriptions = self.subscriptions.get(event.event_type, [])
            for subscription in subscriptions:
                if subscription.agent_id == event.target_agent:
                    handlers.append(subscription.handler)
                    break

            if not handlers:
                self.logger.warning(
                    "target_agent_not_found",
                    event_id=event.id,
                    target_agent=event.target_agent,
                    event_type=event.event_type.value,
                )

        else:
            # Use discovery service to find best agent(s)
            handlers = await self._discover_target_handlers(event)

            # Fallback to broadcast if discovery fails
            if not handlers:
                self.logger.warning(
                    "discovery_failed_fallback_to_broadcast",
                    event_id=event.id,
                    event_type=event.event_type.value,
                )
                handlers = await self._get_broadcast_handlers(event)

        return handlers

    async def _deliver_event(
        self, event: AgentEvent, handlers: List[EventHandler]
    ) -> List[AgentResponse]:
        """Deliver event to handlers concurrently."""
        # Create delivery tasks
        delivery_tasks = [self._safe_deliver_to_handler(event, handler) for handler in handlers]

        # Execute all deliveries concurrently with timeout
        try:
            responses = await asyncio.wait_for(
                asyncio.gather(*delivery_tasks, return_exceptions=True),
                timeout=self._delivery_timeout,
            )

            # Filter out exceptions and None responses
            valid_responses = [
                response for response in responses if isinstance(response, AgentResponse)
            ]

            return valid_responses

        except asyncio.TimeoutError:
            self.logger.error(
                "event_delivery_timeout", event_id=event.id, timeout=self._delivery_timeout
            )
            return []

    async def _safe_deliver_to_handler(
        self, event: AgentEvent, handler: EventHandler
    ) -> Optional[AgentResponse]:
        """Safely deliver event to a single handler."""
        try:
            response = await handler.handle_event(event)
            return response

        except Exception as e:
            self.logger.error(
                "handler_error",
                event_id=event.id,
                handler_type=type(handler).__name__,
                error_type=type(e).__name__,
                error_message=str(e),
            )

            # Return error response
            return AgentResponse(
                original_event_id=event.id,
                responder_id="event_bus",
                success=False,
                error_message=f"Handler error: {str(e)}",
            )

    def get_subscription_info(self) -> Dict[str, Dict]:
        """Get information about current subscriptions."""
        info = {}

        for agent_id in self.agents:
            agent_event_types = []
            for event_type, subscriptions in self.subscriptions.items():
                for sub in subscriptions:
                    if sub.agent_id == agent_id:
                        agent_event_types.append(event_type.value)

            info[agent_id] = {
                "event_types": agent_event_types,
                "subscription_count": len(agent_event_types),
            }

        return info

    async def shutdown(self) -> None:
        """Shutdown the event bus and clean up resources."""
        self.logger.info("event_bus_shutting_down", agents_count=len(self.agents))

        # Clear all subscriptions
        self.subscriptions.clear()
        self.agents.clear()

        self.logger.info("event_bus_shutdown_complete")

    async def _discover_target_handlers(self, event: AgentEvent) -> List[EventHandler]:
        """Use discovery service to find the best handlers for an event."""
        handlers = []

        try:
            # Extract event-specific parameters
            kwargs = self._extract_discovery_params(event)

            # Find the best agent using discovery service
            best_agent = await self.discovery_service.find_best_agent(event.event_type, **kwargs)

            if best_agent:
                # Find the handler for the selected agent
                subscriptions = self.subscriptions.get(event.event_type, [])
                for subscription in subscriptions:
                    if subscription.agent_id == best_agent.agent_id:
                        handlers.append(subscription.handler)
                        self.logger.info(
                            "agent_selected_by_discovery",
                            event_id=event.id,
                            selected_agent=best_agent.agent_id,
                            agent_type=best_agent.agent_type,
                            event_type=event.event_type.value,
                        )
                        break

                if not handlers:
                    self.logger.warning(
                        "selected_agent_not_subscribed",
                        event_id=event.id,
                        selected_agent=best_agent.agent_id,
                        event_type=event.event_type.value,
                    )

        except Exception as e:
            self.logger.error(
                "discovery_service_error",
                event_id=event.id,
                event_type=event.event_type.value,
                error=str(e),
            )

        return handlers

    async def _get_broadcast_handlers(self, event: AgentEvent) -> List[EventHandler]:
        """Get all subscribed handlers for broadcast (except sender)."""
        handlers = []
        subscriptions = self.subscriptions.get(event.event_type, [])

        for subscription in subscriptions:
            if subscription.agent_id != event.sender_id:
                handlers.append(subscription.handler)

        return handlers

    def _extract_discovery_params(self, event: AgentEvent) -> Dict[str, any]:
        """Extract parameters from event for discovery service."""
        kwargs = {}

        # Extract common parameters from event payload
        if hasattr(event, "payload") and event.payload:
            # For help requests
            if event.event_type == EventType.REQUEST_HELP:
                kwargs["help_type"] = event.payload.get("help_type", "general")
                kwargs["user_language_level"] = event.payload.get("user_language_level", "beginner")

            # For handoff requests
            elif event.event_type == EventType.REQUEST_HANDOFF:
                kwargs["reason"] = event.payload.get("reason", "general")
                kwargs["conversation_complexity"] = event.payload.get(
                    "conversation_complexity", "medium"
                )
                kwargs["current_agent_type"] = event.payload.get("current_agent_type")

            # For correction reviews
            elif event.event_type == EventType.REQUEST_CORRECTION_REVIEW:
                kwargs["correction_type"] = event.payload.get("correction_type", "general")

        return kwargs
