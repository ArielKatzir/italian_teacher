"""
Coordinator Service implementation for intelligent conversation orchestration.

The Coordinator Service works behind the scenes to orchestrate multi-agent conversations,
manage context switching, track learning progress, and facilitate smooth agent handoffs.
Users never interact directly with the coordinator - it monitors and manages agent interactions.

Architecture Flow:
1. User talks directly to agents (Marco, Professoressa, etc.)
2. Agents publish events when they need help or context switching
3. Coordinator monitors events and orchestrates transitions
4. Coordinator facilitates natural handoffs between agents
5. Users experience seamless conversations with "Italian family members"
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from pydantic import BaseModel, Field

from .agent_discovery import AgentDiscoveryService
from .agent_events import AgentEvent, AgentResponse, EventType
from .conversation_state import ConversationStateManager
from .event_bus import AgentEventBus
from .logging_config import get_agent_logger


class ConversationPhase(Enum):
    """Phases of a learning conversation."""

    GREETING = "greeting"
    GOAL_SETTING = "goal_setting"
    LEARNING = "learning"
    PRACTICE = "practice"
    CORRECTION = "correction"
    REINFORCEMENT = "reinforcement"
    WRAP_UP = "wrap_up"


class LearningGoal(BaseModel):
    """Represents a user's learning goal for the session."""

    goal_type: str = Field(..., description="Type of learning goal (grammar, conversation, etc.)")
    target_level: str = Field(..., description="Target proficiency level")
    specific_topics: List[str] = Field(default_factory=list)
    estimated_duration: Optional[int] = Field(None, description="Estimated minutes")
    priority: int = Field(default=1, description="Goal priority (1=highest)")


class SessionState(BaseModel):
    """Current session state and progress tracking."""

    session_id: str
    user_id: str
    started_at: datetime
    current_phase: ConversationPhase
    current_agent_id: str  # Currently active agent
    learning_goals: List[LearningGoal]
    completed_goals: List[str]

    # Conversation tracking
    conversation_history: List[str] = Field(default_factory=list)
    messages_exchanged: int = 0
    corrections_made: int = 0
    topics_covered: Set[str] = Field(default_factory=set)
    engagement_score: float = 0.0

    # Context for handoffs
    last_user_message: Optional[str] = None
    conversation_context: Dict[str, Any] = Field(default_factory=dict)


class CoordinatorService:
    """
    Background Coordinator Service for orchestrating multi-agent conversations.

    This service:
    - Monitors agent conversations through event subscriptions
    - Analyzes context and determines when handoffs are needed
    - Facilitates smooth transitions between agents
    - Tracks learning progress across sessions
    - Never directly interacts with users
    """

    def __init__(
        self,
        coordinator_id: str = "coordinator_001",
        discovery_service: Optional[AgentDiscoveryService] = None,
        event_bus: Optional[AgentEventBus] = None,
        state_manager: Optional[ConversationStateManager] = None,
    ):
        """Initialize the Coordinator Service."""
        self.coordinator_id = coordinator_id
        self.discovery_service = discovery_service
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.logger = get_agent_logger(coordinator_id, "CoordinatorService")

        # Active sessions managed by coordinator
        self.active_sessions: Dict[str, SessionState] = {}

        # Event handlers
        self.event_handlers = {
            EventType.REQUEST_HANDOFF: self._handle_handoff_request,
            EventType.REQUEST_HELP: self._handle_help_request,
            EventType.SHARE_CONTEXT: self._handle_context_share,
            EventType.REQUEST_CORRECTION_REVIEW: self._handle_correction_request,
        }

        # Configuration thresholds
        self.handoff_confidence_threshold = 0.7
        self.max_session_duration = timedelta(hours=2)
        self.inactivity_timeout = timedelta(minutes=15)

        # Subscribe to agent events if event bus available
        if self.event_bus:
            self._setup_event_subscriptions()

        self.logger.info("Coordinator Service initialized")

    def _setup_event_subscriptions(self):
        """Set up event subscriptions to monitor agent interactions."""
        for event_type in self.event_handlers.keys():
            self.event_bus.subscribe(
                event_type=event_type,
                handler=self._handle_agent_event,
                subscriber_id=self.coordinator_id,
            )

        self.logger.info("Event subscriptions configured")

    async def _handle_agent_event(self, event: AgentEvent) -> Optional[AgentResponse]:
        """Central event handler that routes to specific handlers."""
        try:
            handler = self.event_handlers.get(event.event_type)
            if handler:
                return await handler(event)
            else:
                self.logger.warning(f"No handler for event type: {event.event_type}")
                return None
        except Exception as e:
            self.logger.error(f"Error handling event {event.id}: {e}")
            return AgentResponse(
                original_event_id=event.id,
                responder_id=self.coordinator_id,
                success=False,
                error_message=str(e),
            )

    async def create_session(
        self,
        user_id: str,
        initial_agent_id: str,
        learning_goals: Optional[List[LearningGoal]] = None,
    ) -> str:
        """
        Create a new session when a user starts talking to an agent.
        This is called by the first agent the user interacts with.
        """
        session_id = str(uuid4())

        session_state = SessionState(
            session_id=session_id,
            user_id=user_id,
            started_at=datetime.now(),
            current_phase=ConversationPhase.GREETING,
            current_agent_id=initial_agent_id,
            learning_goals=learning_goals or [],
            completed_goals=[],
        )

        self.active_sessions[session_id] = session_state

        self.logger.info(f"Created session {session_id} with agent {initial_agent_id}")
        return session_id

    async def _handle_handoff_request(self, event: AgentEvent) -> AgentResponse:
        """Handle request to handoff conversation to another agent."""
        try:
            session_id = event.session_id
            if not session_id or session_id not in self.active_sessions:
                return AgentResponse(
                    original_event_id=event.id,
                    responder_id=self.coordinator_id,
                    success=False,
                    error_message="Invalid session ID",
                )

            session = self.active_sessions[session_id]

            # Find best agent for handoff
            required_specialization = event.payload.get("required_specialization")
            user_message = event.payload.get("user_message", "")

            if self.discovery_service and required_specialization:
                suitable_agents = await self.discovery_service.find_agent_for_handoff(
                    from_agent_id=event.sender_id,
                    required_specialization=required_specialization,
                    user_message=user_message,
                )

                if suitable_agents:
                    best_agent = suitable_agents.registration

                    # Update session state
                    session.current_agent_id = best_agent.agent_id
                    session.conversation_context["handoff_reason"] = event.payload.get("reason", "")
                    session.conversation_context["previous_agent"] = event.sender_id

                    # Create handoff instructions for agents
                    handoff_context = {
                        "session_id": session_id,
                        "previous_agent_id": event.sender_id,
                        "new_agent_id": best_agent.agent_id,
                        "user_message": user_message,
                        "conversation_context": session.conversation_context,
                        "transition_message": event.payload.get("transition_message", ""),
                    }

                    self.logger.info(
                        f"Orchestrating handoff from {event.sender_id} to {best_agent.agent_id}"
                    )

                    return AgentResponse(
                        original_event_id=event.id,
                        responder_id=self.coordinator_id,
                        success=True,
                        payload=handoff_context,
                    )

            return AgentResponse(
                original_event_id=event.id,
                responder_id=self.coordinator_id,
                success=False,
                error_message="No suitable agent found for handoff",
            )

        except Exception as e:
            self.logger.error(f"Error in handoff request: {e}")
            return AgentResponse(
                original_event_id=event.id,
                responder_id=self.coordinator_id,
                success=False,
                error_message=str(e),
            )

    async def _handle_help_request(self, event: AgentEvent) -> AgentResponse:
        """Handle request for help while agent remains primary."""
        try:
            # Find expert agent for specific help
            help_type = event.payload.get("help_type")

            if self.discovery_service and help_type:
                expert_agents = await self.discovery_service.find_agent_for_help_request(
                    requesting_agent_id=event.sender_id,
                    required_specialization=help_type,
                    user_message=event.payload.get("user_message", ""),
                )

                if expert_agents:
                    expert = expert_agents.registration

                    # Create help context
                    help_context = {
                        "requesting_agent": event.sender_id,
                        "expert_agent": expert.agent_id,
                        "help_type": help_type,
                        "user_message": event.payload.get("user_message", ""),
                        "context": event.payload.get("context", {}),
                    }

                    self.logger.info(
                        f"Facilitating help request: {event.sender_id} -> {expert.agent_id}"
                    )

                    return AgentResponse(
                        original_event_id=event.id,
                        responder_id=self.coordinator_id,
                        success=True,
                        payload=help_context,
                    )

            return AgentResponse(
                original_event_id=event.id,
                responder_id=self.coordinator_id,
                success=False,
                error_message="No expert agent available for help",
            )

        except Exception as e:
            return AgentResponse(
                original_event_id=event.id,
                responder_id=self.coordinator_id,
                success=False,
                error_message=str(e),
            )

    async def _handle_context_share(self, event: AgentEvent) -> AgentResponse:
        """Handle context sharing between agents."""
        session_id = event.session_id
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]

            # Update session context
            shared_context = event.payload.get("context", {})
            session.conversation_context.update(shared_context)

            # Track topics
            topics = event.payload.get("topics", [])
            session.topics_covered.update(topics)

            self.logger.info(f"Updated context for session {session_id}")

            return AgentResponse(
                original_event_id=event.id,
                responder_id=self.coordinator_id,
                success=True,
                payload={"context_updated": True},
            )

        return AgentResponse(
            original_event_id=event.id,
            responder_id=self.coordinator_id,
            success=False,
            error_message="Session not found",
        )

    async def _handle_correction_request(self, event: AgentEvent) -> AgentResponse:
        """Handle grammar/language correction requests."""
        # Find grammar expert
        if self.discovery_service:
            grammar_experts = await self.discovery_service.find_agent_for_correction_review(
                requesting_agent_id=event.sender_id, text_to_review=event.payload.get("text", "")
            )

            if grammar_experts:
                expert = grammar_experts.registration

                correction_context = {
                    "requesting_agent": event.sender_id,
                    "grammar_expert": expert.agent_id,
                    "text_to_review": event.payload.get("text", ""),
                    "context": event.payload.get("context", {}),
                }

                return AgentResponse(
                    original_event_id=event.id,
                    responder_id=self.coordinator_id,
                    success=True,
                    payload=correction_context,
                )

        return AgentResponse(
            original_event_id=event.id,
            responder_id=self.coordinator_id,
            success=False,
            error_message="No grammar expert available",
        )

    async def update_session_progress(self, session_id: str, message_content: str, agent_id: str):
        """Update session progress when agents report activity."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]

            # Update message count and history
            session.messages_exchanged += 1
            session.conversation_history.append(message_content)
            session.last_user_message = message_content

            # Extract and track topics
            topics = await self._extract_topics(message_content)
            session.topics_covered.update(topics)

            # Update engagement score
            session.engagement_score = min(1.0, session.messages_exchanged * 0.1)

            # Keep history manageable
            if len(session.conversation_history) > 10:
                session.conversation_history.pop(0)

            self.logger.debug(f"Updated progress for session {session_id}")

    async def _extract_topics(self, message: str) -> Set[str]:
        """Extract topics/themes from message."""
        # Simple keyword extraction (could be enhanced with NLP)
        # Support both English and Italian words
        topic_keywords = {
            "food": [
                "cibo",
                "mangiare",
                "ristorante",
                "pasta",
                "pizza",
                "food",
                "eat",
                "restaurant",
                "cooking",
                "meal",
            ],
            "travel": [
                "viaggio",
                "aeroporto",
                "stazione",
                "hotel",
                "treno",
                "travel",
                "airport",
                "station",
                "hotel",
                "train",
                "trip",
            ],
            "family": [
                "famiglia",
                "madre",
                "padre",
                "fratello",
                "sorella",
                "family",
                "mother",
                "father",
                "brother",
                "sister",
            ],
            "work": [
                "lavoro",
                "ufficio",
                "colleghi",
                "riunione",
                "work",
                "office",
                "colleagues",
                "meeting",
                "job",
            ],
        }

        message_lower = message.lower()
        found_topics = set()

        for topic, keywords in topic_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                found_topics.add(topic)

        return found_topics

    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current session status and progress."""
        if session_id not in self.active_sessions:
            return None

        session_state = self.active_sessions[session_id]

        return {
            "session_id": session_id,
            "current_phase": session_state.current_phase.value,
            "current_agent": session_state.current_agent_id,
            "messages_exchanged": session_state.messages_exchanged,
            "topics_covered": list(session_state.topics_covered),
            "engagement_score": session_state.engagement_score,
            "session_duration": (datetime.now() - session_state.started_at).total_seconds() / 60,
            "goals_progress": {
                "total": len(session_state.learning_goals),
                "completed": len(session_state.completed_goals),
            },
        }

    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a learning session and return summary."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session_state = self.active_sessions[session_id]

        # Create session summary
        summary = {
            "session_id": session_id,
            "duration_minutes": (datetime.now() - session_state.started_at).total_seconds() / 60,
            "messages_exchanged": session_state.messages_exchanged,
            "topics_covered": list(session_state.topics_covered),
            "engagement_score": session_state.engagement_score,
            "goals_completed": len(session_state.completed_goals),
            "total_goals": len(session_state.learning_goals),
        }

        # Clean up
        del self.active_sessions[session_id]

        self.logger.info(f"Ended session {session_id}")
        return summary


# For backward compatibility, create an alias
CoordinatorAgent = CoordinatorService
