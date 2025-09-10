"""
Base Agent class providing common interfaces for all Italian Teacher agents.

This module defines the abstract base class that all agents (Marco, Professoressa Rossi,
Nonna Giulia, Lorenzo) inherit from, ensuring consistent interfaces and behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from .logging_config import get_agent_logger
from .retention_policy import RetentionPreference, default_retention_manager


class MessageType(Enum):
    """Types of messages agents can send and receive."""

    CONVERSATION = "conversation"
    CORRECTION = "correction"
    CULTURAL_NOTE = "cultural_note"
    ENCOURAGEMENT = "encouragement"
    QUESTION = "question"
    SYSTEM = "system"


class AgentStatus(Enum):
    """Current status of an agent."""

    INACTIVE = "inactive"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"


@dataclass
class AgentActivityEvent:
    """Lightweight log entry for agent system events (not conversation content)."""

    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # activated, deactivated, error, initialized, etc.
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Standard message format for agent communication."""

    id: str = field(default_factory=lambda: str(uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: MessageType = MessageType.CONVERSATION
    content: str = ""
    priority: int = 1  # 1-10, higher = more urgent
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context information for ongoing conversations."""

    user_id: str
    session_id: str
    conversation_history: List[AgentMessage] = field(default_factory=list)
    current_topic: Optional[str] = None
    user_language_level: str = "beginner"  # beginner, intermediate, advanced
    learning_goals: List[str] = field(default_factory=list)
    cultural_preferences: Dict[str, Any] = field(default_factory=dict)
    last_interaction: Optional[datetime] = None

    # Retention policy configuration
    retention_preference: RetentionPreference = RetentionPreference.BALANCED
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_deletion: Optional[datetime] = None
    cleanup_job_id: Optional[str] = None  # Track cleanup job for cancellation


class AgentPersonality(BaseModel):
    """Configuration for an agent's personality traits."""

    name: str = Field(..., min_length=1, max_length=50, description="Agent's display name")
    role: str = Field(..., min_length=1, max_length=100, description="Agent's role/profession")
    speaking_style: str = Field(..., min_length=1, description="How the agent communicates")
    personality_traits: List[str] = Field(
        default_factory=list, description="Key personality characteristics"
    )
    expertise_areas: List[str] = Field(
        default_factory=list, description="Areas of specialized knowledge"
    )
    response_patterns: Dict[str, str] = Field(
        default_factory=dict, description="Template responses for common situations"
    )
    cultural_knowledge: Dict[str, Any] = Field(
        default_factory=dict, description="Cultural context and knowledge"
    )

    correction_style: str = Field(default="gentle", description="How corrections are delivered")
    enthusiasm_level: int = Field(
        default=5, ge=1, le=10, description="Energy level (1=calm, 10=very energetic)"
    )

    # Configurable behavior parameters
    formality_level: int = Field(
        default=5, ge=1, le=10, description="Language formality (1=very informal, 10=very formal)"
    )
    correction_frequency: int = Field(
        default=5, ge=1, le=10, description="How often to correct (1=rarely, 10=always)"
    )
    topic_focus: List[str] = Field(
        default_factory=list, description="Preferred conversation topics"
    )
    patience_level: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Patience with slow learners (1=impatient, 10=very patient)",
    )
    encouragement_frequency: int = Field(
        default=5, ge=1, le=10, description="How often to encourage (1=rarely, 10=constantly)"
    )

    @validator("correction_style")
    def validate_correction_style(cls, v):
        valid_styles = ["gentle", "direct", "encouraging"]
        if v not in valid_styles:
            raise ValueError(f"correction_style must be one of: {valid_styles}")
        return v

    @validator("personality_traits", "expertise_areas", "topic_focus")
    def validate_string_lists(cls, v):
        if not isinstance(v, list):
            raise ValueError("must be a list")
        for item in v:
            if not isinstance(item, str) or len(item.strip()) == 0:
                raise ValueError("list items must be non-empty strings")
        return v

    class Config:
        frozen = True  # Make immutable to prevent accidental changes
        json_schema_extra = {
            "example": {
                "name": "Marco",
                "role": "Friendly Conversationalist",
                "speaking_style": "casual_friendly",
                "personality_traits": ["encouraging", "patient", "enthusiastic"],
                "expertise_areas": ["conversation", "pronunciation"],
                "correction_style": "gentle",
                "enthusiasm_level": 8,
                "formality_level": 3,
                "correction_frequency": 6,
                "topic_focus": ["travel", "food", "daily_life"],
                "patience_level": 8,
                "encouragement_frequency": 8,
            }
        }


class BaseAgent(ABC):
    """
    Abstract base class for all Italian Teacher agents.

    This class defines the core interface that all agents must implement,
    ensuring consistent behavior across Marco, Professoressa Rossi,
    Nonna Giulia, and Lorenzo.
    """

    def __init__(
        self, agent_id: str, personality: AgentPersonality, config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the base agent.

        Args:
            agent_id: Unique identifier for this agent
            personality: Personality configuration for this agent
            config: Additional configuration options
        """
        self.agent_id = agent_id
        self.personality = personality
        self.config = config or {}
        self.status = AgentStatus.INACTIVE
        self.context: Optional[ConversationContext] = None
        self.activity_log: List[AgentActivityEvent] = []  # System events only
        self.created_at = datetime.now()
        self.last_active = datetime.now()

        # Agent state
        self._is_initialized = False
        self._error_count = 0
        self._max_errors = 5

        # Retention policy manager
        self.retention_manager = default_retention_manager

        # Set up structured logging
        self.logger = get_agent_logger(agent_id=agent_id, agent_name=personality.name)

    async def initialize(self) -> bool:
        """
        Initialize the agent with necessary resources.

        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.info(
            "agent_initializing",
            formality_level=self.personality.formality_level,
            correction_frequency=self.personality.correction_frequency,
        )
        try:
            await self._load_personality_config()
            await self._prepare_response_templates()
            self._is_initialized = True
            self.status = AgentStatus.ACTIVE
            self.logger.info("agent_initialized_successfully")
            return True
        except Exception as e:
            self.status = AgentStatus.ERROR
            self._error_count += 1
            self.logger.error(
                "agent_initialization_failed", error_type=type(e).__name__, error_message=str(e)
            )
            raise RuntimeError(f"Agent {self.agent_id} initialization failed: {e}")

    async def activate(self, context: ConversationContext) -> None:
        """
        Activate the agent for a conversation session.

        Args:
            context: Conversation context to work with
        """
        if not self._is_initialized:
            await self.initialize()

        self.context = context
        self.status = AgentStatus.ACTIVE
        self.last_active = datetime.now()

        # Reset cleanup timer on reactivation
        await self._reset_cleanup_timer()

        # Log activation
        self.logger.info(
            "agent_activated",
            session_id=context.session_id,
            user_id=context.user_id,
            user_language_level=context.user_language_level,
            learning_goals=context.learning_goals,
        )

        # Log activation in activity log
        self._log_activity(
            event_type="activated",
            description=f"Agent {self.personality.name} activated",
            metadata={"session_id": context.session_id},
        )

    async def deactivate(self) -> None:
        """Deactivate the agent and clean up resources."""
        session_id = self.context.session_id if self.context else None
        conversation_message_count = len(self.context.conversation_history) if self.context else 0

        self.status = AgentStatus.INACTIVE
        await self._save_conversation_state()

        # Schedule cleanup for this conversation
        await self._schedule_cleanup()

        # Log deactivation
        self.logger.info(
            "agent_deactivated",
            session_id=session_id,
            conversation_message_count=conversation_message_count,
        )

        # Log deactivation in activity log
        if self.context:
            self._log_activity(
                event_type="deactivated",
                description=f"Agent {self.personality.name} deactivated",
                metadata={"session_id": self.context.session_id},
            )

    @abstractmethod
    async def generate_response(
        self, user_message: str, context: Optional[ConversationContext] = None
    ) -> AgentMessage:
        """
        Generate a response to user input.

        This is the core method each agent must implement based on their
        personality and expertise.

        Args:
            user_message: The user's input message
            context: Optional conversation context

        Returns:
            AgentMessage with the agent's response
        """

    @abstractmethod
    async def can_handle_message(
        self, message: str, context: Optional[ConversationContext] = None
    ) -> float:
        """
        Determine if this agent can handle a given message.

        Args:
            message: The message to evaluate
            context: Optional conversation context

        Returns:
            Confidence score 0.0-1.0 (higher = more confident)
        """

    def add_message_to_conversation(self, message: AgentMessage) -> None:
        """Add a message to the conversation history."""
        if self.context:
            self.context.conversation_history.append(message)
            self.context.last_interaction = datetime.now()

    def get_recent_messages(self, count: int = 10) -> List[AgentMessage]:
        """Get recent messages from conversation history."""
        if not self.context or not self.context.conversation_history:
            return []
        return self.context.conversation_history[-count:]

    def get_my_messages(self, count: Optional[int] = None) -> List[AgentMessage]:
        """Get messages sent by this agent from conversation history."""
        if not self.context:
            return []

        my_messages = [
            msg for msg in self.context.conversation_history if msg.sender_id == self.agent_id
        ]

        if count is not None:
            return my_messages[-count:]
        return my_messages

    def is_available(self) -> bool:
        """Check if agent is available for new conversations."""
        return (
            self._is_initialized
            and self.status == AgentStatus.ACTIVE
            and self._error_count < self._max_errors
        )

    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information about the agent."""
        conversation_message_count = len(self.context.conversation_history) if self.context else 0
        my_message_count = len(self.get_my_messages())

        return {
            "agent_id": self.agent_id,
            "name": self.personality.name,
            "role": self.personality.role,
            "status": self.status.value,
            "is_available": self.is_available(),
            "error_count": self._error_count,
            "last_active": self.last_active.isoformat(),
            "conversation_message_count": conversation_message_count,
            "my_message_count": my_message_count,
            "activity_events": len(self.activity_log),
            "current_session": self.context.session_id if self.context else None,
        }

    async def handle_error(self, error: Exception, context: str) -> None:
        """Handle errors that occur during agent operations."""
        self._error_count += 1
        self.status = (
            AgentStatus.ERROR if self._error_count >= self._max_errors else AgentStatus.ACTIVE
        )

        # Log error with structured logging
        self.logger.error(
            "agent_error",
            error_type=type(error).__name__,
            error_message=str(error),
            error_count=self._error_count,
            context=context,
            max_errors_reached=self._error_count >= self._max_errors,
        )

        # Log error in activity log
        self._log_activity(
            event_type="error",
            description=f"Error in {context}: {str(error)}",
            metadata={
                "error_type": type(error).__name__,
                "error_count": self._error_count,
                "context": context,
            },
        )

    # Protected methods for subclasses to override

    async def _load_personality_config(self) -> None:
        """Load agent-specific personality configuration."""
        # Default implementation - subclasses can override

    async def _prepare_response_templates(self) -> None:
        """Prepare response templates based on personality."""
        # Default implementation - subclasses can override

    async def _save_conversation_state(self) -> None:
        """Save current conversation state for persistence."""
        # Default implementation - subclasses can override

    def _create_response_message(
        self,
        content: str,
        message_type: MessageType = MessageType.CONVERSATION,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        """Helper to create properly formatted response messages."""
        return AgentMessage(
            sender_id=self.agent_id,
            message_type=message_type,
            content=content,
            priority=priority,
            metadata=metadata or {},
        )

    async def _schedule_cleanup(self) -> None:
        """Schedule automatic deletion of conversation data."""
        if not self.context:
            return

        # Get the next cleanup date based on user's retention preference
        next_cleanup_date = self.retention_manager.get_next_cleanup_date(
            preference=self.context.retention_preference, deactivation_date=datetime.now()
        )

        if not next_cleanup_date:
            return  # No cleanup scheduled (permanent retention)

        # Cancel existing cleanup job if any
        if self.context.cleanup_job_id:
            await self._cancel_cleanup_job(self.context.cleanup_job_id)

        # Schedule new cleanup job
        cleanup_job_id = await self._create_cleanup_job(
            session_id=self.context.session_id,
            deletion_date=next_cleanup_date,
            retention_preference=self.context.retention_preference,
        )

        # Update context with new cleanup info
        self.context.scheduled_deletion = next_cleanup_date
        self.context.cleanup_job_id = cleanup_job_id

    async def _reset_cleanup_timer(self) -> None:
        """Reset cleanup timer when agent is reactivated."""
        if not self.context:
            return

        # Cancel existing cleanup job
        if self.context.cleanup_job_id:
            await self._cancel_cleanup_job(self.context.cleanup_job_id)

        # Clear scheduled deletion
        self.context.scheduled_deletion = None
        self.context.cleanup_job_id = None

        # Note: New cleanup will be scheduled when agent is deactivated again

    async def _create_cleanup_job(
        self, session_id: str, deletion_date: datetime, retention_preference: RetentionPreference
    ) -> str:
        """
        Create a cleanup job for automatic data deletion.

        This is a placeholder that should be implemented by a concrete cleanup service.
        Could use: Celery, Redis, database scheduled jobs, cloud functions, etc.

        Args:
            session_id: Session to clean up
            deletion_date: When to delete the data
            retention_preference: User's retention preference for determining what to delete

        Returns:
            Job ID for tracking/cancellation
        """
        # Default implementation - just generate a job ID
        # In production, this would interface with a real job scheduler
        job_id = (
            f"cleanup_{session_id}_{retention_preference.value}_{int(deletion_date.timestamp())}"
        )

        # TODO: Integrate with actual job scheduler (Celery, etc.)
        # The job scheduler would use retention_preference to determine what data to delete
        print(f"Scheduled {retention_preference.value} cleanup job {job_id} for {deletion_date}")

        return job_id

    async def _cancel_cleanup_job(self, job_id: str) -> None:
        """
        Cancel a scheduled cleanup job.

        Args:
            job_id: Job to cancel
        """
        # Default implementation - log cancellation
        # In production, this would interface with the job scheduler
        print(f"Cancelled cleanup job {job_id}")

        # TODO: Integrate with actual job scheduler to cancel the job

    def _log_activity(
        self, event_type: str, description: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an activity event for this agent."""
        activity_event = AgentActivityEvent(
            event_type=event_type, description=description, metadata=metadata or {}
        )
        self.activity_log.append(activity_event)

        # Keep activity log from getting too large (keep last 100 events)
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[-100:]

    def get_recent_activities(self, count: int = 10) -> List[AgentActivityEvent]:
        """Get recent activity events for this agent."""
        return self.activity_log[-count:] if self.activity_log else []

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.agent_id}, {self.personality.name}, {self.status.value})>"
