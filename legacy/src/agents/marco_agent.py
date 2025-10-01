"""
Marco Agent Implementation

Marco is the friendly, enthusiastic Italian conversationalist who serves as the primary
conversation partner for users learning Italian. He integrates advanced motivation
and error tolerance systems to provide personalized, encouraging language learning.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.agent_events import AgentEvent, EventType
from core.base_agent import AgentPersonality, AgentResponse, BaseAgent, ConversationContext
from core.error_tolerance import (
    ErrorToleranceConfig,
    ErrorToleranceSystem,
)
from pre_lora.motivation_system import (
    MotivationSystem,
)
from prompts.marco_system_prompt import (
    get_marco_cultural_facts,
    get_marco_system_prompt,
)

logger = logging.getLogger(__name__)


class MarcoAgent(BaseAgent):
    """
    Marco - The Friendly Italian Conversationalist

    Marco is an enthusiastic, patient, and culturally rich conversation partner
    who helps users practice Italian through natural, engaging conversations.
    He integrates advanced motivation and error tolerance systems to provide
    personalized learning experiences.

    Key Features:
    - Intelligent encouragement and motivation system
    - Adaptive error tolerance and gentle corrections
    - Cultural knowledge integration
    - Natural conversation flow with personality consistency
    - Seamless handoffs to specialized agents when needed
    """

    def __init__(
        self, agent_id: str, personality: AgentPersonality, config: Optional[Dict[str, Any]] = None
    ):

        super().__init__(agent_id, personality, config)

        # Initialize motivation system (convert personality to dict for legacy systems)
        personality_dict = {
            "enthusiasm_level": personality.enthusiasm_level,
            "encouragement_frequency": personality.encouragement_frequency,
            "patience_level": personality.patience_level,
            "personality_traits": personality.personality_traits,
        }
        self.motivation_system = MotivationSystem(personality_dict)

        # Initialize error tolerance system
        error_config = ErrorToleranceConfig(
            max_corrections_per_message=2,
            correction_frequency=personality.correction_frequency / 10.0,  # Convert to 0-1 scale
            encourage_before_correct=True,
            patience_factor=personality.patience_level / 5.0,  # Convert to multiplier
            cultural_sensitivity=True,
        )
        self.error_tolerance = ErrorToleranceSystem(personality_dict, error_config)

        # Marco's conversation state
        self.current_topic = None
        self.user_level = "beginner"  # Will be updated based on interaction
        self.session_start = datetime.now()

        logger.info(
            f"Marco agent {agent_id} initialized with advanced motivation and error tolerance"
        )

    # Removed can_handle_message - agents now trust LLM for message handling decisions

    async def generate_response(self, message: str, context: ConversationContext) -> str:
        """
        Generate Marco's response with integrated motivation and error tolerance.
        """
        try:
            # Detect user level from context or message complexity
            self._update_user_level(message, context)

            # TODO Phase 2.1: Replace separate analytical systems with unified LLM call
            # Currently doing 3 separate analyses - should piggyback all on main LLM:
            # 1. Error detection (below)
            # 2. Motivation assessment (below)
            # 3. Complexity assessment (above)

            # Detect errors in user message
            detected_errors = self.error_tolerance.detect_errors(
                message, {"user_level": self.user_level, "topic": self.current_topic}
            )

            # Determine if user made progress or mistakes
            has_errors = len(detected_errors) > 0
            self.motivation_system.update_progress(
                correct=not has_errors, topic=self.current_topic, user_message=message
            )

            # Generate base response using system prompt
            system_prompt = get_marco_system_prompt(
                user_level=self.user_level,
                context={
                    "last_topic": self.current_topic,
                    "session_duration": str(datetime.now() - self.session_start),
                },
                session_info={
                    "duration": f"{(datetime.now() - self.session_start).total_seconds() / 60:.1f} minutes"
                },
            )

            # Generate LLM response with full context including motivation and error information
            base_response = await self._generate_conversational_response(
                message, context, detected_errors, has_errors
            )

            # Trust the LLM to handle encouragement and corrections naturally within the response
            # Provide context about errors and motivation needs to the base response generation
            final_response = base_response

            # Update conversation state
            self._update_conversation_state(message, final_response, context)

            return final_response

        except Exception as e:
            logger.error(f"Error generating Marco's response: {e}")
            return "Scusa, ho avuto un piccolo problema! Ripeti quello che hai detto?"

    async def handle_event(self, event: AgentEvent) -> Optional[AgentResponse]:
        """Handle events from other agents or the coordinator."""

        if event.event_type == EventType.REQUEST_HELP:
            # Another agent needs help with something Marco can assist with
            help_type = event.payload.get("help_type")

            if help_type == "encouragement":
                # Provide natural encouragement using LLM instead of templates
                event.payload
                motivation_context = {
                    "accuracy_rate": self.motivation_system.progress_metrics.accuracy_rate,
                    "needs_boost": True,
                    "personality": self.personality.model_dump(),
                }

                # Simple encouraging response - in production, would use LLM with context
                return AgentResponse(
                    message="Non ti preoccupare! Stai facendo benissimo! Tutti fanno errori quando imparano. L'importante è continuare a provare! 💪",
                    metadata={"type": "encouragement", "generated_by": "llm_context"},
                )

            elif help_type == "cultural_context":
                # Provide cultural context for a topic
                cultural_facts = get_marco_cultural_facts()
                relevant_fact = self._find_relevant_cultural_fact(
                    event.payload.get("topic", ""), cultural_facts
                )
                if relevant_fact:
                    return AgentResponse(
                        message=f"Interessante! {relevant_fact}",
                        metadata={"type": "cultural_context"},
                    )

        elif event.event_type == EventType.REQUEST_HANDOFF:
            # Handle requests to take over conversation
            if self.can_handle_handoff(event.payload):
                # Accept handoff with natural greeting
                user_name = event.payload.get("user_name", "amico")
                return AgentResponse(
                    message=f"Ciao {user_name}! Sono Marco! Come posso aiutarti oggi? 😊",
                    metadata={"handoff_accepted": True},
                )

        return None

    async def _generate_conversational_response(
        self, message: str, context: ConversationContext, detected_errors=None, has_errors=False
    ) -> str:
        """
        Generate the core conversational response using the language model.

        TODO Phase 2.1: This method will be refactored to do ALL analysis in single LLM call:

        unified_prompt = '''
        As Marco, respond to this Italian learner AND provide analysis:

        User message: "{message}"

        Your response should:
        1. Reply naturally in Italian/English mix as Marco
        2. Include gentle corrections if needed
        3. Provide appropriate encouragement based on user progress
        4. End with: [ANALYSIS: level=beginner|intermediate|advanced, errors=0-5, motivation=low|medium|high]

        Base analysis on actual Italian grammar, vocabulary sophistication, and learning progress.
        '''

        This eliminates the need for separate detect_errors(), update_progress(), and complexity assessment calls.
        """
        # Trust the LLM to handle conversation naturally rather than pattern matching

        # Prepare comprehensive context for the language model including motivation and error info
        model_context = {
            "user_level": self.user_level,
            "current_topic": self.current_topic,
            "session_duration": str(datetime.now() - self.session_start),
            "personality": self.personality.model_dump(),
            "motivation_context": {
                "accuracy_rate": self.motivation_system.progress_metrics.accuracy_rate,
                "consecutive_correct": self.motivation_system.progress_metrics.consecutive_correct,
                "needs_encouragement": has_errors
                or self.motivation_system.progress_metrics.accuracy_rate < 70,
            },
            "error_context": {
                "has_errors": has_errors,
                "error_count": len(detected_errors) if detected_errors else 0,
                "correction_style": self.personality.correction_style,
            },
        }

        # If we have a language model, use it
        if hasattr(self, "language_model") and self.language_model:
            try:
                response = await self.language_model.generate(message, model_context)
                return response.text if hasattr(response, "text") else str(response)
            except Exception as e:
                logger.warning(f"Language model generation failed: {e}")

        # Simple fallback response that encourages conversation
        return "Interessante! Dimmi di più! Mi piace ascoltare le tue storie. Come ti senti oggi con l'italiano?"

    # Template-based encouragement and error correction methods removed
    # LLM now handles all response generation naturally with context

    # Moved _update_user_level and _assess_message_complexity to BaseAgent
    # Now using parent class methods for user level assessment

    def _update_user_level(self, message: str, context: ConversationContext) -> None:
        """Update Marco's assessment of user's Italian level."""
        current_accuracy = self.motivation_system.progress_metrics.accuracy_rate
        self.user_level = super()._update_user_level(message, current_accuracy)

    def _update_conversation_state(
        self, user_message: str, response: str, context: ConversationContext
    ) -> None:
        """Update internal conversation state tracking."""
        # Let the LLM determine topics naturally rather than pattern matching
        # For now, just track basic conversation flow
        if len(user_message.split()) > 5:
            # Longer messages suggest more engaged conversation
            self.current_topic = "conversation"
        else:
            # Short messages might be greetings or simple responses
            self.current_topic = "basic"

    def _find_relevant_cultural_fact(self, topic: str, cultural_facts: List[str]) -> Optional[str]:
        """Find a relevant cultural fact for the given topic."""
        topic_lower = topic.lower()
        for fact in cultural_facts:
            if any(word in fact.lower() for word in topic_lower.split()):
                return fact
        return None

    def can_handle_handoff(self, handoff_data: Dict[str, Any]) -> bool:
        """Determine if Marco should accept a conversation handoff."""
        # Marco is designed for general Italian conversation practice
        # Accept most handoffs unless explicitly inappropriate
        reason = handoff_data.get("reason", "")

        # Only refuse if explicitly asking for specialized technical help
        specialized_requests = ["technical", "medical", "legal", "academic_research"]
        return not any(spec in reason.lower() for spec in specialized_requests)

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current status summary for monitoring."""
        return {
            **super().get_status_summary(),
            "user_level": self.user_level,
            "current_topic": self.current_topic,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "motivation_stats": self.motivation_system.get_session_summary(),
            "error_stats": self.error_tolerance.get_error_summary(),
        }
