"""
Marco Agent Implementation

Marco is the friendly, enthusiastic Italian conversationalist who serves as the primary
conversation partner for users learning Italian. He integrates advanced motivation
and error tolerance systems to provide personalized, encouraging language learning.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.agent_events import AgentEvent, EventType
from src.core.base_agent import AgentResponse, BaseAgent, ConversationContext
from src.core.error_tolerance import (
    CorrectionResponse,
    DetectedError,
    ErrorToleranceConfig,
    ErrorToleranceSystem,
)
from src.pre_lora.motivation_system import (
    EncouragementResponse,
    MotivationalTrigger,
    MotivationSystem,
)
from src.prompts.marco_system_prompt import (
    get_marco_conversation_starters,
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

    def __init__(self, agent_id: str, personality_config: Dict[str, Any]):
        super().__init__(agent_id, personality_config)

        # Initialize motivation system
        self.motivation_system = MotivationSystem(personality_config)

        # Initialize error tolerance system
        error_config = ErrorToleranceConfig(
            max_corrections_per_message=2,
            correction_frequency=0.6,  # Marco is moderately corrective
            encourage_before_correct=True,
            patience_factor=1.2,  # Marco is quite patient
            cultural_sensitivity=True,
        )
        self.error_tolerance = ErrorToleranceSystem(personality_config, error_config)

        # Marco's conversation state
        self.current_topic = None
        self.user_level = "beginner"  # Will be updated based on interaction
        self.session_start = datetime.now()

        logger.info(
            f"Marco agent {agent_id} initialized with advanced motivation and error tolerance"
        )

    async def can_handle_message(self, message: str, context: ConversationContext) -> bool:
        """
        Marco can handle most conversational messages, especially:
        - General conversation practice
        - Beginner to intermediate level interactions
        - Cultural questions about Italy
        - Encouragement and motivation needs
        """
        message_lower = message.lower()

        # Always handle greetings and basic conversation
        greeting_indicators = ["ciao", "buongiorno", "buonasera", "hello", "hi", "come stai"]
        if any(greeting in message_lower for greeting in greeting_indicators):
            return True

        # Handle conversation practice
        conversation_indicators = ["parliamo", "chiacchiera", "conversation", "talk", "practice"]
        if any(indicator in message_lower for indicator in conversation_indicators):
            return True

        # Handle cultural topics
        cultural_indicators = ["italia", "italian", "cultura", "tradizioni", "food", "travel"]
        if any(indicator in message_lower for indicator in cultural_indicators):
            return True

        # Handle encouragement needs (detected through context)
        if context.metadata.get("needs_encouragement", False):
            return True

        # Default: Marco can handle most general conversation
        return True

    async def generate_response(self, message: str, context: ConversationContext) -> str:
        """
        Generate Marco's response with integrated motivation and error tolerance.
        """
        try:
            # Detect user level from context or message complexity
            self._update_user_level(message, context)

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

            # Simulate LLM response (in real implementation, this would call Llama 3.1)
            base_response = await self._generate_conversational_response(message, context)

            # Add motivational elements
            encouragement = await self._add_encouragement(message, context, has_errors)

            # Add error corrections
            corrections = await self._add_error_corrections(detected_errors, context)

            # Combine response components naturally
            final_response = self._combine_response_elements(
                base_response, encouragement, corrections
            )

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
                # Provide encouragement for struggling user
                encouragement = self.motivation_system.get_encouragement(
                    MotivationalTrigger.CONFIDENCE_BOOST_NEEDED, event.payload
                )
                if encouragement:
                    return AgentResponse(
                        message=encouragement.message_template,
                        metadata={"type": "encouragement", "tone": encouragement.emotional_tone},
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
        self, message: str, context: ConversationContext
    ) -> str:
        """Generate the core conversational response (simulated LLM call)."""
        # This would be replaced with actual Llama 3.1 8B call in production
        message_lower = message.lower()

        # Greeting responses
        if any(greeting in message_lower for greeting in ["ciao", "buongiorno", "hello"]):
            import random

            greetings = get_marco_conversation_starters()
            return random.choice(greetings)

        # Food-related responses
        if any(food in message_lower for food in ["pizza", "pasta", "cibo", "mangiare"]):
            return "Ah, il cibo italiano! È la mia passione! Qual è il tuo piatto italiano preferito? Io adoro la carbonara autentica di Roma!"

        # Travel responses
        if any(travel in message_lower for travel in ["viaggio", "italia", "visit", "travel"]):
            return "L'Italia è bellissima! Hai mai visitato Milano? È la mia città! C'è il Duomo, la Scala... Dove vorresti andare in Italia?"

        # Default encouraging response
        return "Interessante! Dimmi di più! Mi piace ascoltare le tue storie. Come ti senti oggi con l'italiano?"

    async def _add_encouragement(
        self, message: str, context: ConversationContext, has_errors: bool
    ) -> Optional[EncouragementResponse]:
        """Add appropriate encouragement based on user performance."""

        # Determine trigger
        if has_errors:
            trigger = MotivationalTrigger.MISTAKE_MADE
        elif self.motivation_system.progress_metrics.consecutive_correct >= 3:
            trigger = MotivationalTrigger.PROGRESS_MILESTONE
        elif context.metadata.get("user_struggling", False):
            trigger = MotivationalTrigger.STRUGGLE_DETECTED
        else:
            trigger = MotivationalTrigger.CORRECT_ANSWER

        # Get encouragement
        encouragement = self.motivation_system.get_encouragement(
            trigger, {"topic": self.current_topic, "user_level": self.user_level}
        )

        return encouragement

    async def _add_error_corrections(
        self, detected_errors: List[DetectedError], context: ConversationContext
    ) -> List[CorrectionResponse]:
        """Add appropriate error corrections based on tolerance settings."""
        corrections = []

        for error in detected_errors:
            if self.error_tolerance.should_correct_error(
                error, {"user_level": self.user_level, "topic": self.current_topic}
            ):
                correction = self.error_tolerance.generate_correction(
                    error,
                    {"user_level": self.user_level, "agent_personality": self.personality.dict()},
                )
                corrections.append(correction)

        return corrections

    def _combine_response_elements(
        self,
        base_response: str,
        encouragement: Optional[EncouragementResponse],
        corrections: List[CorrectionResponse],
    ) -> str:
        """Naturally combine response elements into coherent message."""
        response_parts = []

        # Add encouragement first if present
        if encouragement:
            response_parts.append(encouragement.message_template)

        # Add corrections naturally
        for correction in corrections:
            if correction.encouragement:
                response_parts.append(correction.encouragement)
            response_parts.append(correction.correction_text)

        # Add main response
        response_parts.append(base_response)

        # Add follow-up suggestions
        for correction in corrections:
            if correction.follow_up_practice:
                response_parts.append(correction.follow_up_practice)

        if encouragement and encouragement.follow_up_suggestion:
            response_parts.append(encouragement.follow_up_suggestion)

        # Join naturally with Italian connectors
        connectors = [" ", " E poi, ", " Inoltre, ", " "]
        result = ""
        for i, part in enumerate(response_parts):
            if i > 0:
                connector = connectors[min(i - 1, len(connectors) - 1)]
                result += connector
            result += part

        return result

    def _update_user_level(self, message: str, context: ConversationContext) -> None:
        """Update assessment of user's Italian level based on message complexity."""
        # Simple heuristic - in production this would use more sophisticated analysis
        message_complexity = self._assess_message_complexity(message)
        current_accuracy = self.motivation_system.progress_metrics.accuracy_rate

        if message_complexity >= 0.8 and current_accuracy >= 80:
            self.user_level = "advanced"
        elif message_complexity >= 0.5 and current_accuracy >= 60:
            self.user_level = "intermediate"
        else:
            self.user_level = "beginner"

    def _assess_message_complexity(self, message: str) -> float:
        """Simple complexity assessment (0-1 scale)."""
        # Count Italian words, complex grammar, etc.
        italian_indicators = ["che", "quando", "perché", "dove", "come", "mentre"]
        italian_count = sum(1 for indicator in italian_indicators if indicator in message.lower())

        word_count = len(message.split())
        if word_count == 0:
            return 0.0

        complexity = (italian_count + word_count / 10) / 5
        return min(1.0, complexity)

    def _update_conversation_state(
        self, user_message: str, response: str, context: ConversationContext
    ) -> None:
        """Update internal conversation state tracking."""
        # Extract topic from message (simplified)
        topics = {
            "cibo": ["pizza", "pasta", "cibo", "mangiare", "ristorante"],
            "viaggio": ["viaggio", "italia", "visit", "travel", "città"],
            "famiglia": ["famiglia", "family", "parents", "mother", "father"],
            "lavoro": ["lavoro", "work", "job", "profession"],
        }

        for topic, keywords in topics.items():
            if any(keyword in user_message.lower() for keyword in keywords):
                self.current_topic = topic
                break

    def _find_relevant_cultural_fact(self, topic: str, cultural_facts: List[str]) -> Optional[str]:
        """Find a relevant cultural fact for the given topic."""
        topic_lower = topic.lower()
        for fact in cultural_facts:
            if any(word in fact.lower() for word in topic_lower.split()):
                return fact
        return None

    def can_handle_handoff(self, handoff_data: Dict[str, Any]) -> bool:
        """Determine if Marco should accept a conversation handoff."""
        handoff_data.get("reason", "")
        topics = handoff_data.get("topics", [])

        # Marco accepts handoffs for conversation practice
        marco_topics = ["conversation", "general", "encouragement", "cultural", "food", "travel"]
        return any(topic in marco_topics for topic in topics)

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
