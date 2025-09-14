"""
Encouragement and Motivation System for Italian Teacher Agents

This module provides intelligent encouragement and motivation systems that:
1. Track user progress and emotional state
2. Provide contextual encouragement based on performance
3. Adapt motivation strategies to individual learning patterns
4. Integrate with agent personalities for authentic responses
"""

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MotivationalTrigger(Enum):
    """Events that trigger motivational responses."""

    CORRECT_ANSWER = "correct_answer"
    MISTAKE_MADE = "mistake_made"
    PROGRESS_MILESTONE = "progress_milestone"
    STRUGGLE_DETECTED = "struggle_detected"
    COMEBACK_AFTER_BREAK = "comeback_after_break"
    NEW_TOPIC_STARTED = "new_topic_started"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    FRUSTRATION_DETECTED = "frustration_detected"
    CONFIDENCE_BOOST_NEEDED = "confidence_boost_needed"


class MotivationLevel(Enum):
    """User's current motivation/energy state."""

    HIGH = "high"  # Confident, engaged, making progress
    MEDIUM = "medium"  # Steady progress, occasional mistakes
    LOW = "low"  # Struggling, frequent errors, may need support
    FRUSTRATED = "frustrated"  # Showing signs of frustration or giving up


@dataclass
class ProgressMetrics:
    """Tracks user learning progress for motivation calculations."""

    correct_responses: int = 0
    total_responses: int = 0
    consecutive_correct: int = 0
    consecutive_incorrect: int = 0
    topics_mastered: List[str] = field(default_factory=list)
    current_streak: int = 0
    best_streak: int = 0
    session_duration: timedelta = field(default_factory=lambda: timedelta(0))
    mistakes_this_session: int = 0
    improvements_noted: List[str] = field(default_factory=list)

    @property
    def accuracy_rate(self) -> float:
        """Calculate current accuracy percentage."""
        if self.total_responses == 0:
            return 0.0
        return (self.correct_responses / self.total_responses) * 100

    @property
    def is_struggling(self) -> bool:
        """Detect if user is currently struggling."""
        return (
            self.consecutive_incorrect >= 3
            or self.accuracy_rate < 30
            or self.mistakes_this_session > 5
        )

    @property
    def is_excelling(self) -> bool:
        """Detect if user is performing well."""
        return self.consecutive_correct >= 3 or self.accuracy_rate > 80 or self.current_streak >= 5


@dataclass
class EncouragementResponse:
    """Structured encouragement response with context."""

    message_template: str
    emotional_tone: str  # excited, gentle, proud, supportive, etc.
    personalization_data: Dict[str, Any] = field(default_factory=dict)
    follow_up_suggestion: Optional[str] = None
    confidence_boost: bool = False


class MotivationStrategy(BaseModel):
    """Configuration for different motivation approaches."""

    name: str
    triggers: List[MotivationalTrigger]
    personality_fit: List[str]  # Which personality types this works for
    frequency_limit: int = Field(default=3, description="Max uses per session")
    cooldown_minutes: int = Field(default=5, description="Minimum time between uses")


class MotivationSystem:
    """
    Core motivation and encouragement system.

    Provides intelligent, context-aware encouragement that:
    - Adapts to user's current performance and emotional state
    - Integrates with agent personalities
    - Tracks effectiveness of different motivation strategies
    - Prevents over-encouragement or robotic responses
    """

    def __init__(self, agent_personality: Dict[str, Any]):
        self.agent_personality = agent_personality
        self.progress_metrics = ProgressMetrics()
        self.strategy_usage = {}  # Track strategy usage and effectiveness
        self.last_encouragement_time = {}
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize motivation strategies based on agent personality."""
        self.strategies = {
            "celebration": MotivationStrategy(
                name="celebration",
                triggers=[
                    MotivationalTrigger.CORRECT_ANSWER,
                    MotivationalTrigger.PROGRESS_MILESTONE,
                ],
                personality_fit=["enthusiastic", "encouraging", "warm"],
                frequency_limit=5,
                cooldown_minutes=2,
            ),
            "gentle_support": MotivationStrategy(
                name="gentle_support",
                triggers=[MotivationalTrigger.MISTAKE_MADE, MotivationalTrigger.STRUGGLE_DETECTED],
                personality_fit=["patient", "supportive", "understanding"],
                frequency_limit=10,
                cooldown_minutes=1,
            ),
            "progress_acknowledgment": MotivationStrategy(
                name="progress_acknowledgment",
                triggers=[MotivationalTrigger.PROGRESS_MILESTONE, MotivationalTrigger.SESSION_END],
                personality_fit=["encouraging", "optimistic"],
                frequency_limit=3,
                cooldown_minutes=10,
            ),
            "confidence_building": MotivationStrategy(
                name="confidence_building",
                triggers=[
                    MotivationalTrigger.CONFIDENCE_BOOST_NEEDED,
                    MotivationalTrigger.FRUSTRATION_DETECTED,
                ],
                personality_fit=["supportive", "motivational", "inspiring"],
                frequency_limit=4,
                cooldown_minutes=5,
            ),
        }

    def update_progress(
        self, correct: bool, topic: Optional[str] = None, user_message: Optional[str] = None
    ) -> None:
        """Update progress tracking with latest user interaction."""
        self.progress_metrics.total_responses += 1

        if correct:
            self.progress_metrics.correct_responses += 1
            self.progress_metrics.consecutive_correct += 1
            self.progress_metrics.consecutive_incorrect = 0
            self.progress_metrics.current_streak += 1
            self.progress_metrics.best_streak = max(
                self.progress_metrics.best_streak, self.progress_metrics.current_streak
            )
        else:
            self.progress_metrics.consecutive_incorrect += 1
            self.progress_metrics.consecutive_correct = 0
            self.progress_metrics.current_streak = 0
            self.progress_metrics.mistakes_this_session += 1

        if topic and topic not in self.progress_metrics.topics_mastered:
            # Simple mastery detection - 3+ correct in a row for this topic
            if correct and self.progress_metrics.consecutive_correct >= 3:
                self.progress_metrics.topics_mastered.append(topic)

    def detect_motivation_level(self) -> MotivationLevel:
        """Analyze current user state and return motivation level."""
        metrics = self.progress_metrics

        # Check for frustration indicators
        if (
            metrics.consecutive_incorrect >= 4
            or metrics.mistakes_this_session > 7
            or (metrics.total_responses > 5 and metrics.accuracy_rate < 25)
        ):
            return MotivationLevel.FRUSTRATED

        # Check for low motivation
        if metrics.is_struggling or metrics.accuracy_rate < 50:
            return MotivationLevel.LOW

        # Check for high motivation
        if metrics.is_excelling or metrics.current_streak >= 5 or len(metrics.topics_mastered) > 0:
            return MotivationLevel.HIGH

        return MotivationLevel.MEDIUM

    def get_encouragement(
        self, trigger: MotivationalTrigger, context: Optional[Dict[str, Any]] = None
    ) -> Optional[EncouragementResponse]:
        """
        Generate contextual encouragement based on trigger and current state.

        Args:
            trigger: What prompted the need for encouragement
            context: Additional context (topic, user message, etc.)

        Returns:
            EncouragementResponse with personalized message, or None if no encouragement needed
        """
        context = context or {}
        motivation_level = self.detect_motivation_level()

        # Find appropriate strategy
        strategy = self._select_strategy(trigger, motivation_level)
        if not strategy or not self._should_encourage(strategy):
            return None

        # Generate response based on agent personality and context
        response = self._generate_response(strategy, trigger, motivation_level, context)

        # Track usage
        self._record_encouragement(strategy)

        return response

    def _select_strategy(
        self, trigger: MotivationalTrigger, motivation_level: MotivationLevel
    ) -> Optional[MotivationStrategy]:
        """Select most appropriate motivation strategy."""
        candidate_strategies = []

        for strategy in self.strategies.values():
            if trigger in strategy.triggers:
                # Check if strategy fits agent personality
                personality_traits = self.agent_personality.get("personality_traits", [])
                if any(trait in strategy.personality_fit for trait in personality_traits):
                    candidate_strategies.append(strategy)

        if not candidate_strategies:
            return None

        # Prefer strategies that haven't been overused
        candidate_strategies.sort(key=lambda s: self.strategy_usage.get(s.name, 0))
        return candidate_strategies[0]

    def _should_encourage(self, strategy: MotivationStrategy) -> bool:
        """Check if we should use this strategy (frequency limits, cooldowns)."""
        strategy_name = strategy.name
        now = datetime.now()

        # Check frequency limit
        usage_count = self.strategy_usage.get(strategy_name, 0)
        if usage_count >= strategy.frequency_limit:
            return False

        # Check cooldown
        last_use = self.last_encouragement_time.get(strategy_name)
        if last_use:
            time_since_last = now - last_use
            if time_since_last < timedelta(minutes=strategy.cooldown_minutes):
                return False

        return True

    def _generate_response(
        self,
        strategy: MotivationStrategy,
        trigger: MotivationalTrigger,
        motivation_level: MotivationLevel,
        context: Dict[str, Any],
    ) -> EncouragementResponse:
        """Generate actual encouragement response."""
        agent_name = self.agent_personality.get("name", "Agent")
        enthusiasm = self.agent_personality.get("enthusiasm_level", 5)

        # Get base templates for Marco specifically
        if agent_name.lower() == "marco":
            templates = self._get_marco_templates(strategy.name, trigger, motivation_level)
        else:
            templates = self._get_generic_templates(strategy.name, trigger, motivation_level)

        # Select template based on context and randomization
        template = random.choice(templates)

        # Determine emotional tone
        tone = self._determine_tone(trigger, motivation_level, enthusiasm)

        # Add personalization data
        personalization = {
            "streak": self.progress_metrics.current_streak,
            "accuracy": f"{self.progress_metrics.accuracy_rate:.0f}%",
            "topics_mastered": len(self.progress_metrics.topics_mastered),
            "agent_enthusiasm": enthusiasm,
            **context,
        }

        # Generate follow-up suggestion if appropriate
        follow_up = self._generate_follow_up(trigger, motivation_level, context)

        return EncouragementResponse(
            message_template=template,
            emotional_tone=tone,
            personalization_data=personalization,
            follow_up_suggestion=follow_up,
            confidence_boost=motivation_level in [MotivationLevel.LOW, MotivationLevel.FRUSTRATED],
        )

    def _get_marco_templates(
        self, strategy: str, trigger: MotivationalTrigger, motivation_level: MotivationLevel
    ) -> List[str]:
        """Get Marco-specific encouragement templates."""
        if strategy == "celebration" and motivation_level == MotivationLevel.HIGH:
            return [
                "Fantastico! Stai andando benissimo! 🎉",
                "Perfetto! Hai capito tutto! Bravissimo!",
                "Che bravo! I tuoi progressi sono incredibili!",
                "Eccellente! Stai diventando sempre più fluente!",
                "Magnifico! Continua così, sei sulla strada giusta!",
            ]
        elif strategy == "gentle_support" and motivation_level == MotivationLevel.LOW:
            return [
                "Non ti preoccupare, succede a tutti! Proviamo insieme...",
                "Tranquillo, gli errori sono parte dell'apprendimento!",
                "Dai, forza! Stai imparando, è normale sbagliare!",
                "Pazienza, stai facendo progressi anche se non li vedi!",
                "Nessun problema! Ogni errore è un passo avanti!",
            ]
        elif strategy == "confidence_building":
            return [
                "Ricorda quanto sei migliorato dall'inizio!",
                "Hai già imparato tantissimo, continua così!",
                "I tuoi progressi sono fantastici, non mollare!",
                "Stai diventando sempre più bravo, fidati!",
                "Ogni giorno impari qualcosa di nuovo - è bellissimo!",
            ]
        else:
            return [
                "Molto bene! Continua così!",
                "Bravissimo! Stai migliorando!",
                "Perfetto! Hai fatto bene!",
            ]

    def _get_generic_templates(
        self, strategy: str, trigger: MotivationalTrigger, motivation_level: MotivationLevel
    ) -> List[str]:
        """Generic encouragement templates for other agents."""
        return [
            "Well done! Keep up the great work!",
            "Excellent progress! You're doing wonderfully!",
            "That's correct! You're learning so well!",
        ]

    def _determine_tone(
        self, trigger: MotivationalTrigger, motivation_level: MotivationLevel, enthusiasm: int
    ) -> str:
        """Determine appropriate emotional tone for the response."""
        if motivation_level == MotivationLevel.FRUSTRATED:
            return "gentle" if enthusiasm <= 5 else "supportive"
        elif motivation_level == MotivationLevel.LOW:
            return "encouraging"
        elif motivation_level == MotivationLevel.HIGH:
            return "excited" if enthusiasm >= 7 else "proud"
        else:
            return "positive"

    def _generate_follow_up(
        self,
        trigger: MotivationalTrigger,
        motivation_level: MotivationLevel,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Generate follow-up suggestions when appropriate."""
        if motivation_level == MotivationLevel.FRUSTRATED:
            return "Vuoi provare qualcosa di più semplice?"
        elif motivation_level == MotivationLevel.HIGH:
            return "Sei pronto per una sfida più difficile?"
        elif trigger == MotivationalTrigger.PROGRESS_MILESTONE:
            return "Quale argomento ti piacerebbe esplorare ora?"
        return None

    def _record_encouragement(self, strategy: MotivationStrategy) -> None:
        """Record that we used this encouragement strategy."""
        strategy_name = strategy.name
        self.strategy_usage[strategy_name] = self.strategy_usage.get(strategy_name, 0) + 1
        self.last_encouragement_time[strategy_name] = datetime.now()

    def reset_session(self) -> None:
        """Reset session-specific metrics for new conversation."""
        self.progress_metrics.mistakes_this_session = 0
        self.progress_metrics.session_duration = timedelta(0)
        self.strategy_usage.clear()
        self.last_encouragement_time.clear()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session progress for motivation insights."""
        return {
            "accuracy": self.progress_metrics.accuracy_rate,
            "total_responses": self.progress_metrics.total_responses,
            "current_streak": self.progress_metrics.current_streak,
            "best_streak": self.progress_metrics.best_streak,
            "topics_mastered": self.progress_metrics.topics_mastered,
            "motivation_level": self.detect_motivation_level().value,
            "encouragements_given": sum(self.strategy_usage.values()),
        }
