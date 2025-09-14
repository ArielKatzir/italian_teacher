"""
Motivation Context System - Post-LoRA Compatible

This module provides lightweight motivation context and metrics tracking that will
remain after LoRA training. The heavy template-based systems in pre_lora/ will be
removed once the model is fine-tuned.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


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


class MotivationContext:
    """
    Lightweight motivation context system for post-LoRA architecture.

    This system will survive the LoRA migration and provides:
    - Progress metrics tracking
    - Motivation level detection
    - Context preparation for the fine-tuned model

    Does NOT include:
    - Template generation (removed post-LoRA)
    - Response assembly (removed post-LoRA)
    - Explicit encouragement logic (internalized in model)
    """

    def __init__(self):
        self.progress_metrics = ProgressMetrics()
        self.session_start = datetime.now()

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
            if correct and self.progress_metrics.consecutive_correct >= 4:
                self.progress_metrics.topics_mastered.append(topic)

    def detect_motivation_level(self) -> MotivationLevel:
        """Analyze current user state and return motivation level."""
        metrics = self.progress_metrics

        # Check for frustration indicators
        if (
            metrics.consecutive_incorrect >= 3
            or metrics.mistakes_this_session > 5
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

    def get_context_for_model(self) -> Dict[str, Any]:
        """
        Get lightweight context data for the LoRA fine-tuned model.

        This is what replaces the heavy prompt engineering post-LoRA.
        The fine-tuned model will know what to do with this context.
        """
        return {
            "user_metrics": {
                "accuracy": round(self.progress_metrics.accuracy_rate, 1),
                "consecutive_correct": self.progress_metrics.consecutive_correct,
                "consecutive_incorrect": self.progress_metrics.consecutive_incorrect,
                "current_streak": self.progress_metrics.current_streak,
                "motivation_level": self.detect_motivation_level().value,
            },
            "session_info": {
                "total_responses": self.progress_metrics.total_responses,
                "mistakes_this_session": self.progress_metrics.mistakes_this_session,
                "topics_mastered": len(self.progress_metrics.topics_mastered),
                "session_duration_minutes": (datetime.now() - self.session_start).total_seconds()
                / 60,
            },
            "behavioral_triggers": {
                "needs_encouragement": self.detect_motivation_level()
                in [MotivationLevel.LOW, MotivationLevel.FRUSTRATED],
                "celebrating_success": self.progress_metrics.is_excelling,
                "struggling": self.progress_metrics.is_struggling,
            },
        }

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session progress for analytics."""
        return {
            "accuracy": self.progress_metrics.accuracy_rate,
            "total_responses": self.progress_metrics.total_responses,
            "current_streak": self.progress_metrics.current_streak,
            "best_streak": self.progress_metrics.best_streak,
            "topics_mastered": self.progress_metrics.topics_mastered,
            "motivation_level": self.detect_motivation_level().value,
            "session_duration": (datetime.now() - self.session_start).total_seconds() / 60,
        }

    def reset_session(self) -> None:
        """Reset session-specific metrics for new conversation."""
        self.progress_metrics.mistakes_this_session = 0
        self.progress_metrics.session_duration = timedelta(0)
        self.session_start = datetime.now()
