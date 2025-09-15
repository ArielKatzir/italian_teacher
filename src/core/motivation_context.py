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

# Import error types for detailed tracking
from .error_tolerance import DetectedError, ErrorSeverity, ErrorType


class MotivationLevel(Enum):
    """User's current motivation/energy state."""

    HIGH = "high"  # Confident, engaged, making progress
    MEDIUM = "medium"  # Steady progress, occasional mistakes
    LOW = "low"  # Struggling, frequent errors, may need support
    FRUSTRATED = "frustrated"  # Showing signs of frustration or giving up


@dataclass
class ProgressMetrics:
    """Tracks user learning progress for motivation calculations."""

    # Message-level tracking
    correct_responses: int = 0
    total_responses: int = 0
    consecutive_incorrect: int = 0

    # Individual error tracking
    total_errors_detected: int = 0
    errors_by_type: Dict[ErrorType, int] = field(default_factory=dict)
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    recent_error_types: List[ErrorType] = field(default_factory=list)  # Last 10 errors

    # Learning tracking
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
        return self.current_streak >= 3 or self.accuracy_rate > 80

    @property
    def error_rate(self) -> float:
        """Calculate errors per message."""
        if self.total_responses == 0:
            return 0.0
        return self.total_errors_detected / self.total_responses

    @property
    def most_common_error_type(self) -> Optional[ErrorType]:
        """Get the most frequent error type."""
        if not self.errors_by_type:
            return None
        return max(self.errors_by_type.items(), key=lambda x: x[1])[0]

    @property
    def needs_grammar_focus(self) -> bool:
        """Detect if user needs grammar-focused help."""
        grammar_errors = (
            self.errors_by_type.get(ErrorType.GRAMMAR, 0)
            + self.errors_by_type.get(ErrorType.VERB_CONJUGATION, 0)
            + self.errors_by_type.get(ErrorType.GENDER_AGREEMENT, 0)
        )
        return grammar_errors > 3 or grammar_errors / max(1, self.total_errors_detected) > 0.4


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
            self.progress_metrics.consecutive_incorrect = 0
            self.progress_metrics.current_streak += 1
            self.progress_metrics.best_streak = max(
                self.progress_metrics.best_streak, self.progress_metrics.current_streak
            )
        else:
            self.progress_metrics.consecutive_incorrect += 1
            self.progress_metrics.current_streak = 0
            self.progress_metrics.mistakes_this_session += 1

        if topic and topic not in self.progress_metrics.topics_mastered:
            # Simple mastery detection - 3+ correct in a row for this topic
            if correct and self.progress_metrics.current_streak >= 4:
                self.progress_metrics.topics_mastered.append(topic)

    def update_progress_with_errors(
        self,
        detected_errors: List[DetectedError],
        topic: Optional[str] = None,
        user_message: Optional[str] = None,
    ) -> None:
        """
        Enhanced progress tracking that handles multiple errors per message.

        This method provides detailed error analysis for educational use:
        - Tracks individual errors by type and severity
        - Calculates message-level success/failure
        - Provides better analytics for schools

        Args:
            detected_errors: List of errors found in the user's message
            topic: Current conversation topic (optional)
            user_message: The user's original message (optional)
        """
        self.progress_metrics.total_responses += 1

        # Track individual errors
        error_count = len(detected_errors)
        self.progress_metrics.total_errors_detected += error_count

        # Update error tracking by type and severity
        for error in detected_errors:
            # Track by type
            if error.error_type not in self.progress_metrics.errors_by_type:
                self.progress_metrics.errors_by_type[error.error_type] = 0
            self.progress_metrics.errors_by_type[error.error_type] += 1

            # Track by severity
            if error.severity not in self.progress_metrics.errors_by_severity:
                self.progress_metrics.errors_by_severity[error.severity] = 0
            self.progress_metrics.errors_by_severity[error.severity] += 1

            # Keep recent error types (last 10)
            self.progress_metrics.recent_error_types.append(error.error_type)
            if len(self.progress_metrics.recent_error_types) > 10:
                self.progress_metrics.recent_error_types.pop(0)

        # Message-level success determination
        message_correct = error_count == 0

        if message_correct:
            self.progress_metrics.correct_responses += 1
            self.progress_metrics.consecutive_incorrect = 0
            self.progress_metrics.current_streak += 1
            self.progress_metrics.best_streak = max(
                self.progress_metrics.best_streak, self.progress_metrics.current_streak
            )
        else:
            self.progress_metrics.consecutive_incorrect += 1
            self.progress_metrics.current_streak = 0
            self.progress_metrics.mistakes_this_session += error_count  # Count individual errors

        # Topic mastery tracking (requires error-free messages)
        if topic and topic not in self.progress_metrics.topics_mastered:
            if message_correct and self.progress_metrics.current_streak >= 4:
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
                "consecutive_incorrect": self.progress_metrics.consecutive_incorrect,
                "current_streak": self.progress_metrics.current_streak,
                "motivation_level": self.detect_motivation_level().value,
                # NEW: Educational analytics
                "error_rate": round(self.progress_metrics.error_rate, 2),
                "total_errors": self.progress_metrics.total_errors_detected,
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
                # NEW: Educational focus areas
                "needs_grammar_focus": self.progress_metrics.needs_grammar_focus,
            },
            # NEW: Educational error analytics for schools
            "educational_analytics": {
                "common_error_type": (
                    self.progress_metrics.most_common_error_type.value
                    if self.progress_metrics.most_common_error_type
                    else None
                ),
                "error_distribution": {
                    error_type.value: count
                    for error_type, count in self.progress_metrics.errors_by_type.items()
                },
                "severity_distribution": {
                    severity.value: count
                    for severity, count in self.progress_metrics.errors_by_severity.items()
                },
                "recent_error_pattern": [
                    error_type.value for error_type in self.progress_metrics.recent_error_types[-5:]
                ],
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
