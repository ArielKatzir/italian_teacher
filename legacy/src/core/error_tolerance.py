"""
Error Tolerance System for Italian Teacher Agents (Post-LoRA Core)

This module provides the core error handling framework that will work with
LoRA-based correction. All hardcoded logic has been moved to pre_lora/.

Core responsibilities:
1. Define error classification and data structures
2. Manage correction policies and tolerance settings
3. Integrate with agent personalities
4. Coordinate between detection engines (pre-LoRA or LoRA-based)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .interfaces.error_correction import ErrorCorrectionEngine

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of different error types."""

    GRAMMAR = "grammar"
    VOCABULARY = "vocabulary"
    PRONUNCIATION = "pronunciation"
    SPELLING = "spelling"
    WORD_ORDER = "word_order"
    VERB_CONJUGATION = "verb_conjugation"
    GENDER_AGREEMENT = "gender_agreement"
    PREPOSITION = "preposition"
    CULTURAL = "cultural"
    COMPREHENSION = "comprehension"


class ErrorSeverity(Enum):
    """How severe is the error for communication."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CRITICAL = "critical"


class CorrectionStyle(Enum):
    """Different approaches to delivering corrections."""

    IMMEDIATE = "immediate"
    GENTLE = "gentle"
    ENCOURAGING = "encouraging"
    DELAYED = "delayed"
    IMPLICIT = "implicit"
    IGNORE = "ignore"


@dataclass
class DetectedError:
    """Represents a detected error with context."""

    original_text: str
    error_type: ErrorType
    severity: ErrorSeverity
    position: Tuple[int, int]  # start, end indices
    suggested_correction: str
    explanation: Optional[str] = None
    confidence: float = 1.0  # How confident we are this is an error (0-1)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionResponse:
    """Structured correction response."""

    correction_text: str
    style: CorrectionStyle
    explanation: Optional[str] = None
    encouragement: Optional[str] = None
    follow_up_practice: Optional[str] = None
    emotional_tone: str = "supportive"


@dataclass
class ErrorPattern:
    """Track recurring error patterns for personalized learning."""

    error_type: ErrorType
    frequency: int = 0
    last_occurrence: Optional[datetime] = None
    examples: List[str] = field(default_factory=list)
    improvement_rate: float = 0.0
    needs_focus: bool = False


class ErrorToleranceConfig(BaseModel):
    """Configuration for error tolerance behavior."""

    max_corrections_per_message: int = Field(default=2)
    correction_frequency: float = Field(default=0.7, ge=0.0, le=1.0)
    prioritize_major_errors: bool = Field(default=True)
    encourage_before_correct: bool = Field(default=True)
    allow_natural_mistakes: bool = Field(default=True)
    patience_factor: float = Field(default=1.0, ge=0.1, le=2.0)
    cultural_sensitivity: bool = Field(default=True)


class ErrorToleranceSystem:
    """
    Core error tolerance and correction system.

    This is the post-LoRA core that delegates actual error detection and
    correction to pluggable engines (pre-LoRA or LoRA-based).
    """

    def __init__(
        self,
        agent_personality: Dict[str, Any],
        config: Optional[ErrorToleranceConfig] = None,
        correction_engine: Optional["ErrorCorrectionEngine"] = None,
    ):

        self.agent_personality = agent_personality
        self.config = config or ErrorToleranceConfig()
        self.error_patterns: Dict[ErrorType, ErrorPattern] = {}
        self.session_corrections = 0
        self.recent_corrections: List[datetime] = []

        # Initialize correction engine (pre-LoRA by default, LoRA when available)
        if correction_engine:
            self.correction_engine = correction_engine
        else:
            # Default to pre-LoRA, legacy engine
            from pre_lora.correction_engine import pre_lora_engine

            self.correction_engine = pre_lora_engine

    def detect_errors(
        self, user_text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedError]:
        """
        Detect potential errors in user text using the configured engine.
        """
        context = context or {}

        # Delegate to the correction engine
        detected_errors = self.correction_engine.detect_errors(user_text, context)

        # Update error patterns tracking
        self._update_error_patterns(detected_errors)

        return detected_errors

    def should_correct_error(self, error: DetectedError, message_context: Dict[str, Any]) -> bool:
        """
        Decide whether to correct this specific error based on tolerance settings.
        """
        # Always correct critical and major errors
        if error.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.MAJOR]:
            return True

        # Check if we've already made too many corrections
        if self.session_corrections >= self.config.max_corrections_per_message:
            return False

        # Check correction frequency settings
        if self.session_corrections > 0:
            correction_rate = len(self.recent_corrections) / max(1, self.session_corrections)
            if correction_rate > self.config.correction_frequency:
                return False

        # Consider agent personality
        patience = self.agent_personality.get("patience_level", 5)
        correction_freq = self.agent_personality.get("correction_frequency", 5)

        # High patience agents correct less frequently
        if patience >= 8 and error.severity == ErrorSeverity.MINOR:
            return False

        # High correction frequency agents correct more
        if correction_freq >= 8:
            return True

        # Check if this error type needs focus
        error_pattern = self.error_patterns.get(error.error_type)
        if error_pattern and error_pattern.needs_focus:
            return True

        # Default behavior based on severity
        return error.severity == ErrorSeverity.MODERATE

    def generate_correction(
        self, error: DetectedError, context: Dict[str, Any]
    ) -> CorrectionResponse:
        """
        Generate a correction response for the detected error.
        """
        # Determine correction style based on agent personality
        style = self._determine_correction_style(error, context)

        # Get correction from engine
        corrected_text, explanation = self.correction_engine.generate_correction(
            error.original_text, error.error_type, context
        )

        # Generate response based on agent personality (delegate to correction engine)
        correction_text = self.correction_engine.format_correction_for_agent(
            corrected_text, explanation, style, context
        )

        # Add encouragement if configured (delegate to correction engine)
        encouragement = None
        if self.config.encourage_before_correct and style != CorrectionStyle.IGNORE:
            encouragement = self.correction_engine.generate_encouragement(error, context)

        # Determine emotional tone (delegate to correction engine)
        tone = self.correction_engine.determine_emotional_tone(error, style)

        # Record this correction
        self._record_correction(error)

        return CorrectionResponse(
            correction_text=correction_text,
            style=style,
            explanation=explanation,
            encouragement=encouragement,
            emotional_tone=tone,
        )

    def _determine_correction_style(
        self, error: DetectedError, context: Dict[str, Any]
    ) -> CorrectionStyle:
        """Determine the best style for delivering this correction."""
        correction_style = self.agent_personality.get("correction_style", "gentle")
        enthusiasm = self.agent_personality.get("enthusiasm_level", 5)
        patience = self.agent_personality.get("patience_level", 5)

        # High severity errors get immediate correction
        if error.severity == ErrorSeverity.CRITICAL:
            return CorrectionStyle.IMMEDIATE

        # Very patient agents use gentler approaches
        if patience >= 8:
            if error.severity == ErrorSeverity.MINOR:
                return CorrectionStyle.IMPLICIT
            else:
                return CorrectionStyle.GENTLE

        # Enthusiastic agents use encouraging style
        if enthusiasm >= 7 and correction_style == "encouraging":
            return CorrectionStyle.ENCOURAGING

        # Map personality correction style
        style_map = {
            "gentle": CorrectionStyle.GENTLE,
            "direct": CorrectionStyle.IMMEDIATE,
            "encouraging": CorrectionStyle.ENCOURAGING,
        }

        return style_map.get(correction_style, CorrectionStyle.GENTLE)

    def _update_error_patterns(self, errors: List[DetectedError]) -> None:
        """Update tracking of user's error patterns."""
        for error in errors:
            if error.error_type not in self.error_patterns:
                self.error_patterns[error.error_type] = ErrorPattern(error_type=error.error_type)

            pattern = self.error_patterns[error.error_type]
            pattern.frequency += 1
            pattern.last_occurrence = datetime.now()
            pattern.examples.append(error.original_text)

            # Mark as needing focus if frequently occurring
            if pattern.frequency >= 3:
                pattern.needs_focus = True

    def _record_correction(self, error: DetectedError) -> None:
        """Record that we made a correction."""
        self.session_corrections += 1
        self.recent_corrections.append(datetime.now())

        # Keep only recent corrections (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        self.recent_corrections = [time for time in self.recent_corrections if time > cutoff]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of user's error patterns."""
        return {
            "total_error_types": len(self.error_patterns),
            "most_common_errors": [
                {
                    "type": pattern.error_type.value,
                    "frequency": pattern.frequency,
                    "needs_focus": pattern.needs_focus,
                }
                for pattern in sorted(
                    self.error_patterns.values(), key=lambda p: p.frequency, reverse=True
                )[:5]
            ],
            "corrections_this_session": self.session_corrections,
            "improvement_areas": [
                pattern.error_type.value
                for pattern in self.error_patterns.values()
                if pattern.needs_focus
            ],
        }

    def reset_session(self) -> None:
        """Reset session-specific tracking."""
        self.session_corrections = 0
        self.recent_corrections.clear()

    def adjust_tolerance(self, patience_factor: float, correction_frequency: float) -> None:
        """Dynamically adjust tolerance settings."""
        self.config.patience_factor = max(0.1, min(2.0, patience_factor))
        self.config.correction_frequency = max(0.0, min(1.0, correction_frequency))

    def switch_to_lora_engine(self, lora_engine: "ErrorCorrectionEngine") -> None:
        """Switch from pre-LoRA to LoRA-based correction engine."""
        logger.info("Switching to LoRA-based correction engine")
        self.correction_engine = lora_engine

    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about error correction."""
        base_stats = self.get_error_summary()
        base_stats["correction_engine"] = type(self.correction_engine).__name__
        return base_stats
