"""
Pre-LoRA Correction Engine

This module provides comprehensive Italian error correction using rule-based
approaches and pattern matching. It's designed to be easily replaceable with
LoRA-based correction when that's implemented.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..core.error_tolerance import DetectedError, ErrorSeverity, ErrorType
from .italian_corrections import italian_corrections_db

logger = logging.getLogger(__name__)


class PreLoRACorrectionEngine:
    """
    Rule-based correction engine for Italian errors.

    This engine uses comprehensive pattern matching and linguistic rules
    to detect and correct Italian errors. It's designed as a temporary
    solution until LoRA fine-tuning is implemented.
    """

    def __init__(self):
        self.corrections_db = italian_corrections_db
        self.detection_stats = {error_type: 0 for error_type in ErrorType}

    def detect_errors(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedError]:
        """
        Detect errors in Italian text using pattern matching.

        Args:
            text: Input text to analyze
            context: Additional context (user level, topic, etc.)

        Returns:
            List of detected errors with corrections
        """
        detected_errors = []
        context = context or {}

        # Check each error type's patterns
        for error_type, patterns in self.corrections_db.error_patterns.items():
            for pattern, explanation, _ in patterns:
                matches = pattern.finditer(text)
                for match in matches:
                    # Generate correction
                    original_text = match.group()
                    corrected_text, _ = self.corrections_db.get_correction(
                        original_text, error_type
                    )

                    if corrected_text != original_text:
                        error = DetectedError(
                            original_text=original_text,
                            error_type=error_type,
                            severity=self._determine_severity(error_type, original_text, context),
                            position=(match.start(), match.end()),
                            suggested_correction=corrected_text,
                            explanation=explanation,
                            confidence=self._calculate_confidence(error_type, original_text),
                            context=context,
                        )
                        detected_errors.append(error)
                        self.detection_stats[error_type] += 1

        # Also check direct corrections database
        words_and_phrases = self._extract_words_and_phrases(text)
        for phrase in words_and_phrases:
            corrected, explanation = self.corrections_db.get_correction(phrase)
            if corrected != phrase:
                # Find position in original text
                start_pos = text.lower().find(phrase.lower())
                if start_pos != -1:
                    error_type = self._infer_error_type(phrase, corrected)
                    error = DetectedError(
                        original_text=phrase,
                        error_type=error_type,
                        severity=self._determine_severity(error_type, phrase, context),
                        position=(start_pos, start_pos + len(phrase)),
                        suggested_correction=corrected,
                        explanation=explanation,
                        confidence=0.9,  # High confidence for direct matches
                        context=context,
                    )
                    detected_errors.append(error)
                    self.detection_stats[error_type] += 1

        # Remove duplicates and overlapping errors
        return self._deduplicate_errors(detected_errors)

    def generate_correction(
        self, original_text: str, error_type: ErrorType, context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Generate a corrected version of the text with explanation.

        Args:
            original_text: The incorrect text
            error_type: Type of error detected
            context: Context information

        Returns:
            Tuple of (corrected_text, explanation)
        """
        return self.corrections_db.get_correction(original_text, error_type)

    def format_correction_for_agent(
        self, corrected_text: str, explanation: str, style, context: Dict[str, Any]
    ) -> str:
        """Format correction text according to agent personality (pre-LoRA templates)."""
        # Template-based response generation (will be removed post-LoRA)
        if style.value == "gentle":
            return f"Try: '{corrected_text}' - {explanation}"
        elif style.value == "encouraging":
            return f"Great attempt! Try: '{corrected_text}'"
        elif style.value == "implicit":
            return f"Ah yes, '{corrected_text}' - that's a good way to say it!"
        else:
            return f"'{corrected_text}'"

    def generate_encouragement(self, error, context: Dict[str, Any]) -> str:
        """Generate encouragement (pre-LoRA templates)."""
        # Template-based encouragement (will be removed post-LoRA)
        return "You're doing great!"

    def determine_emotional_tone(self, error, style) -> str:
        """Determine appropriate emotional tone for the correction."""
        if style.value == "encouraging":
            return "excited"
        elif style.value == "gentle":
            return "supportive"
        elif style.value == "implicit":
            return "casual"
        else:
            return "neutral"

    def _determine_severity(
        self, error_type: ErrorType, text: str, context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Determine severity based on error type and context."""
        # Base severity mapping
        severity_map = {
            ErrorType.VERB_CONJUGATION: ErrorSeverity.MODERATE,
            ErrorType.GENDER_AGREEMENT: ErrorSeverity.MINOR,
            ErrorType.PREPOSITION: ErrorSeverity.MODERATE,
            ErrorType.WORD_ORDER: ErrorSeverity.MODERATE,
            ErrorType.SPELLING: ErrorSeverity.MINOR,
            ErrorType.VOCABULARY: ErrorSeverity.MAJOR,
            ErrorType.GRAMMAR: ErrorSeverity.MODERATE,
            ErrorType.PRONUNCIATION: ErrorSeverity.MINOR,
            ErrorType.CULTURAL: ErrorSeverity.MINOR,
            ErrorType.COMPREHENSION: ErrorSeverity.CRITICAL,
        }

        base_severity = severity_map.get(error_type, ErrorSeverity.MODERATE)

        # Adjust based on user level
        user_level = context.get("user_level", "beginner")
        if user_level == "beginner":
            # More lenient with beginners
            if base_severity == ErrorSeverity.MODERATE:
                return ErrorSeverity.MINOR
            elif base_severity == ErrorSeverity.MAJOR:
                return ErrorSeverity.MODERATE
        elif user_level == "advanced":
            # More strict with advanced learners
            if base_severity == ErrorSeverity.MINOR:
                return ErrorSeverity.MODERATE
            elif base_severity == ErrorSeverity.MODERATE:
                return ErrorSeverity.MAJOR

        # Increase severity for fundamental errors (essere/avere)
        if "essere" in text.lower() or "avere" in text.lower():
            if error_type == ErrorType.VERB_CONJUGATION:
                return ErrorSeverity.MAJOR

        return base_severity

    def _calculate_confidence(self, error_type: ErrorType, text: str) -> float:
        """Calculate confidence score for error detection."""
        # Base confidence by error type
        confidence_map = {
            ErrorType.SPELLING: 0.95,  # Very high for spelling with accents
            ErrorType.VERB_CONJUGATION: 0.85,  # High for clear conjugation errors
            ErrorType.GENDER_AGREEMENT: 0.80,  # High for article-noun mismatches
            ErrorType.PREPOSITION: 0.75,  # Medium-high for place/time prepositions
            ErrorType.WORD_ORDER: 0.70,  # Medium for word order
            ErrorType.VOCABULARY: 0.60,  # Lower for vocabulary (more subjective)
        }

        base_confidence = confidence_map.get(error_type, 0.75)

        # Adjust based on text characteristics
        if len(text) <= 3:  # Very short text, likely a spelling error
            base_confidence += 0.1
        elif "ho essere" in text.lower() or "sono avere" in text.lower():
            # Very common, clear errors
            base_confidence = 0.95

        return min(1.0, base_confidence)

    def _extract_words_and_phrases(self, text: str) -> List[str]:
        """Extract words and common phrases for correction lookup."""
        import re

        # Split into words
        words = re.findall(r"\b\w+\b", text.lower())

        # Also extract common 2-3 word phrases
        phrases = []
        words_list = text.lower().split()

        # 2-word phrases
        for i in range(len(words_list) - 1):
            phrase = f"{words_list[i]} {words_list[i + 1]}"
            phrases.append(phrase)

        # 3-word phrases
        for i in range(len(words_list) - 2):
            phrase = f"{words_list[i]} {words_list[i + 1]} {words_list[i + 2]}"
            phrases.append(phrase)

        return words + phrases

    def _infer_error_type(self, original: str, corrected: str) -> ErrorType:
        """Infer error type from original and corrected text."""
        # Simple heuristics to categorize corrections
        if any(word in original.lower() for word in ["essere", "avere", "andare", "fare", "stare"]):
            return ErrorType.VERB_CONJUGATION
        elif any(word in original.lower() for word in ["un ", "una ", "il ", "la "]):
            return ErrorType.GENDER_AGREEMENT
        elif any(word in original.lower() for word in ["in ", "a ", "da ", "di ", "per "]):
            return ErrorType.PREPOSITION
        elif len(original) == len(corrected) and original != corrected:
            # Likely spelling (accent marks)
            return ErrorType.SPELLING
        else:
            return ErrorType.VOCABULARY

    def _deduplicate_errors(self, errors: List[DetectedError]) -> List[DetectedError]:
        """Remove duplicate and overlapping errors."""
        if not errors:
            return errors

        # Sort by position
        errors.sort(key=lambda e: e.position[0])

        deduplicated = []
        last_end = -1

        for error in errors:
            # Skip if this error overlaps with the previous one
            if error.position[0] < last_end:
                # Keep the error with higher confidence
                if deduplicated and error.confidence > deduplicated[-1].confidence:
                    deduplicated[-1] = error
                continue

            deduplicated.append(error)
            last_end = error.position[1]

        return deduplicated

    def get_detection_statistics(self) -> Dict[str, int]:
        """Get statistics about error detection."""
        return {error_type.value: count for error_type, count in self.detection_stats.items()}

    def get_training_data(self) -> List[Dict[str, Any]]:
        """
        Generate training data for LoRA fine-tuning.

        Returns:
            List of training examples with error type annotations
        """
        training_data = []

        # Get basic examples
        for incorrect, correct, category in self.corrections_db.get_training_examples():
            training_data.append(
                {
                    "incorrect": incorrect,
                    "correct": correct,
                    "error_type": category,
                    "context": "basic_correction",
                }
            )

        # Get contextual examples
        for (
            incorrect_sentence,
            correct_sentence,
            error_type,
            explanation,
        ) in self.corrections_db.get_contextual_examples():
            training_data.append(
                {
                    "incorrect": incorrect_sentence,
                    "correct": correct_sentence,
                    "error_type": error_type,
                    "explanation": explanation,
                    "context": "sentence_level",
                }
            )

        return training_data

    def reset_statistics(self):
        """Reset detection statistics."""
        self.detection_stats = {error_type: 0 for error_type in ErrorType}


# Global instance
pre_lora_engine = PreLoRACorrectionEngine()
