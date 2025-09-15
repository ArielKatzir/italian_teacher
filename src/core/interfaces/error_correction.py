"""
Error Correction Engine Interface

This module defines the Protocol for error correction engines that can be
plugged into the ErrorToleranceSystem. This allows for different implementations
(pre-LoRA, LoRA-based, or other future correction engines).
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple

from ..error_tolerance import DetectedError, ErrorType


class ErrorCorrectionEngine(Protocol):
    """Protocol for error correction engines (pre-LoRA or LoRA-based)."""

    def detect_errors(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> List[DetectedError]:
        """Detect errors in text."""
        ...

    def generate_correction(
        self, original_text: str, error_type: ErrorType, context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Generate correction with explanation."""
        ...

    def format_correction_for_agent(
        self, corrected_text: str, explanation: str, style, context: Dict[str, Any]
    ) -> str:
        """Format correction text according to agent personality."""
        ...

    def generate_encouragement(self, error, context: Dict[str, Any]) -> str:
        """Generate encouragement message."""
        ...

    def determine_emotional_tone(self, error, style) -> str:
        """Determine appropriate emotional tone for the correction."""
        ...
