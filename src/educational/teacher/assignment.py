"""
Homework Assignment Configuration

Simple dataclass for manual homework assignment creation.
Teachers specify parameters directly via UI instead of natural language parsing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class CEFRLevel(Enum):
    """Common European Framework of Reference for Languages levels."""

    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient


class GrammarFocus(Enum):
    """Grammar topics for exercises."""

    PRESENT_TENSE = "present_tense"
    PAST_TENSE = "past_tense"
    FUTURE_TENSE = "future_tense"
    SUBJUNCTIVE = "subjunctive"
    CONDITIONAL = "conditional"
    IMPERATIVE = "imperative"
    PASSIVE_VOICE = "passive_voice"
    PRONOUNS = "pronouns"
    ARTICLES = "articles"
    PREPOSITIONS = "prepositions"
    CONJUNCTIONS = "conjunctions"


class ExerciseType(Enum):
    """Types of exercises to generate."""

    FILL_IN_BLANK = "fill_in_blank"
    TRANSLATION = "translation"
    SENTENCE_COMPLETION = "sentence_completion"
    MULTIPLE_CHOICE = "multiple_choice"
    ESSAY = "essay"
    CONVERSATION = "conversation"


@dataclass
class HomeworkAssignment:
    """
    Homework assignment configuration for manual input.

    Example:
        >>> assignment = HomeworkAssignment(
        ...     cefr_level=CEFRLevel.A2,
        ...     quantity=5,
        ...     grammar_focus=GrammarFocus.PAST_TENSE,
        ...     topic="history of Milan"
        ... )
    """

    # Required fields
    cefr_level: CEFRLevel
    quantity: int = 5  # Number of exercises to generate

    # Optional fields
    grammar_focus: Optional[GrammarFocus] = None
    topic: Optional[str] = None  # Custom topic (e.g., "history of Milan", "Italian food")
    student_groups: List[str] = field(default_factory=lambda: ["all"])
    exercise_types: List[ExerciseType] = field(
        default_factory=lambda: [
            ExerciseType.FILL_IN_BLANK,
            ExerciseType.TRANSLATION,
            ExerciseType.SENTENCE_COMPLETION,
            ExerciseType.MULTIPLE_CHOICE,
        ]
    )
    difficulty_scaling: bool = True

    def __post_init__(self):
        """Validate assignment parameters."""
        if self.quantity < 1 or self.quantity > 20:
            raise ValueError("Quantity must be between 1 and 20")

        if not self.student_groups:
            self.student_groups = ["all"]

        if not self.exercise_types:
            self.exercise_types = [
                ExerciseType.FILL_IN_BLANK,
                ExerciseType.TRANSLATION,
                ExerciseType.SENTENCE_COMPLETION,
                ExerciseType.MULTIPLE_CHOICE,
            ]

    def to_prompt_context(self) -> str:
        """
        Convert assignment to context string for homework generator prompts.

        Returns:
            Formatted string to include in exercise generation prompts.
        """
        context_parts = [f"CEFR Level: {self.cefr_level.value}"]

        if self.grammar_focus:
            grammar_name = self.grammar_focus.value.replace("_", " ").title()
            context_parts.append(f"Grammar Focus: {grammar_name}")

        if self.topic:
            context_parts.append(f"Topic: {self.topic}")

        context_parts.append(f"Number of Exercises: {self.quantity}")

        return "\n".join(context_parts)

    def __repr__(self) -> str:
        """Human-readable representation."""
        parts = [f"{self.cefr_level.value}"]
        if self.grammar_focus:
            parts.append(self.grammar_focus.value)
        if self.topic:
            parts.append(f"topic='{self.topic}'")
        parts.append(f"{self.quantity}x")
        return f"HomeworkAssignment({', '.join(parts)})"
