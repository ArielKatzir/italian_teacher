"""
Advanced curriculum units configuration.
"""

from dataclasses import dataclass, field
from typing import List

from pre_lora.educational_questions import DifficultyLevel, LearningTopic, QuestionType


@dataclass
class TopicRequirement:
    """Defines requirements for completing a topic."""

    topic: LearningTopic
    min_accuracy: float = 80.0
    min_correct_answers: int = 5
    min_questions_attempted: int = 8
    required_question_types: List[QuestionType] = field(default_factory=list)
    estimated_time_minutes: int = 15


@dataclass
class CurriculumUnit:
    """A unit of curriculum with topics and dependencies."""

    unit_id: str
    name: str
    description: str
    difficulty_level: DifficultyLevel
    topics: List[TopicRequirement]
    prerequisites: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    estimated_hours: float = 2.0


ADVANCED_UNITS = [
    # Unit 9: Subjunctive and Complex Grammar
    CurriculumUnit(
        unit_id="unit_09_subjunctive",
        name="Subjunctive Mood",
        description="Master the subjunctive for complex expression",
        difficulty_level=DifficultyLevel.ADVANCED,
        topics=[
            TopicRequirement(
                topic=LearningTopic.GRAMMAR_VERBS,
                min_accuracy=85.0,
                min_correct_answers=10,
                min_questions_attempted=15,
                required_question_types=[
                    QuestionType.GRAMMAR_CORRECTION,
                    QuestionType.MULTIPLE_CHOICE,
                ],
            )
        ],
        prerequisites=["unit_08_prepositions"],
        learning_objectives=[
            "Use subjunctive after opinion expressions",
            "Form present and past subjunctive",
            "Understand when subjunctive is required",
        ],
    ),
    # Unit 10: Cultural Mastery
    CurriculumUnit(
        unit_id="unit_10_culture",
        name="Italian Culture and Idioms",
        description="Understand cultural nuances and idiomatic expressions",
        difficulty_level=DifficultyLevel.ADVANCED,
        topics=[
            TopicRequirement(
                topic=LearningTopic.CULTURAL_CONTEXT,
                min_accuracy=80.0,
                min_correct_answers=6,
                min_questions_attempted=10,
                required_question_types=[QuestionType.CULTURAL_CONTEXT],
            )
        ],
        prerequisites=["unit_09_subjunctive"],
        learning_objectives=[
            "Understand regional cultural differences",
            "Use idiomatic expressions naturally",
            "Navigate complex social situations",
        ],
    ),
]
