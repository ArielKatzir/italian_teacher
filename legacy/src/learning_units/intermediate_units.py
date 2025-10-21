"""
Intermediate curriculum units configuration.
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


INTERMEDIATE_UNITS = [
    # Unit 6: Past Tenses
    CurriculumUnit(
        unit_id="unit_06_past_tenses",
        name="Past Tenses: Passato Prossimo",
        description="Express past actions and experiences",
        difficulty_level=DifficultyLevel.INTERMEDIATE,
        topics=[
            TopicRequirement(
                topic=LearningTopic.GRAMMAR_VERBS,
                min_accuracy=80.0,
                min_correct_answers=8,
                min_questions_attempted=12,
                required_question_types=[
                    QuestionType.VERB_CONJUGATION,
                    QuestionType.GRAMMAR_CORRECTION,
                ],
            )
        ],
        prerequisites=["unit_05_food"],  # Must complete beginner path
        learning_objectives=[
            "Form passato prossimo with essere/avere",
            "Choose correct auxiliary verb",
            "Use past participle agreement rules",
        ],
    ),
    # Unit 7: Travel and Transportation
    CurriculumUnit(
        unit_id="unit_07_travel",
        name="Travel and Transportation",
        description="Navigate Italy and discuss travel experiences",
        difficulty_level=DifficultyLevel.INTERMEDIATE,
        topics=[
            TopicRequirement(
                topic=LearningTopic.TRAVEL,
                min_accuracy=75.0,
                min_correct_answers=7,
                min_questions_attempted=10,
            ),
            TopicRequirement(
                topic=LearningTopic.TRANSPORTATION,
                min_accuracy=75.0,
                min_correct_answers=5,
                min_questions_attempted=8,
            ),
        ],
        prerequisites=["unit_06_past_tenses"],
        learning_objectives=[
            "Book transportation tickets",
            "Ask for directions",
            "Describe travel experiences",
        ],
    ),
    # Unit 8: Prepositions and Complex Grammar
    CurriculumUnit(
        unit_id="unit_08_prepositions",
        name="Prepositions and Location",
        description="Master Italian prepositions and spatial relationships",
        difficulty_level=DifficultyLevel.INTERMEDIATE,
        topics=[
            TopicRequirement(
                topic=LearningTopic.GRAMMAR_PREPOSITIONS,
                min_accuracy=80.0,
                min_correct_answers=8,
                min_questions_attempted=12,
                required_question_types=[
                    QuestionType.GRAMMAR_CORRECTION,
                    QuestionType.FILL_IN_BLANK,
                ],
            )
        ],
        prerequisites=["unit_07_travel"],
        learning_objectives=[
            "Use prepositions correctly with places",
            "Understand articulated prepositions",
            "Express spatial and temporal relationships",
        ],
    ),
]
