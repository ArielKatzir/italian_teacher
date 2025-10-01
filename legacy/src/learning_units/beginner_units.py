"""
Beginner curriculum units configuration.
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


BEGINNER_UNITS = [
    # Unit 1: First Contact
    CurriculumUnit(
        unit_id="unit_01_greetings",
        name="Greetings and Basic Phrases",
        description="Essential greetings and polite expressions",
        difficulty_level=DifficultyLevel.BEGINNER,
        topics=[
            TopicRequirement(
                topic=LearningTopic.GREETINGS,
                min_accuracy=75.0,
                min_correct_answers=4,
                min_questions_attempted=6,
                required_question_types=[
                    QuestionType.FILL_IN_BLANK,
                    QuestionType.CONVERSATION_PROMPT,
                ],
            )
        ],
        learning_objectives=[
            "Greet people appropriately in different contexts",
            "Use basic polite expressions",
            "Understand formal vs informal address",
        ],
    ),
    # Unit 2: Essential Verbs
    CurriculumUnit(
        unit_id="unit_02_essential_verbs",
        name="Essential Verbs: Essere and Avere",
        description="Master the two most important Italian verbs",
        difficulty_level=DifficultyLevel.BEGINNER,
        topics=[
            TopicRequirement(
                topic=LearningTopic.GRAMMAR_VERBS,
                min_accuracy=80.0,
                min_correct_answers=6,
                min_questions_attempted=10,
                required_question_types=[QuestionType.VERB_CONJUGATION, QuestionType.FILL_IN_BLANK],
            )
        ],
        prerequisites=["unit_01_greetings"],
        learning_objectives=[
            "Conjugate 'essere' in present tense",
            "Conjugate 'avere' in present tense",
            "Use essere/avere in basic sentences",
            "Understand subject-verb agreement",
        ],
    ),
    # Unit 3: Numbers and Time
    CurriculumUnit(
        unit_id="unit_03_numbers_time",
        name="Numbers and Time",
        description="Learn numbers, dates, and time expressions",
        difficulty_level=DifficultyLevel.BEGINNER,
        topics=[
            TopicRequirement(
                topic=LearningTopic.NUMBERS,
                min_accuracy=75.0,
                min_correct_answers=5,
                min_questions_attempted=8,
            ),
            TopicRequirement(
                topic=LearningTopic.TIME,
                min_accuracy=75.0,
                min_correct_answers=4,
                min_questions_attempted=6,
            ),
        ],
        prerequisites=["unit_02_essential_verbs"],
        learning_objectives=[
            "Count from 1-100",
            "Tell time in Italian",
            "Express dates and days of the week",
        ],
    ),
    # Unit 4: Family and Descriptions
    CurriculumUnit(
        unit_id="unit_04_family",
        name="Family and Personal Descriptions",
        description="Describe family members and personal characteristics",
        difficulty_level=DifficultyLevel.BEGINNER,
        topics=[
            TopicRequirement(
                topic=LearningTopic.FAMILY,
                min_accuracy=75.0,
                min_correct_answers=5,
                min_questions_attempted=8,
            ),
            TopicRequirement(
                topic=LearningTopic.GRAMMAR_ADJECTIVES,
                min_accuracy=70.0,
                min_correct_answers=4,
                min_questions_attempted=7,
                required_question_types=[QuestionType.GRAMMAR_CORRECTION],
            ),
        ],
        prerequisites=["unit_03_numbers_time"],
        learning_objectives=[
            "Name family members",
            "Use adjectives to describe people",
            "Understand gender agreement with adjectives",
        ],
    ),
    # Unit 5: Food and Dining
    CurriculumUnit(
        unit_id="unit_05_food",
        name="Food and Dining",
        description="Order food and discuss Italian cuisine",
        difficulty_level=DifficultyLevel.BEGINNER,
        topics=[
            TopicRequirement(
                topic=LearningTopic.FOOD,
                min_accuracy=75.0,
                min_correct_answers=6,
                min_questions_attempted=10,
                required_question_types=[
                    QuestionType.CONVERSATION_PROMPT,
                    QuestionType.VOCABULARY_MATCH,
                ],
            )
        ],
        prerequisites=["unit_04_family"],
        learning_objectives=[
            "Order food in a restaurant",
            "Discuss food preferences",
            "Understand Italian dining culture",
        ],
    ),
]
