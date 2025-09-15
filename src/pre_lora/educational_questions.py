"""
Educational Question System - Pre-LoRA Implementation

This module provides structured learning questions for school use. It includes
template-based question generation that will be replaced by LoRA-based natural
question generation in later phases.

Designed for educational environments with:
- Structured difficulty progression
- Topic-based organization
- Assessment-friendly format
- Integration with error tolerance and motivation systems
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.error_tolerance import ErrorType


class QuestionType(Enum):
    """Types of educational questions available."""

    FILL_IN_BLANK = "fill_in_blank"
    TRANSLATION = "translation"
    CONVERSATION_PROMPT = "conversation_prompt"
    MULTIPLE_CHOICE = "multiple_choice"
    VERB_CONJUGATION = "verb_conjugation"
    GRAMMAR_CORRECTION = "grammar_correction"
    VOCABULARY_MATCH = "vocabulary_match"
    CULTURAL_CONTEXT = "cultural_context"


class DifficultyLevel(Enum):
    """Learning difficulty levels aligned with educational standards."""

    BEGINNER = "beginner"  # A1-A2 CEFR
    INTERMEDIATE = "intermediate"  # B1-B2 CEFR
    ADVANCED = "advanced"  # C1-C2 CEFR


class LearningTopic(Enum):
    """Educational topics for structured learning."""

    GREETINGS = "greetings"
    FAMILY = "family"
    FOOD = "food"
    TRAVEL = "travel"
    SHOPPING = "shopping"
    WEATHER = "weather"
    TIME = "time"
    NUMBERS = "numbers"
    COLORS = "colors"
    BODY_PARTS = "body_parts"
    EMOTIONS = "emotions"
    HOBBIES = "hobbies"
    WORK = "work"
    HEALTH = "health"
    TRANSPORTATION = "transportation"
    GRAMMAR_VERBS = "grammar_verbs"
    GRAMMAR_NOUNS = "grammar_nouns"
    GRAMMAR_ADJECTIVES = "grammar_adjectives"
    GRAMMAR_PREPOSITIONS = "grammar_prepositions"
    CULTURAL_CONTEXT = "cultural_context"


@dataclass
class LearningQuestion:
    """Represents a structured learning question."""

    question_id: str
    question_type: QuestionType
    difficulty: DifficultyLevel
    topic: LearningTopic
    question_text: str
    correct_answer: str
    alternatives: List[str] = field(default_factory=list)  # For multiple choice
    hints: List[str] = field(default_factory=list)
    explanation: str = ""
    target_errors: List[ErrorType] = field(default_factory=list)  # Errors this question tests
    learning_objectives: List[str] = field(default_factory=list)
    estimated_time_minutes: int = 2


@dataclass
class QuestionSet:
    """A collection of questions for a learning session."""

    set_id: str
    name: str
    topic: LearningTopic
    difficulty: DifficultyLevel
    questions: List[LearningQuestion] = field(default_factory=list)
    total_time_minutes: int = 0
    learning_objectives: List[str] = field(default_factory=list)


class EducationalQuestionEngine:
    """
    Pre-LoRA question generation engine for structured learning.

    This will be replaced by LoRA-based natural question generation
    but provides immediate educational functionality for schools.
    """

    def __init__(self):
        self.question_templates = self._build_question_templates()
        self.question_sets = self._build_predefined_sets()

    def _build_question_templates(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Build comprehensive question templates for all topics and difficulties."""
        return {
            # GREETINGS TOPIC
            "greetings": {
                "beginner": [
                    {
                        "type": QuestionType.FILL_IN_BLANK,
                        "template": "Complete the greeting: 'Buon______, come stai?'",
                        "answer": "giorno",
                        "alternatives": ["giorno", "sera", "notte", "mattina"],
                        "explanation": "'Buongiorno' is used for 'good day/morning' in Italian.",
                        "target_errors": [ErrorType.VOCABULARY],
                        "learning_objectives": [
                            "Basic Italian greetings",
                            "Time-appropriate greetings",
                        ],
                    },
                    {
                        "type": QuestionType.TRANSLATION,
                        "template": "Translate to Italian: 'How are you?'",
                        "answer": "Come stai?",
                        "explanation": "'Come stai?' is the informal way to ask how someone is.",
                        "target_errors": [ErrorType.VOCABULARY],
                        "learning_objectives": ["Question formation", "Informal register"],
                    },
                    {
                        "type": QuestionType.CONVERSATION_PROMPT,
                        "template": "Respond appropriately to: 'Ciao, come va?'",
                        "answer": "Ciao! Tutto bene, grazie. E tu?",
                        "alternatives": ["Bene, grazie!", "Tutto ok!", "Non c'è male!"],
                        "explanation": "A friendly response shows you're well and asks back.",
                        "target_errors": [ErrorType.CULTURAL],
                        "learning_objectives": ["Conversational flow", "Politeness expressions"],
                    },
                ],
                "intermediate": [
                    {
                        "type": QuestionType.MULTIPLE_CHOICE,
                        "template": "Which greeting is most appropriate for a business meeting at 2 PM?",
                        "answer": "Buon pomeriggio",
                        "alternatives": ["Buongiorno", "Buon pomeriggio", "Buonasera", "Ciao"],
                        "explanation": "Buon pomeriggio is formal and appropriate for afternoon business.",
                        "target_errors": [ErrorType.CULTURAL],
                        "learning_objectives": ["Formal register", "Cultural appropriateness"],
                    },
                    {
                        "type": QuestionType.GRAMMAR_CORRECTION,
                        "template": "Correct this greeting: 'Buongiorno! Come state oggi?'",
                        "answer": "Buongiorno! Come sta oggi? (formal) or Come stai oggi? (informal)",
                        "explanation": "Choose 'sta' (formal) or 'stai' (informal), not 'state' (plural).",
                        "target_errors": [ErrorType.VERB_CONJUGATION],
                        "learning_objectives": ["Formal/informal address", "Verb conjugation"],
                    },
                ],
            },
            # VERB CONJUGATIONS
            "grammar_verbs": {
                "beginner": [
                    {
                        "type": QuestionType.VERB_CONJUGATION,
                        "template": "Conjugate 'essere' for 'io': Io ______ italiano.",
                        "answer": "sono",
                        "explanation": "'Io sono' is the first person singular of the verb 'essere'.",
                        "target_errors": [ErrorType.VERB_CONJUGATION],
                        "learning_objectives": ["Present tense essere", "Subject-verb agreement"],
                    },
                    {
                        "type": QuestionType.FILL_IN_BLANK,
                        "template": "Complete: 'Tu ______ una pizza.' (to have)",
                        "answer": "hai",
                        "alternatives": ["hai", "ho", "ha", "hanno"],
                        "explanation": "'Tu hai' is the second person singular of 'avere'.",
                        "target_errors": [ErrorType.VERB_CONJUGATION],
                        "learning_objectives": ["Present tense avere", "Common verbs"],
                    },
                ],
                "intermediate": [
                    {
                        "type": QuestionType.GRAMMAR_CORRECTION,
                        "template": "Correct: 'Ieri io ho essere al cinema.'",
                        "answer": "Ieri io sono stato/a al cinema.",
                        "explanation": "Use 'essere' with 'stato/a' for past tense of location.",
                        "target_errors": [ErrorType.VERB_CONJUGATION, ErrorType.GRAMMAR],
                        "learning_objectives": ["Past tense formation", "Auxiliary verb choice"],
                    }
                ],
                "advanced": [
                    {
                        "type": QuestionType.MULTIPLE_CHOICE,
                        "template": "Choose the correct subjunctive: 'Penso che lui ______ ragione.'",
                        "answer": "abbia",
                        "alternatives": ["ha", "abbia", "aveva", "avrebbe"],
                        "explanation": "After 'penso che' use the subjunctive 'abbia'.",
                        "target_errors": [ErrorType.GRAMMAR],
                        "learning_objectives": ["Subjunctive mood", "Complex sentence structure"],
                    }
                ],
            },
            # FOOD TOPIC
            "food": {
                "beginner": [
                    {
                        "type": QuestionType.VOCABULARY_MATCH,
                        "template": "Match the Italian food to English: 'pasta'",
                        "answer": "pasta",
                        "alternatives": ["pasta", "bread", "rice", "pizza"],
                        "explanation": "'Pasta' is the same in both languages!",
                        "target_errors": [ErrorType.VOCABULARY],
                        "learning_objectives": ["Basic food vocabulary", "Cognates"],
                    },
                    {
                        "type": QuestionType.FILL_IN_BLANK,
                        "template": "Complete: 'Vorrei ______ pizza, per favore.' (a/one)",
                        "answer": "una",
                        "alternatives": ["una", "un", "uno", "delle"],
                        "explanation": "'Pizza' is feminine, so use 'una'.",
                        "target_errors": [ErrorType.GENDER_AGREEMENT],
                        "learning_objectives": ["Articles", "Gender agreement", "Ordering food"],
                    },
                ],
                "intermediate": [
                    {
                        "type": QuestionType.CONVERSATION_PROMPT,
                        "template": "Order your favorite Italian dish politely at a restaurant.",
                        "answer": "Scusi, potrei avere... per favore?",
                        "explanation": "Use polite forms when ordering in restaurants.",
                        "target_errors": [ErrorType.CULTURAL],
                        "learning_objectives": ["Restaurant etiquette", "Polite requests"],
                    }
                ],
            },
            # CULTURAL CONTEXT
            "cultural_context": {
                "intermediate": [
                    {
                        "type": QuestionType.CULTURAL_CONTEXT,
                        "template": "Explain why Italians often say 'In bocca al lupo!' before exams.",
                        "answer": "It's an Italian way to wish good luck, like 'break a leg' in English.",
                        "explanation": "Literally 'into the wolf's mouth' - response is 'Crepi il lupo!'",
                        "target_errors": [ErrorType.CULTURAL],
                        "learning_objectives": ["Cultural expressions", "Idiomatic phrases"],
                    }
                ]
            },
        }

    def _build_predefined_sets(self) -> Dict[str, QuestionSet]:
        """Build predefined question sets for common learning scenarios."""
        return {
            "beginner_intro": QuestionSet(
                set_id="bg_intro_01",
                name="Italian Basics - Introductions",
                topic=LearningTopic.GREETINGS,
                difficulty=DifficultyLevel.BEGINNER,
                total_time_minutes=15,
                learning_objectives=[
                    "Basic greetings",
                    "Personal introductions",
                    "Polite expressions",
                ],
            ),
            "verb_essere_intro": QuestionSet(
                set_id="verb_essere_01",
                name="The Verb 'Essere' (To Be)",
                topic=LearningTopic.GRAMMAR_VERBS,
                difficulty=DifficultyLevel.BEGINNER,
                total_time_minutes=20,
                learning_objectives=[
                    "Essere conjugation",
                    "Basic sentences",
                    "Identity expressions",
                ],
            ),
            "food_ordering": QuestionSet(
                set_id="food_order_01",
                name="Ordering Food in Italian",
                topic=LearningTopic.FOOD,
                difficulty=DifficultyLevel.INTERMEDIATE,
                total_time_minutes=25,
                learning_objectives=[
                    "Restaurant vocabulary",
                    "Polite requests",
                    "Food preferences",
                ],
            ),
        }

    def generate_question(
        self,
        topic: LearningTopic,
        difficulty: DifficultyLevel,
        question_type: Optional[QuestionType] = None,
        user_weak_areas: Optional[List[ErrorType]] = None,
    ) -> Optional[LearningQuestion]:
        """
        Generate a specific question based on parameters.

        Args:
            topic: The learning topic to focus on
            difficulty: Student's current level
            question_type: Specific type of question (optional)
            user_weak_areas: Error types the student struggles with

        Returns:
            A generated learning question or None if no match found
        """
        topic_key = topic.value
        difficulty_key = difficulty.value

        if topic_key not in self.question_templates:
            return None

        if difficulty_key not in self.question_templates[topic_key]:
            return None

        templates = self.question_templates[topic_key][difficulty_key]

        # Filter by question type if specified
        if question_type:
            templates = [t for t in templates if t["type"] == question_type]

        # Prioritize questions that address user's weak areas
        if user_weak_areas:
            relevant_templates = [
                t
                for t in templates
                if any(error in t.get("target_errors", []) for error in user_weak_areas)
            ]
            if relevant_templates:
                templates = relevant_templates

        if not templates:
            return None

        # Select random template
        template = random.choice(templates)

        # Generate unique question ID
        question_id = (
            f"{topic_key}_{difficulty_key}_{template['type'].value}_{random.randint(1000, 9999)}"
        )

        return LearningQuestion(
            question_id=question_id,
            question_type=template["type"],
            difficulty=difficulty,
            topic=topic,
            question_text=template["template"],
            correct_answer=template["answer"],
            alternatives=template.get("alternatives", []),
            hints=template.get("hints", []),
            explanation=template.get("explanation", ""),
            target_errors=template.get("target_errors", []),
            learning_objectives=template.get("learning_objectives", []),
            estimated_time_minutes=template.get("time_minutes", 2),
        )

    def get_question_set(self, set_id: str) -> Optional[QuestionSet]:
        """Get a predefined question set by ID."""
        if set_id not in self.question_sets:
            return None

        question_set = self.question_sets[set_id]

        # Generate questions for the set if not already populated
        if not question_set.questions:
            for _ in range(5):  # Generate 5 questions per set
                question = self.generate_question(
                    topic=question_set.topic, difficulty=question_set.difficulty
                )
                if question:
                    question_set.questions.append(question)

        return question_set

    def generate_adaptive_questions(
        self,
        user_level: DifficultyLevel,
        recent_errors: List[ErrorType],
        topics_covered: List[LearningTopic],
        num_questions: int = 3,
    ) -> List[LearningQuestion]:
        """
        Generate adaptive questions based on user performance.

        Args:
            user_level: Current difficulty level
            recent_errors: Recent error types to focus on
            topics_covered: Topics already covered in session
            num_questions: Number of questions to generate

        Returns:
            List of adaptive learning questions
        """
        questions = []

        # Prioritize topics the user hasn't covered much
        all_topics = list(LearningTopic)
        uncovered_topics = [t for t in all_topics if t not in topics_covered]

        # Mix of review (covered topics) and new content (uncovered topics)
        topic_pool = (
            uncovered_topics[:2] + topics_covered[-1:] if topics_covered else uncovered_topics[:3]
        )

        for i in range(num_questions):
            # Focus on user's weak areas for first question
            if i == 0 and recent_errors:
                topic = random.choice(topic_pool) if topic_pool else LearningTopic.GREETINGS
                question = self.generate_question(
                    topic=topic, difficulty=user_level, user_weak_areas=recent_errors
                )
            else:
                # Mix of different question types for variety
                topic = random.choice(topic_pool) if topic_pool else LearningTopic.GREETINGS
                question = self.generate_question(topic=topic, difficulty=user_level)

            if question:
                questions.append(question)

        return questions

    def validate_answer(
        self, question: LearningQuestion, user_answer: str, partial_credit: bool = True
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate user's answer and provide detailed feedback.

        Args:
            question: The learning question
            user_answer: User's response
            partial_credit: Whether to give partial credit

        Returns:
            Tuple of (is_correct, score_percentage, feedback_messages)
        """
        feedback = []
        score = 0.0

        correct_answer = question.correct_answer.lower().strip()
        user_answer = user_answer.lower().strip()

        # Exact match
        if user_answer == correct_answer:
            score = 100.0
            feedback.append("Perfetto! Correct answer!")
            return True, score, feedback

        # Partial credit scenarios
        if partial_credit and question.question_type in [
            QuestionType.FILL_IN_BLANK,
            QuestionType.TRANSLATION,
        ]:
            # Check if answer is in alternatives (for multiple choice style)
            if question.alternatives and user_answer in [
                alt.lower() for alt in question.alternatives
            ]:
                if user_answer == correct_answer:
                    score = 100.0
                    feedback.append("Correct!")
                else:
                    score = 25.0
                    feedback.append(f"Close, but the best answer is '{question.correct_answer}'.")

            # Check for common variations or typos
            elif self._is_close_answer(user_answer, correct_answer):
                score = 75.0
                feedback.append(f"Very close! The exact answer is '{question.correct_answer}'.")

        # Incorrect
        if score == 0.0:
            feedback.append(f"Not quite. The correct answer is '{question.correct_answer}'.")
            if question.explanation:
                feedback.append(question.explanation)

        return score >= 50.0, score, feedback

    def _is_close_answer(self, user_answer: str, correct_answer: str) -> bool:
        """Check if user answer is close to correct (handles typos, etc.)."""
        # Simple edit distance check for typos
        if len(user_answer) == len(correct_answer):
            differences = sum(c1 != c2 for c1, c2 in zip(user_answer, correct_answer))
            return differences <= 1  # Allow 1 character difference

        # Check if it's the same words in different order (for longer phrases)
        user_words = set(user_answer.split())
        correct_words = set(correct_answer.split())
        return user_words == correct_words


# Global instance for pre-LoRA use
educational_question_engine = EducationalQuestionEngine()
