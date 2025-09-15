"""
Educational Validator - Advanced Question Validation and Feedback System

This module provides sophisticated validation for student answers with detailed
feedback generation, progress tracking, and adaptive assessment capabilities.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pre_lora.educational_questions import LearningQuestion, QuestionType


class FeedbackType(Enum):
    """Types of educational feedback."""

    ENCOURAGEMENT = "encouragement"
    CORRECTION = "correction"
    HINT = "hint"
    EXPLANATION = "explanation"
    CULTURAL_NOTE = "cultural_note"
    GRAMMAR_TIP = "grammar_tip"
    PROGRESS_UPDATE = "progress_update"


class AnswerQuality(Enum):
    """Quality assessment of student answers."""

    EXCELLENT = "excellent"  # 95-100%
    GOOD = "good"  # 80-94%
    SATISFACTORY = "satisfactory"  # 60-79%
    NEEDS_IMPROVEMENT = "needs_improvement"  # 40-59%
    POOR = "poor"  # 0-39%


@dataclass
class ValidationResult:
    """Comprehensive validation result with detailed feedback."""

    is_correct: bool
    score_percentage: float
    answer_quality: AnswerQuality
    primary_feedback: str
    detailed_feedback: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    grammar_notes: List[str] = field(default_factory=list)
    cultural_insights: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


@dataclass
class AnswerAnalysis:
    """Detailed analysis of student's answer."""

    original_answer: str
    normalized_answer: str
    detected_errors: List[str] = field(default_factory=list)
    partial_matches: List[str] = field(default_factory=list)
    alternative_interpretations: List[str] = field(default_factory=list)
    effort_indicators: List[str] = field(default_factory=list)  # Shows student tried


class EducationalValidator:
    """
    Advanced validation system for educational questions with rich feedback.

    Provides detailed analysis, multiple feedback types, and adaptive
    responses based on student performance patterns.
    """

    def __init__(self):
        self.italian_patterns = self._build_italian_patterns()
        self.feedback_templates = self._build_feedback_templates()

    def validate_answer(
        self,
        question: LearningQuestion,
        user_answer: str,
        student_context: Dict[str, Any],
        agent_personality: Dict[str, Any],
    ) -> ValidationResult:
        """
        Comprehensive answer validation with personalized feedback.

        Args:
            question: The learning question being answered
            user_answer: Student's response
            student_context: Student's learning context and history
            agent_personality: Current agent's personality for feedback style

        Returns:
            Detailed validation result with rich feedback
        """
        # Analyze the answer
        analysis = self._analyze_answer(question, user_answer)

        # Calculate score and correctness
        score, is_correct = self._calculate_score(question, analysis)

        # Determine answer quality
        quality = self._assess_quality(score, analysis, student_context)

        # Generate personalized feedback
        feedback_components = self._generate_feedback_components(
            question, analysis, score, student_context, agent_personality
        )

        # Compile primary feedback message
        primary_feedback = self._compile_primary_feedback(
            feedback_components, agent_personality, is_correct
        )

        # Generate suggestions and next steps
        suggestions = self._generate_suggestions(question, analysis, student_context)
        next_steps = self._generate_next_steps(question, score, student_context)

        return ValidationResult(
            is_correct=is_correct,
            score_percentage=score,
            answer_quality=quality,
            primary_feedback=primary_feedback,
            detailed_feedback=feedback_components,
            suggestions=suggestions,
            grammar_notes=self._extract_grammar_notes(question, analysis),
            cultural_insights=self._extract_cultural_insights(question, analysis),
            next_steps=next_steps,
        )

    def _analyze_answer(self, question: LearningQuestion, user_answer: str) -> AnswerAnalysis:
        """Perform detailed analysis of student's answer."""
        original = user_answer.strip()
        normalized = self._normalize_italian_text(original)
        correct_normalized = self._normalize_italian_text(question.correct_answer)

        analysis = AnswerAnalysis(original_answer=original, normalized_answer=normalized)

        # Check for exact match
        if normalized == correct_normalized:
            return analysis  # Perfect match, no further analysis needed

        # Detect common errors
        analysis.detected_errors = self._detect_common_errors(
            normalized, correct_normalized, question.question_type
        )

        # Find partial matches
        analysis.partial_matches = self._find_partial_matches(
            normalized, correct_normalized, question.alternatives
        )

        # Look for alternative valid interpretations
        analysis.alternative_interpretations = self._check_alternative_answers(normalized, question)

        # Assess effort and understanding
        analysis.effort_indicators = self._assess_effort(original, question, correct_normalized)

        return analysis

    def _calculate_score(
        self, question: LearningQuestion, analysis: AnswerAnalysis
    ) -> Tuple[float, bool]:
        """Calculate numerical score and correctness determination."""
        correct_normalized = self._normalize_italian_text(question.correct_answer)

        # Perfect match
        if analysis.normalized_answer == correct_normalized:
            return 100.0, True

        # Check alternatives for exact match
        for alt in question.alternatives:
            if analysis.normalized_answer == self._normalize_italian_text(alt):
                return 100.0, True

        # Alternative valid interpretations
        if analysis.alternative_interpretations:
            return 90.0, True

        # Partial credit based on different factors
        score = 0.0

        # Partial matches (close but not exact)
        if analysis.partial_matches:
            best_match = max(analysis.partial_matches, key=len)
            similarity = len(best_match) / max(
                len(correct_normalized), len(analysis.normalized_answer)
            )
            score = max(score, similarity * 75)  # Up to 75% for good partial matches

        # Effort recognition (student tried, shows understanding)
        if analysis.effort_indicators:
            score = max(score, 25.0)  # Minimum 25% for genuine effort

        # Adjust based on question type
        if question.question_type == QuestionType.CONVERSATION_PROMPT:
            # More lenient scoring for conversation practice
            if any("comunicativ" in indicator for indicator in analysis.effort_indicators):
                score = max(score, 60.0)
        elif question.question_type == QuestionType.CULTURAL_CONTEXT:
            # Cultural questions allow more interpretation
            if analysis.effort_indicators:
                score = max(score, 50.0)

        # Minor errors shouldn't completely fail the answer
        if len(analysis.detected_errors) == 1 and len(analysis.effort_indicators) > 1:
            score = max(score, 65.0)

        is_correct = score >= 60.0  # 60% threshold for "correct"
        return min(score, 100.0), is_correct

    def _assess_quality(
        self, score: float, analysis: AnswerAnalysis, context: Dict[str, Any]
    ) -> AnswerQuality:
        """Assess overall quality of the answer."""
        if score >= 95:
            return AnswerQuality.EXCELLENT
        elif score >= 80:
            return AnswerQuality.GOOD
        elif score >= 60:
            return AnswerQuality.SATISFACTORY
        elif score >= 40:
            return AnswerQuality.NEEDS_IMPROVEMENT
        else:
            return AnswerQuality.POOR

    def _generate_feedback_components(
        self,
        question: LearningQuestion,
        analysis: AnswerAnalysis,
        score: float,
        student_context: Dict[str, Any],
        agent_personality: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate detailed feedback components."""
        components = []
        enthusiasm = agent_personality.get("enthusiasm_level", 5)
        patience = agent_personality.get("patience_level", 5)

        # Main correctness feedback
        if score >= 90:
            if enthusiasm >= 7:
                components.append(
                    {
                        "type": FeedbackType.ENCOURAGEMENT.value,
                        "message": "Fantastico! Perfetto! 🎉",
                        "priority": 1,
                    }
                )
            else:
                components.append(
                    {
                        "type": FeedbackType.ENCOURAGEMENT.value,
                        "message": "Excellent work!",
                        "priority": 1,
                    }
                )
        elif score >= 60:
            if patience >= 7:
                components.append(
                    {
                        "type": FeedbackType.ENCOURAGEMENT.value,
                        "message": "Bravo! Quasi perfetto!",
                        "priority": 1,
                    }
                )
            else:
                components.append(
                    {
                        "type": FeedbackType.ENCOURAGEMENT.value,
                        "message": "Good job! Very close!",
                        "priority": 1,
                    }
                )
        else:
            if patience >= 8:
                components.append(
                    {
                        "type": FeedbackType.ENCOURAGEMENT.value,
                        "message": "Non ti preoccupare, succede a tutti! Proviamo insieme.",
                        "priority": 1,
                    }
                )
            else:
                components.append(
                    {
                        "type": FeedbackType.ENCOURAGEMENT.value,
                        "message": "Let's work on this together.",
                        "priority": 1,
                    }
                )

        # Specific corrections
        if analysis.detected_errors and score < 90:
            correction_msg = f"Try: '{question.correct_answer}'"
            if question.explanation:
                correction_msg += f" - {question.explanation}"

            components.append(
                {"type": FeedbackType.CORRECTION.value, "message": correction_msg, "priority": 2}
            )

        # Grammar tips for grammar questions
        if question.question_type in [
            QuestionType.GRAMMAR_CORRECTION,
            QuestionType.VERB_CONJUGATION,
        ]:
            grammar_tip = self._generate_grammar_tip(question, analysis)
            if grammar_tip:
                components.append(
                    {"type": FeedbackType.GRAMMAR_TIP.value, "message": grammar_tip, "priority": 3}
                )

        # Cultural insights for cultural questions
        if question.question_type == QuestionType.CULTURAL_CONTEXT:
            cultural_insight = self._generate_cultural_insight(question, analysis)
            if cultural_insight:
                components.append(
                    {
                        "type": FeedbackType.CULTURAL_NOTE.value,
                        "message": cultural_insight,
                        "priority": 3,
                    }
                )

        return sorted(components, key=lambda x: x["priority"])

    def _compile_primary_feedback(
        self, components: List[Dict[str, Any]], agent_personality: Dict[str, Any], is_correct: bool
    ) -> str:
        """Compile the main feedback message from components."""
        if not components:
            return "Keep practicing!"

        # Take the highest priority components
        primary_components = [c for c in components if c["priority"] <= 2]

        if len(primary_components) == 1:
            return primary_components[0]["message"]
        elif len(primary_components) == 2:
            return f"{primary_components[0]['message']} {primary_components[1]['message']}"
        else:
            # Combine encouragement with most important correction
            encouragement = next(
                (c for c in primary_components if c["type"] == "encouragement"), None
            )
            correction = next((c for c in primary_components if c["type"] == "correction"), None)

            if encouragement and correction:
                return f"{encouragement['message']} {correction['message']}"

            return primary_components[0]["message"]

    def _normalize_italian_text(self, text: str) -> str:
        """Normalize Italian text for comparison."""
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Handle common punctuation variations
        normalized = re.sub(r"[.!?]+$", "", normalized)  # Remove ending punctuation
        normalized = re.sub(r'[\'"' '""«»]', "", normalized)  # Remove quotes

        # Normalize apostrophes
        normalized = normalized.replace("'", "'").replace("'", "'")

        return normalized

    def _detect_common_errors(
        self, user_answer: str, correct_answer: str, question_type: QuestionType
    ) -> List[str]:
        """Detect common Italian language errors."""
        errors = []

        # Gender agreement errors
        if question_type in [QuestionType.FILL_IN_BLANK, QuestionType.GRAMMAR_CORRECTION]:
            if self._check_gender_agreement_error(user_answer, correct_answer):
                errors.append("gender_agreement")

        # Verb conjugation errors
        if question_type == QuestionType.VERB_CONJUGATION:
            if self._check_verb_conjugation_error(user_answer, correct_answer):
                errors.append("verb_conjugation")

        # Article errors
        if self._check_article_error(user_answer, correct_answer):
            errors.append("article_choice")

        # Preposition errors
        if self._check_preposition_error(user_answer, correct_answer):
            errors.append("preposition_choice")

        return errors

    def _find_partial_matches(
        self, user_answer: str, correct_answer: str, alternatives: List[str]
    ) -> List[str]:
        """Find partial matches in user's answer."""
        matches = []

        # Check word-level matches
        user_words = set(user_answer.split())
        correct_words = set(correct_answer.split())

        common_words = user_words.intersection(correct_words)
        if common_words:
            matches.extend(common_words)

        # Check against alternatives
        for alt in alternatives:
            alt_normalized = self._normalize_italian_text(alt)
            alt_words = set(alt_normalized.split())
            alt_common = user_words.intersection(alt_words)
            if alt_common:
                matches.extend(alt_common)

        return list(set(matches))  # Remove duplicates

    def _check_alternative_answers(self, user_answer: str, question: LearningQuestion) -> List[str]:
        """Check for alternative valid answers not in the provided alternatives."""
        alternatives = []

        # For conversation prompts, check for communicatively appropriate responses
        if question.question_type == QuestionType.CONVERSATION_PROMPT:
            if self._is_communicatively_appropriate(user_answer, question):
                alternatives.append("communicatively_appropriate")

        # For cultural questions, check for culturally aware responses
        if question.question_type == QuestionType.CULTURAL_CONTEXT:
            if self._is_culturally_aware(user_answer, question):
                alternatives.append("culturally_aware")

        return alternatives

    def _assess_effort(
        self, user_answer: str, question: LearningQuestion, correct_answer: str
    ) -> List[str]:
        """Assess indicators that student made genuine effort."""
        indicators = []

        # Check answer length (not just random short answers)
        if len(user_answer.strip()) >= 3:
            indicators.append("adequate_length")

        # Check if answer is in Italian (contains Italian words/patterns)
        if self._contains_italian_patterns(user_answer):
            indicators.append("italian_attempt")

        # Check if answer shows understanding of question topic
        if self._shows_topic_understanding(user_answer, question):
            indicators.append("topic_understanding")

        # Check if answer shows grammatical structure attempt
        if self._shows_grammar_attempt(user_answer):
            indicators.append("grammar_attempt")

        return indicators

    def _generate_suggestions(
        self, question: LearningQuestion, analysis: AnswerAnalysis, context: Dict[str, Any]
    ) -> List[str]:
        """Generate helpful suggestions for improvement."""
        suggestions = []

        if "gender_agreement" in analysis.detected_errors:
            suggestions.append(
                "Remember: nouns ending in -a are usually feminine (la/una), -o are usually masculine (il/un)"
            )

        if "verb_conjugation" in analysis.detected_errors:
            suggestions.append(
                "Practice verb conjugation patterns - each pronoun has its specific ending"
            )

        if "article_choice" in analysis.detected_errors:
            suggestions.append(
                "Focus on definite articles (il, la, lo, l') vs indefinite (un, una, uno)"
            )

        if not analysis.effort_indicators:
            suggestions.append(
                "Try to answer in Italian - even if not perfect, it helps with learning!"
            )

        return suggestions

    def _generate_next_steps(
        self, question: LearningQuestion, score: float, context: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized next steps for the student."""
        steps = []

        if score < 40:
            steps.append("Review the basics of this topic before trying more questions")
            steps.append("Ask for hints if you're stuck")
        elif score < 70:
            steps.append("You're on the right track! Practice similar questions")
            steps.append("Focus on the specific area that needs improvement")
        else:
            steps.append("Great progress! Ready for slightly more challenging questions")
            if question.learning_objectives:
                steps.append(f"You've mastered: {', '.join(question.learning_objectives[:2])}")

        return steps

    def _build_italian_patterns(self) -> Dict[str, List[str]]:
        """Build patterns for Italian language recognition."""
        return {
            "common_words": [
                "il",
                "la",
                "lo",
                "gli",
                "le",
                "un",
                "una",
                "uno",
                "essere",
                "avere",
                "fare",
                "dire",
                "andare",
                "sapere",
                "come",
                "cosa",
                "dove",
                "quando",
                "perché",
                "chi",
                "ciao",
                "buongiorno",
                "buonasera",
                "grazie",
                "prego",
            ],
            "verb_endings": ["o", "i", "a", "iamo", "ate", "ano"],
            "common_endings": ["zione", "mente", "ello", "etto", "ino", "ina"],
        }

    def _build_feedback_templates(self) -> Dict[str, List[str]]:
        """Build templates for different types of feedback."""
        return {
            "encouragement": {
                "high_enthusiasm": [
                    "Fantastico! 🎉",
                    "Perfetto!",
                    "Bravissimo!",
                    "Che bravo!",
                    "Ottimo lavoro!",
                    "Sei incredibile!",
                ],
                "medium_enthusiasm": [
                    "Bravo!",
                    "Bene!",
                    "Molto bene!",
                    "Giusto!",
                    "Excellent!",
                    "Well done!",
                ],
                "supportive": [
                    "Non ti preoccupare, succede a tutti!",
                    "Proviamo insieme!",
                    "Ci sei quasi!",
                    "Keep trying, you're learning!",
                ],
            }
        }

    # Helper methods for error detection
    def _check_gender_agreement_error(self, user: str, correct: str) -> bool:
        """Check for gender agreement errors."""
        # Simplified check - in real implementation, this would be more sophisticated
        gender_pairs = [("la", "il"), ("una", "un"), ("bella", "bello")]
        for fem, masc in gender_pairs:
            if fem in correct and masc in user:
                return True
            if masc in correct and fem in user:
                return True
        return False

    def _check_verb_conjugation_error(self, user: str, correct: str) -> bool:
        """Check for verb conjugation errors."""
        # Basic pattern check
        verb_roots = ["esser", "aver", "far", "andar"]
        return any(root in user and root in correct for root in verb_roots)

    def _check_article_error(self, user: str, correct: str) -> bool:
        """Check for article errors."""
        articles = ["il", "la", "lo", "l'", "gli", "le", "un", "una", "uno"]
        user_articles = [word for word in user.split() if word in articles]
        correct_articles = [word for word in correct.split() if word in articles]
        return user_articles != correct_articles

    def _check_preposition_error(self, user: str, correct: str) -> bool:
        """Check for preposition errors."""
        prepositions = ["di", "a", "da", "in", "con", "su", "per", "tra", "fra"]
        user_prep = [word for word in user.split() if word in prepositions]
        correct_prep = [word for word in correct.split() if word in prepositions]
        return user_prep != correct_prep

    def _is_communicatively_appropriate(self, answer: str, question: LearningQuestion) -> bool:
        """Check if answer is communicatively appropriate for conversation."""
        # Basic heuristic - contains greeting words, polite expressions, etc.
        appropriate_words = ["ciao", "grazie", "prego", "scusa", "bene", "tutto", "come"]
        return any(word in answer.lower() for word in appropriate_words)

    def _is_culturally_aware(self, answer: str, question: LearningQuestion) -> bool:
        """Check if answer shows cultural awareness."""
        # Basic check for cultural knowledge indicators
        cultural_words = ["famiglia", "tradizione", "cultura", "italiano", "italia"]
        return any(word in answer.lower() for word in cultural_words)

    def _contains_italian_patterns(self, text: str) -> bool:
        """Check if text contains recognizable Italian patterns."""
        for word_list in self.italian_patterns.values():
            if any(pattern in text.lower() for pattern in word_list):
                return True
        return False

    def _shows_topic_understanding(self, answer: str, question: LearningQuestion) -> bool:
        """Check if answer shows understanding of the topic."""
        # Check if answer contains topic-related vocabulary
        topic_keywords = {
            "greetings": ["ciao", "buongiorno", "saluti", "come"],
            "family": ["famiglia", "madre", "padre", "fratello", "sorella"],
            "food": ["mangiare", "cibo", "pizza", "pasta", "ristorante"],
        }

        question_topic = (
            question.topic.value if hasattr(question.topic, "value") else str(question.topic)
        )
        keywords = topic_keywords.get(question_topic, [])
        return any(keyword in answer.lower() for keyword in keywords)

    def _shows_grammar_attempt(self, answer: str) -> bool:
        """Check if answer shows grammatical structure attempt."""
        # Basic check for sentence structure
        words = answer.strip().split()
        return len(words) >= 2 and not all(word.isdigit() for word in words)

    def _generate_grammar_tip(
        self, question: LearningQuestion, analysis: AnswerAnalysis
    ) -> Optional[str]:
        """Generate grammar-specific tip."""
        if question.question_type == QuestionType.VERB_CONJUGATION:
            return "Remember: -o (io), -i (tu), -a (lui/lei), -iamo (noi), -ate (voi), -ano (loro)"
        elif "gender_agreement" in analysis.detected_errors:
            return "Tip: Adjectives must agree with nouns in gender and number"
        return None

    def _generate_cultural_insight(
        self, question: LearningQuestion, analysis: AnswerAnalysis
    ) -> Optional[str]:
        """Generate cultural insight for cultural questions."""
        if question.question_type == QuestionType.CULTURAL_CONTEXT:
            return "Cultural note: This expression reflects Italian values of family and community"
        return None

    def _extract_grammar_notes(
        self, question: LearningQuestion, analysis: AnswerAnalysis
    ) -> List[str]:
        """Extract relevant grammar notes."""
        notes = []
        if "verb_conjugation" in analysis.detected_errors:
            notes.append("Verb conjugation varies by person and number")
        if "gender_agreement" in analysis.detected_errors:
            notes.append("Articles and adjectives must match noun gender")
        return notes

    def _extract_cultural_insights(
        self, question: LearningQuestion, analysis: AnswerAnalysis
    ) -> List[str]:
        """Extract cultural insights."""
        insights = []
        if question.question_type == QuestionType.CULTURAL_CONTEXT:
            insights.append("Understanding cultural context helps with natural communication")
        return insights


# Global validator instance
educational_validator = EducationalValidator()
