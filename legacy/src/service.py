"""
Educational Service - Core Integration Layer

This service integrates the educational question system with the agent framework,
providing a bridge between pre-LoRA question templates and the core agent system.
This layer will remain post-LoRA but delegate to different question generators.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.error_tolerance import DetectedError, ErrorToleranceSystem
from core.motivation_context import MotivationContext, MotivationLevel
from pre_lora.educational_questions import (
    DifficultyLevel,
    LearningQuestion,
    LearningTopic,
    QuestionType,
    educational_question_engine,
)

from educational.curriculum import CurriculumManager
from educational.validator import educational_validator


class LearningSession(Enum):
    """Types of learning sessions available."""

    FREE_CONVERSATION = "free_conversation"
    STRUCTURED_LESSON = "structured_lesson"
    ASSESSMENT = "assessment"
    REVIEW_PRACTICE = "review_practice"
    ADAPTIVE_DRILLING = "adaptive_drilling"


@dataclass
class EducationalConfig:
    """Configuration for educational features."""

    questions_per_session: int = 3
    adaptive_questioning: bool = True
    difficulty_auto_adjust: bool = True
    focus_on_weak_areas: bool = True
    include_cultural_questions: bool = True
    max_consecutive_questions: int = 5
    question_timeout_minutes: int = 3


@dataclass
class StudentProfile:
    """Tracks student's learning progress and preferences."""

    student_id: str
    current_level: DifficultyLevel = DifficultyLevel.BEGINNER
    topics_mastered: List[LearningTopic] = field(default_factory=list)
    preferred_topics: List[LearningTopic] = field(default_factory=list)
    weak_areas: List[LearningTopic] = field(default_factory=list)
    learning_goals: List[str] = field(default_factory=list)
    total_questions_answered: int = 0
    total_correct_answers: int = 0
    last_assessment_date: Optional[datetime] = None


@dataclass
class QuestionSession:
    """Tracks a current questioning session."""

    session_id: str
    session_type: LearningSession
    started_at: datetime
    current_question: Optional[LearningQuestion] = None
    questions_asked: List[LearningQuestion] = field(default_factory=list)
    answers_given: List[Dict[str, Any]] = field(default_factory=list)
    session_complete: bool = False
    target_questions: int = 3


class EducationalService:
    """
    Core educational service that integrates question generation with agent system.

    This service coordinates between:
    - Question generation (pre-LoRA templates, later LoRA-based)
    - Motivation tracking
    - Error tolerance
    - Agent personality systems
    """

    def __init__(
        self,
        motivation_context: MotivationContext,
        error_tolerance: ErrorToleranceSystem,
        config: Optional[EducationalConfig] = None,
        curriculum_manager: Optional[CurriculumManager] = None,
    ):
        self.motivation_context = motivation_context
        self.error_tolerance = error_tolerance
        self.config = config or EducationalConfig()

        # Curriculum management for structured learning
        self.curriculum_manager = curriculum_manager or curriculum_manager

        # Question engine (pre-LoRA, will be replaced)
        self.question_engine = educational_question_engine

        # Advanced validator with rich feedback (core system, will remain post-LoRA)
        self.validator = educational_validator

        # Session tracking
        self.active_sessions: Dict[str, QuestionSession] = {}
        self.student_profiles: Dict[str, StudentProfile] = {}

    def start_learning_session(
        self,
        student_id: str,
        session_type: LearningSession = LearningSession.ADAPTIVE_DRILLING,
        target_topics: Optional[List[LearningTopic]] = None,
        target_questions: int = 3,
    ) -> str:
        """
        Start a new educational session for a student.

        Args:
            student_id: Unique student identifier
            session_type: Type of learning session
            target_topics: Specific topics to focus on (optional)
            target_questions: Number of questions to ask

        Returns:
            Session ID for tracking
        """
        # Get or create student profile
        if student_id not in self.student_profiles:
            self.student_profiles[student_id] = StudentProfile(student_id=student_id)
            # Auto-enroll in beginner curriculum if not already enrolled
            if student_id not in self.curriculum_manager.student_progress:
                self.curriculum_manager.enroll_student(student_id, "italian_beginner")

        self.student_profiles[student_id]

        # Create session
        session_id = f"{student_id}_{session_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        session = QuestionSession(
            session_id=session_id,
            session_type=session_type,
            started_at=datetime.now(),
            target_questions=min(target_questions, self.config.max_consecutive_questions),
        )

        self.active_sessions[session_id] = session

        return session_id

    def get_next_question(
        self, session_id: str, conversation_context: Optional[Dict[str, Any]] = None
    ) -> Optional[LearningQuestion]:
        """
        Get the next appropriate question for the session.

        Args:
            session_id: Active session identifier
            conversation_context: Current conversation context

        Returns:
            Next learning question or None if session complete
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        student = self.student_profiles.get(session_id.split("_")[0])

        if not student or session.session_complete:
            return None

        # Check if session should continue
        if len(session.questions_asked) >= session.target_questions:
            session.session_complete = True
            return None

        # Get user's current state from motivation context
        motivation_context = self.motivation_context.get_context_for_model()
        user_metrics = motivation_context["user_metrics"]
        behavioral_triggers = motivation_context["behavioral_triggers"]

        # Determine question parameters based on user state and session type
        if session.session_type == LearningSession.ADAPTIVE_DRILLING:
            question = self._generate_adaptive_question(student, user_metrics, behavioral_triggers)
        elif session.session_type == LearningSession.REVIEW_PRACTICE:
            question = self._generate_review_question(student, user_metrics)
        elif session.session_type == LearningSession.ASSESSMENT:
            question = self._generate_assessment_question(student)
        else:
            # Default to adaptive
            question = self._generate_adaptive_question(student, user_metrics, behavioral_triggers)

        if question:
            session.current_question = question
            session.questions_asked.append(question)

        return question

    def process_answer(
        self, session_id: str, user_answer: str, agent_personality: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process student's answer and provide educational feedback.

        Args:
            session_id: Active session identifier
            user_answer: Student's response
            agent_personality: Current agent's personality for feedback style

        Returns:
            Dictionary with feedback, score, and next steps
        """
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]
        if not session.current_question:
            return {"error": "No active question"}

        question = session.current_question
        student = self.student_profiles[session_id.split("_")[0]]

        # Use advanced validator for comprehensive answer analysis
        student_context = {
            "accuracy": (student.total_correct_answers / max(1, student.total_questions_answered))
            * 100,
            "total_questions": student.total_questions_answered,
            "weak_areas": [topic.value for topic in student.weak_areas],
            "motivation_level": self.motivation_context.detect_motivation_level().value,
            "learning_goals": student.learning_goals,
        }

        validation_result = self.validator.validate_answer(
            question, user_answer, student_context, agent_personality
        )

        is_correct = validation_result.is_correct
        score = validation_result.score_percentage

        # Update student progress
        student.total_questions_answered += 1
        if is_correct:
            student.total_correct_answers += 1

        # Create detailed detected errors for motivation system
        detected_errors = self._create_detected_errors_from_validation(
            validation_result, question, user_answer
        )

        # Update motivation context with comprehensive error analysis
        if detected_errors:
            self.motivation_context.update_progress_with_errors(
                detected_errors, topic=question.topic.value
            )
        else:
            self.motivation_context.update_progress(correct=True, topic=question.topic.value)

        # Use advanced validator's feedback instead of simple agent feedback
        educational_feedback = validation_result.primary_feedback

        # Record answer with rich validation data
        answer_record = {
            "question_id": question.question_id,
            "user_answer": user_answer,
            "correct_answer": question.correct_answer,
            "is_correct": is_correct,
            "score": score,
            "answer_quality": validation_result.answer_quality.value,
            "timestamp": datetime.now(),
            "feedback": validation_result.detailed_feedback,
            "suggestions": validation_result.suggestions,
        }
        session.answers_given.append(answer_record)

        # Update curriculum progress
        self._update_curriculum_progress(student, question, is_correct, score, session)

        # Update student weak areas
        if not is_correct and question.topic not in student.weak_areas:
            student.weak_areas.append(question.topic)
        elif is_correct and question.topic in student.weak_areas:
            # Remove from weak areas if they're improving
            if self._check_topic_improvement(student, question.topic):
                student.weak_areas.remove(question.topic)

        # Clear current question
        session.current_question = None

        return {
            "is_correct": is_correct,
            "score": score,
            "answer_quality": validation_result.answer_quality.value,
            "feedback": educational_feedback,
            "detailed_feedback": validation_result.detailed_feedback,
            "suggestions": validation_result.suggestions,
            "grammar_notes": validation_result.grammar_notes,
            "cultural_insights": validation_result.cultural_insights,
            "next_steps": validation_result.next_steps,
            "explanation": question.explanation,
            "learning_objectives": question.learning_objectives,
            "session_progress": {
                "questions_completed": len(session.answers_given),
                "target_questions": session.target_questions,
                "session_complete": len(session.answers_given) >= session.target_questions,
            },
            "student_progress": {
                "total_accuracy": (
                    student.total_correct_answers / max(1, student.total_questions_answered)
                )
                * 100,
                "questions_answered": student.total_questions_answered,
                "current_level": student.current_level.value,
            },
            # Educational analytics integration
            "educational_analytics": self.motivation_context.get_context_for_model().get(
                "educational_analytics", {}
            ),
            "motivation_context": {
                "motivation_level": self.motivation_context.detect_motivation_level().value,
                "needs_encouragement": self.motivation_context.detect_motivation_level()
                in [MotivationLevel.LOW, MotivationLevel.FRUSTRATED],
            },
        }

    def _generate_adaptive_question(
        self,
        student: StudentProfile,
        user_metrics: Dict[str, Any],
        behavioral_triggers: Dict[str, Any],
    ) -> Optional[LearningQuestion]:
        """Generate an adaptive question based on curriculum progression and student state."""
        # Get curriculum-recommended topics and difficulty
        next_topics = self.curriculum_manager.get_next_topics(student.student_id)

        if not next_topics:
            # Fallback if curriculum is complete or not available
            next_topics = [LearningTopic.GREETINGS]

        # Use curriculum-recommended difficulty
        primary_topic = next_topics[0]
        target_level = self.curriculum_manager.get_recommended_difficulty(
            student.student_id, primary_topic
        )

        # Adjust difficulty based on recent performance (but stay within curriculum bounds)
        if behavioral_triggers["struggling"] or user_metrics["accuracy"] < 50:
            # Student struggling - use easier questions within the same topic
            # Don't change difficulty level, just make questions more supportive
            pass
        elif user_metrics["accuracy"] > 90 and not behavioral_triggers["struggling"]:
            # Student excelling - could try slightly harder questions
            # But still stay within curriculum progression
            pass

        # Prioritize topics based on curriculum vs weak areas
        target_topic = primary_topic
        if student.weak_areas and behavioral_triggers.get("struggling", False):
            # If struggling, focus on weak areas that are in curriculum path
            curriculum_weak_areas = [topic for topic in student.weak_areas if topic in next_topics]
            if curriculum_weak_areas:
                target_topic = curriculum_weak_areas[0]

        # Get recent error types from motivation context
        recent_errors = []
        if "educational_analytics" in self.motivation_context.get_context_for_model():
            analytics = self.motivation_context.get_context_for_model()["educational_analytics"]
            recent_errors = [error_type for error_type in analytics.get("recent_error_pattern", [])]

        return self.question_engine.generate_question(
            topic=target_topic, difficulty=target_level, user_weak_areas=recent_errors
        )

    def _generate_review_question(
        self, student: StudentProfile, user_metrics: Dict[str, Any]
    ) -> Optional[LearningQuestion]:
        """Generate a review question from previously covered topics."""
        if not student.topics_mastered:
            return self._generate_adaptive_question(student, user_metrics, {})

        # Review a previously mastered topic
        review_topic = student.topics_mastered[-1]  # Most recently mastered
        return self.question_engine.generate_question(
            topic=review_topic, difficulty=student.current_level
        )

    def _generate_assessment_question(self, student: StudentProfile) -> Optional[LearningQuestion]:
        """Generate an assessment question for formal evaluation."""
        return self.question_engine.generate_question(
            topic=LearningTopic.GRAMMAR_VERBS,  # Core assessment topic
            difficulty=student.current_level,
            question_type=QuestionType.GRAMMAR_CORRECTION,  # Formal assessment type
        )

    def _generate_agent_feedback(
        self,
        question: LearningQuestion,
        is_correct: bool,
        score: float,
        feedback_messages: List[str],
        agent_personality: Dict[str, Any],
    ) -> str:
        """Generate personality-appropriate feedback for the answer."""
        enthusiasm = agent_personality.get("enthusiasm_level", 5)
        patience = agent_personality.get("patience_level", 5)
        correction_style = agent_personality.get("correction_style", "gentle")

        if is_correct:
            if enthusiasm >= 7:
                return f"Fantastico! {feedback_messages[0]} 🎉"
            else:
                return f"Bravo! {feedback_messages[0]}"
        else:
            if patience >= 8:
                encouragement = "Non ti preoccupare, succede a tutti! "
            else:
                encouragement = ""

            if correction_style == "gentle":
                return f"{encouragement}{feedback_messages[0]} Proviamo ancora!"
            else:
                return f"{encouragement}{feedback_messages[0]}"

    def _determine_error_severity(self, score: float):
        """Determine error severity based on answer score."""
        from .error_tolerance import ErrorSeverity

        if score >= 75:
            return ErrorSeverity.MINOR
        elif score >= 50:
            return ErrorSeverity.MODERATE
        elif score >= 25:
            return ErrorSeverity.MAJOR
        else:
            return ErrorSeverity.CRITICAL

    def _get_next_difficulty_level(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get next higher difficulty level."""
        if current == DifficultyLevel.BEGINNER:
            return DifficultyLevel.INTERMEDIATE
        elif current == DifficultyLevel.INTERMEDIATE:
            return DifficultyLevel.ADVANCED
        return DifficultyLevel.ADVANCED

    def _get_previous_difficulty_level(self, current: DifficultyLevel) -> DifficultyLevel:
        """Get previous lower difficulty level."""
        if current == DifficultyLevel.ADVANCED:
            return DifficultyLevel.INTERMEDIATE
        elif current == DifficultyLevel.INTERMEDIATE:
            return DifficultyLevel.BEGINNER
        return DifficultyLevel.BEGINNER

    def _check_topic_improvement(self, student: StudentProfile, topic: LearningTopic) -> bool:
        """Check if student has improved in a specific topic."""
        # Simple heuristic: if they've answered 3+ questions correctly in this topic recently
        # In a real implementation, this would track topic-specific accuracy
        return student.total_correct_answers > 3

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of completed learning session."""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]
        student = self.student_profiles[session_id.split("_")[0]]

        # Calculate session statistics
        total_questions = len(session.answers_given)
        correct_answers = sum(1 for answer in session.answers_given if answer["is_correct"])
        average_score = sum(answer["score"] for answer in session.answers_given) / max(
            1, total_questions
        )

        # Topics covered
        topics_covered = list(set(q.topic.value for q in session.questions_asked))

        return {
            "session_id": session_id,
            "session_type": session.session_type.value,
            "duration_minutes": (datetime.now() - session.started_at).total_seconds() / 60,
            "questions_asked": total_questions,
            "correct_answers": correct_answers,
            "session_accuracy": (correct_answers / max(1, total_questions)) * 100,
            "average_score": average_score,
            "topics_covered": topics_covered,
            "student_progress": {
                "current_level": student.current_level.value,
                "total_accuracy": (
                    student.total_correct_answers / max(1, student.total_questions_answered)
                )
                * 100,
                "weak_areas": [topic.value for topic in student.weak_areas],
                "topics_mastered": [topic.value for topic in student.topics_mastered],
            },
            "recommendations": self._generate_recommendations(student, session),
        }

    def _generate_recommendations(
        self, student: StudentProfile, session: QuestionSession
    ) -> List[str]:
        """Generate learning recommendations based on session performance."""
        recommendations = []

        # Analyze session performance
        session_accuracy = sum(1 for answer in session.answers_given if answer["is_correct"]) / max(
            1, len(session.answers_given)
        )

        if session_accuracy < 0.5:
            recommendations.append("Consider reviewing basic concepts before moving to new topics")
            recommendations.append("Practice with easier questions to build confidence")
        elif session_accuracy > 0.8:
            recommendations.append("Great progress! Ready for more challenging questions")
            recommendations.append("Consider exploring new topics or advanced grammar")

        # Weak area recommendations
        if student.weak_areas:
            weak_topics = [topic.value for topic in student.weak_areas[:2]]
            recommendations.append(f"Focus practice on: {', '.join(weak_topics)}")

        return recommendations

    def _update_curriculum_progress(
        self,
        student: StudentProfile,
        question: LearningQuestion,
        is_correct: bool,
        score: float,
        session: QuestionSession,
    ) -> None:
        """Update student's curriculum progress based on question performance."""
        # Count questions answered for this topic in this session
        topic_questions_in_session = [
            q for q in session.questions_asked if q.topic == question.topic
        ]
        topic_answers_in_session = [
            a
            for a in session.answers_given
            if any(
                q.question_id == a["question_id"] and q.topic == question.topic
                for q in session.questions_asked
            )
        ]

        # Calculate topic performance metrics
        questions_answered = len(topic_answers_in_session)
        correct_answers = sum(1 for a in topic_answers_in_session if a["is_correct"])
        question_types_used = list(set(q.question_type for q in topic_questions_in_session))

        # Check curriculum completion for this topic
        is_topic_completed, progress_details = self.curriculum_manager.check_topic_completion(
            student.student_id,
            question.topic,
            questions_answered,
            correct_answers,
            question_types_used,
        )

        # Update student profile if topic completed
        if is_topic_completed and question.topic not in student.topics_mastered:
            student.topics_mastered.append(question.topic)

            # Check if this completes a curriculum unit
            self._check_unit_completion(student)

    def _check_unit_completion(self, student: StudentProfile) -> None:
        """Check if student has completed current curriculum unit."""
        curriculum_status = self.curriculum_manager.get_student_curriculum_status(
            student.student_id
        )

        if "current_unit" in curriculum_status:
            current_unit = curriculum_status["current_unit"]
            if current_unit.get("unit_progress", 0) >= 100:
                # Unit completed! This would trigger advancement to next unit
                # The curriculum manager handles this automatically in get_next_topics()
                pass

    def get_curriculum_overview(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive curriculum overview for student dashboard."""
        curriculum_status = self.curriculum_manager.get_student_curriculum_status(student_id)

        # Add educational service specific data
        if student_id in self.student_profiles:
            student = self.student_profiles[student_id]
            curriculum_status["educational_profile"] = {
                "preferred_topics": [topic.value for topic in student.preferred_topics],
                "learning_goals": student.learning_goals,
                "total_questions_answered": student.total_questions_answered,
                "overall_accuracy": (
                    student.total_correct_answers / max(1, student.total_questions_answered)
                )
                * 100,
            }

        return curriculum_status

    def set_student_learning_path(self, student_id: str, path_id: str) -> bool:
        """Enroll student in a specific learning path."""
        success = self.curriculum_manager.enroll_student(student_id, path_id)

        if success and student_id in self.student_profiles:
            student = self.student_profiles[student_id]
            # Update difficulty level based on path
            if path_id == "italian_beginner":
                student.current_level = DifficultyLevel.BEGINNER
            elif path_id == "italian_intermediate":
                student.current_level = DifficultyLevel.INTERMEDIATE
            elif path_id == "italian_advanced":
                student.current_level = DifficultyLevel.ADVANCED

        return success

    def get_available_learning_paths(self) -> Dict[str, Dict[str, Any]]:
        """Get all available learning paths for student selection."""
        paths = {}
        for path_id, path in self.curriculum_manager.learning_paths.items():
            paths[path_id] = {
                "name": path.name,
                "description": path.description,
                "target_level": path.target_level.value,
                "total_units": len(path.units),
                "estimated_hours": path.total_estimated_hours,
                "unit_names": [unit.name for unit in path.units],
            }
        return paths

    def _create_detected_errors_from_validation(
        self, validation_result, question: LearningQuestion, user_answer: str
    ) -> List[DetectedError]:
        """Create DetectedError objects from validation result for motivation system."""
        detected_errors = []

        if not validation_result.is_correct and validation_result.score_percentage < 90:
            # Create primary error based on question type and validation feedback
            error_type = self._map_question_to_error_type(question.question_type)
            severity = self._map_score_to_severity(validation_result.score_percentage)

            # Extract specific error information from detailed feedback
            error_explanation = ""
            if validation_result.detailed_feedback:
                correction_feedback = next(
                    (
                        f
                        for f in validation_result.detailed_feedback
                        if f.get("type") == "correction"
                    ),
                    None,
                )
                if correction_feedback:
                    error_explanation = correction_feedback.get("message", "")

            # Create the detected error
            detected_error = DetectedError(
                original_text=user_answer,
                error_type=error_type,
                severity=severity,
                position=(0, len(user_answer)),
                suggested_correction=question.correct_answer,
                explanation=error_explanation or question.explanation or "",
                confidence=0.85,  # High confidence from educational validator
                context={
                    "educational_question": True,
                    "question_type": question.question_type.value,
                    "question_topic": question.topic.value,
                    "answer_quality": validation_result.answer_quality.value,
                    "validation_score": validation_result.score_percentage,
                    "grammar_notes": validation_result.grammar_notes,
                    "cultural_insights": validation_result.cultural_insights,
                },
            )

            detected_errors.append(detected_error)

        return detected_errors

    def _map_question_to_error_type(self, question_type: QuestionType):
        """Map question types to appropriate error types."""
        from .error_tolerance import ErrorType

        mapping = {
            QuestionType.VERB_CONJUGATION: ErrorType.VERB_CONJUGATION,
            QuestionType.GRAMMAR_CORRECTION: ErrorType.GRAMMAR,
            QuestionType.FILL_IN_BLANK: ErrorType.VOCABULARY,
            QuestionType.TRANSLATION: ErrorType.VOCABULARY,
            QuestionType.MULTIPLE_CHOICE: ErrorType.VOCABULARY,
            QuestionType.VOCABULARY_MATCH: ErrorType.VOCABULARY,
            QuestionType.CONVERSATION_PROMPT: ErrorType.CULTURAL,
            QuestionType.CULTURAL_CONTEXT: ErrorType.CULTURAL,
        }

        return mapping.get(question_type, ErrorType.VOCABULARY)

    def _map_score_to_severity(self, score: float):
        """Map validation score to error severity."""
        from .error_tolerance import ErrorSeverity

        if score >= 75:
            return ErrorSeverity.MINOR
        elif score >= 50:
            return ErrorSeverity.MODERATE
        elif score >= 25:
            return ErrorSeverity.MAJOR
        else:
            return ErrorSeverity.CRITICAL
