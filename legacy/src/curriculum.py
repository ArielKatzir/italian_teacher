"""
Curriculum Manager - Structured Learning Progression System

This module defines educational curricula with topic dependencies, learning paths,
and progression requirements. It ensures students learn topics in the correct order
and advance only when ready.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pre_lora.educational_questions import DifficultyLevel, LearningTopic, QuestionType

from .learning_units import ADVANCED_UNITS, BEGINNER_UNITS, INTERMEDIATE_UNITS

logger = logging.getLogger(__name__)


class ProgressionCriteria(Enum):
    """Criteria for advancing to next topic/level."""

    ACCURACY_BASED = "accuracy_based"  # Must achieve X% accuracy
    QUESTION_COUNT = "question_count"  # Must answer X questions correctly
    TIME_BASED = "time_based"  # Must study for X minutes
    MASTERY_DEMONSTRATION = "mastery_demonstration"  # Must demonstrate understanding


@dataclass
class TopicRequirement:
    """Defines requirements for completing a topic."""

    topic: LearningTopic
    min_accuracy: float = 80.0  # Minimum accuracy percentage required
    min_correct_answers: int = 5  # Minimum correct answers needed
    min_questions_attempted: int = 8  # Minimum questions that must be attempted
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
    prerequisites: List[str] = field(default_factory=list)  # Unit IDs that must be completed first
    learning_objectives: List[str] = field(default_factory=list)
    estimated_hours: float = 2.0


@dataclass
class LearningPath:
    """Complete learning path with curriculum units."""

    path_id: str
    name: str
    description: str
    target_level: DifficultyLevel
    units: List[CurriculumUnit]
    total_estimated_hours: float = 0.0


@dataclass
class StudentProgress:
    """Tracks student's progress through curriculum."""

    student_id: str
    current_path_id: str
    completed_units: Set[str] = field(default_factory=set)
    current_unit_id: Optional[str] = None
    topic_progress: Dict[LearningTopic, Dict[str, Any]] = field(default_factory=dict)
    unit_start_times: Dict[str, Any] = field(default_factory=dict)
    last_updated: Optional[Any] = None


class CurriculumManager:
    """
    Manages structured learning paths and topic progression.

    Ensures students follow appropriate learning sequences and advance
    only when they've mastered prerequisites.
    """

    def __init__(self):
        self.learning_paths = self._build_curriculum_paths()
        self.student_progress: Dict[str, StudentProgress] = {}

    def _build_curriculum_paths(self) -> Dict[str, LearningPath]:
        """Build comprehensive curriculum paths for different learning goals."""

        # BEGINNER PATH: Complete Italian Basics
        beginner_path = LearningPath(
            path_id="italian_beginner",
            name="Italian Basics for Beginners",
            description="Complete introduction to Italian language fundamentals",
            target_level=DifficultyLevel.BEGINNER,
            units=BEGINNER_UNITS,
        )

        # INTERMEDIATE PATH: Expanding Communication
        intermediate_path = LearningPath(
            path_id="italian_intermediate",
            name="Intermediate Italian Communication",
            description="Build fluency and handle complex conversations",
            target_level=DifficultyLevel.INTERMEDIATE,
            units=INTERMEDIATE_UNITS,
        )

        # ADVANCED PATH: Mastery and Culture
        advanced_path = LearningPath(
            path_id="italian_advanced",
            name="Advanced Italian Mastery",
            description="Achieve native-like fluency and cultural understanding",
            target_level=DifficultyLevel.ADVANCED,
            units=ADVANCED_UNITS,
        )

        # Calculate total hours for each path
        for path in [beginner_path, intermediate_path, advanced_path]:
            path.total_estimated_hours = sum(unit.estimated_hours for unit in path.units)

        return {
            "italian_beginner": beginner_path,
            "italian_intermediate": intermediate_path,
            "italian_advanced": advanced_path,
        }

    def enroll_student(self, student_id: str, path_id: str = "italian_beginner") -> bool:
        """
        Enroll a student in a learning path.

        Args:
            student_id: Unique student identifier
            path_id: Learning path to enroll in

        Returns:
            True if enrollment successful, False otherwise
        """
        if path_id not in self.learning_paths:
            logger.error(f"Learning path {path_id} not found")
            return False

        # Initialize student progress
        path = self.learning_paths[path_id]
        first_unit = path.units[0] if path.units else None

        self.student_progress[student_id] = StudentProgress(
            student_id=student_id,
            current_path_id=path_id,
            current_unit_id=first_unit.unit_id if first_unit else None,
        )

        logger.info(f"Enrolled student {student_id} in path {path_id}")
        return True

    def get_next_topics(self, student_id: str) -> List[LearningTopic]:
        """
        Get the next topics the student should study.

        Args:
            student_id: Student identifier

        Returns:
            List of topics in order of priority
        """
        if student_id not in self.student_progress:
            return [LearningTopic.GREETINGS]  # Default starting topic

        progress = self.student_progress[student_id]
        if not progress.current_unit_id:
            return [LearningTopic.GREETINGS]

        path = self.learning_paths[progress.current_path_id]
        current_unit = self._get_unit_by_id(path, progress.current_unit_id)

        if not current_unit:
            return [LearningTopic.GREETINGS]

        # Get topics from current unit that aren't completed yet
        incomplete_topics = []
        for topic_req in current_unit.topics:
            if not self._is_topic_completed(student_id, topic_req):
                incomplete_topics.append(topic_req.topic)

        if incomplete_topics:
            return incomplete_topics

        # If current unit is complete, advance to next unit
        next_unit = self._get_next_available_unit(student_id, path)
        if next_unit:
            progress.current_unit_id = next_unit.unit_id
            return [topic_req.topic for topic_req in next_unit.topics[:2]]  # First 2 topics

        return []  # Path completed

    def check_topic_completion(
        self,
        student_id: str,
        topic: LearningTopic,
        questions_answered: int,
        correct_answers: int,
        question_types_used: List[QuestionType],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if student has completed a topic according to curriculum requirements.

        Args:
            student_id: Student identifier
            topic: Topic to check
            questions_answered: Total questions answered for this topic
            correct_answers: Number of correct answers
            question_types_used: Types of questions answered

        Returns:
            Tuple of (is_completed, progress_details)
        """
        if student_id not in self.student_progress:
            return False, {"error": "Student not enrolled"}

        progress = self.student_progress[student_id]
        path = self.learning_paths[progress.current_path_id]
        current_unit = self._get_unit_by_id(path, progress.current_unit_id)

        if not current_unit:
            return False, {"error": "No current unit"}

        # Find topic requirement in current unit
        topic_req = None
        for req in current_unit.topics:
            if req.topic == topic:
                topic_req = req
                break

        if not topic_req:
            return False, {"error": "Topic not in current unit"}

        # Calculate progress
        accuracy = (correct_answers / max(1, questions_answered)) * 100

        # Check all requirements
        meets_accuracy = accuracy >= topic_req.min_accuracy
        meets_correct_count = correct_answers >= topic_req.min_correct_answers
        meets_attempt_count = questions_answered >= topic_req.min_questions_attempted

        # Check required question types
        meets_question_types = True
        if topic_req.required_question_types:
            required_types_set = set(topic_req.required_question_types)
            used_types_set = set(question_types_used)
            meets_question_types = required_types_set.issubset(used_types_set)

        is_completed = all(
            [meets_accuracy, meets_correct_count, meets_attempt_count, meets_question_types]
        )

        # Update student progress tracking
        if topic not in progress.topic_progress:
            progress.topic_progress[topic] = {}

        progress.topic_progress[topic].update(
            {
                "questions_answered": questions_answered,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "completed": is_completed,
                "question_types_used": [qt.value for qt in question_types_used],
            }
        )

        progress_details = {
            "accuracy": accuracy,
            "required_accuracy": topic_req.min_accuracy,
            "correct_answers": correct_answers,
            "required_correct": topic_req.min_correct_answers,
            "questions_answered": questions_answered,
            "required_attempts": topic_req.min_questions_attempted,
            "meets_requirements": {
                "accuracy": meets_accuracy,
                "correct_count": meets_correct_count,
                "attempt_count": meets_attempt_count,
                "question_types": meets_question_types,
            },
            "is_completed": is_completed,
        }

        return is_completed, progress_details

    def get_recommended_difficulty(self, student_id: str, topic: LearningTopic) -> DifficultyLevel:
        """
        Get recommended difficulty level for a student and topic.

        Args:
            student_id: Student identifier
            topic: Topic to get difficulty for

        Returns:
            Recommended difficulty level
        """
        if student_id not in self.student_progress:
            return DifficultyLevel.BEGINNER

        progress = self.student_progress[student_id]
        path = self.learning_paths[progress.current_path_id]

        # Return the difficulty level of the current unit
        current_unit = self._get_unit_by_id(path, progress.current_unit_id)
        if current_unit:
            return current_unit.difficulty_level

        return DifficultyLevel.BEGINNER

    def get_student_curriculum_status(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive curriculum status for a student."""
        if student_id not in self.student_progress:
            return {"error": "Student not enrolled"}

        progress = self.student_progress[student_id]
        path = self.learning_paths[progress.current_path_id]

        # Calculate overall progress
        total_units = len(path.units)
        completed_units = len(progress.completed_units)
        progress_percentage = (completed_units / max(1, total_units)) * 100

        # Get current unit status
        current_unit = self._get_unit_by_id(path, progress.current_unit_id)
        current_unit_progress = {}
        if current_unit:
            completed_topics = 0
            total_topics = len(current_unit.topics)

            for topic_req in current_unit.topics:
                if self._is_topic_completed(student_id, topic_req):
                    completed_topics += 1

            current_unit_progress = {
                "unit_name": current_unit.name,
                "topics_completed": completed_topics,
                "total_topics": total_topics,
                "unit_progress": (completed_topics / max(1, total_topics)) * 100,
            }

        return {
            "path_name": path.name,
            "overall_progress": progress_percentage,
            "units_completed": completed_units,
            "total_units": total_units,
            "current_unit": current_unit_progress,
            "next_topics": self.get_next_topics(student_id),
            "completed_units": list(progress.completed_units),
        }

    def _get_unit_by_id(
        self, path: LearningPath, unit_id: Optional[str]
    ) -> Optional[CurriculumUnit]:
        """Get unit by ID from a learning path."""
        if not unit_id:
            return None

        for unit in path.units:
            if unit.unit_id == unit_id:
                return unit
        return None

    def _is_topic_completed(self, student_id: str, topic_req: TopicRequirement) -> bool:
        """Check if a topic requirement is completed."""
        progress = self.student_progress[student_id]

        if topic_req.topic not in progress.topic_progress:
            return False

        topic_progress = progress.topic_progress[topic_req.topic]
        return topic_progress.get("completed", False)

    def _get_next_available_unit(
        self, student_id: str, path: LearningPath
    ) -> Optional[CurriculumUnit]:
        """Get the next unit the student can access."""
        progress = self.student_progress[student_id]

        for unit in path.units:
            # Skip if already completed
            if unit.unit_id in progress.completed_units:
                continue

            # Check if prerequisites are met
            prerequisites_met = all(
                prereq_id in progress.completed_units for prereq_id in unit.prerequisites
            )

            if prerequisites_met:
                return unit

        return None  # No available units


# Global instance
curriculum_manager = CurriculumManager()
