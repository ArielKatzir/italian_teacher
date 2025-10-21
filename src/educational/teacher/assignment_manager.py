"""
Assignment Manager

Handles distribution and management of homework assignments to students.
Integrates with the Teacher Command Processing system to deliver homework to student interfaces.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .homework_generator import HomeworkSet

logger = logging.getLogger(__name__)


@dataclass
class StudentProfile:
    """Student profile information."""

    student_id: str
    name: str
    cefr_level: str
    preferred_language: str = "en"  # Interface language
    active: bool = True
    learning_preferences: Optional[Dict] = None

    def __post_init__(self):
        if self.learning_preferences is None:
            self.learning_preferences = {
                "exercise_types": [],
                "topics_of_interest": [],
                "difficulty_preference": "standard",
            }


@dataclass
class AssignmentDistribution:
    """Assignment distribution record."""

    assignment_id: str
    student_id: str
    homework_set: HomeworkSet
    assigned_at: datetime
    due_date: Optional[datetime] = None
    status: str = "assigned"  # assigned, in_progress, completed, overdue
    progress: Dict = None

    def __post_init__(self):
        if self.progress is None:
            self.progress = {
                "exercises_completed": 0,
                "total_exercises": len(self.homework_set.exercises),
                "score": None,
                "started_at": None,
                "completed_at": None,
            }


class AssignmentManager:
    """Manages homework assignment distribution and tracking."""

    def __init__(self, storage_path: str = "data/assignments"):
        """
        Initialize assignment manager.

        Args:
            storage_path: Path to store assignment data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.students_file = self.storage_path / "students.json"
        self.assignments_file = self.storage_path / "assignments.json"

        self.students: Dict[str, StudentProfile] = self._load_students()
        self.assignments: Dict[str, AssignmentDistribution] = self._load_assignments()

        logger.info(f"AssignmentManager initialized with {len(self.students)} students")

    def register_student(
        self,
        student_id: str,
        name: str,
        cefr_level: str,
        preferred_language: str = "en",
        learning_preferences: Optional[Dict] = None,
    ) -> StudentProfile:
        """
        Register a new student in the system.

        Args:
            student_id: Unique identifier for the student
            name: Student's name
            cefr_level: Student's CEFR level (A1, A2, B1, B2, C1, C2)
            preferred_language: Interface language preference
            learning_preferences: Optional learning preferences

        Returns:
            StudentProfile object
        """
        student = StudentProfile(
            student_id=student_id,
            name=name,
            cefr_level=cefr_level,
            preferred_language=preferred_language,
            learning_preferences=learning_preferences,
        )

        self.students[student_id] = student
        self._save_students()

        logger.info(f"Registered new student: {name} ({student_id}) - Level {cefr_level}")
        return student

    def distribute_assignment(
        self, homework_set: HomeworkSet, target_students: List[str] = None, due_days: int = 7
    ) -> List[AssignmentDistribution]:
        """
        Distribute homework assignment to students.

        Args:
            homework_set: The homework set to distribute
            target_students: List of student IDs, or None for all students
            due_days: Number of days until due date

        Returns:
            List of AssignmentDistribution records
        """
        logger.info(f"Distributing assignment to students")

        # Determine target students
        if target_students is None:
            # Distribute to all active students at appropriate level
            target_students = self._get_students_for_level(homework_set.assignment.cefr_level)
        else:
            # Filter for active students only
            target_students = [
                sid for sid in target_students if sid in self.students and self.students[sid].active
            ]

        distributions = []
        assignment_time = datetime.now()
        due_date = assignment_time + timedelta(days=due_days)

        for student_id in target_students:
            assignment_id = f"{student_id}_{assignment_time.strftime('%Y%m%d_%H%M%S')}"

            distribution = AssignmentDistribution(
                assignment_id=assignment_id,
                student_id=student_id,
                homework_set=homework_set,
                assigned_at=assignment_time,
                due_date=due_date,
            )

            self.assignments[assignment_id] = distribution
            distributions.append(distribution)

            # Send notification to student interface
            self._notify_student(student_id, distribution)

        self._save_assignments()

        logger.info(f"Distributed assignment to {len(distributions)} students")
        return distributions

    def get_student_assignments(
        self, student_id: str, status_filter: Optional[List[str]] = None
    ) -> List[AssignmentDistribution]:
        """
        Get assignments for a specific student.

        Args:
            student_id: Student ID
            status_filter: Optional list of statuses to filter by

        Returns:
            List of assignments for the student
        """
        student_assignments = [
            assignment
            for assignment in self.assignments.values()
            if assignment.student_id == student_id
        ]

        if status_filter:
            student_assignments = [
                assignment
                for assignment in student_assignments
                if assignment.status in status_filter
            ]

        return sorted(student_assignments, key=lambda x: x.assigned_at, reverse=True)

    def update_assignment_progress(
        self,
        assignment_id: str,
        exercises_completed: int,
        score: Optional[float] = None,
        completed: bool = False,
    ) -> bool:
        """
        Update progress on an assignment.

        Args:
            assignment_id: Assignment ID
            exercises_completed: Number of exercises completed
            score: Optional score (0-100)
            completed: Whether assignment is fully completed

        Returns:
            True if update was successful
        """
        if assignment_id not in self.assignments:
            logger.error(f"Assignment not found: {assignment_id}")
            return False

        assignment = self.assignments[assignment_id]

        # Update progress
        assignment.progress["exercises_completed"] = exercises_completed
        if score is not None:
            assignment.progress["score"] = score

        # Update timestamps
        if assignment.progress["started_at"] is None and exercises_completed > 0:
            assignment.progress["started_at"] = datetime.now()

        if completed:
            assignment.status = "completed"
            assignment.progress["completed_at"] = datetime.now()
        elif exercises_completed > 0:
            assignment.status = "in_progress"

        # Check for overdue
        if (
            assignment.due_date
            and datetime.now() > assignment.due_date
            and assignment.status != "completed"
        ):
            assignment.status = "overdue"

        self._save_assignments()

        logger.info(f"Updated progress for assignment {assignment_id}")
        return True

    def get_assignment_statistics(self) -> Dict:
        """
        Get statistics about assignments and student performance.

        Returns:
            Dictionary with assignment statistics
        """
        total_assignments = len(self.assignments)
        completed_assignments = sum(1 for a in self.assignments.values() if a.status == "completed")
        in_progress_assignments = sum(
            1 for a in self.assignments.values() if a.status == "in_progress"
        )
        overdue_assignments = sum(1 for a in self.assignments.values() if a.status == "overdue")

        # Calculate average scores
        completed_with_scores = [
            a
            for a in self.assignments.values()
            if a.status == "completed" and a.progress.get("score") is not None
        ]

        avg_score = None
        if completed_with_scores:
            avg_score = sum(a.progress["score"] for a in completed_with_scores) / len(
                completed_with_scores
            )

        return {
            "total_assignments": total_assignments,
            "completed": completed_assignments,
            "in_progress": in_progress_assignments,
            "overdue": overdue_assignments,
            "completion_rate": (
                completed_assignments / total_assignments if total_assignments > 0 else 0
            ),
            "average_score": avg_score,
            "total_students": len(self.students),
            "active_students": sum(1 for s in self.students.values() if s.active),
        }

    def export_assignment_data(self, output_path: str) -> str:
        """
        Export assignment data for analysis.

        Args:
            output_path: Path to save exported data

        Returns:
            Path to exported file
        """
        export_data = {
            "students": {sid: asdict(student) for sid, student in self.students.items()},
            "assignments": {
                aid: self._assignment_to_dict(assignment)
                for aid, assignment in self.assignments.items()
            },
            "statistics": self.get_assignment_statistics(),
            "exported_at": datetime.now().isoformat(),
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Assignment data exported to: {output_file}")
        return str(output_file)

    def _get_students_for_level(self, cefr_level) -> List[str]:
        """Get list of student IDs at the specified CEFR level."""
        return [
            student_id
            for student_id, student in self.students.items()
            if student.active and student.cefr_level == cefr_level.value
        ]

    def _notify_student(self, student_id: str, distribution: AssignmentDistribution):
        """Send notification to student about new assignment."""
        # This would integrate with the student interface system
        # For now, we'll just log the notification
        student = self.students.get(student_id)
        if student:
            logger.info(
                f"📚 Notification sent to {student.name}: New homework assignment available"
            )
            # TODO: Integrate with actual student notification system

    def _load_students(self) -> Dict[str, StudentProfile]:
        """Load students from storage."""
        if not self.students_file.exists():
            return {}

        try:
            with open(self.students_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {sid: StudentProfile(**student_data) for sid, student_data in data.items()}
        except Exception as e:
            logger.error(f"Error loading students: {e}")
            return {}

    def _load_assignments(self) -> Dict[str, AssignmentDistribution]:
        """Load assignments from storage."""
        if not self.assignments_file.exists():
            return {}

        try:
            with open(self.assignments_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                assignments = {}
                for aid, assignment_data in data.items():
                    # Convert datetime strings back to datetime objects
                    assignment_data["assigned_at"] = datetime.fromisoformat(
                        assignment_data["assigned_at"]
                    )
                    if assignment_data.get("due_date"):
                        assignment_data["due_date"] = datetime.fromisoformat(
                            assignment_data["due_date"]
                        )

                    # Reconstruct homework set (simplified - in practice might load from separate files)
                    # For now, we'll store a reference to the homework data
                    assignments[aid] = AssignmentDistribution(**assignment_data)

                return assignments
        except Exception as e:
            logger.error(f"Error loading assignments: {e}")
            return {}

    def _save_students(self):
        """Save students to storage."""
        try:
            data = {sid: asdict(student) for sid, student in self.students.items()}
            with open(self.students_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving students: {e}")

    def _save_assignments(self):
        """Save assignments to storage."""
        try:
            data = {
                aid: self._assignment_to_dict(assignment)
                for aid, assignment in self.assignments.items()
            }
            with open(self.assignments_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving assignments: {e}")

    def _assignment_to_dict(self, assignment: AssignmentDistribution) -> Dict:
        """Convert assignment to dictionary for serialization."""
        assignment_dict = asdict(assignment)
        # Convert datetime objects to ISO format strings
        assignment_dict["assigned_at"] = assignment.assigned_at.isoformat()
        if assignment.due_date:
            assignment_dict["due_date"] = assignment.due_date.isoformat()

        return assignment_dict
