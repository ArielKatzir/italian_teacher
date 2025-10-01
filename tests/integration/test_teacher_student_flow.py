"""
Integration test for teacher-to-student homework flow.

Tests the complete flow:
1. Teacher creates assignment
2. Homework is generated
3. Homework is distributed to students
4. Students receive correct homework matching specifications
"""

import sys
from pathlib import Path

import pytest

# Add demos to path for importing staging environment
demos_path = Path(__file__).parent.parent.parent / "demos"
sys.path.insert(0, str(demos_path))

from staging_teacher_flow import (
    CEFRLevel,
    GrammarFocus,
    HomeworkAssignment,
    StagingEnvironment,
)


@pytest.mark.integration
class TestTeacherStudentFlow:
    """Integration tests for teacher-to-student homework flow."""

    def test_complete_flow_simple(self):
        """Test complete flow with simple assignment."""
        # Setup
        env = StagingEnvironment()
        env.add_student("s001", "Test Student", "A2")

        # Create assignment
        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=5)

        # Generate homework
        homework_set = env.create_homework(assignment)

        # Verify homework matches assignment
        assert homework_set.cefr_level == assignment.cefr_level.value
        assert len(homework_set.exercises) == assignment.quantity

        # Distribute
        distributions = env.distribute_homework(homework_set)
        assert len(distributions) == 1

        # Verify student received it
        student_assignments = env.get_student_assignments("s001")
        assert len(student_assignments) == 1
        assert student_assignments[0].homework_set.cefr_level == "A2"

    def test_homework_matches_grammar_focus(self):
        """Test that homework exercises match specified grammar focus."""
        env = StagingEnvironment()
        env.add_student("s001", "Student 1", "A2")

        # Create assignment with past tense focus
        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.A2, quantity=3, grammar_focus=GrammarFocus.PAST_TENSE
        )

        homework_set = env.create_homework(assignment)

        # Verify
        assert homework_set.grammar_focus == "past_tense"
        assert len(homework_set.exercises) == 3

        # All exercises should be related to past tense
        for exercise in homework_set.exercises:
            assert (
                "past tense" in exercise.question.lower() or "passato" in exercise.question.lower()
            )

    def test_homework_includes_topic(self):
        """Test that homework exercises include specified topic."""
        env = StagingEnvironment()
        env.add_student("s001", "Student 1", "A2")

        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.A2, quantity=3, topic="history of Milan"
        )

        homework_set = env.create_homework(assignment)

        # Verify topic is included
        assert homework_set.topic == "history of Milan"

        # Exercises should reference the topic
        for exercise in homework_set.exercises:
            assert "history of Milan" in exercise.question or "Milano" in exercise.question

    def test_multiple_students_receive_same_homework(self):
        """Test that multiple students receive the same homework."""
        env = StagingEnvironment()
        env.add_student("s001", "Student 1", "A2")
        env.add_student("s002", "Student 2", "A2")
        env.add_student("s003", "Student 3", "A2")

        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.A2, quantity=5, grammar_focus=GrammarFocus.PRESENT_TENSE
        )

        homework_set = env.create_homework(assignment)
        distributions = env.distribute_homework(homework_set)

        # All 3 students should receive it
        assert len(distributions) == 3

        # Verify each student has the assignment
        for student_id in ["s001", "s002", "s003"]:
            assignments = env.get_student_assignments(student_id)
            assert len(assignments) == 1
            assert assignments[0].homework_set.assignment_id == homework_set.assignment_id

    def test_homework_verification_passes(self):
        """Test that homework verification correctly validates matching homework."""
        env = StagingEnvironment()

        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.B1,
            quantity=10,
            grammar_focus=GrammarFocus.SUBJUNCTIVE,
            topic="Italian cinema",
        )

        homework_set = env.create_homework(assignment)

        # Verification should pass
        result = env.verify_homework_correctness(assignment, homework_set)
        assert result is True

    def test_homework_quantity_matches(self):
        """Test that homework quantity matches assignment."""
        env = StagingEnvironment()

        for quantity in [1, 3, 5, 10]:
            assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=quantity)

            homework_set = env.create_homework(assignment)
            assert len(homework_set.exercises) == quantity

    def test_different_cefr_levels(self):
        """Test homework generation for different CEFR levels."""
        env = StagingEnvironment()

        levels = [CEFRLevel.A1, CEFRLevel.A2, CEFRLevel.B1, CEFRLevel.B2]

        for level in levels:
            assignment = HomeworkAssignment(cefr_level=level, quantity=3)

            homework_set = env.create_homework(assignment)
            assert homework_set.cefr_level == level.value
            assert len(homework_set.exercises) == 3

    def test_assignment_status_tracking(self):
        """Test that assignment status is tracked correctly."""
        env = StagingEnvironment()
        env.add_student("s001", "Student 1", "A2")

        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=3)

        homework_set = env.create_homework(assignment)
        distributions = env.distribute_homework(homework_set)

        # Check initial status
        assert distributions[0].status == "assigned"

        # Verify student can see their assignment
        student_assignments = env.get_student_assignments("s001")
        assert len(student_assignments) == 1
        assert student_assignments[0].status == "assigned"

    def test_real_world_scenario_a2_past_tense_milan(self):
        """
        Real-world scenario: A2 class, past tense, history of Milan.

        This is the exact scenario from the demo.
        """
        env = StagingEnvironment()

        # Setup class
        env.add_student("s001", "Maria Rossi", "A2")
        env.add_student("s002", "Giovanni Bianchi", "A2")
        env.add_student("s003", "Lucia Verde", "A2")

        # Create assignment
        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.A2,
            quantity=5,
            grammar_focus=GrammarFocus.PAST_TENSE,
            topic="history of Milan",
        )

        # Generate and verify
        homework_set = env.create_homework(assignment)
        assert env.verify_homework_correctness(assignment, homework_set)

        # Distribute
        distributions = env.distribute_homework(homework_set)
        assert len(distributions) == 3

        # Verify all students received it
        for student_id in ["s001", "s002", "s003"]:
            assignments = env.get_student_assignments(student_id)
            assert len(assignments) == 1
            assert assignments[0].homework_set.cefr_level == "A2"
            assert assignments[0].homework_set.grammar_focus == "past_tense"
            assert assignments[0].homework_set.topic == "history of Milan"
            assert len(assignments[0].homework_set.exercises) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
