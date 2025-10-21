"""
Tests for HomeworkAssignment dataclass.
"""

import pytest

from educational.teacher.assignment import CEFRLevel, ExerciseType, GrammarFocus, HomeworkAssignment


class TestHomeworkAssignment:
    """Test HomeworkAssignment creation and validation."""

    def test_minimal_assignment(self):
        """Test creating assignment with only required fields."""
        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2)

        assert assignment.cefr_level == CEFRLevel.A2
        assert assignment.quantity == 5  # Default
        assert assignment.grammar_focus is None
        assert assignment.topic is None
        assert assignment.student_groups == ["all"]
        assert assignment.difficulty_scaling is True
        assert len(assignment.exercise_types) == 4  # Default types

    def test_full_assignment(self):
        """Test creating assignment with all fields specified."""
        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.B1,
            quantity=10,
            grammar_focus=GrammarFocus.SUBJUNCTIVE,
            topic="Italian cinema",
            student_groups=["group_a", "group_b"],
            exercise_types=[ExerciseType.ESSAY, ExerciseType.TRANSLATION],
            difficulty_scaling=False,
        )

        assert assignment.cefr_level == CEFRLevel.B1
        assert assignment.quantity == 10
        assert assignment.grammar_focus == GrammarFocus.SUBJUNCTIVE
        assert assignment.topic == "Italian cinema"
        assert assignment.student_groups == ["group_a", "group_b"]
        assert assignment.exercise_types == [ExerciseType.ESSAY, ExerciseType.TRANSLATION]
        assert assignment.difficulty_scaling is False

    def test_all_cefr_levels(self):
        """Test that all CEFR levels work."""
        levels = [
            CEFRLevel.A1,
            CEFRLevel.A2,
            CEFRLevel.B1,
            CEFRLevel.B2,
            CEFRLevel.C1,
            CEFRLevel.C2,
        ]

        for level in levels:
            assignment = HomeworkAssignment(cefr_level=level)
            assert assignment.cefr_level == level

    def test_all_grammar_focuses(self):
        """Test that all grammar focus options work."""
        grammar_options = [
            GrammarFocus.PRESENT_TENSE,
            GrammarFocus.PAST_TENSE,
            GrammarFocus.FUTURE_TENSE,
            GrammarFocus.SUBJUNCTIVE,
            GrammarFocus.CONDITIONAL,
            GrammarFocus.IMPERATIVE,
            GrammarFocus.PASSIVE_VOICE,
            GrammarFocus.PRONOUNS,
            GrammarFocus.ARTICLES,
            GrammarFocus.PREPOSITIONS,
            GrammarFocus.CONJUNCTIONS,
        ]

        for grammar in grammar_options:
            assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, grammar_focus=grammar)
            assert assignment.grammar_focus == grammar

    def test_all_exercise_types(self):
        """Test that all exercise types work."""
        exercise_options = [
            ExerciseType.FILL_IN_BLANK,
            ExerciseType.TRANSLATION,
            ExerciseType.SENTENCE_COMPLETION,
            ExerciseType.MULTIPLE_CHOICE,
            ExerciseType.ESSAY,
            ExerciseType.CONVERSATION,
        ]

        for exercise_type in exercise_options:
            assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, exercise_types=[exercise_type])
            assert exercise_type in assignment.exercise_types

    def test_quantity_validation(self):
        """Test that quantity is validated."""
        # Valid quantities
        for qty in [1, 5, 10, 20]:
            assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=qty)
            assert assignment.quantity == qty

        # Invalid quantities should raise ValueError
        with pytest.raises(ValueError, match="between 1 and 20"):
            HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=0)

        with pytest.raises(ValueError, match="between 1 and 20"):
            HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=21)

    def test_default_student_groups(self):
        """Test that student_groups defaults to ['all']."""
        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, student_groups=None)
        assert assignment.student_groups == ["all"]

        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, student_groups=[])
        assert assignment.student_groups == ["all"]

    def test_default_exercise_types(self):
        """Test that exercise_types gets sensible defaults."""
        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, exercise_types=None)
        assert ExerciseType.FILL_IN_BLANK in assignment.exercise_types
        assert ExerciseType.TRANSLATION in assignment.exercise_types
        assert ExerciseType.SENTENCE_COMPLETION in assignment.exercise_types
        assert ExerciseType.MULTIPLE_CHOICE in assignment.exercise_types

        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, exercise_types=[])
        assert len(assignment.exercise_types) == 4  # Should get defaults

    def test_to_prompt_context_minimal(self):
        """Test prompt context generation with minimal fields."""
        assignment = HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=5)

        context = assignment.to_prompt_context()

        assert "CEFR Level: A2" in context
        assert "Number of Exercises: 5" in context
        assert "Grammar Focus" not in context  # Should not appear if None
        assert "Topic" not in context  # Should not appear if None

    def test_to_prompt_context_full(self):
        """Test prompt context generation with all fields."""
        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.B1,
            quantity=10,
            grammar_focus=GrammarFocus.PAST_TENSE,
            topic="history of Milan",
        )

        context = assignment.to_prompt_context()

        assert "CEFR Level: B1" in context
        assert "Grammar Focus: Past Tense" in context
        assert "Topic: history of Milan" in context
        assert "Number of Exercises: 10" in context

    def test_repr(self):
        """Test string representation."""
        assignment = HomeworkAssignment(
            cefr_level=CEFRLevel.A2,
            quantity=5,
            grammar_focus=GrammarFocus.PAST_TENSE,
            topic="Italian food",
        )

        repr_str = repr(assignment)

        assert "A2" in repr_str
        assert "past_tense" in repr_str
        assert "Italian food" in repr_str
        assert "5x" in repr_str

    def test_common_use_cases(self):
        """Test common real-world use cases."""
        # Use case 1: Beginner homework about numbers
        assignment1 = HomeworkAssignment(cefr_level=CEFRLevel.A1, quantity=3, topic="numbers")
        assert assignment1.cefr_level == CEFRLevel.A1
        assert assignment1.quantity == 3
        assert assignment1.topic == "numbers"
        assert assignment1.grammar_focus is None

        # Use case 2: A2 past tense about history of Milan
        assignment2 = HomeworkAssignment(
            cefr_level=CEFRLevel.A2,
            quantity=5,
            grammar_focus=GrammarFocus.PAST_TENSE,
            topic="history of Milan",
        )
        assert assignment2.cefr_level == CEFRLevel.A2
        assert assignment2.grammar_focus == GrammarFocus.PAST_TENSE
        assert assignment2.topic == "history of Milan"

        # Use case 3: B1 subjunctive with custom exercise types
        assignment3 = HomeworkAssignment(
            cefr_level=CEFRLevel.B1,
            quantity=10,
            grammar_focus=GrammarFocus.SUBJUNCTIVE,
            topic="Italian cinema",
            exercise_types=[ExerciseType.ESSAY, ExerciseType.TRANSLATION],
        )
        assert assignment3.cefr_level == CEFRLevel.B1
        assert assignment3.grammar_focus == GrammarFocus.SUBJUNCTIVE
        assert assignment3.topic == "Italian cinema"
        assert len(assignment3.exercise_types) == 2
        assert ExerciseType.ESSAY in assignment3.exercise_types


class TestCEFRLevel:
    """Test CEFRLevel enum."""

    def test_all_levels_exist(self):
        """Test that all CEFR levels are defined."""
        assert CEFRLevel.A1.value == "A1"
        assert CEFRLevel.A2.value == "A2"
        assert CEFRLevel.B1.value == "B1"
        assert CEFRLevel.B2.value == "B2"
        assert CEFRLevel.C1.value == "C1"
        assert CEFRLevel.C2.value == "C2"

    def test_level_from_string(self):
        """Test creating levels from strings."""
        assert CEFRLevel("A1") == CEFRLevel.A1
        assert CEFRLevel("B2") == CEFRLevel.B2
        assert CEFRLevel("C1") == CEFRLevel.C1


class TestGrammarFocus:
    """Test GrammarFocus enum."""

    def test_all_grammar_topics_exist(self):
        """Test that all grammar topics are defined."""
        expected_topics = [
            "present_tense",
            "past_tense",
            "future_tense",
            "subjunctive",
            "conditional",
            "imperative",
            "passive_voice",
            "pronouns",
            "articles",
            "prepositions",
            "conjunctions",
        ]

        actual_topics = [g.value for g in GrammarFocus]

        for topic in expected_topics:
            assert topic in actual_topics

    def test_grammar_from_string(self):
        """Test creating grammar focus from strings."""
        assert GrammarFocus("past_tense") == GrammarFocus.PAST_TENSE
        assert GrammarFocus("subjunctive") == GrammarFocus.SUBJUNCTIVE


class TestExerciseType:
    """Test ExerciseType enum."""

    def test_all_exercise_types_exist(self):
        """Test that all exercise types are defined."""
        expected_types = [
            "fill_in_blank",
            "translation",
            "sentence_completion",
            "multiple_choice",
            "essay",
            "conversation",
        ]

        actual_types = [e.value for e in ExerciseType]

        for ex_type in expected_types:
            assert ex_type in actual_types

    def test_exercise_type_from_string(self):
        """Test creating exercise type from strings."""
        assert ExerciseType("fill_in_blank") == ExerciseType.FILL_IN_BLANK
        assert ExerciseType("essay") == ExerciseType.ESSAY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
