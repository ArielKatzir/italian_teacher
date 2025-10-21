"""
Teacher Homework Assignment Module

This module provides the Teacher Interface & Homework Assignment System
for creating and managing homework assignments.
"""

from .assignment import CEFRLevel, ExerciseType, GrammarFocus, HomeworkAssignment

# Note: homework_generator and assignment_manager can be imported separately if needed
# from .homework_generator import HomeworkGenerator
# from .assignment_manager import AssignmentManager

__all__ = [
    "HomeworkAssignment",
    "CEFRLevel",
    "GrammarFocus",
    "ExerciseType",
]
