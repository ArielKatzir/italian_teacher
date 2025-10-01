"""
Educational Domain Module

This module contains educational functionality for the Italian Teacher system.
Currently focused on the teacher homework assignment system.
"""

# Main exports are from the teacher submodule
from .teacher.assignment import CEFRLevel, ExerciseType, GrammarFocus, HomeworkAssignment

__all__ = [
    "HomeworkAssignment",
    "CEFRLevel",
    "GrammarFocus",
    "ExerciseType",
]
