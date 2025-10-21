"""
Italian Teacher - Educational Homework System

A system for creating and managing Italian language homework assignments.
"""

__version__ = "0.2.0"
__author__ = "Italian Teacher Team"

# Main exports for the homework system
from .educational.teacher.assignment import (
    CEFRLevel,
    ExerciseType,
    GrammarFocus,
    HomeworkAssignment,
)

__all__ = [
    "HomeworkAssignment",
    "CEFRLevel",
    "GrammarFocus",
    "ExerciseType",
]
