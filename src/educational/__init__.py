"""
Educational Domain Module

This module contains all educational functionality including:
- Curriculum management and learning paths
- Question generation and validation
- Educational analytics and progress tracking
- Student assessment and feedback systems

The educational domain is separated from core infrastructure to maintain
clean architecture and enable easier testing and maintenance.

Import specific modules as needed:
- from educational.service import EducationalService
- from educational.curriculum import CurriculumManager
- from educational.validator import EducationalValidator
"""

# Lazy imports to avoid dependency issues
__all__ = [
    "EducationalService",
    "CurriculumManager",
    "curriculum_manager",
    "EducationalValidator",
    "educational_validator",
]


def __getattr__(name):
    """Lazy import of educational components."""
    if name == "EducationalService":
        from educational.service import EducationalService

        return EducationalService
    elif name == "CurriculumManager":
        from educational.curriculum import CurriculumManager

        return CurriculumManager
    elif name == "curriculum_manager":
        from educational.curriculum import curriculum_manager

        return curriculum_manager
    elif name == "EducationalValidator":
        from educational.validator import EducationalValidator

        return EducationalValidator
    elif name == "educational_validator":
        from educational.validator import educational_validator

        return educational_validator
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
