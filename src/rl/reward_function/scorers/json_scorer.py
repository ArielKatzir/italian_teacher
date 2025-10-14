"""
JSON validity scorer.

Validates exercise structure and format (0-15 points).
"""

from typing import Any, Dict, List, Set, Tuple

from .base import BaseScorer


class JSONScorer(BaseScorer):
    """
    Scores JSON structure validity (0-15 points).

    Checks:
    - All required fields present (6 pts)
    - Valid exercise type (3 pts)
    - Type-specific fields (6 pts)
    """

    def __init__(self, nlp=None):
        super().__init__(nlp)

        # Expected JSON schema
        self.required_fields: Set[str] = {"type", "question", "correct_answer"}
        self.valid_types: Set[str] = {"fill_in_blank", "translation", "multiple_choice"}
        self.mc_required_fields: Set[str] = {"options", "correct_option"}

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score JSON structure validity."""
        score = 0.0
        errors = []

        # Check required fields (6 pts)
        missing_fields = self.required_fields - set(exercise.keys())
        if not missing_fields:
            score += 6
        else:
            errors.append(f"Missing fields: {missing_fields}")

        # Check exercise type (3 pts)
        ex_type = exercise.get("type")
        if ex_type in self.valid_types:
            score += 3
        else:
            errors.append(f"Invalid type: {ex_type}")

        # Check type-specific fields (6 pts)
        if ex_type == "multiple_choice":
            mc_missing = self.mc_required_fields - set(exercise.keys())
            if not mc_missing:
                score += 6
            else:
                errors.append(f"MC missing: {mc_missing}")
        else:
            score += 6  # Not MC, no extra requirements

        return score, errors

    @property
    def max_score(self) -> float:
        return 15.0

    @property
    def name(self) -> str:
        return "json_validity"
