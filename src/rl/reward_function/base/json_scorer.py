"""
JSON validity scorer.

Validates exercise structure and format (0-15 points).
"""

from typing import Any, Dict, List, Set, Tuple

from .base_scorer import BaseScorer


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
        self.mc_required_fields: Set[str] = {"options"}  # Multiple choice needs options array

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Score JSON structure validity with STRICT type enforcement (Round 3 FIX).

        CRITICAL CHANGES:
        - Type mismatch: -30 points (MASSIVE PENALTY) - This affects overall score heavily!
        - Missing options in MC: -20 points - CRITICAL ERROR

        This ensures type mismatch drops total score from ~86 to ~56 - model MUST learn to match types!
        """
        score = 0.0
        errors = []

        # Check required fields (6 pts)
        missing_fields = self.required_fields - set(exercise.keys())
        if not missing_fields:
            score += 6
        else:
            errors.append(f"Missing fields: {missing_fields}")

        # Check exercise type matches request (CRITICAL - CAN GO NEGATIVE!)
        ex_type = exercise.get("type")
        requested_types = request.get("exercise_types", [])

        if ex_type in self.valid_types:
            # Check if type matches request
            if requested_types and ex_type not in requested_types:
                # Type mismatch - MASSIVE PENALTY (Round 3 FIX)
                errors.append(
                    f"🚨 CRITICAL TYPE MISMATCH: requested {requested_types}, got {ex_type}"
                )
                score -= 30  # NEGATIVE 30 POINTS - drops total from 86 to 56!
                # This makes type mismatch the MOST important error to fix
            else:
                score += 3  # Correct type
        else:
            errors.append(f"Invalid type: {ex_type}")
            score -= 20  # Unknown types also get big penalty

        # Check type-specific fields (6 pts, but can go negative!)
        if ex_type == "multiple_choice":
            mc_missing = self.mc_required_fields - set(exercise.keys())
            if not mc_missing:
                # Verify options is actually a list with exactly 4 items (professional standard)
                options = exercise.get("options")
                if options is None:
                    errors.append("🚨 CRITICAL: Multiple choice MUST have options array, got null")
                    score -= 20  # MASSIVE PENALTY
                elif not isinstance(options, list):
                    errors.append(f"🚨 CRITICAL: Options must be list, got {type(options)}")
                    score -= 15
                elif len(options) != 4:
                    errors.append(
                        f"🚨 CRITICAL: Multiple choice needs exactly 4 options, got {len(options)}"
                    )
                    score -= 10
                elif len(options) != len(set(options)):
                    errors.append("Options must be UNIQUE - no duplicates")
                    score += 3  # Partial credit
                else:
                    score += 6  # Perfect
            else:
                errors.append(f"MC missing required fields: {mc_missing}")
                score -= 15
        elif ex_type == "fill_in_blank" or ex_type == "translation":
            # Verify options is null (not an array)
            # Any value for 'options' is incorrect for these types. It should be null or not present.
            if exercise.get("options") is not None:
                errors.append(f"CRITICAL: {ex_type} should have options=null, not an array")
                score -= 10  # Bigger penalty
            else:
                score += 6  # Correct
        else:
            # Unknown type, still check format
            score += 6

        return score, errors

    def score_batch(self, exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Score batch-level JSON properties, specifically exercise type diversity.
        This is a separate check from individual exercise scoring.
        """
        batch_score = 0.0
        errors = []
        requested_types = request.get("exercise_types", [])
        
        if len(requested_types) > 1: # Only check diversity if multiple types were requested
            generated_types = {ex.get("type") for ex in exercises if ex.get("type")}
            
            # Calculate how many of the requested types were actually generated
            matched_types = generated_types.intersection(set(requested_types))
            
            if len(matched_types) < len(requested_types):
                # Penalize if not all requested types are present
                missing_types = set(requested_types) - matched_types
                errors.append(f"Batch JSON error: Missing requested exercise types: {list(missing_types)}")
                # Penalty scales with the number of missing types
                batch_score -= (len(missing_types) / len(requested_types)) * 10 # Max 10 points penalty for diversity

        return batch_score, errors

    @property
    def max_score(self) -> float: # Max score for JSON is 15, but batch penalty can reduce it.
        return 15.0 

    @property
    def name(self) -> str:
        return "json_validity"
