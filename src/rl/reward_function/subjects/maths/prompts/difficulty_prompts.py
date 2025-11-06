"""
Math difficulty/grade-level evaluation prompts.
Placeholder for future math-specific implementation.
"""

import json
from typing import Any, Dict, List


def get_difficulty_prompt(exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
    """
    Generate prompt for evaluating math problem difficulty appropriateness.

    Args:
        exercises: List of exercises to evaluate
        request: Request context (with target grade level)

    Returns:
        Formatted prompt string for LLM evaluation
    """
    # TODO: Implement math-specific difficulty evaluation prompt
    target_level = request.get("level", "grade_5")
    exercises_json = json.dumps([{"id": i, **ex} for i, ex in enumerate(exercises)], indent=2)

    return f"""
You are an expert math educator evaluating problem difficulty for {target_level}.

Exercises to evaluate:
{exercises_json}

TODO: Add detailed math difficulty evaluation criteria.

Respond with JSON: {{"scores": [{{"id": 0, "score": X, "issue": "description"}}]}}
"""
