"""
Math problem clarity evaluation prompts.
Placeholder for future math-specific implementation.
"""

import json
from typing import Any, Dict, List


def get_clarity_prompt(exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
    """
    Generate prompt for evaluating math problem clarity and understandability.

    Args:
        exercises: List of exercises to evaluate
        request: Request context

    Returns:
        Formatted prompt string for LLM evaluation
    """
    # TODO: Implement math-specific clarity evaluation prompt
    exercises_json = json.dumps([{"id": i, **ex} for i, ex in enumerate(exercises)], indent=2)

    return f"""
You are an expert math educator evaluating problem clarity and understandability.

Exercises to evaluate:
{exercises_json}

TODO: Add detailed math clarity evaluation criteria.

Respond with JSON: {{"scores": [{{"id": 0, "score": X, "issue": "description"}}]}}
"""
