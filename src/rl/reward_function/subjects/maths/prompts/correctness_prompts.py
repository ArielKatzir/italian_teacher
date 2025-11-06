"""
Math correctness evaluation prompts.
Placeholder for future math-specific implementation.
"""

import json
from typing import Any, Dict, List


def get_correctness_prompt(exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
    """
    Generate prompt for evaluating math problem correctness.

    Args:
        exercises: List of exercises to evaluate
        request: Request context

    Returns:
        Formatted prompt string for LLM evaluation
    """
    # TODO: Implement math-specific correctness evaluation prompt
    exercises_json = json.dumps([{"id": i, **ex} for i, ex in enumerate(exercises)], indent=2)

    return f"""
You are an expert math educator evaluating the correctness of math problems.

Exercises to evaluate:
{exercises_json}

TODO: Add detailed math correctness evaluation criteria.

Respond with JSON: {{"scores": [{{"id": 0, "score": X, "issue": "description"}}]}}
"""
