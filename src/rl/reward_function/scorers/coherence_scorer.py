"""
Coherence scorer.
Validates if an exercise is logical and makes sense (0-10 points).
"""

import json
from typing import Any, Dict, List

from .base_llm_scorer import BaseLLMScorer


class CoherenceScorer(BaseLLMScorer):
    """
    Scores exercise coherence (0-10 points) using a batched LLM.
    Uses models configured in SCORER_MODEL_CONFIG (base_llm_scorer.py).
    """

    def __init__(self, llm_handler, **kwargs):
        super().__init__(llm_handler, **kwargs)

    def get_prompt(self, exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
        # For fill-in-the-blank, create the completed sentence for evaluation.
        processed_exercises = []
        for i, ex in enumerate(exercises):
            question = ex.get("question", "")
            answer = ex.get("correct_answer", "")
            if ex.get("type") == "fill_in_blank" and "___" in question and answer:
                completed_text = question.replace("___", answer, 1)
                processed_exercises.append({"id": i, "completed_exercise": completed_text})
            else:
                processed_exercises.append({"id": i, "completed_exercise": f"{question} {answer}".strip()})

        exercises_json_string = json.dumps(
            processed_exercises,
            indent=2,
        )

        return f"""
You are evaluating a batch of Italian exercises for coherence.

Here is the batch of exercises:
For each exercise, I have provided the 'completed_exercise' text. For fill-in-the-blank, this is the question with the answer inserted.
{exercises_json_string}

For each exercise, rate the coherence of the 'completed_exercise' on a scale of 0-10.
- 10: Perfectly logical and makes sense.
- 5: Makes sense, but is awkward or weak.
- 0: Nonsense, contradictory, or grammatically broken to the point of being incomprehensible.

Respond with a single JSON object with a "scores" key, containing a list of objects with "id", "score", and "issue".

"""

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "coherence"