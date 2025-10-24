"""
CEFR level alignment scorer.

Validates exercise complexity matches target CEFR level (0-20 points).
"""

import json
from typing import Any, Dict, List

from .base_llm_scorer import BaseLLMScorer


class CEFRScorer(BaseLLMScorer):
    """
    Scores CEFR level alignment (0-20 points) using a batched LLM.
    Uses models configured in SCORER_MODEL_CONFIG (base_llm_scorer.py).
    """

    def __init__(self, llm_handler):
        super().__init__(llm_handler)

    def get_prompt(self, exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
        level = request.get("level", "A2").upper()
        grammar_focus = request.get("grammar_focus", "any")
        exercise_types = request.get("exercise_types", ["any"])

        exercises_json_string = json.dumps(
            [
                {
                    "id": i,
                    "type": ex.get("type"),
                    "question": ex.get("question"),
                    "correct_answer": ex.get("correct_answer"),
                    "options": ex.get("options"),
                }
                for i, ex in enumerate(exercises)
            ],
            indent=2,
        )

        return f"""
You are a strict and meticulous Italian language professor evaluating a batch of exercises for CEFR level appropriateness. Your primary job is to ensure the exercises are genuinely challenging for the target level and not just superficially correct.

**REQUEST DETAILS:**
- **Target CEFR Level:** {level}
- **Grammar Focus:** {grammar_focus}
- **Exercise Types:** {exercise_types}

Here is the batch of exercises:
{exercises_json_string}

For each exercise, evaluate its appropriateness for a **{level}** student.

**PRIMARY RULE: First, check if the exercise is fundamentally broken.** If the answer is already present in the question, or the exercise is nonsensical, it is **unacceptable** and must receive a score between 0 and 4, regardless of any other factor.

If the exercise is not broken, then evaluate it based on these criteria:
1.  **Grammatical & Task Complexity:** Is the grammar point and the task itself (e.g., filling a blank) genuinely challenging for a {level} student? Simple A1/A2 tasks are not acceptable for B1+ levels.
2.  **Vocabulary & Sentence Structure:** Is the vocabulary and sentence structure appropriate for the {level} level?

**STRICT SCORING SCALE (0-20) - BE HARSH:**
- **18-20:** Perfect. A challenging, well-formed exercise perfectly aligned with the {level} level.
- **10-17:** Acceptable but flawed. Might be slightly too simple or use slightly simple vocabulary.
- **5-9:** Poor. The exercise is significantly too simple for the target level (e.g., an A2-level task for a B2 request).
- **0-4: Unacceptable.** Assign a score in this range if ANY of the following are true:
    - The exercise is **fundamentally broken** (as per the PRIMARY RULE).
    - The exercise tests a grammar point **two or more levels below** the target.

Respond ONLY with a single JSON object with a "scores" key, containing a list of objects with "id", "score", and "issue".

"""

    @property
    def max_score(self) -> float:
        return 20.0

    @property
    def name(self) -> str:
        return "cefr_alignment"
