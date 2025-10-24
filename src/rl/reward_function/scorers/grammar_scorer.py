import asyncio
import json
from typing import Any, Dict, List, Tuple

from .base_llm_scorer import BaseLLMScorer


class GrammarScorer(BaseLLMScorer):
    """
    Scores grammar correctness (0-10 points) using a batched LLM.
    Uses models configured in SCORER_MODEL_CONFIG (base_llm_scorer.py).
    """

    def __init__(self, llm_handler, **kwargs):
        super().__init__(llm_handler, **kwargs)

    def get_prompt(self, exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
        grammar_focus = request.get("grammar_focus", "general")
        
        # Prepare the batch of exercises for the prompt
        exercises_json_string = json.dumps(
            [
                {
                    "id": i,
                    "type": ex.get("type"),
                    "question": ex.get("question"),
                    "correct_answer": ex.get("correct_answer"),
                }
                for i, ex in enumerate(exercises)
            ],
            indent=2,
        )

        return f"""
You are evaluating a batch of Italian language exercises for grammar correctness.

The required grammar focus for all exercises is: "{grammar_focus}"

Here is the batch of exercises to evaluate:
{exercises_json_string}

**PRIMARY RULE: First, check if the exercise is fundamentally broken.** If the answer is already present in the question, or if inserting the answer creates a grammatically nonsensical or impossible sentence, it is **unacceptable** and must receive a score between 0 and 4.

If the exercise is not broken, then evaluate it based on these criteria:
1.  **Grammar Match:** Does the exercise test the requested grammar focus: "{grammar_focus}"? Be specific (e.g., `passato_prossimo` is not the same as `trapassato_prossimo`).
2.  **Pedagogical Quality:** Is the exercise a meaningful and effective test? Is it too simple for the target CEFR level?

**STRICT SCORING SCALE (0-10) - BE HARSH AND CRITICAL:**
- **9-10:** Perfect. Clearly and correctly tests the requested grammar focus ("{grammar_focus}"). The exercise is a meaningful and effective test.
- **5-8:** Partially Correct. Tests the correct grammar, but the exercise is flawed, too simple, or not the primary focus.
- **0-4: Unacceptable.** Assign a score in this range if ANY of the following are true:
    - The exercise is **fundamentally broken** (as per the PRIMARY RULE).
    - The exercise **does not test the requested grammar focus**. (e.g., tests 'articles' when 'pronouns' was requested, or 'passato_prossimo' when 'trapassato_prossimo' was requested). **This is a score of 0.**
    - The complexity of the grammar test is **far below the requested CEFR level**.

Respond ONLY with a single JSON object containing a key "scores", which is a list of objects. Each object must have "id", "score", and "issue". The order must match the input IDs.

"""

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "grammar_correctness"
