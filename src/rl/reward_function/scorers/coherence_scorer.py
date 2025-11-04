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
You are a STRICT evaluator of Italian exercise coherence and naturalness. Be CRITICAL and thorough.

Exercises to evaluate (with answers inserted for fill-in-blanks):
{exercises_json_string}

**EVALUATION CRITERIA - Assess each 'completed_exercise' for:**

1. **Grammatical Completeness:**
   - Is it a complete, well-formed Italian sentence?
   - Are there missing words, articles, or prepositions?
   - Does subject-verb-object structure make sense?

2. **Semantic Coherence:**
   - Does the sentence have a clear, logical meaning?
   - Are the words used correctly in context?
   - Does it express a plausible real-world scenario or concept?

3. **Natural Italian Phrasing:**
   - Does it sound like something a native speaker would say?
   - Are word order and collocations correct?
   - Are there awkward constructions or unnatural translations?

4. **Internal Consistency:**
   - Do all parts of the sentence fit together?
   - Are there contradictions or nonsensical combinations?
   - Do adjectives/articles match their nouns appropriately?

**SPECIFIC RED FLAGS (auto-penalize):**
- Incomplete sentences missing verbs or subjects
- Word order that's clearly wrong (e.g., "velocemente molto corre")
- Semantic nonsense (e.g., "Il tavolo ha mangiato la casa")
- Awkward literal translations from English that don't work in Italian
- Incorrect use of idiomatic expressions
- Preposition misuse that changes meaning drastically

**STRICT SCORING SCALE (0-10):**

- **10:** Perfect. Grammatically complete, semantically coherent, natural-sounding Italian. Could appear in a native text.
- **8-9:** Excellent. Minor stylistic awkwardness but fully coherent and correct.
- **6-7:** Good. Understandable and mostly correct, but somewhat awkward or unnatural phrasing.
- **4-5:** Mediocre. Makes sense but has noticeable issues (missing words, awkward structure, unnatural phrasing).
- **2-3:** Poor. Barely comprehensible, major structural or semantic problems.
- **0-1: Unacceptable.** Nonsensical, grammatically broken, contradictory, or incomprehensible.

**IMPORTANT:**
- Be HARSH on incomplete sentences: missing articles/prepositions = max 5 points
- Be CRITICAL of awkward phrasing: unnatural Italian = max 7 points
- Penalize semantic nonsense heavily: contradiction/impossibility = 0-2 points
- Default to LOWER scores when something feels "off"

Respond ONLY with valid JSON: {{"scores": [{{"id": 0, "score": X, "issue": "specific description of problem"}}]}}
"""

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "coherence"