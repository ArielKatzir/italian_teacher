"""
Italian language fluency evaluation prompts.
Extracted from fluency_scorer.py for better organization and subject-specific customization.
"""

import json
from typing import Any, Dict, List


def get_fluency_prompt(exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
    """
    Generate prompt for evaluating Italian language fluency and natural flow.

    Args:
        exercises: List of exercises to evaluate
        request: Request context

    Returns:
        Formatted prompt string for LLM evaluation
    """
    # Process exercises for fluency evaluation
    processed_exercises = []
    for i, ex in enumerate(exercises):
        question = ex.get("question", "")
        answer = ex.get("correct_answer", "")
        ex_type = ex.get("type", "")

        # Create the completed text
        if ex_type == "fill_in_blank" and "___" in question and answer:
            completed_text = question.replace("___", answer, 1)
        else:
            completed_text = f"{question} {answer}".strip()

        processed_exercises.append({
            "id": i,
            "type": ex_type,
            "text": completed_text
        })

    exercises_json_string = json.dumps(processed_exercises, indent=2)

    return f"""
You are a STRICT evaluator of Italian language fluency and natural flow. Be CRITICAL and thorough.

Exercises to evaluate:
{exercises_json_string}

**EVALUATION CRITERIA - Assess each exercise for:**

1. **Natural Language Flow:**
   - Does the Italian text flow naturally?
   - Are there awkward pauses or unnatural rhythm?
   - Would a native speaker phrase it this way?

2. **Word Choice and Collocations:**
   - Are word combinations natural and idiomatic?
   - Are there better, more common ways to express the same idea?
   - Does vocabulary feel forced or unnatural?

3. **Sentence Construction:**
   - Is the sentence structure smooth and elegant?
   - Are there unnecessarily complex or convoluted constructions?
   - Does the syntax feel native or translated?

4. **Idiomatic Usage:**
   - Are idioms and expressions used correctly?
   - Does it sound like authentic Italian or a literal translation?

**STRICT SCORING SCALE (0-10):**

- **10:** Perfect. Completely natural, fluent Italian that a native speaker would use.
- **8-9:** Excellent. Very natural with only minor areas for improvement.
- **6-7:** Good. Generally fluent but with some awkwardness or unnatural phrasing.
- **4-5:** Mediocre. Comprehensible but clearly not natural Italian, possibly translated literally.
- **2-3:** Poor. Awkward, stilted, or clearly non-native phrasing throughout.
- **0-1: Unacceptable.** Completely unnatural, incomprehensible, or not Italian at all.

**IMPORTANT:**
- Be HARSH on unnatural phrasing: literal translations = max 5 points
- Penalize awkward collocations: unnatural word combinations = max 6 points
- Reward authentic, native-sounding Italian with high scores
- Default to LOWER scores when something feels "translated" rather than "written"

Respond ONLY with valid JSON: {{"scores": [{{"id": 0, "score": X, "issue": "specific fluency issue"}}]}}
"""
