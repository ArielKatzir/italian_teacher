"""
Italian CEFR level alignment evaluation prompts.
Extracted from cefr_scorer.py for better organization and subject-specific customization.
"""

import json
from typing import Any, Dict, List


def get_cefr_prompt(exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
    """
    Generate prompt for evaluating CEFR level appropriateness for Italian exercises.

    Args:
        exercises: List of exercises to evaluate
        request: Request context with level, grammar_focus, exercise_types

    Returns:
        Formatted prompt string for LLM evaluation
    """
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
You are an EXTREMELY STRICT Italian language professor evaluating CEFR level appropriateness. Your job is to HARSHLY penalize exercises that don't match the target level. Be UNFORGIVING.

**REQUEST:**
- **Target CEFR Level:** {level}
- **Grammar Focus:** {grammar_focus}
- **Exercise Types:** {exercise_types}

**Exercises:**
{exercises_json_string}

**CRITICAL: PRIMARY VALIDATION RULES**
Check if the exercise is fundamentally broken FIRST:
1. **Answer Already Visible:** Is the correct_answer already present in the question? → Score: 0
2. **Nonsensical Exercise:** Is the exercise logically impossible or incoherent? → Score: 0
3. **Wrong Grammar Focus:** Does it test completely different grammar than requested? → Score: 0
4. **Non-standard Italian:** Does it use incorrect or non-existent Italian words? → Score: 0

**EVALUATION CRITERIA (for non-broken exercises):**

1. **Grammar Complexity for {level}:**
   - A1: Basic present tense, simple sentences, common verbs
   - A2: Past tenses (passato prossimo), simple future, basic pronouns
   - B1: Complex tenses (imperfetto, trapassato), conditionals, subjunctive introduction
   - B2: Full subjunctive mastery, passive voice, complex subordinate clauses
   - C1/C2: Nuanced tense usage, advanced rhetoric, idiomatic mastery

2. **Vocabulary Sophistication:**
   - Is the vocabulary genuinely challenging for {level}?
   - Common words (casa, mangiare, andare) are A1/A2, not B1+

3. **Task Complexity:**
   - Simple blanks with obvious answers are A1/A2
   - Challenging distractors and nuanced choices are B1+

**ULTRA-HARSH SCORING SCALE (0-30):**

- **27-30:** Perfect. Exceptionally challenging and perfectly calibrated for {level}. Rare.
- **20-26:** Excellent. Appropriately challenging for {level} with good complexity.
- **15-19:** Good. Acceptable for {level} but could be more challenging.
- **10-14:** Mediocre. Slightly below {level} complexity (e.g., uses some B1 grammar for B2 request).
- **5-9:** Poor. Significantly too simple (e.g., A2 task for B1+ level).
- **0-4: Unacceptable/Broken.** Use this range if:
    - Exercise is fundamentally broken (PRIMARY RULES violated)
    - Tests grammar **2+ levels below** target (e.g., A1 grammar for B2)
    - Uses vocabulary far below the target level

**IMPORTANT GUIDELINES:**
- Be RUTHLESS with level mismatches: even 1 level too simple = max 14 points
- Basic vocabulary at high levels (B2+) = automatic penalty
- Simple fill-in-blanks for B1+ students should score ≤10 unless exceptionally clever
- Default to LOWER scores. When in doubt, penalize.

Respond with JSON: {{"scores": [{{"id": 0, "score": X, "issue": "specific issue"}}]}}
"""
