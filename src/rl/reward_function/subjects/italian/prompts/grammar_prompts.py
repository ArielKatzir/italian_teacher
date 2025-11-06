"""
Italian grammar correctness evaluation prompts.
Extracted from grammar_scorer.py for better organization and subject-specific customization.
"""

import json
from typing import Any, Dict, List


def get_grammar_prompt(exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
    """
    Generate prompt for evaluating Italian grammar correctness.

    Args:
        exercises: List of exercises to evaluate
        request: Request context with grammar_focus

    Returns:
        Formatted prompt string for LLM evaluation
    """
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
You are a STRICT evaluator of Italian language exercises for grammar correctness. Be HARSH and CRITICAL.

**REQUESTED GRAMMAR FOCUS: "{grammar_focus}"**

Exercises to evaluate:
{exercises_json_string}

**🚨 CRITICAL: PRIMARY VALIDATION RULES - CHECK THESE FIRST! 🚨**

Before any other evaluation, check if the exercise is fundamentally broken:

1. **Answer Already Present:** Is the correct_answer already visible in the question? → AUTOMATIC Score: 0

2. **Wrong Grammar Category:** Does the answer test a completely different grammar point than "{grammar_focus}"?
   - Example: Request is "imperativo" but answer uses infinitive "Spegnere" → Score: 0
   - Example: Request is "imperfect_tense" but answer uses "sentii" (passato remoto) → Score: 0
   - Example: Request is "verbi_riflessivi" but answer has no reflexive pronoun → Score: 0
   → AUTOMATIC Score: 0

3. **Nonsensical Result:** Does inserting the answer create a grammatically impossible sentence? → Score: 0-2

**COMPREHENSIVE GRAMMAR EVALUATION CRITERIA:**

For "{grammar_focus}", verify the following with EXTREME PRECISION:

A. **Exact Grammar Match** (Most Critical - ANY MISMATCH = 0 POINTS):

   **TENSE/MOOD MATCHING - BE EXTREMELY STRICT:**

   **STEP 1: Identify all verbs in the correct_answer**
   **STEP 2: Check if those verbs match the requested grammar focus**

   - passato_prossimo (ho fatto, sono andato) ≠ passato_remoto (feci, andai) ≠ imperfect (facevo, andavo)
   - If request is "imperfect_tense", answer MUST use imperfect endings: -avo, -evo, -ivo (ero, facevo, andavo, sentivo)
   - If request is "passato_remoto", answer MUST use passato remoto forms (fu, fece, andò, sentì, sentii)
   - If request is "imperativo", answer MUST use imperative forms (va'!, fai!, spegni!, andate!)
   - If request is "past_tense", answer MUST use passato prossimo (ho fatto, sono andato)

   **⚠️ COMMON MISTAKES THAT GET 0 POINTS:**
   - Request: "imperfect_tense" but answer has "sentii" (this is passato remoto, NOT imperfect) → 0 points
   - Request: "imperativo" but answer has "Spegnere" (this is infinitive, NOT imperative) → 0 points
   - Request: "past_tense" but answer has "feci" (this is passato remoto, NOT passato prossimo) → 0 points

   **Wrong tense/mood = AUTOMATIC 0 POINTS**

   **SPECIFIC CHECKS:**
   - For verb tenses: Check the EXACT tense form, not just "past" vs "present"
   - For imperativo: Verify it's TRUE imperative mood, NOT infinitive (e.g., "Spegni!" ✓, "Spegnere!" ✗)
   - For reflexive verbs: Verify reflexive pronoun is present and correctly placed
   - For pronouns: Verify the answer tests pronoun usage, not articles or other elements
   - For conditionals: Verify conditional mood is used, not subjunctive or indicative

   **EXAMPLES OF 0-POINT ANSWERS:**
   - Request: "imperfect_tense", Answer uses "sentii" (passato remoto) → 0 points
   - Request: "imperativo", Answer uses "Spegnere!" (infinitive) → 0 points
   - Request: "verbi_riflessivi", Answer has no "si/mi/ti" → 0 points
   - Request: "passato_prossimo", Answer uses "andò" (passato remoto) → 0 points

B. **Answer Quality:**
   - Is the correct_answer actually correct Italian grammar?
   - Does it demonstrate mastery of the target grammar point?
   - Are there ANY conjugation errors, agreement mistakes, or incorrect forms?

C. **Pedagogical Effectiveness:**
   - Does the exercise meaningfully test the grammar (not just vocabulary)?
   - Is it sophisticated enough for the target level?
   - Are there clear distractors that test understanding (for multiple choice)?

**ULTRA-STRICT SCORING SCALE (0-10):**

- **10:** Perfect. Flawlessly tests "{grammar_focus}" with no errors. Pedagogically excellent.
- **9:** Excellent. Tests "{grammar_focus}" correctly with only minor stylistic issues.
- **7-8:** Good. Tests "{grammar_focus}" but has minor flaws (e.g., slightly unclear, or could be more challenging).
- **5-6:** Mediocre. Tests the correct grammar category but poorly executed or too simple.
- **3-4:** Poor. Barely tests the requested grammar OR has significant correctness issues.
- **0-2: Unacceptable/Broken.** Use this range if:
    - The exercise is fundamentally broken (PRIMARY RULES violated)
    - Tests WRONG grammar entirely (e.g., articles when pronouns requested)
    - The correct_answer is grammatically incorrect
    - Uses wrong verb mood/tense (e.g., infinitive instead of imperative)

**IMPORTANT:**
- Be MERCILESS with mismatched grammar categories: wrong tense/mood = 0 points
- Be HARSH on conjugation/agreement errors: any error = max 4 points
- Default to LOWER scores when in doubt

**🔍 FINAL CHECK BEFORE SCORING:**
For each exercise, ask yourself:
1. "Does the correct_answer actually test '{grammar_focus}'?"
2. "If the grammar focus is a tense (imperfect_tense, passato_remoto, etc.), do the verb forms match EXACTLY?"
3. "If I see verb endings like -ii, -ì, -ò (like sentii, sentì, andò), is passato_remoto requested?"
4. "If I see verb endings like -avo, -evo, -ivo (like facevo, ero, sentivo), is imperfect_tense requested?"

If the answer to questions 2-4 is NO, give a score of 0.

Respond with a JSON object: {{"scores": [{{"id": 0, "score": X, "issue": "description"}}]}}
"""
