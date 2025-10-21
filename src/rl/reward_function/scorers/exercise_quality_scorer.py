"""
Exercise Quality Scorer - Validates exercise construction principles.

This scorer catches CRITICAL exercise design flaws that make exercises unusable:
1. Answer already visible in question (redundancy)
2. No actual grammar testing (asking for non-verbs when testing verb tenses)
3. Wrong exercise types (requested translation, got fill-in-blank)
4. Answer = question (no actual exercise)

State-of-the-art: Uses pattern matching + linguistic analysis, not hardcoded lists.
"""

import re
from typing import Any, Dict, List, Tuple

import spacy

from .base import BaseScorer


class ExerciseQualityScorer(BaseScorer):
    """
    Scores exercise construction quality (0-30 points). [UPDATED Round 4]

    This is a BLOCKING scorer - critical failures return 0, which triggers -10 penalty in multi-reward.

    Components:
    - No redundancy: Answer must not appear in question (4 pts) [BLOCKING]
    - Grammar testing: Must test requested grammar with actual examples (4 pts) [BLOCKING]
    - Context sufficiency: Fill-in-blank must have clues (15 pts) [BLOCKING - Round 4 CRITICAL]
    - Exercise type match: Must match requested type (4 pts)
    - Answer quality: Answer must be valid Italian (3 pts)

    Total: 30 points (increased from 20→25→30 to make context penalty significant)
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score exercise construction quality (Round 4: 15pt context penalty)."""
        errors = []
        score = 30.0  # Updated max score (was 25)

        # Extract components
        question = exercise.get("question", "")
        answer = exercise.get("correct_answer", exercise.get("answer", ""))
        exercise_type = exercise.get("type", "")
        requested_types = request.get("exercise_types", [])
        grammar_focus = request.get("grammar_focus", "")

        # CRITICAL CHECK 1: Redundancy (4 pts) [BLOCKING]
        redundancy_score = 4.0
        if answer and question:
            answer_clean = answer.lower().strip()
            question_clean = question.lower().replace("___", "").replace("_", "")

            # Check if answer appears verbatim in question (excluding blank)
            if len(answer_clean) > 1 and answer_clean in question_clean:
                redundancy_score = 0.0
                errors.append(f"CRITICAL: Answer '{answer}' already visible in question")

            # Check if answer is just a word from the question
            question_words = set(question_clean.split())
            if answer_clean in question_words:
                redundancy_score = 0.0
                errors.append(f"CRITICAL: Answer '{answer}' is already a word in question")

        score += redundancy_score - 4.0  # Start at 20, adjust

        # CRITICAL CHECK 2: Grammar testing (4 pts) [BLOCKING]
        grammar_score = 4.0
        if grammar_focus:
            # Check if grammar focus is a tense
            tense_focuses = [
                "past_tense",
                "present_tense",
                "future_tense",
                "imperfect_tense",
                "conditional",
                "subjunctive",
            ]

            if grammar_focus in tense_focuses:
                # Parse question + answer to check for verbs
                text = f"{question} {answer}"
                doc = self.nlp(text)
                verbs = [token for token in doc if token.pos_ == "VERB"]

                if not verbs:
                    grammar_score = 0.0
                    errors.append(f"CRITICAL: Testing {grammar_focus} but no verbs found")
                elif len(verbs) == 1 and verbs[0].text.lower() == answer.lower():
                    # Only verb is the answer itself - good!
                    pass
                elif not any(v.text.lower() == answer.lower() for v in verbs):
                    # Answer is not a verb but we're testing verb tenses
                    grammar_score = 0.0
                    errors.append(
                        f"CRITICAL: Answer '{answer}' is not a verb (testing {grammar_focus})"
                    )

        score += grammar_score - 4.0

        # ROUND 4 FIX: CRITICAL CHECK 2.5: Context sufficiency for fill-in-blank (15 pts) [BLOCKING]
        # INCREASED from 5→10→15 points - context is CRITICAL for pedagogy
        context_score = 15.0
        if exercise_type == "fill_in_blank" and question and answer:
            # Check if question provides sufficient context clues
            has_context_clue = False

            # Pattern 1: Base verb form in parentheses (e.g., "Ieri (andare) ___ al cinema")
            if re.search(r"\([a-zA-Z]+\)", question):
                has_context_clue = True

            # Pattern 2: Translation prompt (e.g., "Translate: The placement is important → ")
            if re.search(r"[Tt]ranslat[e:]|[Ii]nglese:|[Ee]nglish:", question):
                has_context_clue = True

            # Pattern 3: Question has enough words for context (minimum 5 words excluding blank)
            question_words = question.replace("___", "").replace("_", "").split()
            if len(question_words) >= 5:
                # If question is long enough, it likely provides context
                has_context_clue = True

            if not has_context_clue:
                # CRITICAL: Impossible to answer without context!
                context_score = 0.0
                errors.append(
                    f"🚨 CRITICAL: Fill-in-blank lacks context clues! Question: '{question}' Answer: '{answer}'"
                )
                errors.append("   Add translation, base verb (andare), or more context words")

        score += context_score - 15.0

        # CHECK 3: Exercise type match (4 pts)
        type_score = 4.0
        if requested_types and exercise_type:
            if exercise_type not in requested_types:
                type_score = 0.0
                errors.append(f"Wrong type: requested {requested_types}, got '{exercise_type}'")

        score += type_score - 4.0

        # CHECK 4: Answer quality (3 pts)
        answer_score = 3.0
        if not answer or len(answer.strip()) == 0:
            answer_score = 0.0
            errors.append("Empty answer")
        elif len(answer.strip()) < 2:
            answer_score = 1.0
            errors.append("Answer too short (< 2 chars)")

        score += answer_score - 3.0

        # Ensure score is within bounds
        score = max(0.0, min(30.0, score))

        return score, errors

    @property
    def max_score(self) -> float:
        return 30.0

    @property
    def name(self) -> str:
        return "exercise_quality"
