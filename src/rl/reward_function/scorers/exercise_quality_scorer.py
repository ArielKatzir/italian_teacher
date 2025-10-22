import re
from typing import Any, Dict, List, Tuple

import spacy

from .base import BaseScorer


class ExerciseQualityScorer(BaseScorer):
    """
    Scores exercise construction quality (0-20 points).

    This is a BLOCKING scorer - critical failures result in a score of 0 for that component.

    Components:
    - Context sufficiency (Fill-in-blank): Must have explicit clues (10 pts) [BLOCKING]
    - Structure validity (Translation): Must have distinct English question and Italian answer (10 pts) [BLOCKING]
    - No redundancy: Answer must not appear in question (8 pts) [BLOCKING]
    - Answer quality: Answer must not be empty/too short (2 pts)

    Total: 20 points (re-weighted)
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score exercise construction quality."""
        errors = []
        score = 20.0  # New max score

        # Extract components
        question = exercise.get("question", "")
        answer = exercise.get("correct_answer", exercise.get("answer", ""))
        exercise_type = exercise.get("type", "")

        # CRITICAL CHECK 1: Context sufficiency for fill-in-blank (10 pts) [BLOCKING]
        # And single blank enforcement
        context_score = 10.0
        if exercise_type == "fill_in_blank" and question and answer:
            if question.count("___") != 1:
                errors.append(f"CRITICAL: Fill-in-blank must have exactly one blank ('___'), found {question.count('___')}.")
                context_score = 0.0

            has_context_clue = False
            
            # Check for a full sentence translation prompt first
            if re.search(r"[Tt]ranslat[e:]|[Ii]nglese:|[Ee]nglish:", question):
                has_context_clue = True
            else:
                # Check for a hint in parentheses, e.g., (andare) or (beautiful)
                hint_match = re.search(r"\((.*?)\)", question)
                if hint_match:
                    hint = hint_match.group(1).strip().lower()
                    
                    # Analyze the correct answer to determine its type
                    answer_doc = self.nlp(answer.strip())
                    if answer_doc and len(answer_doc) > 0:
                        answer_token = answer_doc[0]
                        
                        # If the answer is a verb, the hint must be its base form (lemma)
                        if answer_token.pos_ == "VERB":
                            if hint == answer_token.lemma_.lower():
                                has_context_clue = True
                            else:
                                errors.append(f"Hint validation failed: For verb '{answer}', hint '({hint})' is not the correct base form '{answer_token.lemma_}'.")
                        # If it's not a verb, we accept any hint as a potential translation
                        # We'll validate that the hint looks like an English word (only a-z characters)
                        else:
                            if hint.isalpha() and hint.isascii():
                                has_context_clue = True
                            else:
                                errors.append(f"Hint validation failed: For non-verb '{answer}', hint '({hint})' is not a valid translation (e.g., 'beautiful').")


            if not has_context_clue:
                context_score = 0.0
                if not errors: # Avoid duplicate error messages
                    errors.append(f"CRITICAL: Fill-in-blank lacks a valid hint (e.g., base verb in parentheses or a translation).")

        # CRITICAL CHECK 1B: Structure validity for translation (10 pts) [BLOCKING]
        elif exercise_type == "translation" and question and answer:
            is_valid_structure = True
            # Heuristic: Question should contain English-like words and not be identical to the answer
            question_words = set(re.findall(r'\b[a-zA-Z]+\b', question.lower()))
            answer_words = set(re.findall(r'\b[a-zA-Z]+\b', answer.lower()))
            
            # Check for "translation leakage" where answer is in the question
            if answer.lower().strip() in question.lower().strip():
                is_valid_structure = False
                errors.append("CRITICAL: Translation answer is present in the question.")

            # Check if question has a prompt
            if not re.search(r"[Tt]ranslat[e:]|[Ii]nglese:|[Ee]nglish:", question):
                 # This is a minor issue, not critical, but we can note it
                 pass # errors.append("Minor: Translation question lacks a 'Translate:' prompt.")

            # Check if answer contains Italian words (heuristic)
            italian_indicators = {"il", "la", "un", "una", "di", "a", "è", "sono", "ho", "ha"}
            if not any(word in italian_indicators for word in answer_words):
                is_valid_structure = False
                errors.append("CRITICAL: Translation answer does not appear to be valid Italian.")

            if not is_valid_structure:
                context_score = 0.0

        score += context_score - 10.0

        # CRITICAL CHECK 2: Redundancy (8 pts, re-weighted) [BLOCKING]
        redundancy_score = 8.0
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

        score += redundancy_score - 8.0

        # CHECK 4: Answer quality (2 pts)
        answer_score = 2.0
        if not answer or len(answer.strip()) == 0:
            answer_score = 0.0
            errors.append("Empty answer")
        elif len(answer.strip()) < 2:
            answer_score = 1.0
            errors.append("Answer too short (< 2 chars)")

        score += answer_score - 2.0

        # Ensure score is within bounds
        score = max(0.0, min(20.0, score))

        return score, errors

    @property
    def max_score(self) -> float:
        return 20.0

    @property
    def name(self) -> str:
        return "exercise_quality"
