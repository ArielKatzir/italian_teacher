"""
Shared text extraction and analysis utilities for scorers.

These functions are used across multiple scorers to avoid code duplication.
"""

from typing import Any, Dict

import spacy


def extract_italian_text(exercise: Dict[str, Any], nlp: spacy.language.Language) -> str:
    """
    Extract Italian text from exercise for analysis.

    For translation exercises, extracts the Italian answer.
    For other types, extracts the question and answer combined.

    Args:
        exercise: Exercise dictionary with question/answer
        nlp: spaCy language model (unused but kept for compatibility)

    Returns:
        Italian text string for analysis
    """
    ex_type = exercise.get("type", "")
    question = exercise.get("question", "")
    answer = exercise.get("correct_answer", "")

    # For translation, the question is English, answer is Italian
    if ex_type == "translation":
        return answer

    # For fill-in-blank and multiple choice, combine question + answer
    # Remove the blank placeholder for analysis
    question_cleaned = question.replace("___", "").replace("_", "")
    return f"{question_cleaned} {answer}".strip()


def is_fill_in_blank(text: str) -> bool:
    """
    Check if text contains fill-in-blank markers.

    Args:
        text: Text to check

    Returns:
        True if text contains ___ or _ markers
    """
    return "___" in text or " _ " in text


def is_exclamation_or_idiom(text: str, nlp: spacy.language.Language) -> bool:
    """
    Check if text is a valid exclamation or idiom that doesn't need a verb.

    Examples of valid verbless phrases:
    - "Che bello!" (How beautiful!)
    - "Buon giorno!" (Good day!)
    - "Basta!" (Enough!)

    Args:
        text: Text to check
        nlp: spaCy language model for analysis

    Returns:
        True if text is a valid exclamation/idiom
    """
    text_lower = text.lower().strip()

    # Common Italian exclamations without verbs
    exclamations = [
        "basta",
        "buongiorno",
        "buon giorno",
        "buonasera",
        "buona sera",
        "buonanotte",
        "buona notte",
        "ciao",
        "salve",
        "arrivederci",
        "grazie",
        "prego",
        "scusa",
        "scusi",
        "permesso",
        "bravo",
        "brava",
        "complimenti",
    ]

    # Check if text starts with common exclamation patterns
    if any(text_lower.startswith(exc) for exc in exclamations):
        return True

    # Check for "Che + adjective!" pattern (e.g., "Che bello!")
    if text_lower.startswith("che ") and "!" in text:
        return True

    # Check if text is very short (likely an exclamation)
    if len(text.split()) <= 2 and "!" in text:
        return True

    return False
