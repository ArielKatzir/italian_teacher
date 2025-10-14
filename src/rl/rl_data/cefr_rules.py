"""
CEFR level complexity rules.

Defines expected ranges and characteristics for each CEFR level (A1-C2).
"""

from typing import Any, Dict

CEFR_RULES: Dict[str, Dict[str, Any]] = {
    "A1": {
        "sentence_length": (3, 10),  # words per sentence
        "word_complexity": (1, 3),  # syllables per word
        "tenses": ["Pres"],  # Allowed tenses
        "clause_depth": 1,  # Max subordinate clauses
        "vocabulary_level": "basic",
    },
    "A2": {
        "sentence_length": (5, 14),
        "word_complexity": (1, 4),
        "tenses": ["Pres", "Past", "Fut"],
        "clause_depth": 1,
        "vocabulary_level": "basic",
    },
    "B1": {
        "sentence_length": (8, 18),
        "word_complexity": (2, 5),
        "tenses": ["Pres", "Past", "Fut", "Imp"],
        "clause_depth": 2,
        "vocabulary_level": "intermediate",
    },
    "B2": {
        "sentence_length": (10, 22),
        "word_complexity": (2, 6),
        "tenses": ["Pres", "Past", "Fut", "Imp", "Cnd"],  # Conditional
        "clause_depth": 3,
        "vocabulary_level": "intermediate",
    },
    "C1": {
        "sentence_length": (12, 28),
        "word_complexity": (3, 7),
        "tenses": ["Pres", "Past", "Fut", "Imp", "Cnd", "Sub"],  # Subjunctive
        "clause_depth": 4,
        "vocabulary_level": "advanced",
    },
    "C2": {
        "sentence_length": (15, 35),
        "word_complexity": (3, 8),
        "tenses": ["Pres", "Past", "Fut", "Imp", "Cnd", "Sub"],
        "clause_depth": 5,
        "vocabulary_level": "advanced",
    },
}
