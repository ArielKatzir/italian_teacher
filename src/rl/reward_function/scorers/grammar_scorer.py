"""
Grammar correctness scorer.

Validates exercise uses requested grammar focus (0-10 points).
"""

from typing import Any, Dict, List, Tuple

import spacy

from .base import BaseScorer


class GrammarScorer(BaseScorer):
    """
    Scores grammar correctness (0-10 points).

    Checks if exercise uses the requested grammar focus (e.g., past_tense, present_tense).
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)

        # Tense patterns for grammar checking
        self.tense_patterns = {
            "past_tense": {
                "verbs": ["Past"],  # spaCy tense tag
                "indicators": [
                    "ho ",
                    "hai ",
                    "ha ",
                    "abbiamo",
                    "avete",
                    "hanno",
                    "sono ",
                    "è ",
                    "era",
                    "erano",
                ],
            },
            "present_tense": {
                "verbs": ["Pres"],
                "indicators": ["o ", "i ", "a ", "iamo", "ate", "ano"],
            },
            "future_tense": {
                "verbs": ["Fut"],
                "indicators": ["rò", "rai", "rà", "remo", "rete", "ranno"],
            },
            "imperfect_tense": {
                "verbs": ["Imp"],
                "indicators": ["avo", "avi", "ava", "avamo", "avate", "avano"],
            },
        }

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score grammar correctness."""
        grammar_focus = request.get("grammar_focus")

        if not grammar_focus:
            return 10.0, []  # No grammar focus specified, assume correct

        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 0.0, ["No Italian text found"]

        # Parse with spaCy
        doc = self.nlp(text)

        # Check if grammar focus is present
        if grammar_focus in self.tense_patterns:
            # Tense checking
            score, tense_errors = self._check_tense(doc, grammar_focus, text)
            errors.extend(tense_errors)
            return score, errors
        else:
            # Other grammar (subjunctive, articles, etc.)
            # For now, assume correct if we can't validate
            # TODO: Add more grammar checks
            return 10.0, []

    def _check_tense(
        self, doc: spacy.tokens.Doc, grammar_focus: str, text: str
    ) -> Tuple[float, List[str]]:
        """Check if verbs match expected tense."""
        errors = []
        patterns = self.tense_patterns[grammar_focus]

        # Extract verb tenses from spaCy
        verbs = [token for token in doc if token.pos_ == "VERB"]
        if not verbs:
            return 0.0, ["No verbs found"]

        verb_tenses = [v.morph.get("Tense") for v in verbs]
        expected_tense = patterns["verbs"][0]

        # Check if at least one verb is in correct tense
        correct_tense_found = any(expected_tense in tense for tense in verb_tenses)

        # Also check indicators (auxiliary verbs, common patterns)
        indicators = patterns["indicators"]
        indicator_found = any(ind in text.lower() for ind in indicators)

        if correct_tense_found or indicator_found:
            score = 10.0
        else:
            score = 0.0
            errors.append(f"Expected {grammar_focus}, found tenses: {verb_tenses}")

        return score, errors

    def _extract_italian_text(self, exercise: Dict[str, Any]) -> str:
        """
        Extract Italian text from exercise for analysis.

        Filters out English text (like "Translate:" prompts) to focus on Italian only.
        """
        import re

        parts = []

        if "question" in exercise:
            question = exercise["question"]
            # Remove common English prompts
            question = re.sub(
                r"^(Translate|Fill in the blank|Choose the correct answer):\s*",
                "",
                question,
                flags=re.IGNORECASE,
            )
            # Only include if it contains Italian-looking text (has Italian articles/words)
            italian_indicators = [
                "il",
                "la",
                "le",
                "gli",
                "lo",
                "un",
                "una",
                "è",
                "sono",
                "di",
                "a",
                "per",
                "che",
            ]
            if any(indicator in question.lower() for indicator in italian_indicators):
                parts.append(question)

        if "answer" in exercise:
            parts.append(exercise["answer"])

        return " ".join(parts)

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "grammar_correctness"
