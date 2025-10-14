"""
Fluency and naturalness scorer.

Validates natural language flow and construction (0-10 points).
"""

from collections import Counter
from typing import Any, Dict, List, Tuple

import spacy

from .base import BaseScorer


class FluencyScorer(BaseScorer):
    """
    Scores fluency and naturalness (0-10 points).

    Checks for:
    - Fragmented sentences (missing verbs)
    - Excessive word repetition
    - Very short or incomplete responses
    - Unnatural patterns (all caps, excessive punctuation)
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score fluency and naturalness."""
        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 0.0, ["No text found"]

        score = 10.0
        doc = self.nlp(text)

        # 1. Check for sentence fragments (no verbs)
        sentences = list(doc.sents)
        for sent in sentences:
            has_verb = any(token.pos_ == "VERB" for token in sent)
            if not has_verb and len(sent) > 3:  # Allow short phrases without verbs
                score -= 3
                errors.append(f"Sentence fragment (no verb): '{sent.text}'")
                break  # Only penalize once

        # 2. Check for excessive word repetition
        words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
        if words:
            word_counts = Counter(words)
            most_common = word_counts.most_common(1)[0]
            if most_common[1] >= 3 and len(words) < 20:  # Same word 3+ times in short text
                score -= 2
                errors.append(
                    f"Excessive repetition: '{most_common[0]}' appears {most_common[1]} times"
                )

        # 3. Check for very short responses (might be incomplete)
        if len(text.strip()) < 10:
            score -= 3
            errors.append("Text too short (possibly incomplete)")

        # 4. Check for unnatural patterns (all caps, excessive punctuation)
        if text.isupper() and len(text) > 5:
            score -= 2
            errors.append("All caps text (unnatural)")

        if text.count("!") > 3 or text.count("?") > 3:
            score -= 1
            errors.append("Excessive punctuation")

        score = max(0, score)
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
        return "fluency"
