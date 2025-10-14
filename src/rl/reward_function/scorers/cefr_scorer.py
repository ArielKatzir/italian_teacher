"""
CEFR level alignment scorer.

Validates exercise complexity matches target CEFR level (0-20 points).
"""

from typing import Any, Dict, List, Tuple

import spacy

from ...rl_data import CEFR_RULES, get_vocabulary_by_cefr
from .base import BaseScorer


class CEFRScorer(BaseScorer):
    """
    Scores CEFR level alignment (0-20 points).

    Checks if exercise complexity matches the target CEFR level.

    Components:
    - Sentence length appropriate for level (8 pts)
    - Vocabulary complexity matches level (7 pts) - Uses 16,887-word vocabulary
    - Grammar complexity appropriate (5 pts)
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)
        self.cefr_rules = CEFR_RULES

        # Pre-load vocabulary for all levels (cached)
        print("Pre-loading CEFR vocabulary (16,887 words)...")
        self._vocab_cache = {}
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            self._vocab_cache[level] = get_vocabulary_by_cefr(level, cumulative=True)
        print(f"✅ Loaded vocabulary for all CEFR levels")

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score CEFR level alignment."""
        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 0.0, ["No Italian text found"]

        # Get target level
        level = request.get("level", "A2")
        level = level.upper() if level else "A2"

        if level not in self.cefr_rules:
            return 10.0, []  # Unknown level, give partial credit

        rules = self.cefr_rules[level]
        doc = self.nlp(text)

        score = 0.0

        # 1. Sentence length check (8 points)
        sentence_score = 8.0
        sentences = list(doc.sents)
        if sentences:
            avg_sent_length = sum(len(sent) for sent in sentences) / len(sentences)
            min_len, max_len = rules["sentence_length"]

            if min_len <= avg_sent_length <= max_len:
                # Perfect range
                sentence_score = 8.0
            elif avg_sent_length < min_len:
                # Too simple
                diff = min_len - avg_sent_length
                penalty = min(8.0, diff * 1.5)
                sentence_score = max(0, 8.0 - penalty)
                errors.append(
                    f"Sentences too short for {level} (avg: {avg_sent_length:.1f}, expected: {min_len}-{max_len})"
                )
            else:
                # Too complex
                diff = avg_sent_length - max_len
                penalty = min(8.0, diff * 1.0)
                sentence_score = max(0, 8.0 - penalty)
                errors.append(
                    f"Sentences too long for {level} (avg: {avg_sent_length:.1f}, expected: {min_len}-{max_len})"
                )

        score += sentence_score

        # 2. Vocabulary complexity (7 points) - Using comprehensive 16,887-word vocabulary
        vocab_score = 7.0
        words = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]

        if words:
            # Get cumulative vocabulary for level (from cache)
            level_vocab = self._vocab_cache.get(level, set())
            known_words = sum(1 for word in words if word in level_vocab)
            vocab_coverage = known_words / len(words) if words else 0

            # Expect 70-90% coverage for appropriate level
            if vocab_coverage >= 0.7:
                vocab_score = 7.0
            elif vocab_coverage >= 0.5:
                vocab_score = 4.0
                errors.append(
                    f"Vocabulary may be too advanced for {level} (coverage: {vocab_coverage:.0%})"
                )
            else:
                vocab_score = 0.0
                errors.append(
                    f"Vocabulary too advanced for {level} (coverage: {vocab_coverage:.0%})"
                )

        score += vocab_score

        # 3. Grammar complexity (5 points)
        grammar_score = 5.0
        verbs = [token for token in doc if token.pos_ == "VERB"]

        if verbs:
            # Check tense complexity
            tenses_used = set()
            for verb in verbs:
                tense = verb.morph.get("Tense")
                if tense:
                    tenses_used.update(tense)

            allowed_tenses = set(rules["tenses"])

            # Check if tenses are appropriate for level
            if not tenses_used:
                # No tenses detected, give partial credit
                grammar_score = 3.0
            elif tenses_used.issubset(allowed_tenses):
                # All tenses appropriate
                grammar_score = 5.0
            else:
                # Some tenses too advanced
                extra_tenses = tenses_used - allowed_tenses
                if len(extra_tenses) == 1:
                    grammar_score = 2.0
                    errors.append(
                        f"Grammar too complex for {level} (uses: {', '.join(extra_tenses)})"
                    )
                else:
                    grammar_score = 0.0
                    errors.append(
                        f"Grammar too complex for {level} (uses: {', '.join(extra_tenses)})"
                    )

        score += grammar_score

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
        return 20.0

    @property
    def name(self) -> str:
        return "cefr_alignment"
