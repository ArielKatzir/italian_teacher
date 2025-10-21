"""
CEFR level alignment scorer.

Validates exercise complexity matches target CEFR level (0-20 points).
"""

import os
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

        # Check if OpenAI API is available for advanced level validation (B2+)
        self.use_llm = os.environ.get("OPENAI_API_KEY") is not None
        if self.use_llm:
            try:
                from openai import OpenAI

                self.client = OpenAI()
                print("  ✅ OpenAI validation enabled for B2+ levels")
            except ImportError:
                self.use_llm = False
                self.client = None
        else:
            self.client = None

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

        # Check if this is a fill-in-blank exercise (has blanks)
        is_fill_in_blank = "___" in text or "_" * 3 in text

        # 1. Sentence length check (8 points)
        sentence_score = 8.0
        sentences = list(doc.sents)
        if sentences:
            avg_sent_length = sum(len(sent) for sent in sentences) / len(sentences)
            min_len, max_len = rules["sentence_length"]

            # For fill-in-blank exercises, be more lenient with short sentences
            if is_fill_in_blank:
                # Reduce minimum length by 3 words for fill-in-blank
                min_len = max(3, min_len - 3)

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
        # Use lemmas instead of word forms (e.g., "bambini" → "bambino", "mangiato" → "mangiare")
        words = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

        if words:
            # Get cumulative vocabulary for level (from cache)
            level_vocab = self._vocab_cache.get(level, set())
            known_words = sum(1 for word in words if word in level_vocab)
            vocab_coverage = known_words / len(words) if words else 0

            # STRICTER vocabulary expectations - encourage better level-appropriate vocab
            if vocab_coverage >= 0.85:
                vocab_score = 7.0  # Excellent - highly appropriate
            elif vocab_coverage >= 0.7:
                vocab_score = 5.0  # Good - mostly appropriate
                errors.append(
                    f"Some advanced vocabulary for {level} (coverage: {vocab_coverage:.0%})"
                )
            elif vocab_coverage >= 0.5:
                vocab_score = 2.0  # Moderate - too advanced
                errors.append(
                    f"Vocabulary too advanced for {level} (coverage: {vocab_coverage:.0%})"
                )
            else:
                vocab_score = 0.0  # Poor - very advanced
                errors.append(
                    f"Vocabulary much too advanced for {level} (coverage: {vocab_coverage:.0%})"
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

        # For B2+ levels, use OpenAI validation to catch false positives
        # Advanced levels are harder to validate with rules alone
        if self.use_llm and level in ["B2", "C1", "C2"]:
            llm_score = self._validate_with_openai(text, level, exercise)
            # Average rule-based and LLM scores for robustness
            score = (score + llm_score) / 2

        return score, errors

    def _validate_with_openai(self, text: str, level: str, exercise: Dict[str, Any]) -> float:
        """
        Use OpenAI to validate CEFR level appropriateness for advanced levels (B2+).

        Returns:
            Score out of 20 points
        """
        try:
            prompt = f"""You are an expert Italian language teacher evaluating CEFR level appropriateness.

Exercise text: "{text}"
Target level: {level}

Evaluate if this exercise is appropriate for {level} level students in terms of:
1. Vocabulary complexity (is it too simple or too advanced?)
2. Grammar structures (appropriate for this level?)
3. Sentence complexity

Respond with a JSON object:
{{
    "score": <number 0-20>,
    "reasoning": "<brief explanation>"
}}

CEFR Guidelines:
- B2: Intermediate-advanced. Can understand complex texts, express ideas fluently.
- C1: Advanced. Can understand a wide range of demanding texts, express ideas fluently.
- C2: Proficient. Can understand virtually everything, express self spontaneously and precisely.
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=150,
            )

            import json

            result = json.loads(response.choices[0].message.content)
            return float(result.get("score", 10.0))

        except Exception:
            # If OpenAI fails, return neutral score
            return 10.0

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
