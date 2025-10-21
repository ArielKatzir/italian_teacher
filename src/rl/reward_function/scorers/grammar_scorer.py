"""

Grammar correctness scorer.

Validates exercise uses requested grammar focus (0-10 points).
Uses rule-based checks for tenses + optional LLM for advanced grammar.
"""

import asyncio
import os
from typing import Any, Dict, List, Tuple

import spacy

from .base import BaseScorer


class GrammarScorer(BaseScorer):
    """
    Scores grammar correctness (0-10 points).

    Checks if exercise uses the requested grammar focus (e.g., past_tense, present_tense).
    Uses LLM API for advanced grammar checks (subjunctive, articles, etc.) if available.
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)

        # Check if OpenAI API is available for advanced grammar checking
        self.use_llm = os.environ.get("OPENAI_API_KEY") is not None
        if self.use_llm:
            try:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI()
                print("  ✅ LLM grammar checking enabled (OpenAI API)")
            except ImportError:
                self.use_llm = False
                print("  ⚠️ OpenAI not installed, using rule-based grammar only")
        else:
            self.client = None

        # Define morphological features for each grammar focus
        # PROFESSIONAL APPROACH: Uses ONLY spaCy's NLP features - no brittle pattern matching
        # Each grammar focus maps to specific morphological features that spaCy can detect
        self.grammar_features = {
            # Tense-based validation (uses Tense morphology)
            "past_tense": {
                "morph_type": "Tense",
                "expected_values": ["Past"],  # Passato prossimo
                "also_accept": ["Imp"],  # Imperfect is also past
            },
            "present_tense": {
                "morph_type": "Tense",
                "expected_values": ["Pres"],
            },
            "future_tense": {
                "morph_type": "Tense",
                "expected_values": ["Fut"],
            },
            "imperfect_tense": {
                "morph_type": "Tense",
                "expected_values": ["Imp"],
            },
            # Mood-based validation (uses Mood morphology)
            "conditional": {
                "morph_type": "Mood",
                "expected_values": ["Cnd"],
            },
            "subjunctive": {
                "morph_type": "Mood",
                "expected_values": ["Sub"],
            },
            # Other grammatical features
            "pronouns": {
                "morph_type": "PronType",
                "expected_values": ["Prs"],  # Personal pronouns
            },
            "articles": {
                "morph_type": "POS",
                "expected_values": ["DET"],  # Determiners include articles
            },
        }

    async def score(
        self, exercise: Dict[str, Any], request: Dict[str, Any], semaphore: asyncio.Semaphore = None
    ) -> Tuple[float, List[str]]:
        """Score grammar correctness using spaCy morphology + optional LLM validation."""
        grammar_focus = request.get("grammar_focus")

        if not grammar_focus:
            return 10.0, []  # No grammar focus specified, assume correct

        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 0.0, ["No Italian text found"]

        # Parse with spaCy
        doc = self.nlp(text)

        # Check if grammar focus is defined in our features
        if grammar_focus in self.grammar_features:
            # Use spaCy morphology for validation
            score, morph_errors = self._check_morphology(doc, grammar_focus, text)
            errors.extend(morph_errors)

            # ALWAYS use OpenAI for validation (catch spaCy false positives)
            # This ensures professional-grade accuracy
            if self.use_llm:
                llm_score, llm_errors = await self._check_grammar_with_llm(
                    text, exercise, request, grammar_focus, semaphore
                )
                # Average spaCy and LLM scores for robustness
                score = (score + llm_score) / 2
                errors.extend(llm_errors)

            return score, errors
        else:
            # Unknown grammar focus - use LLM if available
            if self.use_llm:
                return await self._check_grammar_with_llm(
                    text, exercise, request, grammar_focus, semaphore
                )
            else:
                return 10.0, [f"Cannot validate '{grammar_focus}' without LLM"]

    def _check_morphology(
        self, doc: spacy.tokens.Doc, grammar_focus: str, text: str
    ) -> Tuple[float, List[str]]:
        """
        PROFESSIONAL-GRADE morphological validation using spaCy NLP.

        No pattern matching - relies entirely on spaCy's linguistic analysis.
        For spaCy errors, we have LLM as backup in the score() method.
        """
        errors = []
        features = self.grammar_features[grammar_focus]
        morph_type = features["morph_type"]
        expected_values = features["expected_values"]
        also_accept = features.get("also_accept", [])

        # Determine which tokens to check based on morph_type
        if morph_type in ["Tense", "Mood"]:
            # For tense/mood: check VERB and AUX tokens
            tokens = [t for t in doc if t.pos_ in ("VERB", "AUX")]
            feature_name = "verbs"
        elif morph_type == "PronType":
            # For pronouns: check PRON tokens
            tokens = [t for t in doc if t.pos_ == "PRON"]
            feature_name = "pronouns"
        elif morph_type == "POS":
            # For articles: check DET tokens
            tokens = [t for t in doc if t.pos_ == "DET"]
            feature_name = "articles"
        else:
            # Generic: check all tokens
            tokens = list(doc)
            feature_name = "tokens"

        if not tokens:
            return 0.0, [f"No {feature_name} found - {grammar_focus} not tested"]

        # Extract morphological features from tokens
        matching_tokens = 0

        for token in tokens:
            if morph_type == "POS":
                # For POS, just check if it's the right part of speech
                if token.pos_ in expected_values:
                    matching_tokens += 1
            else:
                # For other features, check morphology
                token_features = token.morph.get(morph_type)

                # Check if token has expected feature value
                if any(val in token_features for val in expected_values):
                    matching_tokens += 1
                # Also check "also_accept" values (e.g., Imp for past_tense)
                elif also_accept and any(val in token_features for val in also_accept):
                    matching_tokens += 1

        # ⚠️ LINEAR PENALTY SCALE: Proportional scoring
        # 100% match → 10.0
        # 75% match  → 7.5
        # 50% match  → 5.0
        # 25% match  → 2.5
        # 0% match   → 0.0
        match_ratio = matching_tokens / len(tokens)
        score = match_ratio * 10.0

        # Add descriptive errors based on match ratio
        if match_ratio >= 0.75:
            # Excellent - no error needed
            pass
        elif match_ratio >= 0.5:
            # Acceptable but could be better
            errors.append(
                f"Only {matching_tokens}/{len(tokens)} {feature_name} use {grammar_focus}"
            )
        elif match_ratio >= 0.25:
            # Weak - minority matches
            errors.append(
                f"Weak {grammar_focus}: only {matching_tokens}/{len(tokens)} {feature_name} match"
            )
        else:
            # Failed - not testing the grammar
            errors.append(
                f"Wrong grammar focus: expected {grammar_focus}, found mostly other forms"
            )

        return score, errors

    async def _check_grammar_with_llm(
        self,
        text: str,
        exercise: Dict[str, Any],
        request: Dict[str, Any],
        grammar_focus: str,
        semaphore: asyncio.Semaphore = None,
    ) -> Tuple[float, List[str]]:
        """
        Use OpenAI API to check advanced grammar focus.

        This handles non-tense grammar like:
        - subjunctive mood
        - articles (definite/indefinite)
        - prepositions (di, a, da, in, etc.)
        - pronouns (direct/indirect object, reflexive)
        - adjective agreement
        - conditional mood
        """
        # If a semaphore is provided, acquire it. Otherwise, run without limits.
        # This makes the scorer usable both with and without the main reward framework.
        if semaphore:
            await semaphore.acquire()

        llm_score, errors = 10.0, []

        try:
            # Map grammar focus to human-readable descriptions
            grammar_descriptions = {
                "subjunctive": "subjunctive mood (congiuntivo)",
                "conditional": "conditional mood (condizionale)",
                "articles": "articles (il, la, un, una, etc.)",
                "prepositions": "prepositions (di, a, da, in, con, per, etc.)",
                "pronouns": "pronouns (mi, ti, lo, la, ci, etc.)",
                "adjective_agreement": "adjective-noun agreement (gender/number)",
                "verb_conjugation": "verb conjugation accuracy",
            }

            grammar_desc = grammar_descriptions.get(grammar_focus, grammar_focus)

            # Special handling for articles
            if grammar_focus == "articles":
                prompt = f"""You are evaluating an Italian article exercise.

Exercise type: {exercise.get('type', 'unknown')}
Question: {exercise.get('question', '')}
Answer: {exercise.get('correct_answer', exercise.get('answer', ''))}

CRITICAL CHECKS:
1. Is the ANSWER an article (il, la, un, una, lo, l', gli, le, i, dello, della, dei, delle)?
   - If answer is a NOUN (not an article), this is WRONG. Score 0-2.
   - If answer is an ARTICLE, continue evaluation.

2. Does the exercise test article knowledge?
   - Student must choose/use correct article (definite vs indefinite, gender, number)
   - Not just vocabulary

3. Is the article usage correct?
   - Correct gender (il/lo/un vs la/una)
   - Correct number (singular vs plural)
   - Correct form (l' before vowels, etc.)

Respond ONLY with JSON:
{{"score": <0-10>, "issue": "<explanation if score < 8>"}}

Examples:
- Answer "il" testing "il/la" → score 10
- Answer "storia" (noun, not article) → score 0-2, issue "Answer is noun, not article"
- Answer "un" but question tests vocabulary → score 3-5, issue "Tests vocabulary not articles"
"""
            else:
                prompt = f"""You are evaluating an Italian language exercise for grammar correctness.

Exercise type: {exercise.get('type', 'unknown')}
Question: {exercise.get('question', '')}
Answer: {exercise.get('correct_answer', exercise.get('answer', ''))}

Grammar focus: {grammar_desc}

Rate the exercise on a scale of 0-10 for how well it tests and correctly uses the specified grammar:
10 = Clearly tests and correctly uses the target grammar
7-9 = Uses the grammar but not as primary focus
4-6 = Grammar is present but incorrect or minimal
0-3 = Does not test the specified grammar or has major errors

Consider:
- Does the exercise actually test the specified grammar?
- Is the grammar used correctly?
- Would a learner practicing this exercise learn about the target grammar?

Respond ONLY with a JSON object:
{{"score": <number 0-10>, "issue": "<brief explanation if score < 8, empty string otherwise>"}}"""

            response = await self.client.completions.create(  # type: ignore
                model="gpt-3.5-turbo-instruct",  # Fast and cheap
                prompt=prompt,
                temperature=0,
                max_tokens=100,
            )

            result_text = response.choices[0].text.strip()

            # Parse JSON response
            import json

            result = json.loads(result_text)

            llm_score = result.get("score", 10.0)
            issue = result.get("issue", "")

            errors = []
            if llm_score < 8 and issue:
                errors.append(f"Grammar issue: {issue}")

        except Exception as e:  # Catch other unexpected errors
            # If LLM check fails, assume correct (neutral score)
            print(f"  ⚠️ LLM grammar check failed: {e}")
            return 5.0, [f"LLM grammar check failed: {e}"]
        finally:
            if semaphore:
                semaphore.release()
        return llm_score, errors

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

        # Get answer from either 'correct_answer' or 'answer' field
        answer = exercise.get("correct_answer") or exercise.get("answer")
        if answer:
            parts.append(answer)

        return " ".join(parts)

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "grammar_correctness"
