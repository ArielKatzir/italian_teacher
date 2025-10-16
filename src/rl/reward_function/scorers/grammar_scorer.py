"""
Grammar correctness scorer.

Validates exercise uses requested grammar focus (0-10 points).
Uses rule-based checks for tenses + optional LLM for advanced grammar.
"""

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
                from openai import OpenAI

                self.client = OpenAI()
                print("  ✅ LLM grammar checking enabled (OpenAI API)")
            except ImportError:
                self.use_llm = False
                print("  ⚠️ OpenAI not installed, using rule-based grammar only")
        else:
            self.client = None

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
                "indicators": [
                    "rò",
                    "rai",
                    "rà",
                    "remo",
                    "rete",
                    "ranno",
                    "sarà",
                    "saranno",
                    "avrò",
                    "avrai",
                    "avrà",
                    "avremo",
                    "avrete",
                    "avranno",
                ],
            },
            "imperfect_tense": {
                "verbs": ["Imp"],
                "indicators": [
                    "avo",
                    "avi",
                    "ava",
                    "avamo",
                    "avate",
                    "avano",
                    "evo",
                    "evi",
                    "eva",
                    "evamo",
                    "evate",
                    "evano",
                    "ivo",
                    "ivi",
                    "iva",
                    "ivamo",
                    "ivate",
                    "ivano",
                ],
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
            # Tense checking (rule-based)
            score, tense_errors = self._check_tense(doc, grammar_focus, text)
            errors.extend(tense_errors)
            return score, errors
        else:
            # Other grammar (subjunctive, articles, prepositions, etc.)
            # Use LLM if available, otherwise assume correct
            if self.use_llm:
                llm_score, llm_errors = self._check_grammar_with_llm(
                    text, exercise, request, grammar_focus
                )
                return llm_score, llm_errors
            else:
                # Fall back to rule-based validation if no LLM
                return 10.0, [f"Cannot validate '{grammar_focus}' without LLM"]

    def _check_tense(
        self, doc: spacy.tokens.Doc, grammar_focus: str, text: str
    ) -> Tuple[float, List[str]]:
        """Check if verbs match expected tense - STRICT VERSION."""
        errors = []
        patterns = self.tense_patterns[grammar_focus]
        expected_tense = patterns["verbs"][0]

        # Extract all verbs from the text
        verbs = [token for token in doc if token.pos_ == "VERB"]

        if not verbs:
            # ⚠️ CRITICAL: If no verbs found, the grammar focus is NOT being tested
            return 0.0, [f"No verbs found - {grammar_focus} not tested"]

        # Get tenses of all verbs
        verb_tenses = [v.morph.get("Tense") for v in verbs]
        [v.text for v in verbs]

        # Count how many verbs match the expected tense
        matching_verbs = sum(1 for tense in verb_tenses if expected_tense in tense)

        # ⚠️ STRICT REQUIREMENT: At least 50% of verbs must be in target tense
        if len(verbs) > 0:
            match_ratio = matching_verbs / len(verbs)

            if match_ratio >= 0.5:
                score = 10.0  # Good - majority uses target grammar
            elif match_ratio >= 0.25:
                score = 5.0  # Partial - some usage
                errors.append(f"Only {matching_verbs}/{len(verbs)} verbs use {grammar_focus}")
            else:
                score = 0.0  # Fail - not testing the grammar
                errors.append(
                    f"Wrong grammar focus: found {verb_tenses}, expected {expected_tense}"
                )
        else:
            score = 0.0
            errors.append("No verbs found")

        return score, errors

    def _check_grammar_with_llm(
        self, text: str, exercise: Dict[str, Any], request: Dict[str, Any], grammar_focus: str
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

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            import json

            result = json.loads(result_text)

            llm_score = result.get("score", 10.0)
            issue = result.get("issue", "")

            errors = []
            if llm_score < 8 and issue:
                errors.append(f"Grammar issue: {issue}")

            return llm_score, errors

        except Exception as e:
            # If LLM check fails, assume correct (neutral score)
            print(f"  ⚠️ LLM grammar check failed: {e}")
            return 10.0, []

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
