"""
Fluency and naturalness scorer.

Validates natural language flow and construction (0-10 points).
Uses rule-based checks + optional LLM for subtle naturalness issues.
"""

import os
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
    - Optional: LLM-based naturalness check for subtle issues
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)

        # Check if OpenAI API is available for advanced checking
        self.use_llm = os.environ.get("OPENAI_API_KEY") is not None
        if self.use_llm:
            try:
                from openai import OpenAI

                self.client = OpenAI()
                print("  ✅ LLM fluency checking enabled (OpenAI API)")
            except ImportError:
                self.use_llm = False
                print("  ⚠️ OpenAI not installed, using rule-based fluency only")
        else:
            self.client = None

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score fluency and naturalness."""
        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 0.0, ["No text found"]

        score = 10.0
        doc = self.nlp(text)

        # Check if this is a fill-in-blank exercise (has blanks)
        is_fill_in_blank = "___" in text or "_" * 3 in text

        # 1. Check for sentence fragments (no verbs) - skip for fill-in-blank exercises
        sentences = list(doc.sents)
        for sent in sentences:
            has_verb = any(token.pos_ == "VERB" for token in sent)
            # For fill-in-blank, the blank IS the missing verb/word, so don't penalize
            if not has_verb and len(sent) > 3 and not is_fill_in_blank:
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

        # 5. Optional: Use LLM for subtle naturalness issues (if score still high)
        if self.use_llm and score >= 7.0:
            llm_score, llm_errors = self._check_fluency_with_llm(text, exercise)
            # LLM can only reduce score for subtle issues
            if llm_score < score:
                score = llm_score
                errors.extend(llm_errors)

        score = max(0, score)
        return score, errors

    def _check_fluency_with_llm(
        self, text: str, exercise: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Use OpenAI API to check subtle naturalness issues.

        This catches issues like:
        - Awkward word order
        - Unnatural collocations
        - Context-inappropriate language register
        - Stilted or robotic phrasing
        """
        try:
            prompt = f"""You are evaluating an Italian language exercise for fluency and naturalness.

Exercise type: {exercise.get('type', 'unknown')}
Question: {exercise.get('question', '')}
Answer: {exercise.get('correct_answer', exercise.get('answer', ''))}

Rate the Italian text on a scale of 0-10 for fluency and naturalness:
10 = Perfectly natural, something a native speaker would say
7-9 = Natural with minor awkwardness
4-6 = Understandable but unnatural or stilted
0-3 = Very awkward or robotic

Consider:
- Would a native Italian speaker phrase it this way?
- Is the word order natural?
- Are collocations appropriate?
- Is the language register appropriate for the context?

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
                errors.append(f"Fluency issue: {issue}")

            return llm_score, errors

        except Exception as e:
            # If LLM check fails, don't penalize - return neutral score
            print(f"  ⚠️ LLM fluency check failed: {e}")
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
        return "fluency"
