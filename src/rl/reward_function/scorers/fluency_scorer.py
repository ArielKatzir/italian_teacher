"""
Fluency and naturalness scorer.

Validates natural language flow and construction.
Uses rule-based checks and an LLM for subtle naturalness issues.
"""

import json
from collections import Counter
from typing import Any, Dict, List, Tuple

import spacy

from .base_llm_scorer import BaseLLMScorer
from .text_utils import extract_italian_text, is_exclamation_or_idiom

class FluencyScorer(BaseLLMScorer):
    """
    Scores fluency and naturalness (0-10 points).

    Checks for:
    - Fragmented sentences (missing verbs)
    - Excessive word repetition
    - Very short or incomplete responses
    - Unnatural patterns (all caps, etc.)
    - LLM-based naturalness check for subtle issues
    """

    def __init__(self, nlp: spacy.language.Language, llm_handler, use_llm: bool = False, disabled: bool = False):
        # Always initialize the parent class
        super().__init__(llm_handler)
        self.nlp = nlp
        if use_llm and not disabled:
            print("  ✅ FluencyScorer: LLM checking is enabled.")
        
        self.use_llm = use_llm and not disabled
        self.disabled = disabled

    async def score(self, exercise: Dict[str, Any], request: Dict[str, Any], semaphore=None) -> Tuple[float, List[str]]:
        """Score fluency and naturalness."""
        if self.disabled:
            return 0.0, ["Fluency scorer is disabled."]

        errors = []
        text = extract_italian_text(exercise, self.nlp)

        if not text:
            return 0.0, ["No text found"]

        score = 10.0
        doc = self.nlp(text)

        # 1. Check for sentence fragments (no verbs) - with intelligent exceptions
        sentences = list(doc.sents)
        for sent in sentences:
            has_verb = any(token.pos_ == "VERB" for token in sent)
            # Skip penalization if:
            # - fill-in-blank (the blank IS the missing verb)
            is_fill_blank = "___" in exercise.get("question", "")
            # - valid exclamation/idiom
            # - very short (< 4 words)
            if not has_verb and len(sent) > 3:
                if not is_fill_blank and not is_exclamation_or_idiom(sent.text, self.nlp):
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
            # Use the batched scoring method from BaseLLMScorer
            llm_results = await self.score_batch([exercise], request, semaphore)
            llm_score, llm_errors = llm_results[0]

            # LLM can only reduce score for subtle issues
            if llm_score < score:
                score = llm_score
                errors.extend(llm_errors)

        return max(0, score), errors

    def get_prompt(self, exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
        """Generates the prompt for the LLM fluency check."""
        processed_exercises = []
        for i, ex in enumerate(exercises):
            completed_text = extract_italian_text(ex, self.nlp)
            processed_exercises.append({"id": i, "text": completed_text})

        exercises_json_string = json.dumps(processed_exercises, indent=2)

        return f"""You are evaluating Italian text for fluency and naturalness.

Here is a batch of texts to evaluate:
{exercises_json_string}

For each text, rate it on a scale of 0-10 for fluency and naturalness:
10 = Perfectly natural, something a native speaker would say
7-9 = Natural with minor awkwardness
4-6 = Understandable but unnatural or stilted
0-3 = Very awkward or robotic

Respond ONLY with a JSON object:
{{"score": <number 0-10>, "issue": "<brief explanation if score < 8, empty string otherwise>"}}"""

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "fluency"
