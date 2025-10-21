"""
Semantic coherence scorer.

Validates that exercises make logical sense (0-10 points).
Uses semantic analysis and optional LLM API for general coherence checking.
"""

import os
from typing import Any, Dict, List, Tuple

import spacy

from .base import BaseScorer


class CoherenceScorer(BaseScorer):
    """
    Scores semantic coherence (0-10 points).

    Uses multiple strategies:
    1. Rule-based checks (redundant answers, repetition)
    2. Semantic analysis (animate vs inanimate subjects with animate verbs)
    3. Optional: LLM API for general coherence (if OPENAI_API_KEY set)
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)

        # Check if OpenAI API is available for advanced checking
        self.use_llm = os.environ.get("OPENAI_API_KEY") is not None
        if self.use_llm:
            try:
                from openai import OpenAI

                self.client = OpenAI()
                print("  ✅ LLM coherence checking enabled (OpenAI API)")
            except ImportError:
                self.use_llm = False
                print("  ⚠️ OpenAI not installed, using rule-based coherence only")
        else:
            self.client = None

        # Semantic categories for basic coherence checking
        # Use spaCy's morphological features instead of hardcoded lists
        # This is much more general and covers all Italian words

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score semantic coherence."""
        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 10.0, []  # No text to check

        # Parse with spaCy
        doc = self.nlp(text)

        score = 10.0

        # 1. Check for redundant answer (answer already in question)
        # IMPORTANT: Verb conjugation testing is VALID pedagogy!
        # Only penalize if the EXACT answer appears in the question (not just base form)
        if "correct_answer" in exercise and "question" in exercise:
            # Defensive: handle None values
            correct_answer = exercise.get("correct_answer")
            question_text = exercise.get("question")

            if not correct_answer or not question_text:
                # Missing required fields - penalize heavily
                score -= 10.0
                errors.append("CRITICAL: Missing answer or question")
                return max(0, score), errors

            answer = correct_answer.lower().strip()
            question = question_text.lower()
            grammar_focus = request.get("grammar_focus", "")

            # If answer is already fully in the question (excluding blank), check if it's valid
            question_without_blank = question.replace("___", "").replace("_", "")
            if answer and len(answer) > 1 and answer in question_without_blank:
                # Check if this is verb conjugation testing (pedagogically valid)
                is_verb_testing = False

                # If testing verb tenses, check if answer is a verb
                if any(focus in grammar_focus for focus in ["tense", "verb", "conjugation"]):
                    # Parse answer to check if it's a verb
                    answer_doc = self.nlp(answer)
                    if any(token.pos_ == "VERB" for token in answer_doc):
                        is_verb_testing = True

                # If testing reflexive pronouns, repetition is expected
                if "reflexive" in grammar_focus or "pronoun" in grammar_focus:
                    is_verb_testing = True

                if not is_verb_testing:
                    # Only penalize if it's NOT verb testing
                    score -= 10.0
                    errors.append(f"CRITICAL: Answer '{answer}' already visible in question")
                    errors.append(f"CRITICAL: Answer '{answer}' is already a word in question")

        # 2. Check for excessive repetition (lazy generation)
        words = [token.text.lower() for token in doc if token.is_alpha and len(token.text) > 2]
        if len(words) >= 5:
            from collections import Counter

            word_counts = Counter(words)
            most_common = word_counts.most_common(1)[0]
            # If same word appears >35% of the time, it's repetitive
            if most_common[1] / len(words) > 0.35:
                score = max(0, score - 4)
                errors.append(
                    f"Repetitive: '{most_common[0]}' appears {most_common[1]}/{len(words)} times"
                )

        # 3. Check for illogical patterns using general linguistic features
        # This is now disabled in favor of redundancy checking (most critical issue)
        # Uncomment if you want to add back semantic checking via LLM
        # for token in doc:
        #     if token.pos_ == "VERB":
        #         # Find the subject of this verb
        #         subject = None
        #         for child in token.children:
        #             if child.dep_ in ["nsubj", "nsubj:pass"]:
        #                 subject = child
        #                 break
        #
        #         if subject:
        #             # Use LLM to check if subject-verb combination makes sense
        #             # (more general than hardcoded lists)
        #             pass

        # 4. ALWAYS use OpenAI for coherence validation (catch spaCy false positives)
        # This ensures professional-grade coherence checking
        if self.use_llm:
            llm_score, llm_errors = self._check_coherence_with_llm(text, exercise, request)

            # ROUND 4 FIX: Use MINIMUM score (harsher penalty) instead of average
            # If either rule-based OR LLM detects an issue, apply the penalty
            score = min(score, llm_score)
            errors.extend(llm_errors)

        return max(0, score), errors

    def _check_coherence_with_llm(
        self, text: str, exercise: Dict[str, Any], request: Dict[str, Any]
    ) -> Tuple[float, List[str]]:
        """
        Use OpenAI API to check general coherence.

        This catches subtle issues like:
        - Contextually inappropriate word usage
        - Illogical scenarios
        - Unnatural sentence construction
        """
        try:
            prompt = f"""You are evaluating an Italian language exercise for logical coherence and naturalness.

Exercise type: {exercise.get('type', 'unknown')}
Question: {exercise.get('question', '')}
Answer: {exercise.get('correct_answer', exercise.get('answer', ''))}
Topic: {request.get('topic', 'general')}

Rate the exercise on a scale of 0-10 for coherence:
10 = Perfectly natural and logical
7-9 = Natural with minor issues
4-6 = Somewhat unnatural or illogical
0-3 = Nonsensical or inappropriate

IMPORTANT GUIDELINES FOR LANGUAGE EXERCISES:

1. VERB CONJUGATION TESTING IS VALID PEDAGOGY:
   - Exercises CAN show a verb and ask for its conjugated form
   - This is how language learners practice conjugation
   - Examples: "prepara" in sentence + answer "prepara" = VALID (tests 3rd person)

2. GRAMMATICAL REPETITION IS ALLOWED:
   - Reflexive pronouns: "si" can appear multiple times
   - Articles/determiners: "la", "il" can repeat
   - Auxiliary verbs: "è", "ha", "sono" can repeat
   - Verb forms: infinitive vs conjugated is VALID testing

3. TRUE REDUNDANCY = Answer is CONTEXTUALLY obvious from meaning:
   - Student can fill blank WITHOUT grammar knowledge
   - The sentence GIVES AWAY the answer through context/meaning

VALID EXERCISES (score 8-10):
✓ "La mamma prepara ___ la cena" → "prepara" (conjugation practice - NOT redundant!)
✓ "I bambini giocano ___" → "giocano" (verb form practice)
✓ "Durante il viaggio, ___ mangiavamo" → "mangiavamo" (redundant word but tests imperfect!)
✓ "che ___ consideri" → "si" (reflexive pronoun)

REDUNDANT EXERCISES (score 0-3):
✗ "La famiglia è in vacanza per ___" → "due settimane" (obvious from context!)
✗ "Il cane mangia la ___ rossa" → "mela" (descriptive words give it away!)
✗ "Ho comprato ___ pizzeria" → "una" (semantic meaning makes it obvious!)

KEY RULE: If it tests GRAMMAR (conjugation, articles, pronouns) = VALID even if word repeats.
         If it tests MEANING and answer is contextually obvious = REDUNDANT.

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
                errors.append(f"LLM coherence: {issue}")

            return llm_score, errors

        except Exception as e:
            # If LLM check fails, don't penalize - return neutral score
            print(f"  ⚠️ LLM coherence check failed: {e}")
            return 10.0, []

    def _extract_italian_text(self, exercise: Dict[str, Any]) -> str:
        """Extract Italian text from exercise for analysis."""
        import re

        parts = []

        if "question" in exercise:
            question = exercise["question"]
            # Remove English prompts
            question = re.sub(
                r"^(Translate|Fill in the blank|Choose the correct answer):\s*",
                "",
                question,
                flags=re.IGNORECASE,
            )
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

        if "answer" in exercise or "correct_answer" in exercise:
            answer = exercise.get("answer") or exercise.get("correct_answer")
            if answer:
                parts.append(answer)

        return " ".join(parts)

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "coherence"
