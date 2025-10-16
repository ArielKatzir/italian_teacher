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
        # Actions that require consciousness/agency
        self.animate_only_verbs = {
            "pensare",
            "sentire",
            "decidere",
            "scegliere",
            "volere",
            "desiderare",
            "parlare",
            "dire",
            "chiedere",
            "rispondere",
            "gridare",
            "sussurrare",
            "camminare",
            "correre",
            "saltare",
            "nuotare",
            "volare",
            "arrampicare",
            "mangiare",
            "bere",
            "dormire",
            "svegliarsi",
            "riposare",
            "sedersi",
            "amare",
            "odiare",
            "sperare",
            "temere",
            "credere",
            "dubitare",
            "vedere",
            "guardare",
            "ascoltare",
            "toccare",
            "annusare",
            "sentire",
            "lavorare",
            "studiare",
            "leggere",
            "scrivere",
            "giocare",
            "ballare",
        }

        # Nouns that are clearly inanimate
        self.inanimate_nouns = {
            # Body parts (when detached/abstract)
            "pelle",
            "capello",
            "unghia",
            # Furniture & objects
            "tavolo",
            "sedia",
            "scrivania",
            "letto",
            "armadio",
            "divano",
            "libro",
            "penna",
            "matita",
            "quaderno",
            "foglio",
            "carta",
            "porta",
            "finestra",
            "muro",
            "pavimento",
            "soffitto",
            "parete",
            # Buildings & structures
            "casa",
            "edificio",
            "palazzo",
            "chiesa",
            "scuola",
            "negozio",
            # Vehicles (when not personified)
            "automobile",
            "macchina",
            "bicicletta",
            "moto",
            "treno",
            "aereo",
            # Technology
            "computer",
            "telefono",
            "televisione",
            "radio",
            "orologio",
            # Abstract concepts
            "idea",
            "pensiero",
            "concetto",
            "problema",
            "soluzione",
        }

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
        if "correct_answer" in exercise and "question" in exercise:
            answer = exercise["correct_answer"].lower().strip()
            question = exercise["question"].lower()

            # If answer is already fully in the question (excluding blank), it's redundant
            if (
                answer
                and len(answer) > 1
                and answer in question.replace("___", "").replace("_", "")
            ):
                score = 0.0
                errors.append(f"Redundant: Answer '{answer}' already appears in question")
                return score, errors  # Critical error, return immediately

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

        # 3. Check for illogical animate verb + inanimate subject combinations
        illogical_found = False
        for token in doc:
            if token.pos_ == "VERB":
                verb_lemma = token.lemma_.lower()

                # Only check if it's an animate-only verb
                if verb_lemma in self.animate_only_verbs:
                    # Find the subject of this verb
                    subject = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubj:pass"]:
                            subject = child
                            break

                    if subject:
                        subj_lemma = subject.lemma_.lower()

                        # Check if inanimate noun is doing animate action
                        if subj_lemma in self.inanimate_nouns:
                            score = max(0, score - 7)  # Heavy penalty
                            errors.append(
                                f"Illogical: '{subject.text}' cannot '{token.text}' "
                                f"(inanimate object doing conscious action)"
                            )
                            illogical_found = True
                            break

        # 4. Optional: Use LLM for general coherence check (if enabled and not already failed)
        if self.use_llm and score >= 7.0 and not illogical_found:
            llm_score, llm_errors = self._check_coherence_with_llm(text, exercise, request)
            # LLM can only reduce score, not increase it
            if llm_score < score:
                score = llm_score
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

Consider:
- Does the sentence make logical sense?
- Are words used in appropriate contexts?
- Is the scenario realistic and natural?
- Would a native speaker find this strange?

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
