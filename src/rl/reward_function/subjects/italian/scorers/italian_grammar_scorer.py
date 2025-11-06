"""
Italian-specific grammar scorer with rule-based Italian verb tense checks.
"""
import asyncio
from typing import Any, Dict, List, Tuple

from ....base import BaseLLMScorer


class ItalianGrammarScorer(BaseLLMScorer):
    """
    Italian grammar scorer with rule-based pre-checks for Italian verb tenses.

    Includes specific checks for:
    - imperfect_tense vs passato_remoto
    - verbi_riflessivi (reflexive verbs)
    - Other Italian-specific grammar patterns
    """

    def _rule_based_tense_check(self, exercise: Dict[str, Any], grammar_focus: str) -> Tuple[bool, str]:
        """
        Rule-based check for obvious Italian verb tense mismatches.

        Returns: (is_valid, error_message)
        """
        import re

        answer = exercise.get("correct_answer", "").lower()

        # Italian verb tense patterns
        passato_remoto_patterns = [
            r'\w+ii\b',  # sentii, capii
            r'\w+ì\b',   # sentì, capì
            r'\w+ò\b',   # andò, parlò
            r'\w+emmo\b', # facemmo
            r'\w+este\b', # faceste
            r'\w+ero\b',  # fecero (NOT "ero" which is imperfect)
        ]

        imperfect_patterns = [
            r'\bero\b', r'\bera\b', r'\berano\b', r'\beravamo\b', r'\beravate\b',  # essere imperfect
            r'\w+avo\b', r'\w+avi\b', r'\w+ava\b', r'\w+avamo\b', r'\w+avate\b', r'\w+avano\b',  # -are imperfect
            r'\w+evo\b', r'\w+evi\b', r'\w+eva\b', r'\w+evamo\b', r'\w+evate\b', r'\w+evano\b',  # -ere imperfect
            r'\w+ivo\b', r'\w+ivi\b', r'\w+iva\b', r'\w+ivamo\b', r'\w+ivate\b', r'\w+ivano\b',  # -ire imperfect
        ]

        # Check for imperfect_tense request with passato_remoto in answer
        if grammar_focus == "imperfect_tense":
            # Look for passato remoto forms
            for pattern in passato_remoto_patterns:
                if re.search(pattern, answer):
                    match = re.search(pattern, answer)
                    if match:
                        verb_form = match.group()
                        # Exception: "ero" is imperfect, not passato remoto
                        if verb_form not in ["ero", "era", "erano"]:
                            return False, f"Request is 'imperfect_tense' but answer contains passato remoto form '{verb_form}'"

            # Check if answer has ANY imperfect forms (if not, it's probably wrong)
            has_imperfect = any(re.search(pattern, answer) for pattern in imperfect_patterns)
            if not has_imperfect and len(answer) > 5:  # Only check if answer is substantial
                return False, "Request is 'imperfect_tense' but answer has no imperfect verb forms"

        # Check for passato_remoto request with imperfect in answer
        if grammar_focus == "passato_remoto":
            has_passato_remoto = any(re.search(pattern, answer) for pattern in passato_remoto_patterns)
            if not has_passato_remoto and len(answer) > 5:
                return False, "Request is 'passato_remoto' but answer has no passato remoto forms"

        # Check for verbi_riflessivi (reflexive verbs)
        if "riflessiv" in grammar_focus.lower():
            reflexive_pronouns = ["mi", "ti", "si", "ci", "vi"]
            has_reflexive = any(pronoun in answer.split() for pronoun in reflexive_pronouns)
            if not has_reflexive:
                return False, "Request is 'verbi_riflessivi' but answer has no reflexive pronoun (mi/ti/si/ci/vi)"

        return True, ""

    async def score_batch(
        self, exercises: List[Dict[str, Any]], request: Dict[str, Any], semaphore=None
    ) -> List[Tuple[float, List[str]]]:
        """
        Override to add Italian-specific rule-based pre-checks before LLM scoring.
        """
        grammar_focus = request.get("grammar_focus", "general")

        # Pre-check each exercise with Italian rule-based validation
        results = []
        exercises_to_llm_score = []
        exercise_indices = []

        for i, exercise in enumerate(exercises):
            is_valid, error_msg = self._rule_based_tense_check(exercise, grammar_focus)

            if not is_valid:
                # Rule-based check failed - give 0 points
                results.append((0.0, [f"RULE-BASED CHECK: {error_msg}"]))
            else:
                # Passed rule-based check - send to LLM for detailed scoring
                exercises_to_llm_score.append(exercise)
                exercise_indices.append(i)
                results.append(None)  # Placeholder

        # Get LLM scores for exercises that passed rule-based checks
        if exercises_to_llm_score:
            llm_results = await super().score_batch(exercises_to_llm_score, request, semaphore)

            # Fill in the LLM results
            for idx, llm_result in zip(exercise_indices, llm_results):
                results[idx] = llm_result

        return results

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "grammar_correctness"
