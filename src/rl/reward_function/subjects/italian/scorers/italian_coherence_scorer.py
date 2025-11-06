"""
Generic coherence scorer.

Validates if an exercise is logical and makes sense.
"""

from ....base import BaseLLMScorer


class CoherenceScorer(BaseLLMScorer):
    """
    Generic coherence scorer using LLM.

    This scorer REQUIRES a prompt_fn to be provided during initialization.
    The prompt function should be subject-specific.
    """

    def __init__(self, llm_handler, prompt_fn, **kwargs):
        """
        Initialize coherence scorer.

        Args:
            llm_handler: LLM API handler instance
            prompt_fn: Function that generates prompts for coherence evaluation
            **kwargs: Additional arguments passed to BaseLLMScorer
        """
        if prompt_fn is None:
            raise ValueError(
                "CoherenceScorer requires a prompt_fn. "
                "Provide a subject-specific prompt function for coherence evaluation."
            )
        super().__init__(llm_handler, prompt_fn=prompt_fn, **kwargs)

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "coherence"
