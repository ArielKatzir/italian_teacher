"""
Generic CEFR level alignment scorer.

Validates exercise complexity matches target CEFR level.
"""

from ....base import BaseLLMScorer


class CEFRScorer(BaseLLMScorer):
    """
    Generic CEFR level alignment scorer using LLM.

    This scorer REQUIRES a prompt_fn to be provided during initialization.
    The prompt function should be subject-specific (e.g., Italian, Math, etc.).
    """

    def __init__(self, llm_handler, prompt_fn, **kwargs):
        """
        Initialize CEFR scorer.

        Args:
            llm_handler: LLM API handler instance
            prompt_fn: Function that generates prompts for CEFR evaluation
            **kwargs: Additional arguments passed to BaseLLMScorer
        """
        if prompt_fn is None:
            raise ValueError(
                "CEFRScorer requires a prompt_fn. "
                "Provide a subject-specific prompt function for CEFR evaluation."
            )
        super().__init__(llm_handler, prompt_fn=prompt_fn, **kwargs)

    @property
    def max_score(self) -> float:
        return 30.0

    @property
    def name(self) -> str:
        return "cefr_alignment"
