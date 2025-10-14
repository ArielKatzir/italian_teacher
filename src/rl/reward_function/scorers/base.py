"""
Base scorer class for reward function components.

All individual scorers inherit from BaseScorer.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import spacy


class BaseScorer(ABC):
    """
    Base class for all reward function scoring components.

    Each scorer is responsible for evaluating one aspect of exercise quality.
    """

    def __init__(self, nlp: Optional[spacy.language.Language] = None):
        """
        Initialize base scorer.

        Args:
            nlp: spaCy Italian language model (optional, not all scorers need it)
        """
        self.nlp = nlp

    @abstractmethod
    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Score an exercise component.

        Args:
            exercise: Generated exercise dict with fields:
                - type: str (fill_in_blank, translation, multiple_choice)
                - question: str (Italian)
                - answer: str (Italian)
                - options: List[str] (for multiple_choice)
                - correct_option: int (for multiple_choice)
            request: Original request dict with fields:
                - level: str (A1, A2, B1, B2, C1, C2)
                - grammar_focus: str
                - topic: str
                - num_exercises: int

        Returns:
            Tuple of (score, errors):
                - score: float (0 to max_score)
                - errors: List[str] (specific issues found)
        """

    @property
    @abstractmethod
    def max_score(self) -> float:
        """
        Maximum possible score for this component.

        Returns:
            float: Maximum points this scorer can award
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Component name for identification.

        Returns:
            str: Scorer name (e.g., "linguistic_quality", "cefr_alignment")
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_score={self.max_score})"
