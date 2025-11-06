"""
Scorer registry for managing subject-specific scorer configurations.
"""

from typing import Any, Callable, Dict, List, Optional


class ScorerConfig:
    """Configuration for a single scorer."""

    def __init__(
        self,
        name: str,
        scorer_class: Any,
        max_score: float,
        prompt_fn: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize scorer configuration.

        Args:
            name: Scorer name (e.g., "grammar", "cefr")
            scorer_class: The scorer class to instantiate
            max_score: Maximum score this scorer can give
            prompt_fn: Optional function to generate prompts (for LLM scorers)
            **kwargs: Additional arguments to pass to scorer constructor
        """
        self.name = name
        self.scorer_class = scorer_class
        self.max_score = max_score
        self.prompt_fn = prompt_fn
        self.kwargs = kwargs


class ScorerRegistry:
    """
    Registry for managing subject-specific scorer configurations.

    This allows different subjects (Italian, Math, etc.) to configure
    which scorers to use and how to customize their behavior.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._scorers: Dict[str, ScorerConfig] = {}

    def register(
        self,
        name: str,
        scorer_class: Any,
        max_score: float,
        prompt_fn: Optional[Callable] = None,
        **kwargs
    ) -> None:
        """
        Register a scorer configuration.

        Args:
            name: Scorer name (e.g., "grammar", "cefr")
            scorer_class: The scorer class to instantiate
            max_score: Maximum score this scorer can give
            prompt_fn: Optional function to generate prompts (for LLM scorers)
            **kwargs: Additional arguments to pass to scorer constructor
        """
        config = ScorerConfig(
            name=name,
            scorer_class=scorer_class,
            max_score=max_score,
            prompt_fn=prompt_fn,
            **kwargs
        )
        self._scorers[name] = config

    def get(self, name: str) -> Optional[ScorerConfig]:
        """Get a scorer configuration by name."""
        return self._scorers.get(name)

    def get_all(self) -> Dict[str, ScorerConfig]:
        """Get all registered scorer configurations."""
        return self._scorers.copy()

    def get_names(self) -> List[str]:
        """Get list of all registered scorer names."""
        return list(self._scorers.keys())

    def remove(self, name: str) -> None:
        """Remove a scorer from the registry."""
        if name in self._scorers:
            del self._scorers[name]

    def clear(self) -> None:
        """Clear all registered scorers."""
        self._scorers.clear()

    def __contains__(self, name: str) -> bool:
        """Check if a scorer is registered."""
        return name in self._scorers

    def __len__(self) -> int:
        """Get number of registered scorers."""
        return len(self._scorers)
