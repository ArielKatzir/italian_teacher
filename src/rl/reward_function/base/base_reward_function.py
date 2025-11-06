"""
Base reward function class for subject-agnostic exercise evaluation.

This abstract base class defines the interface and common functionality
for all subject-specific reward functions (Italian, Math, etc.).
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from .scorer_registry import ScorerRegistry


@dataclass
class RewardBreakdown:
    """
    Detailed breakdown of reward components.

    This is a flexible structure that can accommodate different
    scoring dimensions based on the subject.
    """

    total: float  # Normalized to 100
    component_scores: Dict[str, float]  # Scores by component name
    errors: List[str]  # Specific issues found

    def __str__(self):
        components_str = "\n".join(
            f"  {name}: {score:.2f}/{self.component_scores.get(f'{name}_max', 'N/A')}"
            for name, score in self.component_scores.items()
            if not name.endswith("_max")
        )
        return (
            f"Score: {self.total:.2f}/100\n"
            f"{components_str}\n"
            f"Errors: {', '.join(self.errors) if self.errors else 'None'}"
        )


class BaseRewardFunction(ABC):
    """
    Abstract base class for subject-specific reward functions.

    Subclasses should:
    1. Override _configure_scorers() to register subject-specific scorers
    2. Optionally override _initialize_resources() for subject-specific setup
    3. Optionally override _validate_exercise() for subject-specific validation
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        disabled_scorers: List[str] = None,
        concurrency_limit: int = 20,
        **subject_specific_kwargs
    ):
        """
        Initialize base reward function.

        Args:
            device: Device for heavy models ('cuda' or 'cpu')
            disabled_scorers: List of scorer names to disable
            concurrency_limit: Max concurrent API calls
            **subject_specific_kwargs: Additional subject-specific configuration
        """
        self.device = device
        self.disabled_scorers = disabled_scorers or []
        self.concurrency_limit = concurrency_limit
        self.subject_kwargs = subject_specific_kwargs

        print(f"Initializing {self.__class__.__name__}...")
        print(f"  Device: {self.device}")

        # Semaphore for API rate limiting
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        # Registry for scorers
        self.scorer_registry = ScorerRegistry()

        # Initialize subject-specific resources (NLP models, LLM handlers, etc.)
        self._initialize_resources()

        # Configure scorers (implemented by subclasses)
        self._configure_scorers()

        # Filter out disabled scorers
        for disabled in self.disabled_scorers:
            if disabled in self.scorer_registry:
                self.scorer_registry.remove(disabled)
                print(f"  Disabled scorer: {disabled}")

        # Calculate max score from active scorers
        self.max_score = sum(
            config.max_score for config in self.scorer_registry.get_all().values()
        )

        active_scorers = self.scorer_registry.get_names()
        print(f"✅ {self.__class__.__name__} initialized")
        print(f"  Active scorers: {active_scorers}")
        print(f"  Max score: {self.max_score}")

    @abstractmethod
    def _configure_scorers(self) -> None:
        """
        Configure subject-specific scorers.

        Subclasses must implement this to register their scorers
        using self.scorer_registry.register().

        Example:
            self.scorer_registry.register(
                name="grammar",
                scorer_class=GrammarScorer,
                max_score=10.0,
                prompt_fn=get_grammar_prompt,
                llm_handler=self.llm_handler
            )
        """
        pass

    def _initialize_resources(self) -> None:
        """
        Initialize subject-specific resources (NLP models, LLM handlers, etc.).

        Subclasses can override this to set up resources like:
        - spaCy models
        - LLM API handlers
        - Sentence transformers
        - Subject-specific data loaders

        Default implementation does nothing.
        """
        pass

    def _validate_exercise(
        self, exercise: Dict[str, Any], request: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Perform subject-specific exercise validation.

        Subclasses can override this to add custom validation logic.

        Args:
            exercise: Exercise to validate
            request: Request context

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return True, []

    async def score(
        self,
        exercise: Dict[str, Any],
        request: Dict[str, Any],
        semaphore: asyncio.Semaphore = None,
    ) -> Tuple[float, RewardBreakdown]:
        """
        Score a single exercise.

        Args:
            exercise: Exercise dict to score
            request: Request context
            semaphore: Optional semaphore for rate limiting

        Returns:
            Tuple of (total_score, breakdown)
        """
        # Use provided semaphore or default
        sem = semaphore or self.semaphore

        # Validate exercise
        is_valid, validation_errors = self._validate_exercise(exercise, request)
        if not is_valid:
            return 0.0, RewardBreakdown(
                total=0.0,
                component_scores={},
                errors=validation_errors,
            )

        # Score with all active scorers
        return await self._score_with_scorers([exercise], request, sem)

    async def score_exercises(
        self,
        exercises: List[Dict[str, Any]],
        request: Dict[str, Any],
        semaphore: asyncio.Semaphore = None,
    ) -> Tuple[float, List[Tuple[float, RewardBreakdown]]]:
        """
        Score multiple exercises and return average score + individual breakdowns.

        Args:
            exercises: List of exercise dicts to score
            request: Request context
            semaphore: Optional semaphore for rate limiting

        Returns:
            Tuple of (average_score, list of (score, breakdown) tuples)
        """
        if not exercises:
            return 0.0, []

        # Use provided semaphore or default
        sem = semaphore or self.semaphore

        # Score all exercises
        results = await self._score_batch(exercises, request, sem)

        # Calculate average
        avg_score = sum(score for score, _ in results) / len(results) if results else 0.0

        return avg_score, results

    async def _score_batch(
        self,
        exercises: List[Dict[str, Any]],
        request: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> List[Tuple[float, RewardBreakdown]]:
        """
        Score a batch of exercises efficiently.

        This method orchestrates parallel scoring across all active scorers.

        Args:
            exercises: List of exercises to score
            request: Request context
            semaphore: Semaphore for rate limiting

        Returns:
            List of (score, breakdown) tuples
        """
        if not exercises:
            return []

        # Get all active scorers
        scorer_configs = self.scorer_registry.get_all()

        # Instantiate scorers (lazy initialization)
        scorers = {}
        for name, config in scorer_configs.items():
            kwargs = config.kwargs.copy()

            # If this is an LLM scorer with a prompt function, inject it
            if config.prompt_fn:
                kwargs["prompt_fn"] = config.prompt_fn

            scorers[name] = config.scorer_class(**kwargs)

        # Score each exercise
        results = []
        for exercise in exercises:
            # Collect scores from all scorers
            component_scores = {}
            all_errors = []

            # Run all scorers (some in parallel)
            tasks = {}
            for name, scorer in scorers.items():
                # Check if scorer has batch scoring capability
                if hasattr(scorer, "score_batch"):
                    # For now, score individually (can optimize later)
                    tasks[name] = scorer.score(exercise, request, semaphore)
                else:
                    tasks[name] = scorer.score(exercise, request)

            # Gather results
            task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            results_map = dict(zip(tasks.keys(), task_results))

            # Extract scores and errors
            for name, result in results_map.items():
                if isinstance(result, Exception):
                    print(f"  ⚠️  {name} scorer failed: {result}")
                    component_scores[name] = 0.0
                    all_errors.append(f"{name} scorer error: {str(result)[:100]}")
                else:
                    score, errors = result
                    component_scores[name] = score
                    component_scores[f"{name}_max"] = scorer_configs[name].max_score
                    all_errors.extend(errors)

            # Calculate total score
            raw_total = sum(
                score for key, score in component_scores.items()
                if not key.endswith("_max")
            )
            total = min(100, (raw_total / self.max_score) * 100) if self.max_score > 0 else 0.0

            breakdown = RewardBreakdown(
                total=total,
                component_scores=component_scores,
                errors=list(filter(None, all_errors)),
            )

            results.append((total, breakdown))

        return results

    @property
    def subject_name(self) -> str:
        """Get the subject name (derived from class name)."""
        class_name = self.__class__.__name__
        # Remove "RewardFunction" suffix if present
        return class_name.replace("RewardFunction", "")
