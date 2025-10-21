"""
Modular Reward Function for Italian Exercise Generation.


Scores exercises on 6 dimensions (0-100 points total):
1. JSON Validity (15 points) - Structure and format
2. Linguistic Quality (35 points) - Comprehensive Italian grammar validation
3. CEFR Level Alignment (20 points) - Appropriate difficulty for target level
4. Fluency (10 points) - Natural language flow and construction
5. Grammar Correctness (10 points) - Matches requested grammar_focus
6. Topic Adherence (10 points) - Relevant to requested topic

This modular version uses individual scorer classes for each component.
"""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Add parent directory to path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import spacy

# Use absolute imports that work both as module and standalone
try:
    from .scorers import (
        CEFRScorer,
        ExerciseQualityScorer,
        FluencyScorer,
        GrammarScorer,
        JSONScorer,
        LinguisticScorer,
        TopicScorer,
    )
except ImportError:
    from src.rl.reward_function.scorers import (
        CEFRScorer,
        ExerciseQualityScorer,
        FluencyScorer,
        GrammarScorer,
        JSONScorer,
        LinguisticScorer,
        TopicScorer,
    )


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    json_validity: float  # 0-15
    exercise_quality: float  # 0-30 (INCREASED: 15pt context penalty!)
    linguistic_quality: float  # 0-25 (Italian grammar rules)
    cefr_alignment: float  # 0-20
    fluency: float  # 0-10
    grammar_correctness: float  # 0-10
    topic_adherence: float  # 0-10
    total: float  # 0-105 (will normalize to 100)
    errors: List[str]  # Specific issues found

    def __str__(self):
        return (
            f"Score: {self.total}/100\n"
            f"  JSON: {self.json_validity}/15\n"
            f"  Quality: {self.exercise_quality}/20\n"
            f"  Linguistic: {self.linguistic_quality}/25\n"
            f"  CEFR: {self.cefr_alignment}/20\n"
            f"  Fluency: {self.fluency}/10\n"
            f"  Grammar: {self.grammar_correctness}/10\n"
            f"  Topic: {self.topic_adherence}/10\n"
            f"Errors: {', '.join(self.errors) if self.errors else 'None'}"
        )


class ExerciseRewardFunction:
    """
    Modular reward function for scoring Italian exercise quality.

    Uses individual scorer components (130 points total → normalized to 100):
    - JSONScorer: Structure validation with STRICT type penalties (15 pts)
    - ExerciseQualityScorer: Context validation, redundancy checks (30 pts) [CRITICAL!]
    - LinguisticScorer: Italian grammar rules (25 pts)
    - CEFRScorer: Level-appropriate complexity (20 pts)
    - FluencyScorer: Natural language flow (10 pts)
    - GrammarScorer: Grammar focus validation using spaCy morphology (10 pts)
    - TopicScorer: Semantic similarity to topic (10 pts)
    - CoherenceScorer: Logical sense and coherence (10 pts)
    """

    def __init__(
        self,
        spacy_model: str = "it_core_news_sm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize modular reward function.

        Args:
            spacy_model: Italian spaCy model name (default: it_core_news_sm for speed)
            device: The device to run heavy models on ('cuda' or 'cpu').
        """
        try:
            # Load Italian NLP model
            print(f"Loading spaCy model: {spacy_model}...")
            self.nlp = spacy.load(spacy_model)
            print("✅ spaCy model loaded")
        except OSError:
            print(f"⚠️  spaCy model '{spacy_model}' not found.")
            print(f"   Install with: python -m spacy download {spacy_model}")
            raise

        self.device = device
        print(f"Reward function will use device: {self.device}")

        # Initialize individual scorers
        print("Initializing scorers...")
        self.scorers = {
            "json": JSONScorer(nlp=None),  # Doesn't need NLP
            "quality": ExerciseQualityScorer(nlp=self.nlp),  # Context validation, redundancy
            "linguistic": LinguisticScorer(nlp=self.nlp),
            "cefr": CEFRScorer(nlp=self.nlp),  # Pre-loads 16,887-word vocabulary
            "fluency": FluencyScorer(nlp=self.nlp),
            "grammar": GrammarScorer(nlp=self.nlp),  # spaCy morphology (no hardcoded words)
            "topic": TopicScorer(nlp=None, device=self.device),  # Uses sentence transformer
        }

        print("✅ Reward function initialized with 8 professional scorers")

    async def score(
        self, exercise: Dict[str, Any], request: Dict[str, Any], semaphore: asyncio.Semaphore = None
    ) -> Tuple[float, RewardBreakdown]:
        """
        Score an exercise.

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
            Tuple of (total_score, breakdown)
        """
        all_errors = []

        loop = asyncio.get_running_loop()

        # --- Run CPU-bound tasks in a thread pool executor to avoid blocking ---
        # The executor runs these synchronous functions in separate threads.
        cpu_tasks = [
            loop.run_in_executor(None, self.scorers["json"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["quality"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["linguistic"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["cefr"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["fluency"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["topic"].score, exercise, request),
        ]

        # --- Run I/O-bound (network) tasks directly on the event loop ---
        io_tasks = [
            self.scorers["grammar"].score(exercise, request, semaphore)  # This is already async
        ]

        # --- Gather results from all tasks concurrently ---
        results = await asyncio.gather(*(cpu_tasks + io_tasks))

        # Unpack results
        (json_score, json_errors) = results[0]
        (quality_score, quality_errors) = results[1]
        (linguistic_score, linguistic_errors) = results[2]
        (cefr_score, cefr_errors) = results[3]
        (fluency_score, fluency_errors) = results[4]
        (topic_score, topic_errors) = results[5]
        (grammar_score, grammar_errors) = results[6]

        all_errors.extend(json_errors)
        all_errors.extend(quality_errors)
        all_errors.extend(linguistic_errors)
        all_errors.extend(cefr_errors)
        all_errors.extend(fluency_errors)
        all_errors.extend(grammar_errors)
        all_errors.extend(topic_errors)

        # Total score (normalized to 100)
        # JSON(15) + Quality(30) + Linguistic(25) + CEFR(20) + Fluency(10) + Grammar(10) + Topic(10) + Coherence(10) = 130
        raw_total = (
            json_score
            + quality_score
            + linguistic_score
            + cefr_score
            + fluency_score
            + grammar_score
            + topic_score
        )

        # Normalize to 100 (130 max → 100)
        total = min(100, (raw_total / 130) * 100)

        # Create breakdown
        breakdown = RewardBreakdown(
            json_validity=json_score,
            exercise_quality=quality_score,
            linguistic_quality=linguistic_score,
            cefr_alignment=cefr_score,
            fluency=fluency_score,
            grammar_correctness=grammar_score,
            topic_adherence=topic_score,
            total=total,
            errors=all_errors,
        )

        return total, breakdown

    async def score_cpu_only(
        self, exercise: Dict[str, Any], request: Dict[str, Any]
    ) -> Tuple[float, RewardBreakdown]:
        """
        Scores an exercise using only CPU-bound scorers.
        This is designed to be run in a thread pool without blocking the main event loop.
        """
        all_errors = []

        loop = asyncio.get_running_loop()

        # --- Run CPU-bound tasks in a thread pool executor ---
        cpu_tasks = [
            loop.run_in_executor(None, self.scorers["json"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["quality"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["linguistic"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["cefr"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["fluency"].score, exercise, request),
            loop.run_in_executor(None, self.scorers["topic"].score, exercise, request),
        ]

        # --- Gather results from all CPU tasks ---
        results = await asyncio.gather(*cpu_tasks)

        # Unpack results
        (json_score, json_errors) = results[0]
        (quality_score, quality_errors) = results[1]
        (linguistic_score, linguistic_errors) = results[2]
        (cefr_score, cefr_errors) = results[3]
        (fluency_score, fluency_errors) = results[4]
        (topic_score, topic_errors) = results[5]

        all_errors.extend(json_errors)
        all_errors.extend(quality_errors)
        all_errors.extend(linguistic_errors)
        all_errors.extend(cefr_errors)
        all_errors.extend(fluency_errors)
        all_errors.extend(topic_errors)

        # The grammar score is handled separately by the async worker pool
        grammar_score = 0.0

        raw_total = (
            json_score + quality_score + linguistic_score + cefr_score + fluency_score + topic_score
        )
        total = min(100, (raw_total / 130) * 100)

        breakdown = RewardBreakdown(
            json_validity=json_score,
            exercise_quality=quality_score,
            linguistic_quality=linguistic_score,
            cefr_alignment=cefr_score,
            fluency=fluency_score,
            grammar_correctness=grammar_score,  # Placeholder
            topic_adherence=topic_score,
            total=total,
            errors=all_errors,
        )
        return total, breakdown

    async def score_exercises(
        self,
        exercises: List[Dict[str, Any]],
        request: Dict[str, Any],
        semaphore: asyncio.Semaphore = None,
    ) -> Tuple[float, List[Tuple[float, RewardBreakdown]]]:
        """
        Score multiple exercises and return average score + individual breakdowns.

        Args:
            exercises: List of exercise dictionaries to score
            request: Request context (same for all exercises)

        Returns:
            Tuple of (average_score, list of (score, breakdown) tuples)
        """
        if not exercises:
            return 0.0, []

        tasks = [self.score(ex, request, semaphore) for ex in exercises]
        results = await asyncio.gather(*tasks)
        total_score = sum(score for score, breakdown in results)

        avg_score = total_score / len(exercises) if exercises else 0.0
        return avg_score, results


# Convenience function
_reward_function = None


async def score_exercise(
    exercise: Dict[str, Any], request: Dict[str, Any], semaphore: asyncio.Semaphore = None
) -> Tuple[float, RewardBreakdown]:
    """
    Score an exercise (convenience function).

    Creates a global reward function instance and scores the exercise.
    For repeated use, create an ExerciseRewardFunction instance directly.
    """
    global _reward_function
    if _reward_function is None:
        _reward_function = ExerciseRewardFunction()

    return await _reward_function.score(exercise, request, semaphore)
