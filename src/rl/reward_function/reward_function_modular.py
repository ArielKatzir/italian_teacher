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
        CoherenceScorer,
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
        CoherenceScorer,
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
    exercise_quality: float  # 0-20
    linguistic_quality: float  # 0-15 (Re-weighted)
    cefr_alignment: float  # 0-20
    fluency: float  # 0-10
    grammar_correctness: float  # 0-10
    coherence: float  # 0-10
    topic_adherence: float  # 0-10
    total: float  # Dynamic total, will be normalized to 100
    errors: List[str]  # Specific issues found

    def __str__(self):
        return (
            f"Score: {self.total}/100\n"
            f"  JSON: {self.json_validity}/15\n"
            f"  Quality: {self.exercise_quality}/20\n"
            f"  Linguistic: {self.linguistic_quality}/15\n"
            f"  CEFR: {self.cefr_alignment}/20\n"
            f"  Fluency: {self.fluency}/10\n" # This will be 0 if disabled
            f"  Grammar: {self.grammar_correctness}/10\n"
            f"  Coherence: {self.coherence}/10\n"
            f"  Topic: {self.topic_adherence}/10\n"
            f"Errors: {', '.join(self.errors) if self.errors else 'None'}"
        )


class ExerciseRewardFunction:
    """
    Modular reward function for scoring Italian exercise quality.

    Uses individual scorer components (130 points total → normalized to 100):
    - JSONScorer: Structure validation with STRICT type penalties (15 pts)
    - ExerciseQualityScorer: Context validation, redundancy checks (20 pts)
    - LinguisticScorer: Italian grammar rules (15 pts)
    - CEFRScorer: Level-appropriate complexity (20 pts)
    - FluencyScorer: Natural language flow (10 pts)
    - GrammarScorer: Grammar focus validation using LLM (10 pts)
    - TopicScorer: Semantic similarity to topic (10 pts)
    - CoherenceScorer: Logical sense and coherence (10 pts)
    """

    def __init__(
        self,
        spacy_model: str = "it_core_news_sm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        disabled_scorers: List[str] = None,
        fluency_use_llm: bool = False, # Keep specific flag for convenience
    ):
        """
        Initialize modular reward function.

        Args:
            spacy_model: Italian spaCy model name (default: it_core_news_sm for speed)
            device: The device to run heavy models on ('cuda' or 'cpu').
            disabled_scorers: A list of scorer names to disable (e.g., ["fluency", "cefr"]).
            fluency_use_llm: Specifically enable the LLM component of the FluencyScorer.
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
        all_scorers = {
            "json": JSONScorer(nlp=None),  # Doesn't need NLP
            "quality": ExerciseQualityScorer(nlp=self.nlp),  # Context validation, redundancy
            "linguistic": LinguisticScorer(nlp=self.nlp),
            "cefr": CEFRScorer(),  # Fully LLM-based, does not use spaCy
            "fluency": FluencyScorer(nlp=self.nlp, use_llm=fluency_use_llm),
            "grammar": GrammarScorer(),  # Fully LLM-based, does not use spaCy
            "coherence": CoherenceScorer(),  # Fully LLM-based, does not use spaCy
            "topic": TopicScorer(nlp=None, device=self.device),  # Uses sentence transformer
        }

        # Filter out disabled scorers
        disabled_scorers = disabled_scorers or []
        self.scorers = {
            name: scorer
            for name, scorer in all_scorers.items()
            if name not in disabled_scorers
        }

        # Calculate the dynamic max score based on active scorers
        self.max_score = sum(scorer.max_score for scorer in self.scorers.values())

        print(f"✅ Reward function initialized. Active scorers: {list(self.scorers.keys())}")
        if disabled_scorers:
            print(f"   Disabled scorers: {disabled_scorers}")

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
            loop.run_in_executor(None, self.scorers["json"].score, exercise, request) if "json" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["quality"].score, exercise, request) if "quality" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["linguistic"].score, exercise, request) if "linguistic" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["fluency"].score, exercise, request) if "fluency" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["topic"].score, exercise, request) if "topic" in self.scorers else asyncio.sleep(0, (0.0, [])),
        ]

        # --- Run I/O-bound (network) tasks directly on the event loop ---
        # These now use the inefficient `score` method which wraps `score_batch` for a single item.
        # This is correct for scoring a single exercise.
        io_tasks = [
            self.scorers["grammar"].score(exercise, request, semaphore) if "grammar" in self.scorers else asyncio.sleep(0, (0.0, [])),
            self.scorers["coherence"].score(exercise, request, semaphore) if "coherence" in self.scorers else asyncio.sleep(0, (0.0, [])),
            self.scorers["cefr"].score(exercise, request, semaphore) if "cefr" in self.scorers else asyncio.sleep(0, (0.0, [])),
        ]

        # --- Gather results from all tasks concurrently ---
        results = await asyncio.gather(*(cpu_tasks + io_tasks))

        # Unpack results
        (json_score, json_errors)           = results[0]
        (quality_score, quality_errors)     = results[1]
        (linguistic_score, linguistic_errors) = results[2]
        (fluency_score, fluency_errors)     = results[3]
        (topic_score, topic_errors)         = results[4]
        (grammar_score, grammar_errors)     = results[5]
        (coherence_score, coherence_errors) = results[6]
        (cefr_score, cefr_errors)           = results[7]

        all_errors.extend(json_errors)
        all_errors.extend(quality_errors)
        all_errors.extend(linguistic_errors)
        all_errors.extend(fluency_errors)
        all_errors.extend(grammar_errors)
        all_errors.extend(coherence_errors)
        all_errors.extend(topic_errors)

        # Total score (normalized to 100)
        # JSON(15) + Quality(30) + Linguistic(25) + CEFR(20) + Fluency(10) + Grammar(10) + Coherence(10) + Topic(10) = 130
        raw_total = (
            json_score
            + quality_score
            + linguistic_score
            + cefr_score
            + fluency_score
            + grammar_score
            + coherence_score
            + topic_score
        )

        # Normalize to 100 (130 max → 100)
        total = min(100, (raw_total / self.max_score) * 100) if self.max_score > 0 else 0.0 # self.max_score is dynamic

        # Create breakdown
        breakdown = RewardBreakdown(
            json_validity=json_score,
            exercise_quality=quality_score,
            linguistic_quality=linguistic_score,
            cefr_alignment=cefr_score,
            fluency=fluency_score,
            grammar_correctness=grammar_score,
            coherence=coherence_score,
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
            loop.run_in_executor(None, self.scorers["json"].score, exercise, request) if "json" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["quality"].score, exercise, request) if "quality" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["linguistic"].score, exercise, request) if "linguistic" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["fluency"].score, exercise, request) if "fluency" in self.scorers else asyncio.sleep(0, (0.0, [])),
            loop.run_in_executor(None, self.scorers["topic"].score, exercise, request) if "topic" in self.scorers else asyncio.sleep(0, (0.0, [])),
        ]

        # --- Gather results from all CPU tasks ---
        results = await asyncio.gather(*cpu_tasks)

        # Unpack results
        (json_score, json_errors) = results[0]
        (quality_score, quality_errors) = results[1]
        (linguistic_score, linguistic_errors) = results[2]
        (fluency_score, fluency_errors) = results[3]
        (topic_score, topic_errors) = results[4]

        all_errors.extend(json_errors)
        all_errors.extend(quality_errors)
        all_errors.extend(linguistic_errors)
        all_errors.extend(fluency_errors)
        all_errors.extend(topic_errors)

        # The grammar score is handled separately by the async worker pool
        grammar_score = 0.0
        cefr_score = 0.0 # Placeholder

        raw_total = (
            json_score + quality_score + linguistic_score + fluency_score + topic_score
        )
        total = min(100, (raw_total / self.max_score) * 100) if self.max_score > 0 else 0.0 # self.max_score is dynamic

        breakdown = RewardBreakdown(
            json_validity=json_score,
            exercise_quality=quality_score,
            linguistic_quality=linguistic_score,
            cefr_alignment=cefr_score,
            fluency=fluency_score,
            grammar_correctness=grammar_score,  # Placeholder
            coherence=0.0,  # Placeholder
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

        # --- Efficiently score a batch of exercises ---
        # 1. Run all CPU-bound scorers for all exercises in parallel
        cpu_tasks = [self.score_cpu_only(ex, request) for ex in exercises]
        cpu_results = await asyncio.gather(*cpu_tasks)

        # Run batch-level JSON checks (e.g., type diversity)
        batch_json_score = 0.0
        batch_json_errors = []
        if "json" in self.scorers:
            batch_json_score, batch_json_errors = self.scorers["json"].score_batch(exercises, request)
            # Apply this batch penalty to all exercises in the breakdown


        # 2. Run all I/O-bound (LLM) scorers in a single batched call per scorer
        io_tasks = [
            self.scorers["grammar"].score_batch(exercises, request, semaphore) if "grammar" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            self.scorers["coherence"].score_batch(exercises, request, semaphore) if "coherence" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            self.scorers["cefr"].score_batch(exercises, request, semaphore) if "cefr" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
        ]
        grammar_results, coherence_results, cefr_results = await asyncio.gather(*io_tasks)

        # 3. Combine the results
        results = []
        for i, (cpu_score, cpu_breakdown) in enumerate(cpu_results):
            cpu_breakdown.grammar_correctness, grammar_errors = grammar_results[i]
            cpu_breakdown.coherence, coherence_errors = coherence_results[i]
            cpu_breakdown.cefr_alignment, cefr_errors = cefr_results[i]
            
            # Add batch-level JSON score and errors to each exercise's breakdown
            cpu_breakdown.json_validity += batch_json_score
            cpu_breakdown.errors.extend(batch_json_errors)

            # --- COMPLETE THE SCORE CALCULATION ---
            # 1. Add new errors
            cpu_breakdown.errors.extend(grammar_errors)
            cpu_breakdown.errors.extend(coherence_errors)
            cpu_breakdown.errors.extend(cefr_errors)

            # 2. Recalculate the raw total score
            raw_total = ( # Recalculate with updated json_validity
                cpu_breakdown.json_validity + cpu_breakdown.exercise_quality +
                cpu_breakdown.linguistic_quality + cpu_breakdown.fluency +
                cpu_breakdown.topic_adherence + cpu_breakdown.grammar_correctness +
                cpu_breakdown.coherence + cpu_breakdown.cefr_alignment
            )

            # 3. Normalize and update the breakdown object
            cpu_breakdown.total = min(100, (raw_total / self.max_score) * 100) if self.max_score > 0 else 0.0
            results.append((cpu_breakdown.total, cpu_breakdown))

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
