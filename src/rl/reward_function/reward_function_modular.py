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
        LLMAPIHandler,
        GrammarScorer,
        JSONScorer,
        LinguisticScorer,
        TopicScorer,
    )
except ImportError:
    from src.rl.reward_function.scorers import (
        CEFRScorer,
        CoherenceScorer,
        LLMAPIHandler,
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
        concurrency_limit: int = 20, # New concurrency limit
    ):
        """
        Initialize modular reward function.

        Args:
            spacy_model: Italian spaCy model name (default: it_core_news_sm for speed)
            device: The device to run heavy models on ('cuda' or 'cpu').
            disabled_scorers: A list of scorer names to disable (e.g., ["fluency", "cefr"]).
            fluency_use_llm: Specifically enable the LLM component of the FluencyScorer.
            concurrency_limit: Max number of concurrent OpenAI API calls.
        """
        # Handle None default for disabled_scorers (avoid mutable default argument)
        if disabled_scorers is None:
            disabled_scorers = []

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
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        # Create a single, shared LLM API handler
        self.llm_handler = LLMAPIHandler()

        # Initialize individual scorers
        print("Initializing scorers...")
        all_scorers = {
            "json": JSONScorer(nlp=None),  # Doesn't need NLP
            "quality": ExerciseQualityScorer(nlp=self.nlp),  # Context validation, redundancy
            "linguistic": LinguisticScorer(nlp=self.nlp),
            # Pass the shared handler to all LLM-based scorers
            "cefr": CEFRScorer(self.llm_handler),
            "fluency": FluencyScorer(nlp=self.nlp, llm_handler=self.llm_handler, use_llm=fluency_use_llm, disabled="fluency" in disabled_scorers),
            "grammar": GrammarScorer(self.llm_handler),
            "coherence": CoherenceScorer(self.llm_handler),
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

        # --- Create a task for each active scorer ---
        tasks = {}
        for name, scorer in self.scorers.items():
            # CPU-bound scorers run in a thread pool to avoid blocking
            if name in ["json", "quality", "linguistic", "topic"]:
                tasks[name] = loop.run_in_executor(None, scorer.score, exercise, request)
            # I/O-bound (LLM) or async scorers run directly
            else:
                tasks[name] = scorer.score(exercise, request, semaphore)

        # --- Gather results concurrently ---
        task_results = await asyncio.gather(*tasks.values())
        
        # --- Map results back to scorer names ---
        results_map = dict(zip(tasks.keys(), task_results))

        # --- Safely extract scores and errors ---
        def get_score(name):
            return results_map.get(name, (0.0, []))

        json_score, json_errors = get_score("json")
        quality_score, quality_errors = get_score("quality")
        linguistic_score, linguistic_errors = get_score("linguistic")
        cefr_score, cefr_errors = get_score("cefr")
        fluency_score, fluency_errors = get_score("fluency")
        grammar_score, grammar_errors = get_score("grammar")
        coherence_score, coherence_errors = get_score("coherence")
        topic_score, topic_errors = get_score("topic")

        # --- Aggregate errors and calculate total score ---
        all_errors = (
            json_errors + quality_errors + linguistic_errors + cefr_errors +
            fluency_errors + grammar_errors + coherence_errors + topic_errors
        )

        raw_total = sum([
            json_score, quality_score, linguistic_score, cefr_score,
            fluency_score, grammar_score, coherence_score, topic_score
        ])

        total = min(100, (raw_total / self.max_score) * 100) if self.max_score > 0 else 0.0

        # --- Create final breakdown ---
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
            errors=list(filter(None, all_errors)), # Filter out empty error strings
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

        # --- 1. Run all CPU-bound scorers for all exercises ---
        cpu_scorers = {name: scorer for name, scorer in self.scorers.items() if name in ["json", "quality", "linguistic", "topic"]}
        cpu_results_by_exercise = [{} for _ in exercises]

        loop = asyncio.get_running_loop()
        # print("  - Running CPU-bound scorers...")  # Commented out to reduce log noise
        for name, scorer in cpu_scorers.items():
            try:
                tasks = [loop.run_in_executor(None, scorer.score, ex, request) for ex in exercises]
                results = await asyncio.gather(*tasks)
                for i, result in enumerate(results):
                    cpu_results_by_exercise[i][name] = result
                # print(f"    ✅ {name}: success")  # Commented out to reduce log noise
            except Exception as e:
                print(f"    ❌ {name}: failed with {e}")


        # --- 2. Run batch-level JSON checks ---
        batch_json_score = 0.0
        batch_json_errors = []
        if "json" in self.scorers:
            try:
                batch_json_score, batch_json_errors = self.scorers["json"].score_batch(exercises, request)
                # print(f"    ✅ json (batch): success")  # Commented out to reduce log noise
            except Exception as e:
                print(f"    ❌ json (batch): failed with {e}")


        # --- 3. Run all I/O-bound (LLM) scorers in a single batched call per scorer ---
        # print("  - Running I/O-bound (LLM) scorers...")  # Commented out to reduce log noise
        io_tasks = {
            "grammar": self.scorers["grammar"].score_batch(exercises, request, semaphore) if "grammar" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            "coherence": self.scorers["coherence"].score_batch(exercises, request, semaphore) if "coherence" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            "cefr": self.scorers["cefr"].score_batch(exercises, request, semaphore) if "cefr" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            "fluency": self.scorers["fluency"].score_batch(exercises, request, semaphore) if "fluency" in self.scorers and self.scorers["fluency"].use_llm else asyncio.sleep(0, [(10.0, [])] * len(exercises)),
        }

        # Add timeout protection to prevent hanging (90s total for all scorers to try multiple providers)
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*io_tasks.values(), return_exceptions=True),
                timeout=90.0
            )
        except asyncio.TimeoutError:
            # print("    ⚠️  LLM scorers timed out after 90s, using default scores")  # Commented out to reduce log noise
            results = [[(5.0, [f"{name} timed out"])] * len(exercises) for name in io_tasks.keys()]
        
        results_map = dict(zip(io_tasks.keys(), results))

        def get_io_results(name):
            result = results_map.get(name)
            if isinstance(result, Exception):
                # print(f"    ❌ {name}: failed with {result}")  # Commented out to reduce log noise
                return [(0.0, [f"{name} scorer failed: {result}"]) for _ in exercises]
            # print(f"    ✅ {name}: success")  # Commented out to reduce log noise
            return result

        grammar_results = get_io_results("grammar")
        coherence_results = get_io_results("coherence")
        cefr_results = get_io_results("cefr")
        fluency_results = get_io_results("fluency")


        # --- 4. Combine all results for each exercise ---
        # print("  - Aggregating results...")  # Commented out to reduce log noise
        results = []
        for i in range(len(exercises)):
            # Get CPU scores
            json_score, json_errors = cpu_results_by_exercise[i].get("json", (0.0, []))
            quality_score, quality_errors = cpu_results_by_exercise[i].get("quality", (0.0, []))
            linguistic_score, linguistic_errors = cpu_results_by_exercise[i].get("linguistic", (0.0, []))
            topic_score, topic_errors = cpu_results_by_exercise[i].get("topic", (0.0, []))

            # Get IO scores
            grammar_score, grammar_errors = grammar_results[i]
            coherence_score, coherence_errors = coherence_results[i]
            cefr_score, cefr_errors = cefr_results[i]
            fluency_score, fluency_errors = fluency_results[i]

            # Apply batch-level JSON penalty
            json_score += batch_json_score

            # Aggregate all scores and errors
            all_errors = list(filter(None, json_errors + quality_errors + linguistic_errors + topic_errors + grammar_errors + coherence_errors + cefr_errors + fluency_errors + batch_json_errors))

            raw_total = sum([
                json_score, quality_score, linguistic_score, topic_score,
                grammar_score, coherence_score, cefr_score, fluency_score
            ])

            total = min(100, (raw_total / self.max_score) * 100) if self.max_score > 0 else 0.0

            breakdown = RewardBreakdown(
                json_validity=json_score, exercise_quality=quality_score, linguistic_quality=linguistic_score,
                cefr_alignment=cefr_score, fluency=fluency_score, grammar_correctness=grammar_score,
                coherence=coherence_score, topic_adherence=topic_score, total=total, errors=all_errors
            )
            results.append((total, breakdown))

        total_score = sum(score for score, breakdown in results)

        avg_score = total_score / len(exercises) if exercises else 0.0
        # print("  - Scoring complete.")  # Commented out to reduce log noise
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
