"""
Italian language exercise reward function.

This implements the complete Italian exercise evaluation system with:
- JSON validation
- Exercise quality checks
- Italian-specific linguistic validation
- CEFR level alignment
- Fluency assessment
- Grammar correctness
- Coherence evaluation
- Topic adherence

This is a refactored version that uses organized prompts and can be extended
for other subjects (e.g., Math).
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import spacy
import torch

from ...base import (
    JSONScorer,
    LLMAPIHandler,
)
from .scorers import (
    ItalianGrammarScorer,
    ItalianLinguisticScorer,
    ItalianCEFRScorer,
    ItalianCoherenceScorer,
    ItalianFluencyScorer,
    ItalianTopicScorer,
    ItalianExerciseQualityScorer,
)
from .prompts import (
    get_grammar_prompt,
    get_cefr_prompt,
    get_coherence_prompt,
    get_fluency_prompt,
)


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    json_validity: float  # 0-15
    exercise_quality: float  # 0-20
    linguistic_quality: float  # 0-15
    cefr_alignment: float  # 0-30 (INCREASED WEIGHT)
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
            f"  CEFR: {self.cefr_alignment}/30\n"
            f"  Fluency: {self.fluency}/10\n"
            f"  Grammar: {self.grammar_correctness}/10\n"
            f"  Coherence: {self.coherence}/10\n"
            f"  Topic: {self.topic_adherence}/10\n"
            f"Errors: {', '.join(self.errors) if self.errors else 'None'}"
        )


class ItalianRewardFunction:
    """
    Italian-specific reward function for exercise evaluation.

    Scoring breakdown (120 points total → normalized to 100):
    - JSON Validity: 15 points
    - Exercise Quality: 20 points
    - Linguistic Quality: 15 points (Italian grammar rules)
    - CEFR Alignment: 30 points (difficulty appropriateness)
    - Fluency: 10 points (natural Italian flow)
    - Grammar Correctness: 10 points (matches requested grammar focus)
    - Coherence: 10 points (logical sense)
    - Topic Adherence: 10 points (semantic relevance)

    This is a refactored version that uses organized prompts from the
    subjects/italian/prompts/ directory.
    """

    def __init__(
        self,
        spacy_model: str = "it_core_news_sm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        disabled_scorers: List[str] = None,
        fluency_use_llm: bool = False,
        concurrency_limit: int = 20,
    ):
        """
        Initialize Italian reward function.

        Args:
            spacy_model: Italian spaCy model name (default: it_core_news_sm for speed)
            device: Device for heavy models ('cuda' or 'cpu')
            disabled_scorers: List of scorer names to disable
            fluency_use_llm: Enable LLM-based fluency scoring
            concurrency_limit: Max concurrent API calls
        """
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

        # Initialize individual scorers with Italian-specific prompts
        print("Initializing scorers...")
        all_scorers = {
            "json": JSONScorer(nlp=None),
            "quality": ItalianExerciseQualityScorer(nlp=self.nlp),
            "linguistic": ItalianLinguisticScorer(nlp=self.nlp),
            # LLM scorers with Italian-specific prompts
            "cefr": ItalianCEFRScorer(self.llm_handler, prompt_fn=get_cefr_prompt),
            "fluency": ItalianFluencyScorer(
                nlp=self.nlp,
                llm_handler=self.llm_handler,
                use_llm=fluency_use_llm,
                disabled="fluency" in disabled_scorers,
                prompt_fn=get_fluency_prompt,
            ),
            "grammar": ItalianGrammarScorer(self.llm_handler, prompt_fn=get_grammar_prompt),
            "coherence": ItalianCoherenceScorer(self.llm_handler, prompt_fn=get_coherence_prompt),
            "topic": ItalianTopicScorer(nlp=None, device=self.device),
        }

        # Filter out disabled scorers
        self.scorers = {
            name: scorer
            for name, scorer in all_scorers.items()
            if name not in disabled_scorers
        }

        # Calculate the dynamic max score based on active scorers
        self.max_score = sum(scorer.max_score for scorer in self.scorers.values())

        print(f"✅ Italian reward function initialized")
        print(f"   Active scorers: {list(self.scorers.keys())}")
        if disabled_scorers:
            print(f"   Disabled scorers: {disabled_scorers}")

    async def score(
        self, exercise: Dict[str, Any], request: Dict[str, Any], semaphore: asyncio.Semaphore = None
    ) -> Tuple[float, RewardBreakdown]:
        """
        Score an exercise.

        Args:
            exercise: Generated exercise dict
            request: Original request dict

        Returns:
            Tuple of (total_score, breakdown)
        """
        all_errors = []
        loop = asyncio.get_running_loop()

        # Create a task for each active scorer
        tasks = {}
        for name, scorer in self.scorers.items():
            # CPU-bound scorers run in a thread pool
            if name in ["json", "quality", "linguistic", "topic"]:
                tasks[name] = loop.run_in_executor(None, scorer.score, exercise, request)
            # I/O-bound (LLM) or async scorers run directly
            else:
                tasks[name] = scorer.score(exercise, request, semaphore)

        # Gather results concurrently
        task_results = await asyncio.gather(*tasks.values())
        results_map = dict(zip(tasks.keys(), task_results))

        # Extract scores and errors
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

        # Aggregate errors and calculate total score
        all_errors = (
            json_errors + quality_errors + linguistic_errors + cefr_errors +
            fluency_errors + grammar_errors + coherence_errors + topic_errors
        )

        raw_total = sum([
            json_score, quality_score, linguistic_score, cefr_score,
            fluency_score, grammar_score, coherence_score, topic_score
        ])

        total = min(100, (raw_total / self.max_score) * 100) if self.max_score > 0 else 0.0

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
            errors=list(filter(None, all_errors)),
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

        This method is optimized for batch scoring with concurrent API calls.

        Args:
            exercises: List of exercise dictionaries to score
            request: Request context (same for all exercises)
            semaphore: Optional semaphore for rate limiting

        Returns:
            Tuple of (average_score, list of (score, breakdown) tuples)
        """
        if not exercises:
            return 0.0, []

        # Use the same optimized batch scoring logic from the original
        # Run all CPU-bound scorers for all exercises
        cpu_scorers = {name: scorer for name, scorer in self.scorers.items() if name in ["json", "quality", "linguistic", "topic"]}
        cpu_results_by_exercise = [{} for _ in exercises]

        loop = asyncio.get_running_loop()
        for name, scorer in cpu_scorers.items():
            try:
                tasks = [loop.run_in_executor(None, scorer.score, ex, request) for ex in exercises]
                results = await asyncio.gather(*tasks)
                for i, result in enumerate(results):
                    cpu_results_by_exercise[i][name] = result
            except Exception as e:
                print(f"    ❌ {name}: failed with {e}")

        # Run batch-level JSON checks
        batch_json_score = 0.0
        batch_json_errors = []
        if "json" in self.scorers:
            try:
                batch_json_score, batch_json_errors = self.scorers["json"].score_batch(exercises, request)
            except Exception as e:
                print(f"    ❌ json (batch): failed with {e}")

        # Run all I/O-bound (LLM) scorers in batched calls
        io_tasks = {
            "grammar": self.scorers["grammar"].score_batch(exercises, request, semaphore) if "grammar" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            "coherence": self.scorers["coherence"].score_batch(exercises, request, semaphore) if "coherence" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            "cefr": self.scorers["cefr"].score_batch(exercises, request, semaphore) if "cefr" in self.scorers else asyncio.sleep(0, [(0.0, [])] * len(exercises)),
            "fluency": self.scorers["fluency"].score_batch(exercises, request, semaphore) if "fluency" in self.scorers and self.scorers["fluency"].use_llm else asyncio.sleep(0, [(10.0, [])] * len(exercises)),
        }

        # Add timeout protection
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*io_tasks.values(), return_exceptions=True),
                timeout=90.0
            )
        except asyncio.TimeoutError:
            results = [[(5.0, [f"{name} timed out"])] * len(exercises) for name in io_tasks.keys()]

        results_map = dict(zip(io_tasks.keys(), results))

        def get_io_results(name):
            result = results_map.get(name)
            if isinstance(result, Exception):
                raise result
            return result

        grammar_results = get_io_results("grammar")
        coherence_results = get_io_results("coherence")
        cefr_results = get_io_results("cefr")
        fluency_results = get_io_results("fluency")

        # Combine all results for each exercise
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
        return avg_score, results
