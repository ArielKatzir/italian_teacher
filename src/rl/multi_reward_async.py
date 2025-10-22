"""
Async Multi-Reward with Parallel OpenAI.

Uses batched async OpenAI calls for professional-grade quality
with acceptable training time (~1-2 hours for full GRPO round).
"""

import asyncio
import json
import os
import random
from typing import Dict

import httpx  # For httpx.TimeoutException
import json5
import torch
from openai import AsyncOpenAI


class AsyncMultiReward:
    """
    Multi-reward with parallel OpenAI coherence checking.

    Performance: ~1.7 hours for 1000 samples with batch_size=20
    Quality: State-of-the-art (uses GPT-4o-mini for coherence)
    """

    __name__ = "AsyncMultiReward"

    def __init__(
        self,
        reward_fn,
        weights=None,
        use_openai=True,
        openai_batch_size=20,
        openai_timeout=60,  # New parameter for OpenAI client timeout
        soft_penalties=True,
    ):
        self.reward_fn = reward_fn
        self.use_openai = use_openai
        self.openai_batch_size = openai_batch_size
        self.soft_penalties = soft_penalties

        self.openai_timeout = openai_timeout  # Store timeout
        self.weights = weights or {
            "grammar": 2.0,
            "coherence": 2.5,
            "topic": 1.5,
            "quality": 1.0,
            "diversity": 0.5,
        }

        if use_openai and os.environ.get("OPENAI_API_KEY"):
            self.openai_client = AsyncOpenAI(
                api_key=os.environ["OPENAI_API_KEY"], timeout=self.openai_timeout
            )
            self.openai_semaphore = asyncio.Semaphore(self.openai_batch_size)
        else:
            self.openai_client = None
            print("⚠️  OpenAI API not available - using rule-based coherence")

    def __call__(
        self, prompts=None, completions=None, completion_ids=None, trainer_state=None, **kwargs
    ):
        """
        TRL-compatible reward function.
        """
        requests = kwargs.get("request", [])

        # Run the async processing from a sync method.
        # nest_asyncio is required for environments with a running event loop (like Jupyter/Colab).
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            print(
                "Warning: nest_asyncio not installed. This may cause issues in Jupyter notebooks."
            )

        rewards = asyncio.run(self._process_batch(completions, requests))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rewards

    async def _process_batch(self, completions, requests):
        """
        Process batch with parallel OpenAI calls.
        """
        import time

        from tqdm import tqdm

        start_time = time.time()

        all_rewards = []
        stats = {"grammar": [], "coherence": [], "topic": [], "quality": [], "diversity": []}

        # Parse all completions first (CPU-bound, sequential)
        print(f"\n⏳ Step 1/3: Parsing {len(completions)} JSON completions...")
        parsed_data = []
        invalid_exercise_count = 0
        for completion, req in tqdm(
            zip(completions, requests), total=len(completions), desc="Parsing JSON", leave=False
        ):
            try:
                exercises = self._parse_exercises(completion)
                # Check if any exercises are not dicts (malformed JSON)
                if exercises and not all(isinstance(ex, dict) for ex in exercises):
                    invalid_exercise_count += 1
                    if invalid_exercise_count == 1:
                        print(f"\n⚠️  WARNING: Parsed exercises contain non-dict items!")
                        print(f"   Types found: {[type(ex).__name__ for ex in exercises]}")
                        print(f"   First 200 chars of completion: '{completion[:200]}'")
                parsed_data.append((exercises, req, True, completion))  # Store completion for debug
            except Exception:
                parsed_data.append((None, req, False, completion))  # Store completion for debug

        # --- Step 2: Score all completions in parallel using the new batched architecture ---
        print(f"⏳ Step 2/3: Scoring {len(parsed_data)} completions with batched reward function...")
        score_tasks = []
        for exercises, req, success, _ in parsed_data:
            if success and exercises:
                # The new `score_exercises` method is highly efficient.
                # It handles both CPU and batched LLM scoring internally.
                score_tasks.append(self.reward_fn.score_exercises(exercises, req, self.openai_semaphore))
            else:
                # For failed parses, create a dummy task that returns a penalty score.
                async def dummy_task():
                    return 0.0, []
                score_tasks.append(dummy_task())

        # Run all scoring tasks concurrently
        all_results = await asyncio.gather(*score_tasks)

        # --- Step 3: Compute CPU-bound scores and aggregate all results ---
        print(f"⏳ Step 3/3: Computing CPU-bound rewards and aggregating results...")
        for i, (avg_score, individual_results) in enumerate(
            tqdm(parsed_data, desc="Aggregating Rewards", leave=False)
        ):
            exercises, req, success, _ = parsed_data[i]
            avg_score, individual_results = all_results[i]

            if not success or not individual_results:
                all_rewards.append(0.0)
                continue

            # Extract scores from the detailed breakdowns
            grammar_scores = [res[1].grammar_correctness / self.reward_fn.scorers['grammar'].max_score for res in individual_results]
            coherence_scores = [res[1].coherence / self.reward_fn.scorers['coherence'].max_score for res in individual_results]
            topic_scores = [res[1].topic_adherence / self.reward_fn.scorers['topic'].max_score for res in individual_results]

            # Re-calculate the composite quality score from the breakdown components
            quality_scores = []
            for _, breakdown in individual_results:
                ling = breakdown.linguistic_quality / self.reward_fn.scorers['linguistic'].max_score
                flu = breakdown.fluency / self.reward_fn.scorers['fluency'].max_score
                exq = breakdown.exercise_quality / self.reward_fn.scorers['quality'].max_score
                # The composite quality score combines multiple aspects
                qual = (ling * 0.4) + (flu * 0.3) + (exq * 0.3)
                if self.soft_penalties:
                    qual = max(0.1, qual)
                quality_scores.append(qual)

            # Compute CPU-bound scores (topic, quality, etc.)
            diversity_score = self._compute_diversity_score(exercises, req)

            # Aggregate all scores
            final_scores = {
                "grammar": sum(grammar_scores) / len(grammar_scores),
                "coherence": sum(coherence_scores) / len(coherence_scores),
                "topic": sum(topic_scores) / len(topic_scores),
                "quality": sum(quality_scores) / len(quality_scores),
                "diversity": diversity_score,
            }

            # Calculate final weighted reward
            total_reward = sum(final_scores[k] * self.weights[k] for k in final_scores)
            all_rewards.append(total_reward)

            # Update stats for logging
            for k, v in final_scores.items():
                stats[k].append(v)

        # Log with timing
        elapsed = time.time() - start_time
        self._log(stats, all_rewards, elapsed)
        return all_rewards

    def _compute_diversity_score(self, exercises: List[Dict], req: Dict) -> float:
        """
        Compute the diversity score for a set of exercises.
        """
        # Filter to only dict exercises (defensive check)
        valid_exercises = [ex for ex in exercises if isinstance(ex, dict)]

        if not valid_exercises:
            return 0.0  # No valid exercises to score

        # Create tasks to run the synchronous, CPU-heavy scorers in a thread pool
        # NOTE: Topic and Quality are now part of the main `score_exercises` call.
        # We only need to calculate diversity here.

        # Diversity (use valid_exercises)
        answer_diversity = 0.5
        if len(valid_exercises) >= 2:
            answers = [ex.get("correct_answer", "") for ex in valid_exercises]
            unique = len(set(answers))
            answer_diversity = unique / len(answers)

        # NEW: Check for exercise type diversity
        requested_types = set(req.get("exercise_types", []))
        generated_types = set(ex.get("type", "") for ex in valid_exercises)
        type_diversity = len(generated_types.intersection(requested_types)) / len(requested_types) if requested_types else 1.0

        # Combine diversity scores (e.g., 60% answer diversity, 40% type diversity)
        diversity = (answer_diversity * 0.6) + (type_diversity * 0.4)

        return diversity

    def _parse_exercises(self, text):
        """
        Fast JSON parser with ROUND 4 CRITICAL FIX: Extract JSON array from completion.

        Issue: TRL GRPOTrainer may include prompt text in completions
        Fix: Detect and extract only the JSON array portion
        """
        original_text = text

        # Find the start of the first JSON array `[`
        start_bracket = original_text.find("[")
        if start_bracket == -1:
            raise ValueError("No JSON array start bracket '[' found.")

        # Find the end of the last JSON array `]`
        end_bracket = original_text.rfind("]")
        if end_bracket == -1 or end_bracket < start_bracket:
            raise ValueError("No JSON array end bracket ']' found after start.")

        # Extract the potential JSON string
        json_str = original_text[start_bracket : end_bracket + 1]

        # Attempt to parse with standard json, then json5 as a fallback
        try:
            data = json.loads(json_str)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            try:
                data = json5.loads(json_str)
                return data if isinstance(data, list) else [data]
            except Exception as e:
                # If both parsers fail, raise an error to be caught upstream
                raise ValueError(
                    f"Failed to parse JSON with both standard and json5 parsers. Error: {e}"
                )

    def _log(self, stats, rewards, elapsed=None):
        if elapsed:
            print(
                f"\n🎯 Multi-Reward (Async OpenAI, batch={self.openai_batch_size}, {elapsed:.1f}s):"
            )
        else:
            print(f"\n🎯 Multi-Reward (Async OpenAI, batch={self.openai_batch_size}):")
        for name, vals in stats.items():
            if vals:
                w = self.weights.get(name, 1.0)
                print(
                    f"   {name.capitalize():10s}: min={min(vals):.3f}, max={max(vals):.3f}, avg={sum(vals)/len(vals):.3f} (weight={w})"
                )
        if rewards:
            print(
                f"   {'TOTAL':10s}: min={min(rewards):.3f}, max={max(rewards):.3f}, avg={sum(rewards)/len(rewards):.3f}"
            )


def create_async_multi_reward(
    reward_fn,
    use_openai=True,
    openai_batch_size=20,
    openai_timeout=60,  # New parameter
    soft_penalties=False,
):
    """
    Create async multi-reward with parallel OpenAI.

    Args:
        reward_fn: ExerciseRewardFunction instance
        use_openai: If True, uses OpenAI for coherence (requires API key)
        openai_batch_size: Batch size for parallel OpenAI calls (10-30)
        soft_penalties: If True, uses 0.1 instead of 0 for failures

    Performance:
        - With OpenAI batch=20: ~1.7 hours for 1000 samples
        - Without OpenAI: ~30 min for 1000 samples

    Returns:
        TRL-compatible reward function
    """
    return AsyncMultiReward(
        reward_fn=reward_fn,
        use_openai=use_openai,
        openai_batch_size=openai_batch_size,
        openai_timeout=openai_timeout,  # Pass timeout
        soft_penalties=soft_penalties,
    )
