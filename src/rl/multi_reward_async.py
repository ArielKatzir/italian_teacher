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

from typing import List, Dict



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
        use_openai=True,
    ):
        self.reward_fn = reward_fn
        self.use_openai = use_openai

        if use_openai and os.environ.get("OPENAI_API_KEY"):
            # The semaphore is now managed inside ExerciseRewardFunction
            pass
        else:
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

        # --- Step 2: Score all completions in parallel (semaphore controls concurrency) ---
        print(f"⏳ Step 2/3: Scoring {len(parsed_data)} completions with batched reward function...")

        # With increased connection pool size, we can process all completions at once
        # The semaphore limits concurrent API calls to stay within comfortable limits
        # Gemini supports 4000 RPM, so high concurrency is fine
        score_tasks = []
        for exercises, req, success, _ in parsed_data:
            if success and exercises:
                score_tasks.append(self.reward_fn.score_exercises(exercises, req, self.reward_fn.semaphore))
            else:
                # For failed parses, create a dummy task that returns a penalty score.
                async def dummy_task():
                    return 0.0, []
                score_tasks.append(dummy_task())

        # Process all tasks (semaphore throttles to safe concurrency level)
        all_results = await asyncio.gather(*score_tasks)

        # Log model usage stats from the handler
        # Access the shared llm_handler directly from the reward_fn_instance
        if hasattr(self.reward_fn, 'llm_handler') and self.reward_fn.llm_handler:
            self.reward_fn.llm_handler.log_stats() # Call the method on the handler instance


        # --- Step 3: Compute CPU-bound scores and aggregate all results ---
        print(f"⏳ Step 3/3: Computing CPU-bound rewards and aggregating results...")
        for i in tqdm(range(len(parsed_data)), desc="Aggregating Rewards", leave=False):
            exercises, req, success, _ = parsed_data[i]
            avg_score, individual_results = all_results[i]

            if not success or not individual_results:
                all_rewards.append(0.0)
                continue

            # Use the final, authoritative score calculated by ExerciseRewardFunction
            all_rewards.append(avg_score)

            # Update stats for logging
            # This part is for logging only and does not affect the reward.
            # Helper to safely calculate average normalized score for a component
            def get_avg_norm_score(component_name: str, attribute_name: str):
                scorer = self.reward_fn.scorers.get(component_name)
                if not scorer or not individual_results: # Check if scorer is active and results exist
                    return 0.0
                
                total_score = sum(getattr(b, attribute_name) for _, b in individual_results)
                max_possible_score = len(individual_results) * scorer.max_score
                return total_score / max_possible_score if max_possible_score > 0 else 0.0

            # Use the helper to safely populate stats
            stats["grammar"].append(get_avg_norm_score("grammar", "grammar_correctness"))
            stats["coherence"].append(get_avg_norm_score("coherence", "coherence"))
            stats["topic"].append(get_avg_norm_score("topic", "topic_adherence"))
            stats["quality"].append(get_avg_norm_score("quality", "exercise_quality"))
            
            # Diversity is calculated differently
            stats["diversity"].append(self._compute_diversity_score(exercises, req))

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
        except json.JSONDecodeError:
            try:
                data = json5.loads(json_str)
            except Exception as e:
                # If both parsers fail, raise an error to be caught upstream
                raise ValueError(
                    f"Failed to parse JSON with both standard and json5 parsers. Error: {e}"
                )

        # Post-parsing validation: Ensure it's a list of dictionaries
        if not isinstance(data, list):
            raise ValueError("Parsed JSON is not a list.")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("Parsed list contains non-dictionary items.")
        
        return data

    def _log(self, stats, rewards, elapsed=None):
        if elapsed:
            print(f"\n🎯 Reward calculation complete ({elapsed:.1f}s):")
        else:
            print(f"\n🎯 Reward calculation complete:")

        for name, vals in stats.items():
            if vals:
                print(
                    # Scale individual stats to be on a 0-100 scale for consistent logging with TOTAL
                    f"   {name.capitalize():10s}: min={min(vals)*100:.1f}, max={max(vals)*100:.1f}, avg={(sum(vals)/len(vals))*100:.1f}"
                )
        if rewards:
            print(
                f"   {'TOTAL':10s}: min={min(rewards):.3f}, max={max(rewards):.3f}, avg={sum(rewards)/len(rewards):.3f}"
            )


def create_async_multi_reward(
    reward_fn,
    use_openai=True, # This parameter is now used to configure the reward_fn itself
):
    """
    Create async multi-reward with parallel OpenAI.

    Args:
        reward_fn: ExerciseRewardFunction instance

    Performance:
        - With OpenAI batch=20: ~1.7 hours for 1000 samples
        - Without OpenAI: ~30 min for 1000 samples

    Returns:
        TRL-compatible reward function
    """
    return AsyncMultiReward(
        reward_fn=reward_fn,
        use_openai=use_openai,
    )
