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

    async def _openai_worker(self, name: str, queue: asyncio.Queue, results: dict):
        """A worker that processes items from the queue until it's empty."""
        while not queue.empty():
            try:
                # Get a work item from the queue
                work_item = queue.get_nowait()
                task_type, index, data = work_item

                if task_type == "coherence":
                    exercise, request = data
                    score = await self._check_coherence_async(
                        exercise, request, self.openai_semaphore
                    )
                    if "coherence" not in results[index]:
                        results[index]["coherence"] = []
                    results[index]["coherence"].append(score)

                elif task_type == "grammar":
                    exercise, request = data
                    # The grammar scorer is part of the modular reward function
                    score, _ = await self.reward_fn.scorers["grammar"].score(
                        exercise, request, self.openai_semaphore
                    )
                    if "grammar" not in results[index]:
                        results[index]["grammar"] = []
                    results[index]["grammar"].append(score / 10.0)  # Normalize

                # Mark the task as done
                queue.task_done()

            except asyncio.QueueEmpty:
                # Queue is empty, worker can exit
                break
            except Exception as e:
                print(f"  ⚠️ Worker {name} encountered an error: {e}")
                # Ensure task_done is called even on error to prevent deadlocks
                if "queue" in locals() and "work_item" in locals():
                    queue.task_done()
                break

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

        # --- PRODUCER-CONSUMER MODEL FOR ALL OPENAI CALLS ---
        # This is a robust pattern for managing high-concurrency I/O.

        # The queue will hold all work items (coherence and grammar checks)
        work_queue = asyncio.Queue()
        total_openai_tasks = 0

        # Dictionary to store results, indexed by the completion's position
        openai_results = [{} for _ in range(len(completions))]

        # --- PRODUCER: Add all OpenAI tasks to the queue ---
        if self.use_openai and self.openai_client:
            print(".. Populating work queue with OpenAI tasks (coherence, grammar)...")
            for i, (exercises, req, success, _) in enumerate(parsed_data):
                if success and exercises:
                    valid_exercises = [ex for ex in exercises if isinstance(ex, dict)]
                    if valid_exercises:
                        # Add coherence check for one random exercise
                        sampled_ex = random.choice(valid_exercises)
                        work_queue.put_nowait(("coherence", i, (sampled_ex, req)))
                        total_openai_tasks += 1

                        # Add grammar checks for ALL exercises
                        for ex in valid_exercises:
                            work_queue.put_nowait(("grammar", i, (ex, req)))
                            total_openai_tasks += 1

        # --- CONSUMERS: Create and run the worker pool ---
        if total_openai_tasks > 0:
            print(
                f"⏳ Step 2/3: Processing {total_openai_tasks} OpenAI tasks with {self.openai_batch_size} parallel workers..."
            )
            workers = [
                asyncio.create_task(self._openai_worker(f"worker-{i}", work_queue, openai_results))
                for i in range(self.openai_batch_size)
            ]
            # Wait for all items in the queue to be processed
            await work_queue.join()

            # Cancel the workers, as they are no longer needed
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

        # --- Step 3: Compute CPU-bound scores and aggregate all results ---
        print(f"⏳ Step 3/3: Computing CPU-bound rewards and aggregating results...")
        for i, (exercises, req, success, _) in enumerate(
            tqdm(parsed_data, desc="Aggregating Rewards", leave=False)
        ):
            if not success or not exercises:
                all_rewards.append(0.0)
                continue

            # Get the pre-computed OpenAI scores for this completion
            grammar_scores = openai_results[i].get("grammar", [0.5])  # Default to neutral score
            coherence_scores = openai_results[i].get("coherence", [0.7])  # Default to neutral score

            # Compute CPU-bound scores (topic, quality, etc.)
            # We no longer need to pass the semaphore here
            cpu_scores = await self._compute_cpu_components(exercises, req)

            # Aggregate all scores
            final_scores = {
                "grammar": sum(grammar_scores) / len(grammar_scores),
                "coherence": sum(coherence_scores) / len(coherence_scores),
                "topic": cpu_scores["topic"],
                "quality": cpu_scores["quality"],
                "diversity": cpu_scores["diversity"],
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

    async def _check_coherence_async(
        self, exercise: Dict, request: Dict, semaphore: asyncio.Semaphore
    ) -> float:
        """
        Async OpenAI coherence check.

        Returns score 0-1 (normalized).
        """
        # Defensive: handle case where exercise is not a dict
        if not isinstance(exercise, dict):
            # Invalid exercise format - return poor score
            return 0.0

        question = exercise.get("question", "")
        answer = exercise.get("correct_answer", exercise.get("answer", ""))
        topic = request.get("topic", "")

        prompt = f"""You are an Italian language expert. Check if this exercise makes sense:

Question: {question}
Answer: {answer}
Topic: {topic}

Rate coherence 0-10:
- 0 = nonsense/redundant (answer already in question)
- 5 = makes sense but weak
- 10 = perfect sense

Respond with ONLY a number 0-10."""

        async with semaphore:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10,
                )
                score = int(response.choices[0].message.content.strip())
                return max(0.0, min(1.0, score / 10.0))
            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                print(f"  ⚠️ LLM coherence check timed out: {e}. Falling back to rule-based.")
                return self._coherence_rule_based(exercise)  # Fallback to rule-based
            except Exception as e:
                print(
                    f"  ⚠️ LLM coherence check failed with unexpected error: {e}. Falling back to rule-based."
                )
                return self._coherence_rule_based(exercise)

    def _coherence_rule_based(self, exercise: Dict) -> float:
        """
        Rule-based coherence check (fallback).
        """
        question = exercise.get("question", "").lower()
        answer = exercise.get("correct_answer", exercise.get("answer", "")).lower()

        # Check redundancy (answer in question)
        if answer and answer in question:
            return 0.0 if not self.soft_penalties else 0.1

        # Check blank consistency
        if "___" not in question and exercise.get("type") == "fill_in_blank":
            return 0.0 if not self.soft_penalties else 0.2

        return 0.7  # Default: probably OK

    async def _compute_cpu_components(self, exercises, req):
        """
        Compute all CPU-bound reward components.
        This runs the non-OpenAI scorers in a thread pool.
        """

        # Filter to only dict exercises (defensive check)
        valid_exercises = [ex for ex in exercises if isinstance(ex, dict)]

        if not valid_exercises:
            return None  # No valid exercises to score

        # Create tasks to run the synchronous, CPU-heavy scorers in a thread pool
        score_tasks = [self.reward_fn.score_cpu_only(ex, req) for ex in valid_exercises]

        # Run all scoring tasks in parallel
        results = await asyncio.gather(*score_tasks)

        topic_scores, quality_scores = [], []
        for _, breakdown in results:
            # Normalize to 0-1
            topic = breakdown.topic_adherence / 10.0

            # Apply soft minimum if enabled
            if self.soft_penalties:
                topic = max(0.1, topic)

            topic_scores.append(topic)

            # Quality (Round 4: updated normalization)
            ling = breakdown.linguistic_quality / 25.0  # Fixed: was /30, should be /25
            flu = breakdown.fluency / 10.0
            cefr = breakdown.cefr_alignment / 20.0
            # Added exercise quality (context validation - CRITICAL in Round 4)
            exq = breakdown.exercise_quality / 30.0
            qual = ling * 0.3 + flu * 0.2 + cefr * 0.2 + exq * 0.3  # Exercise quality now 30%
            if self.soft_penalties:
                qual = max(0.1, qual)
            quality_scores.append(qual)

        # Defensive: handle case where no valid exercises
        if not topic_scores:
            return {"topic": 0.0, "quality": 0.0, "diversity": 0.0}

        # Diversity (use valid_exercises)
        diversity = 0.5
        if len(valid_exercises) >= 2:
            answers = [ex.get("correct_answer", "") for ex in valid_exercises]
            unique = len(set(answers))
            diversity = unique / len(answers)

        return {
            "topic": sum(topic_scores) / len(topic_scores),
            "quality": sum(quality_scores) / len(quality_scores),
            "diversity": diversity,
        }

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
