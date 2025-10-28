"""
Validation callback for GRPO training.
Tests fixed samples at each checkpoint to track actual quality improvements.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from datetime import datetime
import time


class ValidationCallback(TrainerCallback):
    """
    Callback that evaluates model on fixed validation samples during checkpoint saves.
    Tracks reward improvements over training.
    """

    def __init__(
        self,
        validation_samples: List[Dict[str, Any]],
        validation_prompts: List[str],
        reward_function,
        tokenizer,
        output_dir: str,
        num_generations: int = 3,
    ):
        """
        Args:
            validation_samples: List of request dicts (from training_requests.json)
            validation_prompts: List of formatted prompts for each request
            reward_function: The reward function instance
            tokenizer: Model tokenizer
            output_dir: Directory to save validation results
            num_generations: Number of completions to generate per prompt
        """
        self.validation_samples = validation_samples
        self.validation_prompts = validation_prompts
        self.reward_function = reward_function
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.num_generations = num_generations

        # Create validation subdirectory
        self.validation_dir = self.output_dir / "validation_results"
        self.validation_dir.mkdir(parents=True, exist_ok=True)

        # Track results over time
        self.history = []

        print(f"\n📊 Validation Callback initialized:")
        print(f"   {len(validation_samples)} validation samples")
        print(f"   {num_generations} generations per sample")
        print(f"   Results will be saved to: {self.validation_dir}")

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Run validation when checkpoint is saved."""
        model = kwargs.get("model")
        if model is None:
            return control

        step = state.global_step
        print(f"\n{'='*80}")
        print(f"📊 Running validation at step {step}...")
        print(f"{'='*80}")

        # CRITICAL: Add cooldown before validation to let APIs recover from training
        print("⏳ Waiting 15 seconds for API cooldown before validation...")
        time.sleep(15)

        # Generate completions for validation samples
        all_completions = []
        all_rewards = []

        model.eval()

        # Save original padding side and set to left for generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        with torch.no_grad():
            for idx, (prompt, request) in enumerate(zip(self.validation_prompts, self.validation_samples)):
                print(f"\n  Generating for validation sample {idx+1}/{len(self.validation_prompts)}...")

                # Add pacing between samples to avoid API rate limits
                if idx > 0:
                    time.sleep(2)  # 2 second delay between samples

                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(model.device)

                # Generate multiple completions
                sample_completions = []
                sample_rewards = []

                for gen_idx in range(self.num_generations):
                    # Generate (using same settings as training for consistency)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=350,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        top_k=50,  # Add top_k for stable generation
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=128000,
                    )

                    # Decode
                    completion = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )

                    sample_completions.append(completion)

                # Debug: Print first completion to verify format
                if idx == 0:
                    print(f"\n  📝 Sample completion preview (first 200 chars):")
                    print(f"  {sample_completions[0][:200]}...")
                    print()

                # Score all completions for this prompt
                # FIXED: Call reward function with correct signature
                try:
                    # Reward function expects: prompts, completions, trainer_state
                    rewards = self.reward_function(
                        prompts=[prompt] * len(sample_completions),  # Repeat prompt for each completion
                        completions=sample_completions,              # Flat list of completions
                        trainer_state=state                          # Pass trainer state
                    )
                    sample_rewards = rewards if isinstance(rewards, list) else [rewards]
                except Exception as e:
                    print(f"    ⚠️  Error scoring sample {idx}: {e}")
                    import traceback
                    traceback.print_exc()  # Print full traceback for debugging
                    sample_rewards = [0.0] * len(sample_completions)

                all_completions.append({
                    "request": request,
                    "prompt": prompt,
                    "completions": sample_completions,
                    "rewards": sample_rewards,
                    "avg_reward": sum(sample_rewards) / len(sample_rewards) if sample_rewards else 0.0,
                    "max_reward": max(sample_rewards) if sample_rewards else 0.0,
                    "min_reward": min(sample_rewards) if sample_rewards else 0.0,
                })
                all_rewards.extend(sample_rewards)

                print(f"    Rewards: avg={all_completions[-1]['avg_reward']:.2f}, "
                      f"max={all_completions[-1]['max_reward']:.2f}, "
                      f"min={all_completions[-1]['min_reward']:.2f}")

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        model.train()

        # Calculate aggregate statistics
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        max_reward = max(all_rewards) if all_rewards else 0.0
        min_reward = min(all_rewards) if all_rewards else 0.0

        # Save results
        result = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "aggregate_stats": {
                "avg_reward": avg_reward,
                "max_reward": max_reward,
                "min_reward": min_reward,
                "num_samples": len(self.validation_samples),
                "num_generations": self.num_generations,
                "total_completions": len(all_rewards),
            },
            "per_sample_results": all_completions,
        }

        # Save to JSON
        result_file = self.validation_dir / f"validation_step_{step}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Update history
        self.history.append({
            "step": step,
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
        })

        # Save history
        history_file = self.validation_dir / "validation_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # Print summary
        print(f"\n📊 Validation Results at Step {step}:")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Max Reward:     {max_reward:.2f}")
        print(f"   Min Reward:     {min_reward:.2f}")
        print(f"   Saved to: {result_file}")

        # Show progress over time if we have history
        if len(self.history) > 1:
            prev = self.history[-2]
            delta = avg_reward - prev["avg_reward"]
            emoji = "📈" if delta > 0 else "📉" if delta < 0 else "➡️"
            print(f"   {emoji} Change from step {prev['step']}: {delta:+.2f}")

        print(f"{'='*80}\n")

        return control


def select_validation_samples(
    training_requests_path: str,
    num_samples: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Select diverse validation samples from training data.

    Tries to pick:
    - Different exercise types (fill_in_blank, multiple_choice, translation, etc.)
    - Different CEFR levels
    - Different grammar topics

    Args:
        training_requests_path: Path to training_requests.json
        num_samples: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        List of request dicts
    """
    import random

    print(f"\n📊 Selecting {num_samples} validation samples from {training_requests_path}...")

    with open(training_requests_path, 'r') as f:
        all_requests = json.load(f)

    print(f"   Total requests available: {len(all_requests)}")

    # Try to get diverse samples by exercise type
    by_type = {}
    for req in all_requests:
        ex_type = req.get("exercise_type", "unknown")
        if ex_type not in by_type:
            by_type[ex_type] = []
        by_type[ex_type].append(req)

    print(f"   Exercise types: {', '.join(by_type.keys())}")

    # Sample from each type proportionally
    random.seed(seed)
    selected = []
    samples_per_type = max(1, num_samples // len(by_type))

    for ex_type, requests in by_type.items():
        n = min(samples_per_type, len(requests))
        selected.extend(random.sample(requests, n))

    # Fill remaining slots randomly
    while len(selected) < num_samples and len(selected) < len(all_requests):
        candidate = random.choice(all_requests)
        if candidate not in selected:
            selected.append(candidate)

    print(f"   ✅ Selected {len(selected)} diverse samples")

    # Print distribution
    type_counts = {}
    for req in selected:
        ex_type = req.get("exercise_type", "unknown")
        type_counts[ex_type] = type_counts.get(ex_type, 0) + 1

    print(f"   Distribution: {dict(type_counts)}")

    return selected[:num_samples]
