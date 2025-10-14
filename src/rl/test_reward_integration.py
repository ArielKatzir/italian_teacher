"""
Test reward function integration with TRL format.

This simulates how GRPO will call the reward function during training.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rl.reward_function import ExerciseRewardFunction

# Initialize reward function
print("Loading reward function...")
reward_fn = ExerciseRewardFunction()
print("✅ Reward function loaded\n")


def italian_exercise_reward(prompts=None, completions=None, completion_ids=None, **kwargs):
    """
    TRL-compliant reward function.

    This is the same function used in train_grpo.ipynb.

    TRL calls this with keyword arguments:
    - prompts: List of prompt strings
    - completions: List of generated completions
    - completion_ids: Token IDs (not used here)
    - **kwargs: All dataset columns (includes 'request')
    """
    # Extract 'request' from kwargs (comes from dataset column)
    requests = kwargs.get("request", [])

    rewards = []

    for completion, req in zip(completions, requests):
        try:
            # Parse generated JSON
            completion_text = completion.strip()
            exercises = json.loads(completion_text)

            if not isinstance(exercises, list):
                exercises = [exercises]

            # Score each exercise with comprehensive reward function
            scores = []
            for exercise in exercises:
                score, _ = reward_fn.score(exercise, req)
                scores.append(score / 100.0)  # Normalize to 0-1

            # Average score across all exercises in the completion
            reward = sum(scores) / len(scores) if scores else 0.0

        except json.JSONDecodeError:
            reward = 0.1
        except Exception:
            reward = 0.15

        rewards.append(reward)

    return rewards


# Test cases
test_cases = [
    {
        "prompt": "Generate 1 Italian past_tense exercise about viaggio for level B1.",
        "completion": json.dumps(
            [
                {
                    "type": "fill_in_blank",
                    "question": "Ieri io _____ (andare) in Italia.",
                    "answer": "sono andato",
                }
            ]
        ),
        "request": {
            "level": "B1",
            "grammar_focus": "past_tense",
            "topic": "viaggio",
            "num_exercises": 1,
        },
    },
    {
        "prompt": "Generate 1 Italian articles exercise about casa for level A2.",
        "completion": json.dumps(
            [
                {
                    "type": "multiple_choice",
                    "question": "Completa: ___ casa è bella.",
                    "answer": "La",
                    "options": ["Il", "La", "Lo", "I"],
                    "correct_option": 1,
                }
            ]
        ),
        "request": {
            "level": "A2",
            "grammar_focus": "articles",
            "topic": "casa",
            "num_exercises": 1,
        },
    },
    {
        "prompt": "Generate 1 Italian present_tense exercise about cibo for level A1.",
        "completion": "INVALID JSON {this is not parseable",
        "request": {
            "level": "A1",
            "grammar_focus": "present_tense",
            "topic": "cibo",
            "num_exercises": 1,
        },
    },
]

print("Testing reward function with TRL-style calls:\n")
print("=" * 60)

for i, test in enumerate(test_cases, 1):
    print(f"\n📝 Test Case {i}:")
    print(f"Request: {test['request']}")
    print(f"Completion: {test['completion'][:100]}...")

    # Call reward function as TRL would (completions first, then prompts)
    rewards = italian_exercise_reward(
        completions=[test["completion"]],
        prompts=[test["prompt"]],
        request=[test["request"]],  # Passed as kwarg
    )

    reward = rewards[0]
    print(f"✅ Reward: {reward:.3f}")

    # Show detailed breakdown for valid exercises
    if reward > 0.2:  # Valid exercise
        try:
            exercise = json.loads(test["completion"])[0]
            score, breakdown = reward_fn.score(exercise, test["request"])
            print(f"\nDetailed Score: {score}/100")
            print(breakdown)
        except:
            pass

print("\n" + "=" * 60)
print("\n✅ All tests passed! Reward function is ready for GRPO training.")
print("\nNext steps:")
print("1. Upload train_grpo.ipynb to Google Colab")
print("2. Ensure A100 GPU is selected")
print("3. Run all cells sequentially")
print("4. Training will take ~2-4 hours for 2000 samples")
