"""
Iterative GRPO training with error-focused refinement.

Round 1: Train on all data
Evaluate: Find weak areas
Round 2: Train on errors + some good examples (to prevent forgetting)
"""

import json
from typing import Dict, List, Tuple


def evaluate_model_on_requests(
    model, tokenizer, reward_fn, requests: List[Dict], output_path: str = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Evaluate model and separate good vs bad exercises.

    Returns:
        (low_scoring_requests, high_scoring_requests)
    """
    print(f"📊 Evaluating model on {len(requests)} requests...")

    low_scoring = []  # Score < 70
    high_scoring = []  # Score >= 70

    for i, request in enumerate(requests):
        # Generate exercise
        prompt = format_prompt(request)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        try:
            # Parse and score
            exercises = json.loads(generated_text)
            if not isinstance(exercises, list):
                exercises = [exercises]

            scores = []
            for exercise in exercises:
                score, _ = reward_fn.score(exercise, request)
                scores.append(score)

            avg_score = sum(scores) / len(scores)

            # Categorize
            if avg_score < 70:
                low_scoring.append(
                    {"request": request, "generated": generated_text, "score": avg_score}
                )
            else:
                high_scoring.append(
                    {"request": request, "generated": generated_text, "score": avg_score}
                )

        except Exception as e:
            # Invalid JSON - definitely low scoring
            low_scoring.append(
                {"request": request, "generated": generated_text, "score": 10.0, "error": str(e)}
            )

        if (i + 1) % 50 == 0:
            print(f"  Evaluated {i+1}/{len(requests)}")

    print(f"\n✅ Evaluation complete:")
    print(f"  Low scoring (< 70): {len(low_scoring)}")
    print(f"  High scoring (>= 70): {len(high_scoring)}")

    # Save results
    if output_path:
        results = {
            "low_scoring": low_scoring,
            "high_scoring": high_scoring,
            "stats": {
                "total": len(requests),
                "low_count": len(low_scoring),
                "high_count": len(high_scoring),
                "avg_score": sum(r["score"] for r in low_scoring + high_scoring) / len(requests),
            },
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved results to {output_path}")

    return low_scoring, high_scoring


def create_round2_dataset(
    low_scoring: List[Dict], high_scoring: List[Dict], low_weight: float = 0.7
) -> List[Dict]:
    """
    Create Round 2 training dataset.

    Args:
        low_scoring: Exercises that scored < 70
        high_scoring: Exercises that scored >= 70
        low_weight: Proportion of low-scoring examples (0.7 = 70% errors, 30% good)

    Returns:
        Mixed dataset for Round 2
    """
    import random

    # Calculate counts
    total_size = len(low_scoring) + len(high_scoring)
    num_low = int(total_size * low_weight)
    num_high = total_size - num_low

    # Sample (with replacement if needed)
    selected_low = random.choices(low_scoring, k=min(num_low, len(low_scoring)))
    selected_high = random.choices(high_scoring, k=min(num_high, len(high_scoring)))

    # Combine and shuffle
    round2_data = selected_low + selected_high
    random.shuffle(round2_data)

    print(f"\n📚 Round 2 dataset created:")
    print(f"  Low-scoring examples: {len(selected_low)} ({low_weight*100:.0f}%)")
    print(f"  High-scoring examples: {len(selected_high)} ({(1-low_weight)*100:.0f}%)")
    print(f"  Total: {len(round2_data)}")

    return [r["request"] for r in round2_data]


def format_prompt(request: dict) -> str:
    """Format request as prompt for model."""
    return f"""Generate {request['num_exercises']} Italian {request['grammar_focus']} exercise(s) about {request['topic']} for level {request['level']}.

Output as JSON array with fields: type, question, answer (and options/correct_option for multiple choice)."""


# Example usage workflow
if __name__ == "__main__":
    print(
        """
╔══════════════════════════════════════════════════════════════════════════╗
║                    ITERATIVE GRPO TRAINING WORKFLOW                      ║
╚══════════════════════════════════════════════════════════════════════════╝

Round 1: Initial Training
-------------------------
1. Train on all 2000 requests (3 epochs)
2. Save model as "italian_v5_round1"

Evaluation Phase
-----------------
3. Load Round 1 model
4. Generate exercises on validation set (500 requests)
5. Score each exercise with reward function
6. Separate: low-scoring (< 70) vs high-scoring (>= 70)

Round 2: Error-Focused Training
--------------------------------
7. Create mixed dataset:
   - 70% from low-scoring requests (errors to fix)
   - 30% from high-scoring requests (to prevent forgetting)
8. Train Round 1 model on this mixed dataset (2-3 epochs)
9. Save as "italian_v5_round2"

Final Result
------------
- Model learns from mistakes
- Doesn't forget what it learned in Round 1
- Higher average reward score!

═══════════════════════════════════════════════════════════════════════════

To use this in your training:

from src.rl.iterative_training import evaluate_model_on_requests, create_round2_dataset

# After Round 1 training:
low, high = evaluate_model_on_requests(
    model, tokenizer, reward_fn,
    validation_requests,
    output_path="evaluation_round1.json"
)

# Create Round 2 dataset
round2_requests = create_round2_dataset(low, high, low_weight=0.7)

# Train Round 2 (starting from Round 1 weights!)
trainer = GRPOTrainer(model=model_round1, ...)  # Same model, continues training
trainer.train()
"""
    )
