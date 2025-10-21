#!/usr/bin/env python3
"""
Merge all extracted datasets into final training format.
- Textbook exercises (Units 1-23)
- OnlineItalianClub exercises (A1-C2)
"""

import json
from pathlib import Path


def convert_onlineitalianclub_to_training_format(oic_data, level):
    """Convert OIC exercises to training format."""
    training_examples = []

    for item in oic_data:
        exercises = item["exercises"]
        exercise_type = item["exercise_type"]

        # Group exercises by type and create training examples
        # Convert to assistant JSON format
        json_exercises = []
        for ex in exercises:
            exercise_dict = {
                "type": exercise_type,
                "question": ex["question"],
                "answer": ex["answer"],
                "hint": item["metadata"].get("grammar_focus", "grammar"),
            }

            # Add options for multiple choice
            if "options" in ex:
                exercise_dict["options"] = ex["options"]

            json_exercises.append(exercise_dict)

        # Create training example
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Italian language teacher. Generate high-quality exercises based on the assignment specification. Output exercises in JSON format.",
                },
                {
                    "role": "user",
                    "content": f"Generate {len(json_exercises)} exercises:\nCEFR Level: {level}\nGrammar Focus: {item['metadata'].get('grammar_focus', 'grammar')}\nExercise Types: {exercise_type}",
                },
                {"role": "assistant", "content": json.dumps(json_exercises, ensure_ascii=False)},
            ],
            "metadata": {
                "cefr_level": level,
                "grammar_focus": item["metadata"].get("grammar_focus", "grammar"),
                "num_exercises": len(json_exercises),
                "source": "onlineitalianclub.com",
                "exercise_type": exercise_type,
                "filename": item["metadata"]["filename"],
            },
        }

        training_examples.append(training_example)

    return training_examples


def main():
    base_dir = Path(__file__).parent.parent
    output_file = base_dir / "datasets" / "v4" / "final_training_dataset.jsonl"

    all_examples = []
    stats = {"textbook": 0, "onlineitalianclub": {}, "total": 0}

    # Load textbook data (already in correct format)
    print("📚 Loading textbook exercises...")
    textbook_file = base_dir / "datasets" / "v4" / "exercise_generation_train.jsonl"
    with open(textbook_file) as f:
        for line in f:
            all_examples.append(json.loads(line))
            stats["textbook"] += 1

    print(f"   ✅ Loaded {stats['textbook']} textbook exercises")

    # Load OnlineItalianClub data for each level
    levels = ["A1", "A2", "B1", "B2", "C1", "C2"]

    for level in levels:
        print(f"🌐 Loading {level} exercises from OnlineItalianClub...")
        oic_file = (
            base_dir / "datasets" / "v4" / f"onlineitalianclub_{level.lower()}_extracted.json"
        )

        with open(oic_file) as f:
            oic_data = json.load(f)

        # Convert to training format
        training_examples = convert_onlineitalianclub_to_training_format(oic_data, level)
        all_examples.extend(training_examples)

        stats["onlineitalianclub"][level] = len(training_examples)
        print(f"   ✅ Converted {len(training_examples)} {level} exercise sets")

    # Save merged dataset
    print(f"\n💾 Saving merged dataset to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    stats["total"] = len(all_examples)

    # Print summary
    print("\n" + "=" * 70)
    print("📊 FINAL DATASET SUMMARY")
    print("=" * 70)
    print(f"Textbook (Units 1-23):     {stats['textbook']:4} examples")
    print(f"OnlineItalianClub:")
    for level in levels:
        print(f"  - {level}:                     {stats['onlineitalianclub'][level]:4} examples")
    print("-" * 70)
    print(f"TOTAL TRAINING EXAMPLES:   {stats['total']:4}")
    print("=" * 70)
    print(f"\n✅ Merged dataset saved to: {output_file}")
    print(f"📁 File size: {output_file.stat().st_size / 1024:.1f} KB")

    # Save stats
    stats_file = base_dir / "datasets" / "v4" / "final_dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"📊 Stats saved to: {stats_file}")


if __name__ == "__main__":
    main()
