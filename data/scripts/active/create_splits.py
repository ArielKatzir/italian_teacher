#!/usr/bin/env python3
"""
Create train/validation/test splits with stratification by CEFR level.
- Train: 80%
- Validation: 10%
- Test: 10%
"""

import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data by level to maintain level distribution."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01

    # Group by level
    by_level = defaultdict(list)
    for item in data:
        level = item["metadata"]["cefr_level"]
        by_level[level].append(item)

    # Split each level
    train, val, test = [], [], []

    random.seed(42)

    for level, items in by_level.items():
        random.shuffle(items)
        n = len(items)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train.extend(items[:train_end])
        val.extend(items[train_end:val_end])
        test.extend(items[val_end:])

        print(
            f"  {level}: {len(items):3} → Train: {len(items[:train_end]):3}, Val: {len(items[train_end:val_end]):3}, Test: {len(items[val_end:]):3}"
        )

    # Shuffle again
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "datasets" / "v4" / "balanced_training_dataset.jsonl"
    output_dir = base_dir / "datasets" / "v4"

    # Load balanced data
    print("📚 Loading balanced dataset...")
    with open(input_file) as f:
        data = [json.loads(line) for line in f]

    print(f"Total examples: {len(data)}\n")

    # Create splits
    print("📊 Creating stratified splits (80/10/10)...")
    train, val, test = stratified_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Save splits
    splits = {"train": train, "validation": val, "test": test}

    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.jsonl"
        print(f"\n💾 Saving {split_name} split ({len(split_data)} examples)...")

        with open(output_file, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"   ✅ Saved to: {output_file}")
        print(f"   📁 Size: {output_file.stat().st_size / 1024:.1f} KB")

    # Print final statistics
    print("\n" + "=" * 70)
    print("📊 FINAL SPLIT STATISTICS")
    print("=" * 70)

    for split_name, split_data in splits.items():
        # Count exercises per level
        level_stats = defaultdict(int)
        total_exercises = 0

        for item in split_data:
            level = item["metadata"]["cefr_level"]
            exercises = json.loads(item["messages"][2]["content"])
            level_stats[level] += len(exercises)
            total_exercises += len(exercises)

        print(f"\n{split_name.upper()} ({len(split_data)} examples, {total_exercises} exercises):")
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            if level in level_stats:
                pct = level_stats[level] / total_exercises * 100
                print(f"  {level}: {level_stats[level]:4} exercises ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("✅ All splits created successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
