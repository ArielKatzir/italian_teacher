#!/usr/bin/env python3
"""
Downsample A2 level to balance the dataset by:
1. Reducing total A2 exercises to ~800 (from 3719)
2. Balancing exercise types within A2
"""

import json
import random
from collections import defaultdict
from pathlib import Path


def downsample_by_type(data, target_total):
    """
    Downsample while maintaining type diversity.
    Strategy: Sample proportionally from each type, but cap at max per type.
    """
    # Group by exercise type
    by_type = defaultdict(list)
    for item in data:
        ex_type = item["metadata"].get("exercise_type", "unknown")
        by_type[ex_type].append(item)

    # Calculate target per type (proportional but capped)
    type_counts = {t: len(items) for t, items in by_type.items()}
    total_current = sum(type_counts.values())

    # Proportional allocation
    target_per_type = {}
    for ex_type, count in type_counts.items():
        proportion = count / total_current
        target = int(target_total * proportion)
        # Cap at 150 examples per type to prevent any single type dominating
        target_per_type[ex_type] = min(target, 150, count)

    # Adjust if we're under target (distribute remainder)
    current_total = sum(target_per_type.values())
    if current_total < target_total:
        # Add more to types that have room
        remainder = target_total - current_total
        for ex_type in sorted(type_counts.keys(), key=lambda t: type_counts[t], reverse=True):
            if target_per_type[ex_type] < type_counts[ex_type] and remainder > 0:
                add = min(remainder, type_counts[ex_type] - target_per_type[ex_type], 20)
                target_per_type[ex_type] += add
                remainder -= add

    # Sample from each type
    sampled = []
    random.seed(42)  # Reproducible

    for ex_type, items in by_type.items():
        target = target_per_type[ex_type]
        if len(items) <= target:
            sampled.extend(items)
        else:
            sampled.extend(random.sample(items, target))

    print(f"\n📊 Downsampling Summary:")
    print(f"{'Type':<20} | {'Before':>8} | {'After':>8} | {'Change':>8}")
    print("-" * 60)
    for ex_type in sorted(type_counts.keys()):
        before = type_counts[ex_type]
        after = target_per_type[ex_type]
        change = f"{(after-before):+d}"
        print(f"{ex_type:<20} | {before:>8} | {after:>8} | {change:>8}")
    print("-" * 60)
    print(
        f"{'TOTAL':<20} | {total_current:>8} | {len(sampled):>8} | {len(sampled)-total_current:+8d}"
    )

    return sampled


def main():
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "datasets" / "v4" / "final_training_dataset.jsonl"
    output_file = base_dir / "datasets" / "v4" / "balanced_training_dataset.jsonl"

    # Load data
    print("📚 Loading dataset...")
    with open(input_file) as f:
        data = [json.loads(line) for line in f]

    # Separate by level
    by_level = defaultdict(list)
    for item in data:
        level = item["metadata"]["cefr_level"]
        by_level[level].append(item)

    print(f"\n📊 Current distribution:")
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        print(f"  {level}: {len(by_level[level])} examples")

    # Downsample A2
    print(f"\n🎯 Downsampling A2 from {len(by_level['A2'])} to ~150 examples...")
    a2_sampled = downsample_by_type(by_level["A2"], target_total=150)

    # Combine all levels
    balanced_data = []
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        if level == "A2":
            balanced_data.extend(a2_sampled)
        else:
            balanced_data.extend(by_level[level])

    # Shuffle
    random.seed(42)
    random.shuffle(balanced_data)

    # Save
    print(f"\n💾 Saving balanced dataset to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Final stats
    print(f"\n" + "=" * 70)
    print("📊 FINAL BALANCED DATASET")
    print("=" * 70)

    final_by_level = defaultdict(lambda: {"examples": 0, "exercises": 0})
    for item in balanced_data:
        level = item["metadata"]["cefr_level"]
        final_by_level[level]["examples"] += 1
        exercises = json.loads(item["messages"][2]["content"])
        final_by_level[level]["exercises"] += len(exercises)

    print(f"{'Level':<6} | {'Examples':>10} | {'Exercises':>12} | {'% of Total':>12}")
    print("-" * 70)

    total_ex = sum(s["examples"] for s in final_by_level.values())
    total_exs = sum(s["exercises"] for s in final_by_level.values())

    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        s = final_by_level[level]
        pct = (s["exercises"] / total_exs * 100) if total_exs > 0 else 0
        print(f"{level:<6} | {s['examples']:>10} | {s['exercises']:>12} | {pct:>11.1f}%")

    print("=" * 70)
    print(f"{'TOTAL':<6} | {total_ex:>10} | {total_exs:>12} | {100:>11.1f}%")
    print("=" * 70)

    print(f"\n✅ Balanced dataset saved!")
    print(f"📁 File size: {output_file.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
