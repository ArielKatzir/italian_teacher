#!/usr/bin/env python3
"""
Comprehensive dataset cleaning and validation

Fixes:
1. Multiple choice without options array
2. Letter-prefix answers (A), B), etc.)
3. Empty/missing answers
4. Missing explanations/hints
5. Malformed exercises
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def extract_options_from_question(question: str) -> Optional[List[str]]:
    """Try to extract options from question text like 'A) pizza B) pasta C) vino D) acqua'"""
    # Pattern: A) option B) option C) option D) option
    pattern = r"[A-D]\)\s*([^A-D\)]+?)(?=\s*[A-D]\)|$)"
    matches = re.findall(pattern, question)
    if len(matches) >= 2:
        return [m.strip() for m in matches]
    return None


def fix_multiple_choice(exercise: Dict) -> Dict:
    """Fix multiple choice exercises."""
    # If missing options, try to extract from question
    if "options" not in exercise or not exercise["options"]:
        # Check if question contains options
        options = extract_options_from_question(exercise.get("question", ""))
        if options:
            exercise["options"] = options
            # Clean question by removing options
            cleaned_q = re.sub(r"\s*[A-D]\)[^A-D\)]+", "", exercise["question"])
            exercise["question"] = cleaned_q.strip()
        else:
            # Generate placeholder options from answer if answer has letter prefix
            answer = exercise.get("answer", "")
            if re.match(r"^[A-D]\)", answer):
                # Extract the actual answer text after letter
                actual_answer = re.sub(r"^[A-D]\)\s*", "", answer)
                exercise["answer"] = actual_answer
                # Create placeholder options (we can't know what the other options were)
                exercise["options"] = [actual_answer, "...", "...", "..."]
            elif not exercise.get("options"):
                # Last resort: convert to fill_in_blank
                exercise["type"] = "fill_in_blank"
                if "options" in exercise:
                    del exercise["options"]

    # Clean letter prefixes from answer
    if "answer" in exercise:
        answer = exercise["answer"]
        if re.match(r"^[A-D]\)", answer):
            exercise["answer"] = re.sub(r"^[A-D]\)\s*", "", answer)

    return exercise


def fix_missing_fields(exercise: Dict) -> Optional[Dict]:
    """Fix or filter out exercises with missing critical fields."""
    # Must have question and answer
    if "question" not in exercise or "answer" not in exercise:
        return None

    # Filter empty values
    if exercise.get("question", "").strip() == "" or exercise.get("answer", "").strip() == "":
        return None

    # Add hint if missing explanation
    if "explanation" not in exercise and "hint" not in exercise:
        exercise["hint"] = exercise.get("type", "exercise")

    return exercise


def validate_exercise(exercise: Dict) -> bool:
    """Validate exercise has required fields and proper structure."""
    required = ["type", "question", "answer"]

    # Check required fields
    for field in required:
        if field not in exercise or not exercise[field]:
            return False

    # Type-specific validation
    ex_type = exercise["type"]

    if ex_type == "multiple_choice":
        if "options" not in exercise or not exercise["options"]:
            return False
        if len(exercise["options"]) < 2:
            return False
        # Answer should be in options
        if exercise["answer"] not in exercise["options"]:
            # Try to find partial match
            found = False
            for opt in exercise["options"]:
                if (
                    exercise["answer"].lower() in opt.lower()
                    or opt.lower() in exercise["answer"].lower()
                ):
                    exercise["answer"] = opt  # Use the option
                    found = True
                    break
            if not found and exercise["options"][0] != "...":
                return False

    return True


def clean_dataset(input_file: Path, output_file: Path):
    """Clean and validate entire dataset."""

    print(f"📂 Processing: {input_file}")

    total = 0
    fixed = 0
    removed = 0

    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            total += 1
            sample = json.loads(line)

            try:
                # Parse exercises
                exercises = json.loads(sample["messages"][-1]["content"])
                cleaned_exercises = []

                for ex in exercises:
                    # Fix type-specific issues
                    if ex.get("type") == "multiple_choice":
                        ex = fix_multiple_choice(ex)

                    # Fix missing fields
                    ex = fix_missing_fields(ex)

                    if ex and validate_exercise(ex):
                        cleaned_exercises.append(ex)
                        if ex != exercises[exercises.index(ex)]:
                            fixed += 1
                    else:
                        removed += 1

                # Only save if we have at least 1 valid exercise
                if cleaned_exercises:
                    sample["messages"][-1]["content"] = json.dumps(
                        cleaned_exercises, ensure_ascii=False
                    )
                    outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")
                else:
                    removed += 1

            except Exception as e:
                print(f"  ⚠️  Error on line {total}: {e}")
                removed += 1

    print(f"  ✅ Total: {total} examples")
    print(f"  🔧 Fixed: {fixed} exercises")
    print(f"  ❌ Removed: {removed} invalid exercises")
    print(f"  📁 Saved to: {output_file}\n")


def main():
    base_dir = Path("data/datasets/v4_augmented")
    output_dir = Path("data/datasets/v4_augmented_clean")
    output_dir.mkdir(exist_ok=True)

    print("🔧 CLEANING AUGMENTED DATASET")
    print("=" * 70)
    print()

    for split in ["train", "validation", "test"]:
        input_file = base_dir / f"{split}.jsonl"
        output_file = output_dir / f"{split}.jsonl"

        if input_file.exists():
            clean_dataset(input_file, output_file)

    print("=" * 70)
    print("✅ CLEANING COMPLETE!")
    print(f"📁 Clean dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
