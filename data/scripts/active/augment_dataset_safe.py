#!/usr/bin/env python3
"""
SAFE Dataset Augmentation - Saves incrementally to prevent data loss

Key improvements:
1. Saves every N examples (checkpoint)
2. Resume from checkpoint if crashed
3. Validates JSON before saving
4. Error handling with retry logic
"""

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Topic pools for variation
TOPICS = {
    "A1": [
        "food",
        "family",
        "colors",
        "numbers",
        "animals",
        "daily routine",
        "weather",
        "clothing",
    ],
    "A2": ["travel", "hobbies", "shopping", "health", "school", "work", "sports", "technology"],
    "B1": [
        "environment",
        "culture",
        "media",
        "relationships",
        "career",
        "education",
        "leisure",
        "society",
    ],
    "B2": [
        "politics",
        "economics",
        "science",
        "history",
        "art",
        "philosophy",
        "literature",
        "global issues",
    ],
    "C1": [
        "abstract concepts",
        "professional contexts",
        "academic discourse",
        "cultural analysis",
        "ethics",
        "innovation",
    ],
    "C2": [
        "nuanced argumentation",
        "specialized domains",
        "critical analysis",
        "theoretical concepts",
        "advanced rhetoric",
    ],
}

GRAMMAR_FOCUSES = {
    "A1": ["present_tense", "articles", "pronouns", "basic_adjectives", "simple_questions"],
    "A2": ["past_tense", "future_tense", "comparatives", "prepositions", "reflexive_verbs"],
    "B1": ["subjunctive", "conditional", "passive_voice", "relative_clauses", "conjunctions"],
    "B2": ["advanced_subjunctive", "indirect_speech", "complex_tenses", "participles", "gerunds"],
    "C1": [
        "subtle_distinctions",
        "stylistic_variations",
        "idiomatic_expressions",
        "formal_register",
    ],
    "C2": ["nuanced_meanings", "rhetorical_devices", "advanced_syntax", "literary_language"],
}


def load_dataset(file_path: Path) -> List[Dict]:
    """Load dataset from JSONL."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def append_to_dataset(example: Dict, file_path: Path):
    """Append single example to JSONL file (safe incremental save)."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


def validate_example(example: Dict) -> bool:
    """Validate example structure."""
    try:
        # Check required fields
        if "messages" not in example or "metadata" not in example:
            return False
        if not isinstance(example["messages"], list):
            return False
        if len(example["messages"]) < 3:  # system, user, assistant
            return False

        # Try to parse assistant response as JSON
        assistant_content = example["messages"][-1]["content"]
        json.loads(assistant_content)  # Should be valid JSON array

        return True
    except:
        return False


def create_variation_prompt(original_example: Dict) -> str:
    """Create prompt for generating exercise variations."""
    metadata = original_example["metadata"]
    messages = original_example["messages"]
    assistant_msg = messages[-1]["content"]

    return f"""Create a VARIATION of these Italian language exercises. Keep the same:
- CEFR level: {metadata['cefr_level']}
- Grammar focus: {metadata.get('grammar_focus', 'general')}
- Exercise types: {metadata.get('exercise_type', 'mixed')}
- Number of exercises: {metadata.get('num_exercises', 'same as original')}

IMPORTANT:
1. Use DIFFERENT vocabulary and context
2. Change the topic to something new but appropriate for the level
3. Maintain the SAME difficulty level and grammar structures
4. Output in the EXACT SAME JSON format

Original exercises:
{assistant_msg}

Generate the variation now (JSON only, no explanation):"""


def create_new_exercise_prompt(
    level: str, topic: str, grammar: str, ex_type: str, count: int
) -> str:
    """Create prompt for generating completely new exercises."""
    return f"""Generate {count} NEW Italian language exercises:

CEFR Level: {level}
Grammar Focus: {grammar}
Topic: {topic}
Exercise Type: {ex_type}

Requirements:
1. Appropriate difficulty for {level} level
2. Focus on {grammar}
3. Related to topic: {topic}
4. Type: {ex_type}
5. Include explanations and hints
6. For multiple_choice: provide 4 options
7. For translation: include alternative_answers if applicable

Output ONLY valid JSON array of exercises. Format:
[
  {{
    "type": "{ex_type}",
    "question": "...",
    "answer": "...",
    "explanation": "...",
    "hint": "..."
  }}
]"""


def generate_with_retry(func, max_retries: int = 3, **kwargs) -> Optional[Dict]:
    """Generate example with retry logic."""
    for attempt in range(max_retries):
        try:
            example = func(**kwargs)
            if validate_example(example):
                return example
            else:
                print(f"      ⚠️  Validation failed, retrying ({attempt + 1}/{max_retries})...")
        except Exception as e:
            print(f"      ⚠️  Error: {e}, retrying ({attempt + 1}/{max_retries})...")
            time.sleep(1)

    return None


def generate_variation(example: Dict, model: str = "gpt-4o-mini") -> Dict:
    """Generate a variation of an existing example using LLM."""
    prompt = create_variation_prompt(example)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Italian language teacher creating exercise variations.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
        max_tokens=1500,
    )

    new_exercises = response.choices[0].message.content.strip()

    new_example = {
        "messages": [
            example["messages"][0],
            {"role": "user", "content": example["messages"][1]["content"]},
            {"role": "assistant", "content": new_exercises},
        ],
        "metadata": {**example["metadata"], "source": "llm_variation"},
    }

    return new_example


def generate_new_exercise(
    level: str, topic: str, grammar: str, ex_type: str, count: int = 3, model: str = "gpt-4o-mini"
) -> Dict:
    """Generate completely new exercises using LLM."""
    prompt = create_new_exercise_prompt(level, topic, grammar, ex_type, count)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert Italian language teacher. Generate high-quality exercises based on the assignment specification. Output exercises in JSON format.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=1500,
    )

    exercises_json = response.choices[0].message.content.strip()
    user_prompt = f"Generate {count} exercises:\nCEFR Level: {level}\nGrammar Focus: {grammar}\nTopic: {topic}\nExercise Types: {ex_type}"

    example = {
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Italian language teacher. Generate high-quality exercises based on the assignment specification. Output exercises in JSON format.",
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": exercises_json},
        ],
        "metadata": {
            "cefr_level": level,
            "topic": topic,
            "num_exercises": count,
            "grammar_focus": grammar,
            "exercise_type": ex_type,
            "source": "llm_generated",
            "model": model,
        },
    }

    return example


def augment_dataset_safe(
    input_file: Path,
    output_file: Path,
    target_size: int = 4000,
    checkpoint_every: int = 10,
    model: str = "gpt-4o-mini",
):
    """Safe augmentation pipeline with incremental saves."""

    print(f"📊 Loading dataset from {input_file}...")
    original_data = load_dataset(input_file)
    print(f"   Original size: {len(original_data)} examples")

    # Check if output file exists (resume mode)
    current_count = len(original_data)
    if output_file.exists():
        existing_data = load_dataset(output_file)
        current_count = len(existing_data)
        print(f"   📂 Found existing augmented data: {current_count} examples")
        print(f"   🔄 Resuming from {current_count}...")
    else:
        # Initialize with original data
        print(f"   ✨ Starting fresh augmentation...")
        for item in original_data:
            append_to_dataset(item, output_file)

    needed = target_size - current_count
    if needed <= 0:
        print(f"   ✅ Already reached target size!")
        return

    print(f"\n🎯 Target: {target_size} examples")
    print(f"   Need to generate: {needed} examples")

    # Strategy: 50% variations, 50% new
    variation_count = needed // 2
    new_count = needed - variation_count

    # Generate variations
    print(f"\n📝 Generating {variation_count} variations...")
    successful = 0
    failed = 0

    for i in range(variation_count):
        source = random.choice(original_data)

        example = generate_with_retry(generate_variation, example=source, model=model)

        if example:
            append_to_dataset(example, output_file)
            successful += 1
            current_count += 1

            if successful % checkpoint_every == 0:
                print(
                    f"   ✅ Progress: {successful}/{variation_count} variations (total: {current_count})"
                )
        else:
            failed += 1
            print(f"      ❌ Failed to generate variation {i + 1}")

    print(f"   📊 Variations complete: {successful} successful, {failed} failed")

    # Generate new exercises
    print(f"\n✨ Generating {new_count} new exercises...")

    level_dist = defaultdict(int)
    for item in original_data:
        level_dist[item["metadata"]["cefr_level"]] += 1
    total_original = len(original_data)

    successful = 0
    failed = 0

    for i in range(new_count):
        # Sample level proportionally
        rand = random.random()
        cumulative = 0
        selected_level = "A2"

        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            cumulative += level_dist.get(level, 0) / total_original
            if rand <= cumulative:
                selected_level = level
                break

        topic = random.choice(TOPICS.get(selected_level, TOPICS["A2"]))
        grammar = random.choice(GRAMMAR_FOCUSES.get(selected_level, GRAMMAR_FOCUSES["A2"]))
        ex_type = random.choice(
            ["fill_in_blank", "multiple_choice", "translation", "conjugation", "transformation"]
        )
        count = random.randint(2, 5)

        example = generate_with_retry(
            generate_new_exercise,
            level=selected_level,
            topic=topic,
            grammar=grammar,
            ex_type=ex_type,
            count=count,
            model=model,
        )

        if example:
            append_to_dataset(example, output_file)
            successful += 1
            current_count += 1

            if successful % checkpoint_every == 0:
                print(
                    f"   ✅ Progress: {successful}/{new_count} new exercises (total: {current_count})"
                )
        else:
            failed += 1
            print(f"      ❌ Failed to generate new exercise {i + 1}")

    print(f"   📊 New exercises complete: {successful} successful, {failed} failed")

    # Final statistics
    final_data = load_dataset(output_file)
    print(f"\n📈 AUGMENTATION COMPLETE")
    print(f"   Original: {len(original_data)} examples")
    print(f"   Final: {len(final_data)} examples")
    print(f"   Increase: {len(final_data) - len(original_data)} examples")
    print(f"   📁 Saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Safe augmentation with incremental saves")
    parser.add_argument(
        "--input", type=str, default="data/datasets/v4/balanced_training_dataset.jsonl"
    )
    parser.add_argument(
        "--output", type=str, default="data/datasets/v4/augmented_training_dataset.jsonl"
    )
    parser.add_argument("--target-size", type=int, default=4000)
    parser.add_argument(
        "--checkpoint-every", type=int, default=10, help="Save checkpoint every N examples"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")

    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return

    augment_dataset_safe(
        input_path,
        output_path,
        target_size=args.target_size,
        checkpoint_every=args.checkpoint_every,
        model=args.model,
    )


if __name__ == "__main__":
    main()
