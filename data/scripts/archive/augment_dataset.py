#!/usr/bin/env python3
"""
Dataset Augmentation for Exercise Generation

Expands the v4 dataset from 428 to ~4,000+ examples using:
1. Paraphrasing (same grammar, different vocabulary)
2. Topic variation (same structure, different topics)
3. LLM generation (GPT-4o-mini for new exercises)
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

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


def save_dataset(data: List[Dict], file_path: Path):
    """Save dataset to JSONL."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved {len(data)} examples to {file_path}")


def create_variation_prompt(original_example: Dict) -> str:
    """Create prompt for generating exercise variations."""
    metadata = original_example["metadata"]
    messages = original_example["messages"]

    # Extract original exercises from assistant response
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

    # Parse response and create new example
    new_exercises = response.choices[0].message.content.strip()

    # Create new example with same structure
    new_example = {
        "messages": [
            example["messages"][0],  # Same system message
            {
                "role": "user",
                "content": example["messages"][1]["content"],  # Same user prompt structure
            },
            {"role": "assistant", "content": new_exercises},
        ],
        "metadata": example["metadata"].copy(),
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

    # Create training example
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


def augment_dataset(
    input_file: Path, output_file: Path, target_size: int = 4000, model: str = "gpt-4o-mini"
):
    """Main augmentation pipeline."""

    print(f"📊 Loading dataset from {input_file}...")
    original_data = load_dataset(input_file)
    print(f"   Original size: {len(original_data)} examples")

    augmented_data = original_data.copy()

    # Calculate how many we need
    needed = target_size - len(original_data)
    print(f"\n🎯 Target: {target_size} examples")
    print(f"   Need to generate: {needed} examples")

    # Strategy 1: Generate variations of existing examples (50%)
    variation_count = needed // 2
    print(f"\n📝 Generating {variation_count} variations...")

    for i in range(variation_count):
        # Sample random example
        source = random.choice(original_data)

        try:
            variation = generate_variation(source, model=model)
            augmented_data.append(variation)

            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{variation_count} variations")

        except Exception as e:
            print(f"   ⚠️  Error generating variation {i + 1}: {e}")
            continue

    # Strategy 2: Generate new exercises (50%)
    new_count = needed - variation_count
    print(f"\n✨ Generating {new_count} new exercises...")

    # Distribute across levels proportionally
    level_dist = defaultdict(int)
    for item in original_data:
        level_dist[item["metadata"]["cefr_level"]] += 1

    total_original = len(original_data)

    for i in range(new_count):
        # Sample level proportionally
        rand = random.random()
        cumulative = 0
        selected_level = "A2"  # Default

        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
            cumulative += level_dist.get(level, 0) / total_original
            if rand <= cumulative:
                selected_level = level
                break

        # Sample random topic and grammar
        topic = random.choice(TOPICS.get(selected_level, TOPICS["A2"]))
        grammar = random.choice(GRAMMAR_FOCUSES.get(selected_level, GRAMMAR_FOCUSES["A2"]))
        ex_type = random.choice(
            ["fill_in_blank", "multiple_choice", "translation", "conjugation", "transformation"]
        )
        count = random.randint(2, 5)

        try:
            new_exercise = generate_new_exercise(
                selected_level, topic, grammar, ex_type, count, model=model
            )
            augmented_data.append(new_exercise)

            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{new_count} new exercises")

        except Exception as e:
            print(f"   ⚠️  Error generating new exercise {i + 1}: {e}")
            continue

    # Save augmented dataset
    print(f"\n💾 Saving augmented dataset...")
    save_dataset(augmented_data, output_file)

    # Print statistics
    print(f"\n📈 AUGMENTATION COMPLETE")
    print(f"   Original: {len(original_data)} examples")
    print(f"   Augmented: {len(augmented_data)} examples")
    print(
        f"   Increase: {len(augmented_data) - len(original_data)} examples ({(len(augmented_data) / len(original_data) - 1) * 100:.1f}%)"
    )

    return augmented_data


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Augment Italian exercise dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/datasets/v4/balanced_training_dataset.jsonl",
        help="Input dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/datasets/v4/augmented_training_dataset.jsonl",
        help="Output dataset",
    )
    parser.add_argument("--target-size", type=int, default=4000, help="Target dataset size")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")

    args = parser.parse_args()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY=your-api-key")
        return

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        return

    augment_dataset(input_path, output_path, target_size=args.target_size, model=args.model)


if __name__ == "__main__":
    main()
