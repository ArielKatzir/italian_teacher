#!/usr/bin/env python3
"""
Parallel Comprehensive Augmentation

Generates ~5,700 examples using multiple parallel workers.
Speed: ~10x faster than sequential (5-10 minutes instead of 50)
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from augment_comprehensive import (
    CEFR_DISTRIBUTION,
    EXERCISE_TYPES,
    GRAMMAR_FOCUSES,
    SAMPLES_PER_GRAMMAR,
    SAMPLES_PER_TOPIC,
    TOPIC_CATEGORIES,
)
from openai import OpenAI

# Configuration
OUTPUT_FILE = "data/datasets/v4_augmented/train_augmentation_comprehensive.jsonl"
NUM_WORKERS = 20  # Parallel workers (adjust based on rate limits)


def generate_prompts() -> List[Dict]:
    """Generate all prompt specifications."""
    prompts = []

    # Topic-focused
    for topic, description in TOPIC_CATEGORIES.items():
        for _ in range(SAMPLES_PER_TOPIC):
            cefr = random.choices(
                list(CEFR_DISTRIBUTION.keys()), weights=list(CEFR_DISTRIBUTION.values())
            )[0]
            grammar = random.choice(list(GRAMMAR_FOCUSES.keys()))
            num_exercises = random.randint(3, 5)
            selected_types = random.sample(EXERCISE_TYPES, min(num_exercises, len(EXERCISE_TYPES)))

            prompts.append(
                {
                    "cefr_level": cefr,
                    "grammar_focus": grammar,
                    "topic": topic.replace("_", " "),
                    "topic_description": description,
                    "num_exercises": num_exercises,
                    "exercise_types": selected_types,
                    "source": "topic_focused",
                }
            )

    # Grammar-focused
    for grammar, description in GRAMMAR_FOCUSES.items():
        for _ in range(SAMPLES_PER_GRAMMAR):
            cefr = random.choices(
                list(CEFR_DISTRIBUTION.keys()), weights=list(CEFR_DISTRIBUTION.values())
            )[0]
            topic = random.choice(list(TOPIC_CATEGORIES.keys()))
            num_exercises = random.randint(3, 5)
            selected_types = random.sample(EXERCISE_TYPES, min(num_exercises, len(EXERCISE_TYPES)))

            prompts.append(
                {
                    "cefr_level": cefr,
                    "grammar_focus": grammar,
                    "grammar_description": description,
                    "topic": topic.replace("_", " "),
                    "num_exercises": num_exercises,
                    "exercise_types": selected_types,
                    "source": "grammar_focused",
                }
            )

    random.shuffle(prompts)
    return prompts


def create_generation_prompt(spec: Dict) -> str:
    """Create prompt for GPT-4o-mini."""
    types_str = ", ".join(spec["exercise_types"])
    focus_description = ""
    if spec.get("topic_description"):
        focus_description = f"\nTopic vocabulary to use: {spec['topic_description']}"
    elif spec.get("grammar_description"):
        focus_description = f"\nGrammar focus: {spec['grammar_description']}"

    return f"""Generate {spec["num_exercises"]} Italian language exercises in JSON format.

REQUIREMENTS:
- CEFR Level: {spec["cefr_level"]}
- Grammar Focus: {spec["grammar_focus"]}
- Topic: {spec["topic"]}{focus_description}
- Exercise Types: {types_str}

CRITICAL INSTRUCTIONS:
1. Use DIVERSE, NATURAL Italian vocabulary related to {spec["topic"]}
2. Test {spec["grammar_focus"]} at {spec["cefr_level"]} level
3. For tense-based grammar: ALL exercises must use that tense consistently
4. Use realistic, factual scenarios
5. Multiple choice: provide 4 DIFFERENT options

OUTPUT FORMAT - JSON array only, no markdown:
[
  {{"type": "fill_in_blank", "question": "Italian sentence with ___ blank", "answer": "correct form", "explanation": "grammar explanation"}},
  {{"type": "translation", "question": "English sentence", "answer": "Italian translation", "explanation": "grammar note"}},
  {{"type": "multiple_choice", "question": "Italian sentence", "answer": "correct", "options": ["opt1", "opt2", "opt3", "opt4"], "explanation": "why correct"}}
]

Generate {spec["num_exercises"]} exercises now:"""


def generate_single_example(client: OpenAI, spec: Dict, idx: int, total: int) -> tuple:
    """Generate a single example (for parallel execution)."""
    try:
        system_prompt = "You are an expert Italian language teacher. Generate high-quality exercises based on the assignment specification. Output exercises in JSON format."

        user_prompt = f"""Generate {spec["num_exercises"]} exercises:
CEFR Level: {spec["cefr_level"]}
Grammar Focus: {spec["grammar_focus"]}
Topic: {spec["topic"]}
Exercise Types: {', '.join(spec["exercise_types"])}"""

        generation_prompt = create_generation_prompt(spec)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": generation_prompt},
            ],
            temperature=0.7,
            max_tokens=2000,
        )

        content = response.choices[0].message.content.strip()

        # Remove markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        # Validate JSON
        json.loads(content)

        # Create training example
        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": content},
            ],
            "metadata": {
                "cefr_level": spec["cefr_level"],
                "topic": spec["topic"],
                "num_exercises": spec["num_exercises"],
                "grammar_focus": spec["grammar_focus"],
                "source": f"gpt4o_mini_{spec['source']}",
            },
        }

        return (idx, True, example, None)

    except json.JSONDecodeError as e:
        return (idx, False, None, f"JSON error: {str(e)[:50]}")
    except Exception as e:
        return (idx, False, None, f"Error: {str(e)[:50]}")


def main():
    """Main parallel execution."""

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return

    client = OpenAI(api_key=api_key)

    print("🚀 Parallel Comprehensive Italian Exercise Augmentation")
    print(f"   Workers: {NUM_WORKERS}")
    print(
        f"   Topic categories: {len(TOPIC_CATEGORIES)} × {SAMPLES_PER_TOPIC} = {len(TOPIC_CATEGORIES) * SAMPLES_PER_TOPIC}"
    )
    print(
        f"   Grammar focuses: {len(GRAMMAR_FOCUSES)} × {SAMPLES_PER_GRAMMAR} = {len(GRAMMAR_FOCUSES) * SAMPLES_PER_GRAMMAR}"
    )
    print(
        f"   Total target: {len(TOPIC_CATEGORIES) * SAMPLES_PER_TOPIC + len(GRAMMAR_FOCUSES) * SAMPLES_PER_GRAMMAR}"
    )
    print()

    # Generate prompts
    print("📝 Generating prompt specifications...")
    prompts = generate_prompts()
    print(f"✅ Generated {len(prompts)} prompts")
    print()

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Parallel generation
    print(f"🤖 Generating exercises with {NUM_WORKERS} parallel workers...")
    print()

    generated = []
    failed = 0
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(generate_single_example, client, spec, i, len(prompts)): i
            for i, spec in enumerate(prompts)
        }

        # Process completed tasks
        for future in as_completed(futures):
            idx, success, example, error = future.result()
            completed += 1

            if success:
                generated.append(example)
                print(f"✅ [{completed}/{len(prompts)}] Generated example {idx + 1}")
            else:
                failed += 1
                print(f"❌ [{completed}/{len(prompts)}] Failed example {idx + 1}: {error}")

            # Checkpoint save every 100
            if len(generated) % 100 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    for ex in generated:
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                print(f"💾 Checkpoint: Saved {len(generated)} examples")

    print()
    print(f"✅ Generated {len(generated)} examples")
    print(f"❌ Failed {failed} examples ({failed / len(prompts) * 100:.1f}%)")
    print()

    # Final save
    print(f"💾 Saving final dataset to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in generated:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(generated)} examples")
    print()

    # Statistics
    grammar_counts = {}
    topic_counts = {}
    for ex in generated:
        grammar_counts[ex["metadata"]["grammar_focus"]] = (
            grammar_counts.get(ex["metadata"]["grammar_focus"], 0) + 1
        )
        topic_counts[ex["metadata"]["topic"]] = topic_counts.get(ex["metadata"]["topic"], 0) + 1

    print("📊 Final Statistics:")
    print(f"   Total examples: {len(generated)}")
    print(f"   Grammar coverage: {len(grammar_counts)} unique")
    print(f"   Topic coverage: {len(topic_counts)} unique")
    print()

    print("📋 Next steps:")
    print(
        f"   1. Merge: cat data/datasets/final/train.jsonl {OUTPUT_FILE} > data/datasets/v4_augmented/train.jsonl"
    )
    print(f"   2. Verify: wc -l data/datasets/v4_augmented/train.jsonl")
    print(f"   3. Retrain with V4 config")


if __name__ == "__main__":
    main()
