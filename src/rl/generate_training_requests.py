"""
Generate diverse training requests for GRPO training.

Uses full 16,887-word vocabulary to create varied, realistic exercise requests.
"""

import json
import random
from pathlib import Path
from typing import Dict, List

try:
    # Try relative imports first (when used as package)
    from .rl_data import get_vocabulary_by_cefr, load_italian_vocabulary
except ImportError:
    # Fall back to direct imports (when used standalone)
    from rl_data import get_vocabulary_by_cefr, load_italian_vocabulary

# Grammar focuses for Italian exercises
GRAMMAR_FOCUSES = [
    "present_tense",
    "past_tense",
    "future_tense",
    "imperfect_tense",
    "conditional",
    "subjunctive",
    "articles",
    "prepositions",
    "pronouns",
    "adjective_agreement",
    "verb_conjugation",
]

# CEFR levels
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# Exercise types (matching V4 model training)
EXERCISE_TYPES = ["fill_in_blank", "multiple_choice", "translation"]


def load_vocabulary_by_pos() -> Dict[str, List[Dict]]:
    """Load vocabulary organized by part of speech."""
    vocab = load_italian_vocabulary()

    by_pos = {"noun": [], "verb": [], "adjective": [], "adverb": []}
    for entry in vocab:
        pos = entry.get("pos", "")
        if pos in by_pos:
            by_pos[pos].append(entry)

    return by_pos


def generate_training_requests(num_requests: int = 2000, output_path: str = None) -> List[Dict]:
    """
    Generate diverse training requests using full vocabulary.

    Args:
        num_requests: Number of requests to generate
        output_path: Optional path to save JSON file

    Returns:
        List of request dictionaries
    """
    print(f"Loading vocabulary...")
    vocab_by_pos = load_vocabulary_by_pos()

    print(f"Generating {num_requests} training requests...")
    requests = []

    for i in range(num_requests):
        # Random CEFR level
        level = random.choice(CEFR_LEVELS)

        # Get vocabulary for this level
        level_vocab = get_vocabulary_by_cefr(level, cumulative=True)

        # Random grammar focus
        grammar_focus = random.choice(GRAMMAR_FOCUSES)

        # Pick a random topic word (noun) from level vocabulary
        level_nouns = [n for n in vocab_by_pos["noun"] if n["word"] in level_vocab]
        if level_nouns:
            topic_word = random.choice(level_nouns)
            topic = topic_word["word"]
        else:
            topic = "general"

        # Random number of questions (1-5 for variety)
        num_exercises = random.randint(1, 5)

        # Random exercise types (1-3 types)
        num_types = random.randint(1, min(3, num_exercises))
        exercise_types = random.sample(EXERCISE_TYPES, num_types)

        request = {
            "level": level,
            "grammar_focus": grammar_focus,
            "topic": topic,
            "num_exercises": num_exercises,
            "exercise_types": exercise_types,
        }

        requests.append(request)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_requests}")

    # Save if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(requests, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {num_requests} requests to {output_path}")

    # Print statistics
    print("\n📊 Request Statistics:")
    from collections import Counter

    level_counts = Counter(r["level"] for r in requests)
    grammar_counts = Counter(r["grammar_focus"] for r in requests)

    print(f"By CEFR level:")
    for level in CEFR_LEVELS:
        print(f"  {level}: {level_counts[level]}")

    print(f"\nTop 5 grammar focuses:")
    for grammar, count in grammar_counts.most_common(5):
        print(f"  {grammar}: {count}")

    return requests


if __name__ == "__main__":
    # Generate 2000 diverse training requests
    requests = generate_training_requests(num_requests=2000, output_path="training_requests.json")
