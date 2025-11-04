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
    # Tenses
    "present_tense",
    "past_tense",  # Passato prossimo
    "future_tense",
    "imperfect_tense",
    "passato_remoto",  # Simple past (more literary)
    "trapassato_prossimo",  # Pluperfect
    # Moods
    "conditional",
    "subjunctive",
    "imperativo",  # Imperative
    # Structures & Particles
    "articles",
    "prepositions",
    "pronouns",
    "pronomi_combinati",  # Combined pronouns (e.g., me lo)
    "ci_e_ne",  # Particles 'ci' and 'ne'
    "adjective_agreement",
    "verb_conjugation",
    "verbi_riflessivi",  # Reflexive verbs
    "forma_passiva",  # Passive voice
    "comparativi_superlativi",  # Comparatives/Superlatives
    "gerundio",  # Gerund
]

# CEFR levels
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# Exercise types (matching V4 model training)
EXERCISE_TYPES = ["fill_in_blank", "multiple_choice", "translation"]

# Define minimum CEFR level for each grammar focus to prevent impossible requests
MIN_CEFR_FOR_GRAMMAR = {
    # A1: Basics
    "present_tense": "A1",
    "articles": "A1",
    "adjective_agreement": "A1",
    # A2: More complex structures
    "past_tense": "A2",  # Passato prossimo
    "prepositions": "A2",
    "pronouns": "A2",
    "verb_conjugation": "A2",  # More complex conjugations
    "verbi_riflessivi": "A2",  # Reflexive verbs
    "imperativo": "A2",  # Imperative (informal)
    # B1: Intermediate tenses and moods
    "future_tense": "B1",
    "imperfect_tense": "B1",
    "conditional": "B1",
    "pronomi_combinati": "B1",  # Combined pronouns
    "ci_e_ne": "B1",  # Particles 'ci' and 'ne'
    "comparativi_superlativi": "B1",
    "gerundio": "B1",
    # B2: Advanced moods and structures
    "subjunctive": "B2",
    "trapassato_prossimo": "B2",  # Pluperfect
    "forma_passiva": "B1",  # Passive voice (simple forms at B1, complex at B2)
    # C1/C2: Advanced/Literary tenses
    "passato_remoto": "C1",  # Simple past (literary)
}

CEFR_ORDER = {level: i for i, level in enumerate(CEFR_LEVELS)}


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

    # Common thematic topics for variety (beyond just nouns)
    thematic_topics = [
        # Daily Life & People
        "vita quotidiana",
        "famiglia",
        "amici e relazioni",
        "descrivere persone",
        "casa e arredamento",
        # Food & Drink
        "cibo e cucina",
        "al ristorante",
        "fare la spesa",
        # Activities & Hobbies
        "viaggi",
        "lavoro",
        "scuola",
        "sport",
        "tempo libero",
        "musica",
        "film e cinema",
        "libri e lettura",
        "arte e cultura",
        # Places
        "città",
        "natura",
        "vacanze al mare",
        "vacanze in montagna",
        # Work & Study
        "lavoro e professioni",
        "scuola e istruzione",
        "tecnologia e internet",
        # Abstract & Situational
        "salute e benessere",
        "emozioni e sentimenti",
        "meteo e stagioni",
        "trasporti e mobilità",
        "feste e tradizioni",
        "sogni e aspirazioni",
        "ambiente e sostenibilità",
        "esprimere un'opinione",
    ]
    
    i = 0
    while i < num_requests:
        # Loop until a valid request is generated
        while True:
            # Random CEFR level with realistic distribution (more A2/B1, less C2)
            level = random.choices(
                CEFR_LEVELS, weights=[15, 25, 25, 20, 10, 5], k=1  # Favor intermediate levels
            )[0]

            # Random grammar focus with realistic distribution
            # (more focus on tenses, less on advanced grammar)
            # Ensure weights list matches the length of GRAMMAR_FOCUSES
            base_weights = [20, 20, 15, 15, 5, 10, 15, 15, 5, 15, 15, 15, 10, 10, 15, 15, 10, 10, 10, 10]
            if len(base_weights) != len(GRAMMAR_FOCUSES):
                raise ValueError("Length of grammar_weights must match GRAMMAR_FOCUSES")

            # --- FLEXIBLE GRAMMAR FOCUS GENERATION ---
            # 80% of the time, pick one focus. 20% of the time, combine two.
            num_focuses_to_combine = random.choices([1, 2], weights=[0.8, 0.2], k=1)[0]
            
            # Select unique focuses to combine
            selected_focuses = random.sample(GRAMMAR_FOCUSES, k=num_focuses_to_combine)

            if len(selected_focuses) == 1:
                grammar_focus = selected_focuses[0]
            else:
                # Combine two focuses into a natural language string
                grammar_focus = f"{selected_focuses[0]} and {selected_focuses[1]}"

            # --- VALIDATION STEP ---
            # For validation, check the requirement of the MOST advanced focus in the request
            max_min_level_required = "A1"
            for focus in selected_focuses:
                min_level = MIN_CEFR_FOR_GRAMMAR.get(focus, "A1")
                if CEFR_ORDER[min_level] > CEFR_ORDER[max_min_level_required]:
                    max_min_level_required = min_level
            
            if CEFR_ORDER[level] < CEFR_ORDER[max_min_level_required]:
                # This combination is invalid (e.g., 'subjunctive' for 'A1'). Try again.
                continue
            else:
                # Combination is valid, break the inner loop and proceed
                break

        # 70% specific noun topics, 30% thematic topics for better diversity
        if random.random() < 0.7:
            level_vocab = get_vocabulary_by_cefr(level, cumulative=True)
            level_nouns = [n for n in vocab_by_pos["noun"] if n["word"] in level_vocab]
            if level_nouns:
                topic = random.choice(level_nouns)["word"]
            else:
                topic = random.choice(thematic_topics)
        else:
            topic = random.choice(thematic_topics)

        # Random number of questions (weighted toward 3-5 exercises)
        num_exercises = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 25, 30, 30], k=1)[0]
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
        i += 1

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
