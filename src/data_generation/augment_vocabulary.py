#!/usr/bin/env python3
"""
Vocabulary Augmentation Script for Italian Exercise Dataset

Generates new training examples using GPT-4o-mini to cover:
- Priority 1: Diverse vocabulary (animals, nature, food, professions)
- Priority 2: Balanced grammar focus (more past_tense, present_tense)
- Priority 4: Topic diversity (animals, nature, daily_life, science)

Target: 500 new examples to augment existing 3,186 examples
"""

import json
import os
import random
from typing import Dict, List

from openai import OpenAI

# Configuration
OUTPUT_FILE = "data/datasets/v4_augmented/train_augmentation.jsonl"
TARGET_EXAMPLES = 500

# Priority 1: Diverse vocabulary categories
VOCABULARY_FOCUS = {
    "animals": [
        "ragno (spider)",
        "lombrico (earthworm)",
        "aquila (eagle)",
        "serpente (snake)",
        "farfalla (butterfly)",
        "formica (ant)",
        "ape (bee)",
        "vespa (wasp)",
        "libellula (dragonfly)",
        "lucciola (firefly)",
        "cavalletta (grasshopper)",
        "scarabeo (beetle)",
        "pipistrello (bat)",
        "riccio (hedgehog)",
        "talpa (mole)",
        "scoiattolo (squirrel)",
        "volpe (fox)",
        "cervo (deer)",
        "cinghiale (wild boar)",
        "lupo (wolf)",
        "orso (bear)",
        "delfino (dolphin)",
        "balena (whale)",
        "foca (seal)",
        "pinguino (penguin)",
        "fenicottero (flamingo)",
        "struzzo (ostrich)",
    ],
    "nature": [
        "fiore (flower)",
        "albero (tree)",
        "bosco (forest)",
        "fiume (river)",
        "montagna (mountain)",
        "collina (hill)",
        "lago (lake)",
        "cascata (waterfall)",
        "ruscello (stream)",
        "prato (meadow)",
        "valle (valley)",
        "pianura (plain)",
        "spiaggia (beach)",
        "scogliera (cliff)",
        "isola (island)",
        "vulcano (volcano)",
        "deserto (desert)",
        "giungla (jungle)",
        "foglia (leaf)",
        "radice (root)",
        "ramo (branch)",
        "seme (seed)",
        "corteccia (bark)",
        "muschio (moss)",
    ],
    "food": [
        "zucca (pumpkin)",
        "melanzana (eggplant)",
        "carciofo (artichoke)",
        "asparago (asparagus)",
        "cavolo (cabbage)",
        "cavolfiore (cauliflower)",
        "barbabietola (beet)",
        "ravanello (radish)",
        "porro (leek)",
        "sedano (celery)",
        "finocchio (fennel)",
        "zucchina (zucchini)",
        "fungo (mushroom)",
        "tartufo (truffle)",
        "oliva (olive)",
        "fico (fig)",
        "albicocca (apricot)",
        "ciliegia (cherry)",
        "mora (blackberry)",
        "lampone (raspberry)",
        "mirtillo (blueberry)",
        "noce (walnut)",
        "mandorla (almond)",
        "nocciola (hazelnut)",
    ],
    "professions": [
        "architetto (architect)",
        "veterinario (veterinarian)",
        "scienziato (scientist)",
        "ricercatore (researcher)",
        "ingegnere (engineer)",
        "programmatore (programmer)",
        "fotografo (photographer)",
        "giornalista (journalist)",
        "scrittore (writer)",
        "pittore (painter)",
        "scultore (sculptor)",
        "musicista (musician)",
        "cuoco (chef)",
        "pasticciere (pastry chef)",
        "sommelier (sommelier)",
        "agricoltore (farmer)",
        "giardiniere (gardener)",
        "pescatore (fisherman)",
        "falegname (carpenter)",
        "elettricista (electrician)",
        "idraulico (plumber)",
        "parrucchiere (hairdresser)",
        "estetista (beautician)",
        "sarto (tailor)",
    ],
}

# Priority 2 & 4: Grammar focuses and topics to emphasize
GRAMMAR_EMPHASIS = {
    "past_tense": 150,  # Currently only 124, need more
    "present_tense": 100,  # Currently 149, boost to 250+
    "prepositions": 50,  # Currently 144, maintain
    "articles": 50,  # Currently 130, maintain
    "pronouns": 50,  # Currently 108, maintain
    "reflexive_verbs": 50,  # Currently 94, boost
    "adjectives": 50,  # Currently 72+45, consolidate
}

TOPIC_EMPHASIS = {
    "animals": 150,
    "nature_environment": 100,
    "daily_life": 100,
    "food_cooking": 75,
    "professions_work": 75,
}

# CEFR levels (focus on A1-B2 as requested)
CEFR_DISTRIBUTION = {
    "A1": 0.25,
    "A2": 0.35,
    "B1": 0.25,
    "B2": 0.15,
}

# Exercise types
EXERCISE_TYPES = [
    "fill_in_blank",
    "translation",
    "multiple_choice",
    "conjugation",
    "sentence_completion",
]


def generate_prompts() -> List[Dict]:
    """
    Generate prompts for GPT-4o-mini to create diverse exercises.

    Returns:
        List of prompt specifications
    """
    prompts = []

    # Generate prompts based on grammar and topic emphasis
    for grammar, count in GRAMMAR_EMPHASIS.items():
        for topic, topic_count in TOPIC_EMPHASIS.items():
            # Calculate proportional examples
            proportion = count * (topic_count / sum(TOPIC_EMPHASIS.values()))
            num_examples = int(proportion)

            if num_examples == 0:
                continue

            for _ in range(num_examples):
                # Sample CEFR level
                cefr = random.choices(
                    list(CEFR_DISTRIBUTION.keys()), weights=list(CEFR_DISTRIBUTION.values())
                )[0]

                # Sample vocabulary from the topic
                vocab_list = []
                if "animal" in topic:
                    vocab_list = VOCABULARY_FOCUS["animals"]
                elif "nature" in topic:
                    vocab_list = VOCABULARY_FOCUS["nature"]
                elif "food" in topic:
                    vocab_list = VOCABULARY_FOCUS["food"]
                elif "profession" in topic or "work" in topic:
                    vocab_list = VOCABULARY_FOCUS["professions"]
                else:
                    # Mix vocabulary
                    vocab_list = sum(VOCABULARY_FOCUS.values(), [])

                # Sample 3-5 vocabulary words
                selected_vocab = random.sample(
                    vocab_list, min(random.randint(3, 5), len(vocab_list))
                )

                # Sample exercise types
                num_exercises = random.randint(3, 5)
                selected_types = random.sample(
                    EXERCISE_TYPES, min(num_exercises, len(EXERCISE_TYPES))
                )

                prompts.append(
                    {
                        "cefr_level": cefr,
                        "grammar_focus": grammar,
                        "topic": topic.replace("_", " "),
                        "vocabulary": selected_vocab,
                        "num_exercises": num_exercises,
                        "exercise_types": selected_types,
                    }
                )

    # Shuffle and limit to TARGET_EXAMPLES
    random.shuffle(prompts)
    return prompts[:TARGET_EXAMPLES]


def create_generation_prompt(spec: Dict) -> str:
    """
    Create a prompt for GPT-4o-mini to generate exercises.

    Args:
        spec: Prompt specification with cefr_level, grammar_focus, topic, vocabulary

    Returns:
        Formatted prompt string
    """
    vocab_str = "\n".join([f"- {word}" for word in spec["vocabulary"]])
    types_str = ", ".join(spec["exercise_types"])

    return f"""Generate {spec["num_exercises"]} Italian language exercises in JSON format.

REQUIREMENTS:
- CEFR Level: {spec["cefr_level"]}
- Grammar Focus: {spec["grammar_focus"]}
- Topic: {spec["topic"]}
- Exercise Types: {types_str}

CRITICAL: You MUST use these vocabulary words in the exercises:
{vocab_str}

GRAMMAR RULES:
- For "past_tense": Use passato prossimo (ho fatto, sono andato) or imperfetto
- For "present_tense": Use presente indicativo (faccio, vado)
- For "articles": Ensure correct gender agreement (il/la/un/una with masculine/feminine nouns)
- ALL exercises must test {spec["grammar_focus"]} at {spec["cefr_level"]} level

REALISM: Use natural, factual scenarios about {spec["topic"]}

OUTPUT FORMAT - JSON array only, no markdown:
[
  {{"type": "fill_in_blank", "question": "Italian sentence with ___ blank", "answer": "correct form", "explanation": "grammar explanation", "hint": "optional hint"}},
  {{"type": "translation", "question": "English sentence", "answer": "Italian translation", "explanation": "grammar note"}},
  {{"type": "multiple_choice", "question": "Italian sentence with blank", "answer": "correct form", "options": ["opt1", "opt2", "opt3", "opt4"], "explanation": "why this is correct"}}
]

Generate {spec["num_exercises"]} exercises now:"""


def generate_exercise_with_gpt4o_mini(client: OpenAI, spec: Dict) -> Dict:
    """
    Use GPT-4o-mini to generate an exercise example.

    Args:
        client: OpenAI client
        spec: Prompt specification

    Returns:
        Training example in dataset format
    """
    system_prompt = "You are an expert Italian language teacher. Generate high-quality exercises based on the assignment specification. Output exercises in JSON format."

    user_prompt = f"""Generate {spec["num_exercises"]} exercises:
CEFR Level: {spec["cefr_level"]}
Grammar Focus: {spec["grammar_focus"]}
Topic: {spec["topic"]}
Exercise Types: {', '.join(spec["exercise_types"])}"""

    generation_prompt = create_generation_prompt(spec)

    try:
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

        # Remove markdown code blocks if present
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
        return {
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
                "source": "gpt4o_mini_augmented",
                "vocabulary_focus": ", ".join(
                    [v.split("(")[0].strip() for v in spec["vocabulary"][:3]]
                ),
            },
        }

    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parsing failed for {spec['topic']}/{spec['grammar_focus']}: {e}")
        print(f"   Content: {content[:200]}")
        return None
    except Exception as e:
        print(f"❌ Generation failed for {spec['topic']}/{spec['grammar_focus']}: {e}")
        return None


def main():
    """Main execution function."""

    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return

    client = OpenAI(api_key=api_key)

    print("🚀 Italian Exercise Vocabulary Augmentation")
    print(f"   Target: {TARGET_EXAMPLES} new examples")
    print(f"   Output: {OUTPUT_FILE}")
    print()

    # Generate prompts
    print("📝 Generating prompt specifications...")
    prompts = generate_prompts()
    print(f"   Created {len(prompts)} prompts")
    print()

    # Summary
    grammar_counts = {}
    topic_counts = {}
    for p in prompts:
        grammar_counts[p["grammar_focus"]] = grammar_counts.get(p["grammar_focus"], 0) + 1
        topic_counts[p["topic"]] = topic_counts.get(p["topic"], 0) + 1

    print("📊 Distribution:")
    print("   Grammar:")
    for g, c in sorted(grammar_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"     {g}: {c}")
    print("   Topics:")
    for t, c in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"     {t}: {c}")
    print()

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Generate examples
    print(f"🤖 Generating exercises with GPT-4o-mini...")
    generated = []
    failed = 0

    for i, spec in enumerate(prompts, 1):
        print(
            f"   [{i}/{len(prompts)}] {spec['cefr_level']} - {spec['topic']} - {spec['grammar_focus']}",
            end="",
        )

        example = generate_exercise_with_gpt4o_mini(client, spec)

        if example:
            generated.append(example)
            print(" ✅")
        else:
            failed += 1
            print(" ❌")

    print()
    print(f"✅ Generated {len(generated)} examples")
    print(f"❌ Failed {failed} examples")
    print()

    # Save to JSONL
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in generated:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✅ Saved {len(generated)} examples to {OUTPUT_FILE}")
    print()

    # Next steps
    print("📋 Next steps:")
    print(f"   1. Review generated examples: head -5 {OUTPUT_FILE} | jq")
    print(f"   2. Merge with existing training data:")
    print(
        f"      cat data/datasets/final/train.jsonl {OUTPUT_FILE} > data/datasets/v4_augmented/train.jsonl"
    )
    print(f"   3. Retrain with V4 config (alpha=6)")


if __name__ == "__main__":
    main()
