#!/usr/bin/env python3
"""
Comprehensive Vocabulary & Grammar Augmentation Script

Generates training examples using GPT-4o-mini with:
- 20 diverse topic categories × 100 samples = 2,000 examples
- 50+ grammar focuses × 50 samples = 2,500+ examples
- Total: ~4,500 new examples with massive vocabulary coverage

Goal: Prevent catastrophic forgetting by exposing model to wide Italian vocabulary
"""

import json
import os
import random
import time
from typing import Dict, List

from openai import OpenAI

# Configuration
OUTPUT_FILE = "data/datasets/v4_augmented/train_augmentation_comprehensive.jsonl"
SAMPLES_PER_TOPIC = 100
SAMPLES_PER_GRAMMAR = 50
DELAY_BETWEEN_REQUESTS = 0.1  # seconds, to avoid rate limits

# 20 Diverse Topic Categories for Vocabulary Coverage
TOPIC_CATEGORIES = {
    "animals_wildlife": "animals, wildlife, insects, birds, marine life, pets",
    "nature_environment": "trees, plants, flowers, landscapes, weather, seasons, natural phenomena",
    "food_cooking": "ingredients, dishes, cooking methods, restaurants, recipes, kitchen tools",
    "professions_work": "jobs, workplaces, professional activities, careers, business",
    "sports_fitness": "sports, exercises, competitions, athletes, equipment, training",
    "travel_geography": "countries, cities, landmarks, transportation, tourism, directions",
    "home_furniture": "rooms, furniture, appliances, household items, interior design",
    "clothing_fashion": "clothes, accessories, fabrics, styles, shopping, wardrobe",
    "technology_digital": "computers, internet, devices, apps, software, digital communication",
    "arts_culture": "painting, music, theater, cinema, literature, museums, artists",
    "education_learning": "schools, subjects, studying, exams, teachers, students, university",
    "health_medicine": "body parts, illnesses, treatments, doctors, hospitals, wellness",
    "family_relationships": "family members, relationships, emotions, life events, celebrations",
    "hobbies_leisure": "activities, games, entertainment, collections, crafts, pastimes",
    "science_research": "physics, chemistry, biology, experiments, discoveries, scientists",
    "history_society": "historical events, civilizations, politics, social issues, traditions",
    "economy_finance": "money, banking, trade, investments, markets, economy concepts",
    "law_justice": "legal system, rights, courts, laws, regulations, justice",
    "vehicles_transport": "cars, trains, planes, boats, transportation systems, traffic",
    "abstract_concepts": "time, space, emotions, ideas, philosophy, logic, reasoning",
}

# Comprehensive Grammar Focuses (50+ categories covering all Italian grammar)
GRAMMAR_FOCUSES = {
    # 1. Morfologia (Word Forms)
    "noun_gender": "il nome - gender agreement (maschile/femminile)",
    "noun_number": "il nome - singular vs plural forms",
    "irregular_nouns": "nomi irregolari - invariabili",
    "definite_articles": "articoli determinativi (il, lo, la, i, gli, le)",
    "indefinite_articles": "articoli indeterminativi (un, uno, una)",
    "partitive_articles": "articoli partitivi (del, dello, della)",
    "qualitative_adjectives": "aggettivi qualificativi",
    "demonstrative_adjectives": "aggettivi dimostrativi (questo, quello)",
    "possessive_adjectives": "aggettivi possessivi (mio, tuo)",
    "adjective_degrees": "gradi dell'aggettivo (comparativo, superlativo)",
    "subject_pronouns": "pronomi personali soggetto (io, tu, lui)",
    "object_pronouns": "pronomi personali complemento (mi, ti, lo, la)",
    "reflexive_pronouns": "pronomi riflessivi (mi, ti, si)",
    "possessive_pronouns": "pronomi possessivi",
    "demonstrative_pronouns": "pronomi dimostrativi",
    "relative_pronouns": "pronomi relativi (che, cui, il quale)",
    "interrogative_pronouns": "pronomi interrogativi (chi, cosa, quale)",
    "indefinite_pronouns": "pronomi indefiniti (qualcuno, niente)",
    "adverbs": "avverbi (modo, tempo, luogo, quantità)",
    "adverb_comparison": "comparativi e superlativi degli avverbi",
    # 2. Verbi (Verb System - Indicativo)
    "present_tense": "presente indicativo",
    "imperfect_tense": "imperfetto",
    "past_tense": "passato prossimo",
    "pluperfect": "trapassato prossimo",
    "simple_past": "passato remoto",
    "remote_pluperfect": "trapassato remoto",
    "future_tense": "futuro semplice",
    "future_perfect": "futuro anteriore",
    # Congiuntivo
    "subjunctive_present": "congiuntivo presente",
    "subjunctive_imperfect": "congiuntivo imperfetto",
    "subjunctive_past": "congiuntivo passato",
    "subjunctive_pluperfect": "congiuntivo trapassato",
    # Condizionale
    "conditional_present": "condizionale presente",
    "conditional_past": "condizionale passato",
    # Imperativo
    "imperative": "imperativo",
    # Modi indefiniti
    "infinitive": "infinito (presente, passato)",
    "gerund": "gerundio (presente, passato)",
    "participle": "participio (presente, passato)",
    # Altre strutture verbali
    "passive_voice": "forma passiva",
    "reflexive_verbs": "verbi riflessivi",
    "pronominal_verbs": "verbi pronominali",
    "modal_verbs": "verbi modali (potere, dovere, volere)",
    "impersonal_verbs": "verbi impersonali (piove, si dice)",
    "sequence_of_tenses": "concordanza dei tempi",
    # 3. Sintassi
    "simple_sentence": "frase semplice (soggetto + verbo + oggetto)",
    "complex_sentence": "frase complessa (coordinazione e subordinazione)",
    "complements": "complementi (oggetto, termine, luogo, tempo)",
    "si_impersonal": "si impersonale",
    "si_passive": "si passivante",
    "subject_verb_agreement": "concordanza soggetto-verbo",
    "noun_adjective_agreement": "concordanza nome-aggettivo",
    "word_order": "ordine delle parole",
    # 4. Proposizioni (Subordinate Clauses)
    "temporal_clauses": "proposizioni temporali (quando)",
    "causal_clauses": "proposizioni causali (perché)",
    "final_clauses": "proposizioni finali (affinché)",
    "consecutive_clauses": "proposizioni consecutive (così che)",
    "concessive_clauses": "proposizioni concessive (sebbene)",
    "conditional_clauses": "proposizioni condizionali (se)",
    "relative_clauses": "proposizioni relative (che, cui)",
    "declarative_clauses": "proposizioni dichiarative",
    "indirect_questions": "interrogative indirette",
    # 5. Strutture Complesse
    "direct_indirect_speech": "discorso diretto e indiretto",
    "hypothetical_type1": "periodo ipotetico - tipo 1 (reale)",
    "hypothetical_type2": "periodo ipotetico - tipo 2 (possibile)",
    "hypothetical_type3": "periodo ipotetico - tipo 3 (irreale)",
    "subjunctive_in_subordinates": "uso del congiuntivo nelle subordinate",
    # 6. Ortografia
    "accents": "accenti (parole tronche, sdrucciole)",
    "apostrophe_elision": "apostrofo ed elisione",
    "double_consonants": "doppie consonanti",
    "h_usage": "uso di H (ho, ha, hanno)",
    # 7. Lessico
    "synonyms_antonyms": "sinonimi e antonimi",
    "homophones": "parole omofone (ha/a, anno/hanno)",
    "idiomatic_expressions": "espressioni idiomatiche",
    "proverbs": "proverbi",
}

# CEFR distribution
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
    "transformation",
]


def generate_prompts() -> List[Dict]:
    """
    Generate comprehensive prompts for GPT-4o-mini.

    Returns:
        List of prompt specifications
    """
    prompts = []

    # Part 1: Topic-focused examples (vocabulary diversity)
    print(
        f"📝 Generating {len(TOPIC_CATEGORIES)} topic categories × {SAMPLES_PER_TOPIC} samples..."
    )
    for topic, description in TOPIC_CATEGORIES.items():
        for _ in range(SAMPLES_PER_TOPIC):
            # Sample CEFR level
            cefr = random.choices(
                list(CEFR_DISTRIBUTION.keys()), weights=list(CEFR_DISTRIBUTION.values())
            )[0]

            # Sample grammar focus
            grammar = random.choice(list(GRAMMAR_FOCUSES.keys()))

            # Sample exercise types
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

    # Part 2: Grammar-focused examples (grammatical coverage)
    print(
        f"📝 Generating {len(GRAMMAR_FOCUSES)} grammar focuses × {SAMPLES_PER_GRAMMAR} samples..."
    )
    for grammar, description in GRAMMAR_FOCUSES.items():
        for _ in range(SAMPLES_PER_GRAMMAR):
            # Sample CEFR level
            cefr = random.choices(
                list(CEFR_DISTRIBUTION.keys()), weights=list(CEFR_DISTRIBUTION.values())
            )[0]

            # Sample topic
            topic = random.choice(list(TOPIC_CATEGORIES.keys()))

            # Sample exercise types
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

    # Shuffle all prompts
    random.shuffle(prompts)

    print(f"✅ Generated {len(prompts)} total prompts")
    return prompts


def create_generation_prompt(spec: Dict) -> str:
    """
    Create a prompt for GPT-4o-mini to generate exercises.
    """
    types_str = ", ".join(spec["exercise_types"])

    # Topic or grammar focused description
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


def generate_exercise_with_gpt4o_mini(client: OpenAI, spec: Dict) -> Dict:
    """
    Use GPT-4o-mini to generate an exercise example.
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

        # Remove markdown code blocks
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
                "source": f"gpt4o_mini_{spec['source']}",
            },
        }

    except json.JSONDecodeError as e:
        print(f"⚠️  JSON parsing failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Generation failed: {e}")
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

    print("🚀 Comprehensive Italian Exercise Augmentation")
    print(
        f"   Topic categories: {len(TOPIC_CATEGORIES)} × {SAMPLES_PER_TOPIC} = {len(TOPIC_CATEGORIES) * SAMPLES_PER_TOPIC}"
    )
    print(
        f"   Grammar focuses: {len(GRAMMAR_FOCUSES)} × {SAMPLES_PER_GRAMMAR} = {len(GRAMMAR_FOCUSES) * SAMPLES_PER_GRAMMAR}"
    )
    print(
        f"   Total target: {len(TOPIC_CATEGORIES) * SAMPLES_PER_TOPIC + len(GRAMMAR_FOCUSES) * SAMPLES_PER_GRAMMAR}"
    )
    print(f"   Output: {OUTPUT_FILE}")
    print()

    # Generate prompts
    print("📝 Generating prompt specifications...")
    prompts = generate_prompts()
    print()

    # Summary
    grammar_counts = {}
    topic_counts = {}
    for p in prompts:
        grammar_counts[p["grammar_focus"]] = grammar_counts.get(p["grammar_focus"], 0) + 1
        topic_counts[p["topic"]] = topic_counts.get(p["topic"], 0) + 1

    print("📊 Distribution Preview:")
    print(f"   Unique grammars: {len(grammar_counts)}")
    print(f"   Unique topics: {len(topic_counts)}")
    print()

    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Generate examples
    print(f"🤖 Generating exercises with GPT-4o-mini...")
    print(f"   (This will take ~{len(prompts) * 0.5 / 60:.0f} minutes)")
    print()

    generated = []
    failed = 0

    for i, spec in enumerate(prompts, 1):
        print(
            f"   [{i}/{len(prompts)}] {spec['cefr_level']} - {spec['topic']} - {spec['grammar_focus']}",
            end="",
            flush=True,
        )

        example = generate_exercise_with_gpt4o_mini(client, spec)

        if example:
            generated.append(example)
            print(" ✅")
        else:
            failed += 1
            print(" ❌")

        # Rate limiting
        time.sleep(DELAY_BETWEEN_REQUESTS)

        # Save incrementally every 100 examples
        if len(generated) % 100 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                for ex in generated:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            print(f"   💾 Saved {len(generated)} examples (checkpoint)")

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
    print("📊 Final Statistics:")
    print(f"   Total examples: {len(generated)}")
    print(
        f"   Grammar coverage: {len(set(ex['metadata']['grammar_focus'] for ex in generated))} unique"
    )
    print(f"   Topic coverage: {len(set(ex['metadata']['topic'] for ex in generated))} unique")
    print()

    # Next steps
    print("📋 Next steps:")
    print(f"   1. Merge with existing data:")
    print(
        f"      cat data/datasets/final/train.jsonl {OUTPUT_FILE} > data/datasets/v4_augmented/train.jsonl"
    )
    print(f"   2. Update config to point to v4_augmented/train.jsonl")
    print(f"   3. Retrain with V4 config (alpha=6)")


if __name__ == "__main__":
    main()
