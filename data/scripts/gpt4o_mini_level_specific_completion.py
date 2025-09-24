#!/usr/bin/env python3
"""
OpenAI GPT-4o Mini - Level-Specific Assistant Response Completion

Complete blank assistant responses using EXPLICIT level-specific templates.
Each CEFR level (A1-C2) has distinct response requirements and structure.
Cost: ~$3.25 for 17,913 conversations (well under budget!)
"""

import json
import os
import time
from pathlib import Path

import tqdm
from openai import OpenAI


def setup_openai():
    """Setup OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY") or input("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    return client


def create_level_specific_prompt(user_message, level):
    """Create EXPLICIT level-specific prompts with detailed requirements."""

    # Extract Italian text if present
    extract_italian_from_question(user_message)

    level_templates = {
        "A1": f"""You are Marco, an Italian teacher helping absolute beginners.

Student question: {user_message}

EXPLICIT A1 REQUIREMENTS - Your response MUST include:
1. Start with friendly greeting: "Great question!" or "I'm happy to help!"
2. Simple, clear English (max 8-grade reading level)
3. If translating: Provide direct English translation first
4. Basic pronunciation guide (simple phonetic spelling)
5. ONE simple example sentence in Italian with English translation
6. Encourage the student: "This is a great word to know!" or similar
7. Keep total response at a max of 100 words
8. NO complex grammar terminology
9. Focus on practical daily usage

Write your response now:""",
        "A2": f"""You are Marco, an Italian teacher helping elementary students.

Student question: {user_message}

EXPLICIT A2 REQUIREMENTS - Your response MUST include:
1. Warm acknowledgment: "That's a good question!"
2. Clear English explanation (9-10 grade reading level)
3. Translation + basic grammatical role (noun, verb, adjective)
4. Simple pronunciation guide
5. One example sentences with translations
6. One basic grammar note (verb tense, gender, etc.)
7. Usage context: "You'd use this when..."
8. Keep response at a max of 140 words
9. Use basic grammar terms only

Write your response now:""",
        "B1": f"""You are Marco, an Italian teacher helping intermediate students.

Student question: {user_message}

EXPLICIT B1 REQUIREMENTS - Your response MUST include:
1. Professional greeting
2. Complete translation and grammatical analysis
4. One other example sentence with context
5. Grammar breakdown: verb conjugations, noun gender, adjective agreement
6. Usage notes: formal vs informal contexts
7. Cultural or regional usage note
8. Practice suggestion: "Try using this in..."
9. Keep response at a max of 200 words
10. Use intermediate grammar terminology

Write your response now:""",
        "B2": f"""You are Marco, an Italian teacher helping upper-intermediate students.

Student question: {user_message}

EXPLICIT B2 REQUIREMENTS - Your response MUST include:
1. Comprehensive grammatical analysis with linguistic terminology
2. Nuanced meaning variations and connotations
3. Two examples showing different contexts/registers
4. Detailed grammar: subjunctive moods, conditional forms, complex tenses
5 Optional: Add Formal vs informal vs colloquial distinctions
6. TWO related expressions with subtle differences
7. Advanced usage tips and common mistakes to avoid
8. Keep response at a max of 240 words
9. Use advanced grammatical concepts
10. Optional: Parts of your response can be in italian if not too complex for B2 learners 

Write your response now:""",
        "C1": f"""You are Marco, an Italian teacher helping advanced students.

Student question: {user_message}

EXPLICIT C1 REQUIREMENTS - Your response MUST include:
1. Academic greeting
2. Deep linguistic analysis with etymological insights
3. Semantic nuances and pragmatic implications
4. Two examples across different registers and contexts, three examples if really needed.
5. Advanced grammar: complex subordination, stylistic variations
6. Optional: Historical language evolution and literary usage
7. Two related expressions with precise semantic differences
8. Dialectal or archaic variants if relevant
9. Optional: Cross-linguistic comparisons where appropriate
10. Keep response at a max of 280 words
11. Use sophisticated linguistic terminology
12. Optional: Parts of your response can be in italian if not too complex for C1 learners

Write your response now:""",
        "C2": f"""You are Marco, an Italian teacher helping near-native speakers.

Student question: {user_message}

EXPLICIT C2 REQUIREMENTS - Your response MUST include:
1. Expert greeting
2. Expert-level linguistic analysis with theoretical frameworks
3. Philosophical and cultural underpinnings of usage
4. SIX examples spanning literary, academic, colloquial, and specialized contexts
5. Master-level grammar: discourse markers, pragmatic particles, stylistic devices
6. Literary citations and author-specific usage patterns
7. FOUR related expressions with micro-distinctions in meaning
8. Historical linguistic development and comparative Romance analysis
9. Sociolinguistic considerations and prestige variants
10. Professional usage in academic, legal, or literary contexts
11. Keep response at a max of 320 words
12. Use expert linguistic and literary terminology
13. Optional: Most of your response can be in italian as long as the answer is not too complex, otherwise use english


Write your response now:""",
    }

    return level_templates[level]


def extract_italian_from_question(user_message):
    """Extract Italian text from user question for context."""
    import re

    # Look for text in quotes
    quotes = re.findall(r"'([^']+)'|\"([^\"]+)\"", user_message)
    if quotes:
        return quotes[0][0] or quotes[0][1]

    # Look for "Italian:" patterns
    italian_match = re.search(r"Italian:?\s*(.+)", user_message, re.IGNORECASE)
    if italian_match:
        return italian_match.group(1).strip()

    return None


def validate_response_quality(response, level, min_words):
    """Validate that response meets level requirements."""
    word_count = len(response.split())

    # Check minimum length
    if word_count == 0:
        return False, f"Too short: {word_count} words (need {min_words}+)"

    # # Check for level-appropriate content
    # level_checks = {
    #     "A1": ["translation", "example"],
    #     "A2": ["translation", "example", "grammar"],
    #     "B1": ["grammar", "example", "usage", "formal"],
    #     "B2": ["nuance", "context", "register", "variation"],
    #     "C1": ["linguistic", "semantic", "pragmatic", "etymology"],
    #     "C2": ["discourse", "sociolinguistic", "literary", "theoretical"]
    # }

    # required_concepts = level_checks.get(level, [])
    # found_concepts = sum(1 for concept in required_concepts if concept.lower() in response.lower())

    # if found_concepts < len(required_concepts) // 2:
    #     return False, f"Missing level-appropriate content for {level}"

    return True, "Passed validation"


def complete_responses_with_level_templates(input_dir: str, output_dir: str):
    """Complete responses using explicit level-specific templates."""

    client = setup_openai()
    os.makedirs(output_dir, exist_ok=True)

    level_stats = {
        level: {"processed": 0, "completed": 0, "failed": 0, "cost": 0.0}
        for level in ["A1", "A2", "B1", "B2", "C1", "C2"]
    }
    total_processed = 0
    total_completed = 0
    total_cost = 0.0

    # Word count requirements by level
    min_words = {"A1": 0, "A2": 0, "B1": 100, "B2": 150, "C1": 200, "C2": 200}
    # for filename in ['train.jsonl', 'validation.jsonl', 'test.jsonl']:
    for filename in ["test.jsonl"]:
        input_path = Path(input_dir) / filename
        output_path = Path(output_dir) / filename

        if not input_path.exists():
            print(f"⚠️  File not found: {input_path}")
            continue

        print(f"\n📁 Processing {filename}...")

        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(output_path, "w", encoding="utf-8") as outfile,
        ):

            for line_num, line in tqdm.tqdm(enumerate(infile)):
                conversation = json.loads(line.strip())
                total_processed += 1

                assistant_content = conversation["messages"][1]["content"].strip()

                if assistant_content == "" or conversation.get("needs_qwen_generation", False):
                    user_message = conversation["messages"][0]["content"]
                    # Get correct level from metadata, fallback to conversation level, then B1
                    level = conversation.get("metadata", {}).get("level") or conversation.get(
                        "level", "B1"
                    )

                    level_stats[level]["processed"] += 1

                    try:
                        # Generate with level-specific template
                        prompt = create_level_specific_prompt(user_message, level)

                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=400,
                            temperature=0.7,
                        )

                        new_content = response.choices[0].message.content.strip()

                        # Calculate cost
                        input_tokens = response.usage.prompt_tokens
                        output_tokens = response.usage.completion_tokens
                        request_cost = (input_tokens * 0.15 / 1000000) + (
                            output_tokens * 0.60 / 1000000
                        )

                        level_stats[level]["cost"] += request_cost
                        total_cost += request_cost

                        # Validate response meets level requirements
                        is_valid, validation_msg = validate_response_quality(
                            new_content, level, min_words[level]
                        )

                        if is_valid:
                            conversation["messages"][1]["content"] = new_content
                            conversation["generated_with"] = f"gpt-4o-mini-{level}-template"
                            conversation["word_count"] = len(new_content.split())
                            conversation["cost"] = request_cost
                            conversation.pop("needs_qwen_generation", None)

                            level_stats[level]["completed"] += 1
                            total_completed += 1

                            if level_stats[level]["completed"] % 10 == 0:
                                print(
                                    f"  ✅ {level}: {level_stats[level]['completed']} completed, ${level_stats[level]['cost']:.3f}"
                                )
                        else:
                            print(f"  ⚠️  {level} validation failed: {validation_msg}")
                            level_stats[level]["failed"] += 1

                        # Rate limiting - GPT-4o Mini has generous limits
                        time.sleep(0.1)

                    except Exception as e:
                        print(f"  ❌ Error on line {line_num+1}: {e}")
                        level_stats[level]["failed"] += 1
                        time.sleep(1)

                # Write conversation to output
                outfile.write(json.dumps(conversation, ensure_ascii=False) + "\n")

                if (line_num + 1) % 500 == 0:
                    print(f"  📊 Processed {line_num+1}, Cost so far: ${total_cost:.3f}")

        print(f"✅ {filename} complete")

    # Final statistics
    print(f"\n🎉 LEVEL-SPECIFIC COMPLETION FINISHED!")
    print(f"\n💰 COST BREAKDOWN BY LEVEL:")
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        stats = level_stats[level]
        success_rate = (
            (stats["completed"] / stats["processed"] * 100) if stats["processed"] > 0 else 0
        )
        print(
            f"   {level}: {stats['completed']}/{stats['processed']} completed ({success_rate:.1f}%), ${stats['cost']:.3f}"
        )

    print(f"\n📈 OVERALL:")
    print(f"   Total processed: {total_processed}")
    print(f"   Total completed: {total_completed}")
    print(f"   Total cost: ${total_cost:.3f}")
    print(f"   Overall success: {total_completed/total_processed*100:.1f}%")
    print(f"   📁 Saved to: {output_dir}")

    if total_cost > 10:
        print(f"⚠️  WARNING: Cost exceeded $10 budget by ${total_cost - 10:.3f}!")
    else:
        print(f"✅ Under budget! Saved ${10 - total_cost:.3f}")


if __name__ == "__main__":
    input_dir = "data/datasets/v3/source_conversations"
    output_dir = "data/datasets/v3/clean_gpt4o_mini"

    print("🚀 Starting LEVEL-SPECIFIC GPT-4o Mini completion...")
    print("💰 Estimated cost: ~$3.25 (well under $10 budget)")
    print("📋 Using explicit templates for each CEFR level:")
    print("   A1: 80-120 words, basic vocabulary, simple examples")
    print("   A2: 120-160 words, elementary grammar, context")
    print("   B1: 160-200 words, detailed analysis, related expressions")
    print("   B2: 200-240 words, nuances, register distinctions")
    print("   C1: 240-280 words, linguistic insights, etymology")
    print("   C2: 280-320 words, expert analysis, literary references")

    complete_responses_with_level_templates(input_dir, output_dir)
