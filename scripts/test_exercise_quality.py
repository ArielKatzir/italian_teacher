#!/usr/bin/env python3
"""
Quick test script for exercise quality validation.

Usage:
    python test_exercise_quality.py
"""

import os

from utils.exercise_validator import ExerciseValidator, GenerationConfig

# Get API URL from environment
API_URL = os.getenv("INFERENCE_API_URL")
if not API_URL:
    print("❌ Error: INFERENCE_API_URL not set")
    print('   Run: export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"')
    exit(1)

print(f"🔗 Using API: {API_URL}\n")

validator = ExerciseValidator(API_URL)

# Test configuration
config = GenerationConfig(
    cefr_level="A2",
    grammar_focus="present_tense",
    topic="daily routines",
    quantity=5,
    exercise_types=["fill_in_blank", "translation", "multiple_choice"],
    temperature=0.7,
    max_tokens=2500,  # Increased to allow all 5 exercises
)

print("🧪 Testing exercise generation...")
print(f"   CEFR Level: {config.cefr_level}")
print(f"   Grammar: {config.grammar_focus}")
print(f"   Topic: {config.topic}")
print(f"   Quantity: {config.quantity}")
print(f"   Temperature: {config.temperature}\n")

# Generate and get raw response first
print("⏳ Generating exercises...\n")
response = validator.generate_exercises(config)

# Print raw model output
print("=" * 80)
print("RAW MODEL OUTPUT (Parsed Exercises JSON)")
print("=" * 80)
print(f"\n🔢 Total Tokens: {response.get('generated_tokens', 0)}")
print(f"⏱️  Generation Time: {response.get('inference_time', 0):.2f}s")
print(f"📊 Parsing Strategy: {response.get('parsing_strategy', 'unknown')}")
print("\n" + "-" * 80)
print("Parsed exercises as JSON:")
print("-" * 80)

# Show the exercises in formatted JSON
import json

print(json.dumps(response.get("exercises", []), indent=2, ensure_ascii=False))
print("-" * 80 + "\n")

# Now validate
result = validator.validate_exercises(
    exercises=response["exercises"],
    expected_quantity=config.quantity,
    parsing_strategy=response.get("parsing_strategy", "unknown"),
    inference_time=response.get("inference_time", 0),
    tokens_generated=response.get("generated_tokens", 0),
)

# Print validation results
print("=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)
print(f"\n✅ Success: {result.success}")
print(f"📊 Quality Score: {result.quality_score:.1f}/100")
print(f"⚡ Parsing Strategy: {result.parsing_strategy}")
print(f"⏱️  Inference Time: {result.inference_time:.2f}s")
print(f"🔢 Tokens Generated: {result.tokens_generated}")
print(f"\n📝 Exercises Generated: {result.num_exercises}/{config.quantity}")
print(f"🔄 Unique Questions: {result.unique_questions}")
print(f"⚠️  Duplicates: {result.duplicate_count}")
print(f"📏 Avg Question Length: {result.avg_question_length} chars")
print(f"📏 Avg Answer Length: {result.avg_answer_length} chars")

if result.issues:
    print(f"\n🐛 Issues Found ({len(result.issues)}):")
    for issue in result.issues:
        print(f"   - {issue}")
else:
    print("\n✨ No issues found!")

print("\n" + "=" * 80)
print("ALL EXERCISES")
print("=" * 80)

for i, ex in enumerate(result.exercises, 1):
    print(f"\nExercise {i}:")
    print(f"  Type: {ex.get('type', 'N/A')}")
    print(f"  Question: {ex.get('question', 'N/A')}")
    print(f"  Answer: {ex.get('correct_answer', 'N/A')}")
    if ex.get("options"):
        print(f"  Options: {ex['options']}")
    print(f"  Explanation: {ex.get('explanation', 'N/A')}")

print("\n" + "=" * 80)

# Recommendations
print("\n💡 RECOMMENDATIONS:\n")

if result.quality_score < 50:
    print("⚠️  Quality is LOW. Consider:")
    print("   - Lower temperature (try 0.3-0.5)")
    print("   - Improve prompt with more examples")
    print("   - Check if model is loaded correctly")

elif result.quality_score < 70:
    print("📈 Quality is FAIR. Suggestions:")
    print("   - Fine-tune temperature (try 0.4-0.6)")
    print("   - Add few-shot examples to prompt")

else:
    print("✨ Quality is GOOD!")
    print(f"   Current temperature ({config.temperature}) is working well")

if result.has_duplicates:
    print("\n🔄 DUPLICATE ISSUE:")
    print("   - Try lower temperature for more variety")
    print("   - Increase max_tokens to allow more diverse generation")

if result.parsing_strategy.startswith("strategy4") or result.parsing_strategy.startswith(
    "strategy5"
):
    print("\n⚠️  PARSING FALLBACK:")
    print("   - Model not generating valid JSON")
    print("   - Try temperature 0.2-0.4 for better structure")
    print("   - Check prompt formatting")

print("\n" + "=" * 80 + "\n")
