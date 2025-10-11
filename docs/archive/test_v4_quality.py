#!/usr/bin/env python3
"""
Test V4 model quality improvements.

This script tests the V4 model against known problematic cases from V3:
1. Gender errors: "Gli aquile" → should be "Le aquile"
2. Tense mismatches: exercises not matching grammar_focus
3. Vocabulary gaps: ragno, lombrico, aquila
"""

import asyncio
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from src.api.database import get_db_session
from src.api.services.homework_service import HomeworkService

TEST_CASES = [
    {
        "name": "Eagles (aquila) - Past Tense",
        "cefr_level": "A2",
        "grammar_focus": "past_tense",
        "topic": "eagles",
        "expected": {
            "gender": "Le aquile (feminine plural)",
            "tense": "passato prossimo or imperfetto",
            "topic": "eagles/aquila",
        },
    },
    {
        "name": "Spiders (ragno) - Past Tense",
        "cefr_level": "A2",
        "grammar_focus": "past_tense",
        "topic": "spiders",
        "expected": {
            "gender": "Il ragno (masculine singular) / I ragni (masculine plural)",
            "tense": "passato prossimo or imperfetto",
            "topic": "spiders/ragno",
        },
    },
    {
        "name": "Worms (lombrico) - Present Tense",
        "cefr_level": "A2",
        "grammar_focus": "present_tense",
        "topic": "worms",
        "expected": {
            "gender": "Il lombrico (masculine singular) / I lombrichi (masculine plural)",
            "tense": "presente indicativo",
            "topic": "worms/lombrico",
        },
    },
]


async def test_v4_quality():
    """Test V4 model quality."""

    print("=" * 80)
    print("🧪 V4 MODEL QUALITY TEST")
    print("=" * 80)
    print()

    # Check environment
    inference_url = os.getenv("INFERENCE_API_URL")
    if inference_url:
        print(f"✅ Using Colab GPU: {inference_url}")
    else:
        print("⚠️  INFERENCE_API_URL not set - will use mock mode")
        print("   To test real V4 model:")
        print("   1. Start Colab notebook: demos/colab_inference_api.ipynb")
        print('   2. export INFERENCE_API_URL="<ngrok-url>"')
        print("   3. Re-run this script")
        print()
        return

    print()

    # Initialize service
    async for session in get_db_session():
        service = HomeworkService(session)

        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n{'=' * 80}")
            print(f"TEST {i}/{len(TEST_CASES)}: {test_case['name']}")
            print(f"{'=' * 80}")
            print(f"Level: {test_case['cefr_level']}")
            print(f"Grammar: {test_case['grammar_focus']}")
            print(f"Topic: {test_case['topic']}")
            print()

            try:
                # Generate exercises
                print("⏳ Generating exercises with V4 model...")
                exercises = await service.generate_exercises(
                    cefr_level=test_case["cefr_level"],
                    grammar_focus=test_case["grammar_focus"],
                    topic=test_case["topic"],
                    num_exercises=3,
                )

                if not exercises:
                    print("❌ No exercises generated")
                    continue

                print(f"✅ Generated {len(exercises)} exercises\n")

                # Analyze each exercise
                for j, ex in enumerate(exercises, 1):
                    print(f"--- Exercise {j}: {ex['type']} ---")
                    print(f"Question: {ex['question'][:100]}...")
                    print(f"Answer: {ex['answer'][:100]}...")

                    # Check for expected vocabulary
                    full_text = f"{ex['question']} {ex['answer']}".lower()

                    # Gender check
                    if "aquila" in test_case["topic"]:
                        if "aquila" in full_text or "aquile" in full_text:
                            print("✅ Contains 'aquila/aquile'")
                            if "le aquile" in full_text or "l'aquila" in full_text:
                                print("✅ CORRECT GENDER: 'le aquile' or \"l'aquila\" (feminine)")
                            elif "gli aquile" in full_text or "il aquila" in full_text:
                                print("❌ GENDER ERROR: Found 'gli aquile' or 'il aquila'")
                        else:
                            print("⚠️  Word 'aquila' not found in exercise")

                    elif "ragno" in test_case["topic"]:
                        if "ragno" in full_text or "ragni" in full_text:
                            print("✅ Contains 'ragno/ragni'")
                            if "il ragno" in full_text or "i ragni" in full_text:
                                print("✅ CORRECT GENDER: 'il ragno' or 'i ragni' (masculine)")
                            elif "la ragno" in full_text or "le ragni" in full_text:
                                print("❌ GENDER ERROR: Found 'la ragno' or 'le ragni'")
                        else:
                            print("⚠️  Word 'ragno' not found in exercise")

                    elif "lombrico" in test_case["topic"]:
                        if "lombrico" in full_text or "lombrichi" in full_text:
                            print("✅ Contains 'lombrico/lombrichi'")
                            if "il lombrico" in full_text or "i lombrichi" in full_text:
                                print(
                                    "✅ CORRECT GENDER: 'il lombrico' or 'i lombrichi' (masculine)"
                                )
                            elif "la lombrico" in full_text or "le lombrichi" in full_text:
                                print("❌ GENDER ERROR: Found 'la lombrico' or 'le lombrichi'")
                        else:
                            print("⚠️  Word 'lombrico' not found in exercise")

                    # Tense check (basic - look for verb indicators)
                    if test_case["grammar_focus"] == "past_tense":
                        past_indicators = [
                            "ho ",
                            "hai ",
                            "ha ",
                            "abbiamo",
                            "avete",
                            "hanno",
                            "sono andato",
                            "era",
                            "erano",
                            "faceva",
                        ]
                        if any(ind in full_text for ind in past_indicators):
                            print(f"✅ Likely past tense (found past indicators)")
                        else:
                            print(f"⚠️  No clear past tense indicators")

                    print()

            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback

                traceback.print_exc()

        print("\n" + "=" * 80)
        print("🏁 TEST COMPLETE")
        print("=" * 80)
        break  # Only need one session


if __name__ == "__main__":
    asyncio.run(test_v4_quality())
