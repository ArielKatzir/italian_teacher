"""
Quick test script to verify the reward function works correctly.

Run this to test:
    python utils/test_reward_function.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path (italian_teacher directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.reward_function.reward_function_modular import ExerciseRewardFunction


def load_api_keys():
    """Load API keys from secrets file."""
    secrets_path = Path.home() / "Google Drive" / "My Drive" / ".secrets.json"

    if not secrets_path.exists():
        print(f"\n❌ ERROR: Secrets file not found at: {secrets_path}")
        print("Please create this file with your API keys.")
        return False

    try:
        with open(secrets_path, 'r') as f:
            secrets = json.load(f)

        print(f"\n📄 Found secrets file with keys: {list(secrets.keys())}")

        # Try multiple possible key names (be flexible)
        key_mappings = [
            # Format 1: lowercase with underscore
            {
                "OPENAI_API_KEY": ["openai_api_key", "OPENAI_API_KEY", "openai"],
                "GOOGLE_API_KEY": ["gemini_api_key", "GOOGLE_API_KEY", "gemini", "google_api_key"],
                "GROQ_API_KEY": ["groq_api_key", "GROQ_API_KEY", "groq"],
                "ANTHROPIC_API_KEY": ["anthropic_api_key", "ANTHROPIC_API_KEY", "anthropic"],
                "DEEPSEEK_API_KEY": ["deepseek_api_key", "DEEPSEEK_API_KEY", "deepseek"],
            }
        ]

        loaded_keys = []
        for env_var, possible_keys in key_mappings[0].items():
            for secret_key in possible_keys:
                if secret_key in secrets and secrets[secret_key]:
                    os.environ[env_var] = secrets[secret_key]
                    provider = env_var.replace("_API_KEY", "")
                    loaded_keys.append(provider)
                    print(f"  ✅ {provider}: loaded from '{secret_key}'")
                    break  # Found it, stop looking

        if not loaded_keys:
            print("\n❌ No API keys found in secrets file!")
            print("   Available keys in file:", list(secrets.keys()))
            print("\n   Expected one of these key names:")
            print("   - openai_api_key or OPENAI_API_KEY")
            print("   - gemini_api_key or GOOGLE_API_KEY")
            print("   - groq_api_key or GROQ_API_KEY")
            print("   - anthropic_api_key or ANTHROPIC_API_KEY")
            print("   - deepseek_api_key or DEEPSEEK_API_KEY")
            return False

        print(f"\n✅ Total loaded: {len(loaded_keys)} API provider(s)")
        return True

    except Exception as e:
        print(f"\n❌ ERROR loading secrets: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_reward_function():
    """Test the reward function with multiple test cases."""

    print("=" * 80)
    print("TESTING REWARD FUNCTION WITH IMPROVED SCORERS")
    print("=" * 80)

    # Load API keys from secrets file
    if not load_api_keys():
        return False

    # Initialize reward function
    print("\n" + "-" * 80)
    print("Initializing reward function...")
    print("-" * 80)

    reward_fn = ExerciseRewardFunction(
        spacy_model="it_core_news_sm",
        device="cpu",
        disabled_scorers=[],
        fluency_use_llm=False,  # Disable for faster testing
        concurrency_limit=20
    )

    # Test exercises: good, bad, and ugly
    test_cases = [
        {
            "name": "GOOD: Proper fill-in-blank with hint",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Ieri (andare) ___ al mercato.",
                "correct_answer": "sono andato",
                "explanation": "Passato prossimo di andare, 1st person singular"
            },
            "request": {
                "level": "A2",
                "grammar_focus": "past_tense",
                "topic": "viaggi",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (75, 95)  # Should score well
        },
        {
            "name": "BAD: Wrong grammar (infinitive instead of imperative)",
            "exercise": {
                "type": "translation",
                "question": "Turn off the projector!",
                "correct_answer": "Spegnere la proiezione!",
                "explanation": "imperativo presente"
            },
            "request": {
                "level": "B1",
                "grammar_focus": "imperativo",
                "topic": "film e cinema",
                "num_exercises": 1,
                "exercise_types": ["translation"]
            },
            "expected_range": (40, 65)  # Should score low (wrong mood)
        },
        {
            "name": "UGLY: Wrong tense (passato remoto instead of imperfect)",
            "exercise": {
                "type": "translation",
                "question": "When I was on the plane, I felt a wave of nausea.",
                "correct_answer": "Quando ero in aereo, sentii un'onda di nausea.",
                "explanation": "imperfetto di sentire"
            },
            "request": {
                "level": "B2",
                "grammar_focus": "imperfect_tense",
                "topic": "vomito",
                "num_exercises": 1,
                "exercise_types": ["translation"]
            },
            "expected_range": (30, 55)  # Should score very low (wrong tense)
        },
        {
            "name": "BROKEN: Answer already in question",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Le vacanze al mare hanno ___ la famiglia",
                "correct_answer": "riempite di emozioni",
                "explanation": "passato remoto"
            },
            "request": {
                "level": "A2",
                "grammar_focus": "verbi_riflessivi",
                "topic": "vacanze al mare",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (25, 50)  # Should score very low (fundamentally broken)
        }
    ]

    # Score all test cases
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"TEST CASE {i}/{len(test_cases)}: {test_case['name']}")
        print("=" * 80)

        exercise = test_case["exercise"]
        request = test_case["request"]
        expected_min, expected_max = test_case["expected_range"]

        print(f"\nExercise:")
        print(f"  Type: {exercise['type']}")
        print(f"  Question: {exercise['question']}")
        print(f"  Answer: {exercise['correct_answer']}")
        print(f"\nRequest:")
        print(f"  Level: {request['level']}")
        print(f"  Grammar: {request['grammar_focus']}")
        print(f"  Topic: {request['topic']}")
        print(f"\nExpected Score Range: {expected_min}-{expected_max}")

        try:
            score, breakdown = await reward_fn.score(exercise, request)

            print(f"\n{'='*60}")
            print(f"SCORE: {score:.2f}/100")
            print(f"{'='*60}")

            # Check if score is in expected range
            in_range = expected_min <= score <= expected_max
            status = "✅ PASS" if in_range else "❌ FAIL"

            print(f"\nStatus: {status}")
            if not in_range:
                print(f"  Expected: {expected_min}-{expected_max}")
                print(f"  Got: {score:.2f}")

            print(f"\nBreakdown:")
            print(f"{breakdown}")

            results.append({
                "name": test_case["name"],
                "score": score,
                "expected_range": (expected_min, expected_max),
                "passed": in_range
            })

        except Exception as e:
            print(f"\n❌ Error during scoring: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": test_case["name"],
                "score": None,
                "expected_range": (expected_min, expected_max),
                "passed": False,
                "error": str(e)
            })

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print(f"\nTests Passed: {passed}/{total}")
    print("\nDetailed Results:")
    for r in results:
        status = "✅" if r["passed"] else "❌"
        score_str = f"{r['score']:.2f}" if r['score'] is not None else "ERROR"
        expected = f"{r['expected_range'][0]}-{r['expected_range'][1]}"
        print(f"  {status} {r['name']}: {score_str} (expected {expected})")

    if passed == total:
        print("\n🎉 All tests passed! Reward function is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the scores above.")

    return passed == total


if __name__ == "__main__":
    asyncio.run(test_reward_function())
