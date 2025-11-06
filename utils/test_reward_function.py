"""
Quick test script to verify the Italian reward function works correctly.

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

from src.rl.reward_function.subjects.italian import ItalianRewardFunction


def load_api_keys():
    """Load API keys from secrets file (checks multiple locations)."""
    secrets_paths = [
        Path.home() / "Google Drive" / "My Drive" / ".secrets.json",  # Local path
        Path("/content/drive/My Drive/.secrets.json"),                # Colab path
        Path('.secrets.json')                                         # Current directory
    ]

    # Try each path
    secrets_path = None
    for path in secrets_paths:
        if path.exists():
            secrets_path = path
            break

    if not secrets_path:
        print(f"\n❌ ERROR: Secrets file not found in any of these locations:")
        for path in secrets_paths:
            print(f"  - {path}")
        print("\nPlease create a .secrets.json file with your API keys.")
        return False

    try:
        with open(secrets_path, 'r') as f:
            secrets = json.load(f)

        print(f"\n📄 Found secrets file at: {secrets_path}")
        print(f"   Available keys: {list(secrets.keys())}")

        # Try multiple possible key names (be flexible)
        key_mappings = {
            "OPENAI_API_KEY": ["openai_api_key", "OPENAI_API_KEY", "openai"],
            "GOOGLE_API_KEY": ["gemini_api_key", "GOOGLE_API_KEY", "gemini", "google_api_key"],
            "GROQ_API_KEY": ["groq_api_key", "GROQ_API_KEY", "groq"],
            "ANTHROPIC_API_KEY": ["anthropic_api_key", "ANTHROPIC_API_KEY", "anthropic"],
            "DEEPSEEK_API_KEY": ["deepseek_api_key", "DEEPSEEK_API_KEY", "deepseek"],
        }

        loaded_keys = []
        for env_var, possible_keys in key_mappings.items():
            for secret_key in possible_keys:
                if secret_key in secrets and secrets[secret_key]:
                    # Skip placeholder values
                    if secrets[secret_key] not in ["your-key-here", "", "sk-..."]:
                        os.environ[env_var] = secrets[secret_key]
                        provider = env_var.replace("_API_KEY", "")
                        loaded_keys.append(provider)
                        print(f"  ✅ {provider}: loaded from '{secret_key}'")
                        break  # Found it, stop looking

        if not loaded_keys:
            print("\n❌ No valid API keys found in secrets file!")
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
    """Test the Italian reward function with comprehensive test cases."""

    print("=" * 80)
    print("TESTING ITALIAN REWARD FUNCTION (NEW ARCHITECTURE)")
    print("=" * 80)

    # Load API keys from secrets file
    if not load_api_keys():
        return False

    # Initialize reward function
    print("\n" + "-" * 80)
    print("Initializing Italian reward function...")
    print("-" * 80)

    reward_fn = ItalianRewardFunction(
        device="cpu",
        disabled_scorers=[],
        fluency_use_llm=False,  # Disable for faster testing
        concurrency_limit=20
    )

    # Comprehensive test cases: good, bad, ugly, and challenging
    test_cases = [
        {
            "name": "✅ GOOD: Perfect A2 fill-in-blank with hint",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Ieri (andare) ___ al mercato con mia madre.",
                "correct_answer": "sono andato",
                "explanation": "Passato prossimo di andare, 1st person singular"
            },
            "request": {
                "level": "A2",
                "grammar_focus": "past_tense",
                "topic": "shopping",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (75, 95)
        },
        {
            "name": "✅ GOOD: Proper B1 imperfect tense",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Quando ero piccolo, (mangiare) ___ sempre la pizza il venerdì.",
                "correct_answer": "mangiavo",
                "explanation": "Imperfetto di mangiare"
            },
            "request": {
                "level": "B1",
                "grammar_focus": "imperfect_tense",
                "topic": "childhood memories",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (75, 95)
        },
        {
            "name": "❌ BAD: Wrong grammar (infinitive instead of imperative)",
            "exercise": {
                "type": "translation",
                "question": "Turn off the projector!",
                "correct_answer": "Spegnere la proiezione!",
                "explanation": "Should use imperative, not infinitive"
            },
            "request": {
                "level": "B1",
                "grammar_focus": "imperativo",
                "topic": "cinema",
                "num_exercises": 1,
                "exercise_types": ["translation"]
            },
            "expected_range": (20, 50)  # Should score low (wrong mood)
        },
        {
            "name": "❌ BAD: Wrong tense (passato remoto vs imperfect)",
            "exercise": {
                "type": "translation",
                "question": "When I was on the plane, I felt nauseous.",
                "correct_answer": "Quando ero in aereo, sentii la nausea.",
                "explanation": "Should use imperfect 'sentivo', not passato remoto 'sentii'"
            },
            "request": {
                "level": "B2",
                "grammar_focus": "imperfect_tense",
                "topic": "travel",
                "num_exercises": 1,
                "exercise_types": ["translation"]
            },
            "expected_range": (15, 45)  # Should score very low (wrong tense)
        },
        {
            "name": "💀 BROKEN: Answer already in question",
            "exercise": {
                "type": "fill_in_blank",
                "question": "La pizza è ___ pizza",
                "correct_answer": "la",
                "explanation": "Redundant answer"
            },
            "request": {
                "level": "A2",
                "grammar_focus": "articles",
                "topic": "food",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (0, 30)  # Should score very low (broken exercise)
        },
        {
            "name": "💀 BROKEN: Missing reflexive pronoun",
            "exercise": {
                "type": "fill_in_blank",
                "question": "(chiamare) ___ Marco.",
                "correct_answer": "chiamo",
                "explanation": "Should be 'mi chiamo' (reflexive)"
            },
            "request": {
                "level": "A1",
                "grammar_focus": "verbi_riflessivi",
                "topic": "introductions",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (0, 35)  # Should score low (missing reflexive)
        },
        {
            "name": "🔥 HARD: Complex B2 subjunctive",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Penso che tu (dovere) ___ studiare di più prima dell'esame.",
                "correct_answer": "debba",
                "explanation": "Congiuntivo presente di dovere"
            },
            "request": {
                "level": "B2",
                "grammar_focus": "congiuntivo",
                "topic": "education",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (70, 95)
        },
        {
            "name": "🔥 HARD: C1 conditional perfect",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Se avessi saputo, (venire) ___ alla festa.",
                "correct_answer": "sarei venuto",
                "explanation": "Condizionale passato di venire"
            },
            "request": {
                "level": "C1",
                "grammar_focus": "conditional_perfect",
                "topic": "social events",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (70, 95)
        },
        {
            "name": "🔥 HARD: Passato remoto (literary tense)",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Dante (scrivere) ___ la Divina Commedia nel XIV secolo.",
                "correct_answer": "scrisse",
                "explanation": "Passato remoto di scrivere"
            },
            "request": {
                "level": "B2",
                "grammar_focus": "passato_remoto",
                "topic": "literature",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (70, 95)
        },
        {
            "name": "⚡ EDGE: Very short answer",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Mi chiamo ___.",
                "correct_answer": "a",
                "explanation": "Too short to be valid"
            },
            "request": {
                "level": "A1",
                "grammar_focus": "general",
                "topic": "names",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (30, 60)  # Should penalize short answer
        },
        {
            "name": "⚡ EDGE: Level mismatch (A1 exercise for C1 level)",
            "exercise": {
                "type": "fill_in_blank",
                "question": "Il gatto è ___.",
                "correct_answer": "nero",
                "explanation": "Too simple for C1"
            },
            "request": {
                "level": "C1",
                "grammar_focus": "adjectives",
                "topic": "animals",
                "num_exercises": 1,
                "exercise_types": ["fill_in_blank"]
            },
            "expected_range": (25, 55)  # Should score low (too simple for level)
        },
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

    # Categorize results
    good_tests = [r for r in results if r["name"].startswith("✅")]
    bad_tests = [r for r in results if r["name"].startswith("❌")]
    broken_tests = [r for r in results if r["name"].startswith("💀")]
    hard_tests = [r for r in results if r["name"].startswith("🔥")]
    edge_tests = [r for r in results if r["name"].startswith("⚡")]

    print("\nDetailed Results by Category:")

    if good_tests:
        print(f"\n  ✅ GOOD Exercises ({sum(1 for r in good_tests if r['passed'])}/{len(good_tests)} passed):")
        for r in good_tests:
            status = "✅" if r["passed"] else "❌"
            score_str = f"{r['score']:.2f}" if r['score'] is not None else "ERROR"
            print(f"    {status} {score_str}/100 - {r['name'][2:]}")

    if bad_tests:
        print(f"\n  ❌ BAD Exercises ({sum(1 for r in bad_tests if r['passed'])}/{len(bad_tests)} passed):")
        for r in bad_tests:
            status = "✅" if r["passed"] else "❌"
            score_str = f"{r['score']:.2f}" if r['score'] is not None else "ERROR"
            print(f"    {status} {score_str}/100 - {r['name'][2:]}")

    if broken_tests:
        print(f"\n  💀 BROKEN Exercises ({sum(1 for r in broken_tests if r['passed'])}/{len(broken_tests)} passed):")
        for r in broken_tests:
            status = "✅" if r["passed"] else "❌"
            score_str = f"{r['score']:.2f}" if r['score'] is not None else "ERROR"
            print(f"    {status} {score_str}/100 - {r['name'][2:]}")

    if hard_tests:
        print(f"\n  🔥 HARD Exercises ({sum(1 for r in hard_tests if r['passed'])}/{len(hard_tests)} passed):")
        for r in hard_tests:
            status = "✅" if r["passed"] else "❌"
            score_str = f"{r['score']:.2f}" if r['score'] is not None else "ERROR"
            print(f"    {status} {score_str}/100 - {r['name'][2:]}")

    if edge_tests:
        print(f"\n  ⚡ EDGE Cases ({sum(1 for r in edge_tests if r['passed'])}/{len(edge_tests)} passed):")
        for r in edge_tests:
            status = "✅" if r["passed"] else "❌"
            score_str = f"{r['score']:.2f}" if r['score'] is not None else "ERROR"
            print(f"    {status} {score_str}/100 - {r['name'][2:]}")

    if passed == total:
        print("\n🎉 All tests passed! Italian reward function is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review the scores above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(test_reward_function())
    sys.exit(0 if success else 1)
