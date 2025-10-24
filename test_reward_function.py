"""
Quick test script to verify the Gemini-integrated reward function works correctly.

Run this to test:
    python test_reward_function.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rl.reward_function.reward_function_modular import ExerciseRewardFunction


async def test_reward_function():
    """Test the reward function with a simple exercise."""

    print("=" * 80)
    print("TESTING GEMINI-INTEGRATED REWARD FUNCTION")
    print("=" * 80)

    # Check API keys
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_gemini = bool(os.environ.get("GOOGLE_API_KEY"))

    print(f"\nAPI Keys found:")
    print(f"  OpenAI: {'✅' if has_openai else '❌'}")
    print(f"  Gemini: {'✅' if has_gemini else '❌'}")

    if not has_openai and not has_gemini:
        print("\n❌ ERROR: No API keys found!")
        print("Set OPENAI_API_KEY or GOOGLE_API_KEY environment variable.")
        return

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

    # Test exercise and request
    test_exercise = {
        "type": "fill_in_blank",
        "question": "Ieri (andare) ___ al mercato.",
        "correct_answer": "sono andato",
        "explanation": "Passato prossimo di andare, 1st person singular"
    }

    test_request = {
        "level": "A2",
        "grammar_focus": "passato prossimo",
        "topic": "viaggi",
        "num_exercises": 1,
        "exercise_types": ["fill_in_blank"]
    }

    print("\n" + "-" * 80)
    print("Test Exercise:")
    print("-" * 80)
    print(f"Type: {test_exercise['type']}")
    print(f"Question: {test_exercise['question']}")
    print(f"Answer: {test_exercise['correct_answer']}")

    print("\n" + "-" * 80)
    print("Test Request:")
    print("-" * 80)
    print(f"Level: {test_request['level']}")
    print(f"Grammar: {test_request['grammar_focus']}")
    print(f"Topic: {test_request['topic']}")

    # Score the exercise
    print("\n" + "-" * 80)
    print("Scoring exercise...")
    print("-" * 80)

    try:
        score, breakdown = await reward_fn.score(test_exercise, test_request)

        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"\nTotal Score: {score:.2f}/100")
        print(f"\n{breakdown}")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during scoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_reward_function())
