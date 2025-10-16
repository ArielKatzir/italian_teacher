"""
Modular Reward Function for Italian Exercise Generation.

Scores exercises on 6 dimensions (0-100 points total):
1. JSON Validity (15 points) - Structure and format
2. Linguistic Quality (35 points) - Comprehensive Italian grammar validation
3. CEFR Level Alignment (20 points) - Appropriate difficulty for target level
4. Fluency (10 points) - Natural language flow and construction
5. Grammar Correctness (10 points) - Matches requested grammar_focus
6. Topic Adherence (10 points) - Relevant to requested topic

This modular version uses individual scorer classes for each component.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Add parent directory to path for standalone execution
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import spacy

# Use absolute imports that work both as module and standalone
try:
    from .scorers import (
        CEFRScorer,
        CoherenceScorer,
        FluencyScorer,
        GrammarScorer,
        JSONScorer,
        LinguisticScorer,
        TopicScorer,
    )
except ImportError:
    from src.rl.reward_function.scorers import (
        CEFRScorer,
        CoherenceScorer,
        FluencyScorer,
        GrammarScorer,
        JSONScorer,
        LinguisticScorer,
        TopicScorer,
    )


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    json_validity: float  # 0-15
    linguistic_quality: float  # 0-30 (reduced from 35)
    cefr_alignment: float  # 0-20
    fluency: float  # 0-10
    grammar_correctness: float  # 0-10
    topic_adherence: float  # 0-10
    coherence: float  # 0-10 (NEW)
    total: float  # 0-105 (will normalize to 100)

    errors: List[str]  # Specific issues found

    def __str__(self):
        return (
            f"Score: {self.total}/100\n"
            f"  JSON: {self.json_validity}/15\n"
            f"  Linguistic: {self.linguistic_quality}/30\n"
            f"  CEFR: {self.cefr_alignment}/20\n"
            f"  Fluency: {self.fluency}/10\n"
            f"  Grammar: {self.grammar_correctness}/10\n"
            f"  Topic: {self.topic_adherence}/10\n"
            f"  Coherence: {self.coherence}/10\n"
            f"Errors: {', '.join(self.errors) if self.errors else 'None'}"
        )


class ExerciseRewardFunction:
    """
    Modular reward function for scoring Italian exercise quality.

    Uses individual scorer components:
    - JSONScorer: Structure validation (15 pts)
    - LinguisticScorer: Italian grammar rules (30 pts)
    - CEFRScorer: Level-appropriate complexity (20 pts)
    - FluencyScorer: Natural language flow (10 pts)
    - GrammarScorer: Grammar focus validation (10 pts)
    - TopicScorer: Semantic similarity to topic (10 pts)
    - CoherenceScorer: Logical sense and coherence (10 pts) NEW
    """

    def __init__(
        self,
        spacy_model: str = "it_core_news_sm",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize modular reward function.

        Args:
            spacy_model: Italian spaCy model name (default: it_core_news_sm for speed)
            device: The device to run heavy models on ('cuda' or 'cpu').
        """
        try:
            # Load Italian NLP model
            print(f"Loading spaCy model: {spacy_model}...")
            self.nlp = spacy.load(spacy_model)
            print("✅ spaCy model loaded")
        except OSError:
            print(f"⚠️  spaCy model '{spacy_model}' not found.")
            print(f"   Install with: python -m spacy download {spacy_model}")
            raise

        self.device = device
        print(f"Reward function will use device: {self.device}")

        # Initialize individual scorers
        print("Initializing scorers...")
        self.scorers = {
            "json": JSONScorer(nlp=None),  # Doesn't need NLP
            "linguistic": LinguisticScorer(nlp=self.nlp),
            "cefr": CEFRScorer(nlp=self.nlp),  # Pre-loads 16,887-word vocabulary
            "fluency": FluencyScorer(nlp=self.nlp),
            "grammar": GrammarScorer(nlp=self.nlp),
            "topic": TopicScorer(nlp=None, device=self.device),  # Uses sentence transformer
            "coherence": CoherenceScorer(nlp=self.nlp),  # NEW: Semantic coherence
        }

        print("✅ Reward function initialized with all scorers (including coherence)")

    def score(
        self, exercise: Dict[str, Any], request: Dict[str, Any]
    ) -> Tuple[float, RewardBreakdown]:
        """
        Score an exercise.

        Args:
            exercise: Generated exercise dict with fields:
                - type: str (fill_in_blank, translation, multiple_choice)
                - question: str (Italian)
                - answer: str (Italian)
                - options: List[str] (for multiple_choice)
                - correct_option: int (for multiple_choice)
            request: Original request dict with fields:
                - level: str (A1, A2, B1, B2, C1, C2)
                - grammar_focus: str
                - topic: str
                - num_exercises: int

        Returns:
            Tuple of (total_score, breakdown)
        """
        all_errors = []

        # Score each component
        json_score, json_errors = self.scorers["json"].score(exercise, request)
        all_errors.extend(json_errors)

        linguistic_score, linguistic_errors = self.scorers["linguistic"].score(exercise, request)
        all_errors.extend(linguistic_errors)

        cefr_score, cefr_errors = self.scorers["cefr"].score(exercise, request)
        all_errors.extend(cefr_errors)

        fluency_score, fluency_errors = self.scorers["fluency"].score(exercise, request)
        all_errors.extend(fluency_errors)

        grammar_score, grammar_errors = self.scorers["grammar"].score(exercise, request)
        all_errors.extend(grammar_errors)

        topic_score, topic_errors = self.scorers["topic"].score(exercise, request)
        all_errors.extend(topic_errors)

        coherence_score, coherence_errors = self.scorers["coherence"].score(exercise, request)
        all_errors.extend(coherence_errors)

        # Total score (normalized to 100)
        # JSON(15) + Linguistic(30) + CEFR(20) + Fluency(10) + Grammar(10) + Topic(10) + Coherence(10) = 105
        raw_total = (
            json_score
            + linguistic_score
            + cefr_score
            + fluency_score
            + grammar_score
            + topic_score
            + coherence_score
        )

        # Normalize to 100 (105 max → 100)
        total = min(100, (raw_total / 105) * 100)

        # Create breakdown
        breakdown = RewardBreakdown(
            json_validity=json_score,
            linguistic_quality=linguistic_score,
            cefr_alignment=cefr_score,
            fluency=fluency_score,
            grammar_correctness=grammar_score,
            topic_adherence=topic_score,
            coherence=coherence_score,
            total=total,
            errors=all_errors,
        )

        return total, breakdown

    def score_exercises(
        self, exercises: List[Dict[str, Any]], request: Dict[str, Any]
    ) -> Tuple[float, List[Tuple[float, RewardBreakdown]]]:
        """
        Score multiple exercises and return average score + individual breakdowns.

        Args:
            exercises: List of exercise dictionaries to score
            request: Request context (same for all exercises)

        Returns:
            Tuple of (average_score, list of (score, breakdown) tuples)
        """
        results = []
        total_score = 0.0

        for exercise in exercises:
            score, breakdown = self.score(exercise, request)
            results.append((score, breakdown))
            total_score += score

        avg_score = total_score / len(exercises) if exercises else 0.0
        return avg_score, results


# Convenience function
_reward_function = None


def score_exercise(
    exercise: Dict[str, Any], request: Dict[str, Any]
) -> Tuple[float, RewardBreakdown]:
    """
    Score an exercise (convenience function).

    Creates a global reward function instance and scores the exercise.
    For repeated use, create an ExerciseRewardFunction instance directly.
    """
    global _reward_function
    if _reward_function is None:
        _reward_function = ExerciseRewardFunction()

    return _reward_function.score(exercise, request)


if __name__ == "__main__":
    # Test the modular reward function
    print("=" * 80)
    print("TESTING MODULAR REWARD FUNCTION (5 Core Tests)")
    print("=" * 80)

    # Create reward function
    rf = ExerciseRewardFunction()

    def print_test(num, title, exercise, request):
        """Helper to print test with exercise content."""
        print("\n" + "=" * 80)
        print(f"TEST {num}: {title}")
        print("=" * 80)
        print(f"\n📝 Exercise:")
        print(f"  Type: {exercise.get('type')}")
        print(f"  Question: {exercise.get('question')}")
        print(f"  Answer: {exercise.get('answer')}")
        if exercise.get("type") == "multiple_choice":
            print(f"  Options: {exercise.get('options')}")
        print(
            f"\n📋 Request: Level={request['level']}, Grammar={request['grammar_focus']}, Topic={request['topic']}"
        )

        score, breakdown = rf.score(exercise, request)
        print(f"\n{breakdown}")
        return score

    # Test 1: High-quality A2 exercise
    exercise1 = {
        "type": "fill_in_blank",
        "question": "Ieri Maria ha mangiato la pizza con i suoi amici.",
        "answer": "mangiato",
    }
    request1 = {"level": "A2", "grammar_focus": "past_tense", "topic": "cibo"}
    score1 = print_test(1, "High-Quality A2 Exercise", exercise1, request1)

    # Test 2: Multiple choice with good formatting
    exercise2 = {
        "type": "multiple_choice",
        "question": "Il gatto ____ sul divano ogni giorno.",
        "answer": "dorme",
        "options": ["dorme", "dormo", "dormi", "dormono"],
        "correct_option": 0,
    }
    request2 = {"level": "A2", "grammar_focus": "present_tense", "topic": "animali"}
    score2 = print_test(2, "Multiple Choice A2", exercise2, request2)

    # Test 3: Gender error (should get penalized)
    exercise3 = {
        "type": "fill_in_blank",
        "question": "Il mano è molto grande e forte.",
        "answer": "grande",
    }
    request3 = {"level": "A1", "grammar_focus": "present_tense", "topic": "corpo"}
    score3 = print_test(3, "Gender Error (il mano → should be la mano)", exercise3, request3)

    # Test 4: B1 complex with good grammar
    exercise4 = {
        "type": "translation",
        "question": "Quando sono arrivato a Roma, ho visitato il Colosseo e ho mangiato la pasta.",
        "answer": "When I arrived in Rome, I visited the Colosseum and ate pasta.",
    }
    request4 = {"level": "B1", "grammar_focus": "past_tense", "topic": "viaggio"}
    score4 = print_test(4, "B1 Complex Sentence", exercise4, request4)

    # Test 5: Vocabulary too advanced for level
    exercise5 = {
        "type": "fill_in_blank",
        "question": "La conseguenza significativa dell'esperienza particolare.",
        "answer": "conseguenza",
    }
    request5 = {"level": "A1", "grammar_focus": "present_tense", "topic": "generale"}
    score5 = print_test(5, "Vocabulary Too Advanced (B2 words in A1)", exercise5, request5)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    tests = [
        ("High-Quality A2", score1),
        ("Multiple Choice A2", score2),
        ("Gender Error (penalized)", score3),
        ("B1 Complex", score4),
        ("Vocab Too Advanced (penalized)", score5),
    ]

    for name, score in tests:
        status = "✅" if score >= 70 else "⚠️"
        print(f"{status} {name}: {score:.1f}/100")

    avg_score = sum(s for _, s in tests) / len(tests)
    print(f"\n📊 Average Score: {avg_score:.1f}/100")
    print("\n✅ All tests complete!")
