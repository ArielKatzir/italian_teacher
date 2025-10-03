"""
Exercise Generation Validator

Tests different generation parameters and validates output quality.
Helps find the best temperature, prompts, and parsing strategies.
"""

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List

import requests


@dataclass
class GenerationConfig:
    """Configuration for exercise generation test."""

    cefr_level: str
    grammar_focus: str
    topic: str
    quantity: int
    exercise_types: List[str]
    temperature: float
    max_tokens: int = 1500


@dataclass
class ValidationResult:
    """Results from validating generated exercises."""

    success: bool
    parsing_strategy: str
    num_exercises: int
    inference_time: float
    tokens_generated: int

    # Quality metrics
    has_duplicates: bool
    duplicate_count: int
    unique_questions: int
    avg_question_length: int
    avg_answer_length: int

    # Issues found
    issues: List[str]
    exercises: List[Dict[str, Any]]

    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0

        # Penalize duplicates heavily
        if self.has_duplicates:
            duplicate_penalty = (self.duplicate_count / self.num_exercises) * 50
            score -= duplicate_penalty

        # Penalize parsing fallbacks
        if self.parsing_strategy.startswith("strategy4"):
            score -= 20
        elif self.parsing_strategy.startswith("strategy5"):
            score -= 40

        # Penalize short/low-quality content
        if self.avg_question_length < 20:
            score -= 15
        if self.avg_answer_length < 3:
            score -= 15

        # Penalize issues
        score -= len(self.issues) * 5

        return max(0, score)


class ExerciseValidator:
    """Validates exercise generation quality across different configurations."""

    def __init__(self, api_url: str):
        """
        Initialize validator.

        Args:
            api_url: Base URL of the inference API (ngrok URL)
        """
        self.api_url = api_url.rstrip("/")

    def generate_exercises(self, config: GenerationConfig) -> Dict[str, Any]:
        """
        Generate exercises with given config.

        Args:
            config: Generation configuration

        Returns:
            API response dictionary
        """
        response = requests.post(
            f"{self.api_url}/generate",
            json={
                "cefr_level": config.cefr_level,
                "grammar_focus": config.grammar_focus,
                "topic": config.topic,
                "quantity": config.quantity,
                "exercise_types": config.exercise_types,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
            timeout=120,  # Increased timeout for longer generations
        )
        response.raise_for_status()
        return response.json()

    def validate_exercises(
        self,
        exercises: List[Dict[str, Any]],
        expected_quantity: int,
        parsing_strategy: str,
        inference_time: float,
        tokens_generated: int,
    ) -> ValidationResult:
        """
        Validate generated exercises for quality issues.

        Args:
            exercises: List of exercise dictionaries
            expected_quantity: Expected number of exercises
            parsing_strategy: Which parsing strategy was used
            inference_time: Time taken for generation
            tokens_generated: Number of tokens generated

        Returns:
            ValidationResult with quality metrics
        """
        issues = []

        # Check quantity
        if len(exercises) != expected_quantity:
            issues.append(f"Wrong quantity: got {len(exercises)}, expected {expected_quantity}")

        # Check for duplicates
        questions = [ex.get("question", "") for ex in exercises]
        question_counts = Counter(questions)
        duplicates = {q: c for q, c in question_counts.items() if c > 1}

        has_duplicates = len(duplicates) > 0
        duplicate_count = sum(c - 1 for c in duplicates.values())

        if has_duplicates:
            issues.append(f"Found {duplicate_count} duplicate questions")

        # Check for required fields
        required_fields = ["type", "question", "correct_answer", "explanation"]
        for i, ex in enumerate(exercises):
            missing = [f for f in required_fields if f not in ex or not ex[f]]
            if missing:
                issues.append(f"Exercise {i+1} missing fields: {', '.join(missing)}")

        # Check for empty/placeholder content
        for i, ex in enumerate(exercises):
            if "Exercise {i+1}" in ex.get("question", ""):
                issues.append(f"Exercise {i+1} has placeholder question")
            if ex.get("correct_answer") == "See explanation":
                issues.append(f"Exercise {i+1} has placeholder answer")

        # Calculate metrics
        question_lengths = [len(ex.get("question", "")) for ex in exercises]
        answer_lengths = [len(ex.get("correct_answer", "")) for ex in exercises]

        avg_question_length = int(statistics.mean(question_lengths)) if question_lengths else 0
        avg_answer_length = int(statistics.mean(answer_lengths)) if answer_lengths else 0

        return ValidationResult(
            success=len(issues) == 0,
            parsing_strategy=parsing_strategy,
            num_exercises=len(exercises),
            inference_time=inference_time,
            tokens_generated=tokens_generated,
            has_duplicates=has_duplicates,
            duplicate_count=duplicate_count,
            unique_questions=len(question_counts),
            avg_question_length=avg_question_length,
            avg_answer_length=avg_answer_length,
            issues=issues,
            exercises=exercises,
        )

    def test_configuration(self, config: GenerationConfig) -> ValidationResult:
        """
        Test a single configuration.

        Args:
            config: Generation configuration to test

        Returns:
            ValidationResult
        """
        response = self.generate_exercises(config)

        return self.validate_exercises(
            exercises=response["exercises"],
            expected_quantity=config.quantity,
            parsing_strategy=response.get("parsing_strategy", "unknown"),
            inference_time=response.get("inference_time", 0),
            tokens_generated=response.get("generated_tokens", 0),
        )

    def compare_temperatures(
        self, base_config: GenerationConfig, temperatures: List[float], runs_per_temp: int = 3
    ) -> Dict[float, List[ValidationResult]]:
        """
        Compare different temperature settings.

        Args:
            base_config: Base configuration to vary
            temperatures: List of temperatures to test
            runs_per_temp: Number of runs per temperature

        Returns:
            Dictionary mapping temperature to validation results
        """
        results = {}

        for temp in temperatures:
            temp_results = []
            config = GenerationConfig(
                cefr_level=base_config.cefr_level,
                grammar_focus=base_config.grammar_focus,
                topic=base_config.topic,
                quantity=base_config.quantity,
                exercise_types=base_config.exercise_types,
                temperature=temp,
                max_tokens=base_config.max_tokens,
            )

            for _ in range(runs_per_temp):
                result = self.test_configuration(config)
                temp_results.append(result)

            results[temp] = temp_results

        return results

    def print_comparison_report(self, results: Dict[float, List[ValidationResult]]):
        """
        Print a formatted comparison report.

        Args:
            results: Results from compare_temperatures()
        """
        print("\n" + "=" * 80)
        print("TEMPERATURE COMPARISON REPORT")
        print("=" * 80)

        for temp, temp_results in sorted(results.items()):
            avg_score = statistics.mean([r.quality_score for r in temp_results])
            avg_time = statistics.mean([r.inference_time for r in temp_results])

            strategies = [r.parsing_strategy for r in temp_results]
            strategy_counts = Counter(strategies)

            duplicate_rate = sum(r.has_duplicates for r in temp_results) / len(temp_results)

            print(f"\nTemperature: {temp}")
            print(f"  Quality Score: {avg_score:.1f}/100")
            print(f"  Inference Time: {avg_time:.2f}s")
            print(f"  Duplicate Rate: {duplicate_rate*100:.0f}%")
            print(f"  Parsing Strategies: {dict(strategy_counts)}")

            # Show issues
            all_issues = []
            for r in temp_results:
                all_issues.extend(r.issues)
            if all_issues:
                issue_counts = Counter(all_issues)
                print(f"  Common Issues:")
                for issue, count in issue_counts.most_common(3):
                    print(f"    - {issue} ({count}x)")

        print("\n" + "=" * 80)

        # Recommendation
        best_temp = max(
            results.keys(), key=lambda t: statistics.mean([r.quality_score for r in results[t]])
        )
        best_score = statistics.mean([r.quality_score for r in results[best_temp]])

        print(f"\n🎯 RECOMMENDATION: Use temperature {best_temp} (avg score: {best_score:.1f}/100)")
        print("=" * 80 + "\n")


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.exercise_validator <ngrok_url>")
        print("\nExample:")
        print("  python -m utils.exercise_validator https://abc123.ngrok-free.dev")
        sys.exit(1)

    api_url = sys.argv[1]

    validator = ExerciseValidator(api_url)

    # Test configuration
    base_config = GenerationConfig(
        cefr_level="A2",
        grammar_focus="present_tense",
        topic="daily routines",
        quantity=5,
        exercise_types=["fill_in_blank", "translation", "multiple_choice"],
        temperature=0.7,  # Will be varied
        max_tokens=1500,
    )

    # Compare temperatures
    print("🧪 Testing different temperatures...")
    print("This will take a few minutes...\n")

    temperatures = [0.2, 0.4, 0.6, 0.8]
    results = validator.compare_temperatures(base_config, temperatures, runs_per_temp=2)

    # Print report
    validator.print_comparison_report(results)


if __name__ == "__main__":
    main()
