#!/usr/bin/env python3
"""
Interactive CLI demo of manual homework assignment creation.
"""

import sys
from pathlib import Path

# Direct import to avoid package __init__ issues
assignment_path = Path(__file__).parent.parent / "src" / "educational" / "teacher"
sys.path.insert(0, str(assignment_path))

from assignment import CEFRLevel, ExerciseType, GrammarFocus, HomeworkAssignment


def get_user_choice(prompt: str, options: dict, allow_skip: bool = False) -> str:
    """Get user choice from a menu."""
    print(f"\n{prompt}")
    for key, value in options.items():
        print(f"  {key}. {value}")
    if allow_skip:
        print(f"  {len(options) + 1}. Skip (None)")

    while True:
        choice = input("\nYour choice: ").strip()
        if choice in options:
            return options[choice]
        if allow_skip and choice == str(len(options) + 1):
            return None
        print("❌ Invalid choice. Please try again.")


def get_number(prompt: str, min_val: int, max_val: int, default: int) -> int:
    """Get a number from user within range."""
    while True:
        response = input(f"\n{prompt} [{min_val}-{max_val}] (default: {default}): ").strip()
        if not response:
            return default
        try:
            num = int(response)
            if min_val <= num <= max_val:
                return num
            print(f"❌ Please enter a number between {min_val} and {max_val}")
        except ValueError:
            print("❌ Please enter a valid number")


def get_text(prompt: str, allow_empty: bool = True) -> str:
    """Get text input from user."""
    response = input(f"\n{prompt}: ").strip()
    if not response and allow_empty:
        return None
    return response if response else None


def create_interactive_assignment():
    """Interactive assignment creation."""
    print("\n" + "=" * 80)
    print("🎯 CREATE HOMEWORK ASSIGNMENT")
    print("=" * 80)

    # CEFR Level
    cefr_options = {
        "1": "A1 (Beginner)",
        "2": "A2 (Elementary)",
        "3": "B1 (Intermediate)",
        "4": "B2 (Upper Intermediate)",
        "5": "C1 (Advanced)",
        "6": "C2 (Proficient)",
    }
    cefr_mapping = {
        "A1 (Beginner)": CEFRLevel.A1,
        "A2 (Elementary)": CEFRLevel.A2,
        "B1 (Intermediate)": CEFRLevel.B1,
        "B2 (Upper Intermediate)": CEFRLevel.B2,
        "C1 (Advanced)": CEFRLevel.C1,
        "C2 (Proficient)": CEFRLevel.C2,
    }
    cefr_choice = get_user_choice("Select CEFR Level:", cefr_options)
    cefr_level = cefr_mapping[cefr_choice]

    # Quantity
    quantity = get_number("Number of exercises", 1, 20, 5)

    # Grammar Focus (optional)
    grammar_options = {
        "1": "Present Tense",
        "2": "Past Tense",
        "3": "Future Tense",
        "4": "Subjunctive",
        "5": "Conditional",
        "6": "Imperative",
        "7": "Passive Voice",
        "8": "Pronouns",
        "9": "Articles",
        "10": "Prepositions",
        "11": "Conjunctions",
    }
    grammar_mapping = {
        "Present Tense": GrammarFocus.PRESENT_TENSE,
        "Past Tense": GrammarFocus.PAST_TENSE,
        "Future Tense": GrammarFocus.FUTURE_TENSE,
        "Subjunctive": GrammarFocus.SUBJUNCTIVE,
        "Conditional": GrammarFocus.CONDITIONAL,
        "Imperative": GrammarFocus.IMPERATIVE,
        "Passive Voice": GrammarFocus.PASSIVE_VOICE,
        "Pronouns": GrammarFocus.PRONOUNS,
        "Articles": GrammarFocus.ARTICLES,
        "Prepositions": GrammarFocus.PREPOSITIONS,
        "Conjunctions": GrammarFocus.CONJUNCTIONS,
    }
    grammar_choice = get_user_choice(
        "Select Grammar Focus (optional):", grammar_options, allow_skip=True
    )
    grammar_focus = grammar_mapping[grammar_choice] if grammar_choice else None

    # Topic (optional)
    topic = get_text(
        "Enter topic (e.g., 'history of Milan', 'Italian food') or press Enter to skip",
        allow_empty=True,
    )

    # Exercise Types
    print(
        "\n📝 Select Exercise Types (enter numbers separated by commas, or press Enter for defaults):"
    )
    exercise_options = {
        "1": "Fill in the Blank",
        "2": "Translation",
        "3": "Sentence Completion",
        "4": "Multiple Choice",
        "5": "Essay",
        "6": "Conversation",
    }
    for key, value in exercise_options.items():
        print(f"  {key}. {value}")

    exercise_mapping = {
        "Fill in the Blank": ExerciseType.FILL_IN_BLANK,
        "Translation": ExerciseType.TRANSLATION,
        "Sentence Completion": ExerciseType.SENTENCE_COMPLETION,
        "Multiple Choice": ExerciseType.MULTIPLE_CHOICE,
        "Essay": ExerciseType.ESSAY,
        "Conversation": ExerciseType.CONVERSATION,
    }

    exercise_input = input("\nYour choices (e.g., 1,2,3) or Enter for defaults: ").strip()
    if exercise_input:
        selected_indices = [idx.strip() for idx in exercise_input.split(",")]
        exercise_types = [
            exercise_mapping[exercise_options[idx]]
            for idx in selected_indices
            if idx in exercise_options
        ]
    else:
        exercise_types = None  # Will use defaults

    # Create assignment
    assignment = HomeworkAssignment(
        cefr_level=cefr_level,
        quantity=quantity,
        grammar_focus=grammar_focus,
        topic=topic,
        exercise_types=exercise_types,
    )

    return assignment


def display_assignment(assignment: HomeworkAssignment):
    """Display assignment details in a nice format."""
    print("\n" + "=" * 80)
    print("✅ ASSIGNMENT CREATED SUCCESSFULLY")
    print("=" * 80)

    print(f"\n📊 Assignment Details:")
    print(f"  • CEFR Level: {assignment.cefr_level.value} ({assignment.cefr_level.name})")
    print(f"  • Quantity: {assignment.quantity} exercises")

    if assignment.grammar_focus:
        grammar_name = assignment.grammar_focus.value.replace("_", " ").title()
        print(f"  • Grammar Focus: {grammar_name}")
    else:
        print(f"  • Grammar Focus: None (general practice)")

    if assignment.topic:
        print(f"  • Topic: {assignment.topic}")
    else:
        print(f"  • Topic: None (general Italian)")

    print(f"  • Student Groups: {', '.join(assignment.student_groups)}")
    print(f"  • Difficulty Scaling: {'Enabled' if assignment.difficulty_scaling else 'Disabled'}")

    print(f"\n📝 Exercise Types:")
    for ex_type in assignment.exercise_types:
        ex_name = ex_type.value.replace("_", " ").title()
        print(f"  • {ex_name}")

    print(f"\n📋 Prompt Context for Homework Generator:")
    print("-" * 80)
    print(assignment.to_prompt_context())
    print("-" * 80)

    print(f"\n🔍 Python Object Representation:")
    print(f"  {repr(assignment)}")

    print(f"\n🎯 Data Structure (detailed):")
    print(f"  HomeworkAssignment(")
    print(f"    cefr_level={assignment.cefr_level!r},")
    print(f"    quantity={assignment.quantity!r},")
    print(f"    grammar_focus={assignment.grammar_focus!r},")
    print(f"    topic={assignment.topic!r},")
    print(f"    student_groups={assignment.student_groups!r},")
    print(f"    exercise_types={assignment.exercise_types!r},")
    print(f"    difficulty_scaling={assignment.difficulty_scaling!r}")
    print(f"  )")


def show_examples():
    """Show pre-configured examples."""
    examples = [
        HomeworkAssignment(
            cefr_level=CEFRLevel.A2,
            quantity=5,
            grammar_focus=GrammarFocus.PAST_TENSE,
            topic="history of Milan",
        ),
        HomeworkAssignment(cefr_level=CEFRLevel.A1, quantity=3, topic="numbers"),
        HomeworkAssignment(
            cefr_level=CEFRLevel.B1,
            quantity=10,
            grammar_focus=GrammarFocus.SUBJUNCTIVE,
            topic="Italian cinema",
            exercise_types=[ExerciseType.ESSAY, ExerciseType.TRANSLATION],
        ),
    ]

    print("\n" + "=" * 80)
    print("📚 EXAMPLE ASSIGNMENTS")
    print("=" * 80)

    for i, assignment in enumerate(examples, 1):
        print(f"\n{i}. {repr(assignment)}")
        print(f"   Context: {assignment.to_prompt_context().replace(chr(10), ' | ')}")


def main():
    print("=" * 80)
    print("📝 INTERACTIVE HOMEWORK ASSIGNMENT DEMO")
    print("=" * 80)
    print("\nThis demo lets you create homework assignments interactively.")
    print("Choose from predefined examples or create your own custom assignment.")

    while True:
        print("\n" + "=" * 80)
        print("MAIN MENU")
        print("=" * 80)
        print("  1. Create Custom Assignment")
        print("  2. View Examples")
        print("  3. Exit")

        choice = input("\nYour choice: ").strip()

        if choice == "1":
            assignment = create_interactive_assignment()
            display_assignment(assignment)

            # Ask if user wants to create another
            again = input("\n\nCreate another assignment? (y/n): ").strip().lower()
            if again != "y":
                break

        elif choice == "2":
            show_examples()

        elif choice == "3":
            print("\n👋 Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please select 1, 2, or 3.")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
