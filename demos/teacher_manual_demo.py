#!/usr/bin/env python3
"""
Simple demo of manual homework assignment creation.

This replaces the AI-based command parsing with direct manual input,
which is more reliable and better for UI integration.
"""

import sys
from pathlib import Path

# Direct import to avoid package __init__ issues
assignment_path = Path(__file__).parent.parent / "src" / "educational" / "teacher"
sys.path.insert(0, str(assignment_path))

from assignment import CEFRLevel, ExerciseType, GrammarFocus, HomeworkAssignment


def main():
    print("=" * 80)
    print("📝 MANUAL HOMEWORK ASSIGNMENT DEMO")
    print("=" * 80)

    # Example 1: Basic assignment
    print("\n📋 Example 1: Basic A2 assignment")
    print("-" * 80)
    assignment1 = HomeworkAssignment(cefr_level=CEFRLevel.A2, quantity=5)
    print(f"Created: {assignment1}")
    print(f"Prompt context:\n{assignment1.to_prompt_context()}")

    # Example 2: Assignment with grammar focus
    print("\n📋 Example 2: A2 with past tense focus")
    print("-" * 80)
    assignment2 = HomeworkAssignment(
        cefr_level=CEFRLevel.A2, quantity=5, grammar_focus=GrammarFocus.PAST_TENSE
    )
    print(f"Created: {assignment2}")
    print(f"Prompt context:\n{assignment2.to_prompt_context()}")

    # Example 3: Assignment with topic
    print("\n📋 Example 3: A2 with custom topic")
    print("-" * 80)
    assignment3 = HomeworkAssignment(
        cefr_level=CEFRLevel.A2,
        quantity=5,
        grammar_focus=GrammarFocus.PAST_TENSE,
        topic="history of Milan",
    )
    print(f"Created: {assignment3}")
    print(f"Prompt context:\n{assignment3.to_prompt_context()}")

    # Example 4: Complex assignment
    print("\n📋 Example 4: B1 subjunctive with custom settings")
    print("-" * 80)
    assignment4 = HomeworkAssignment(
        cefr_level=CEFRLevel.B1,
        quantity=10,
        grammar_focus=GrammarFocus.SUBJUNCTIVE,
        topic="Italian cinema",
        exercise_types=[ExerciseType.FILL_IN_BLANK, ExerciseType.ESSAY, ExerciseType.TRANSLATION],
    )
    print(f"Created: {assignment4}")
    print(f"Prompt context:\n{assignment4.to_prompt_context()}")
    print(f"Exercise types: {[et.value for et in assignment4.exercise_types]}")

    # Example 5: Beginner assignment
    print("\n📋 Example 5: A1 beginners with numbers topic")
    print("-" * 80)
    assignment5 = HomeworkAssignment(cefr_level=CEFRLevel.A1, quantity=3, topic="numbers")
    print(f"Created: {assignment5}")
    print(f"Prompt context:\n{assignment5.to_prompt_context()}")

    # Show how this would work in a UI
    print("\n" + "=" * 80)
    print("🖥️  UI INTEGRATION EXAMPLE")
    print("=" * 80)
    print(
        """
In a web UI, the teacher would see a form like:

┌─────────────────────────────────────────┐
│ Create Homework Assignment              │
├─────────────────────────────────────────┤
│                                         │
│ CEFR Level: [A1▼] [A2] [B1] [B2] [C1] [C2] │
│                                         │
│ Quantity: [5]                           │
│                                         │
│ Grammar Focus (optional):               │
│ [Select...▼]                            │
│   - Present Tense                       │
│   - Past Tense                          │
│   - Future Tense                        │
│   - Subjunctive                         │
│   - ...                                 │
│                                         │
│ Topic (optional):                       │
│ [history of Milan____________]          │
│                                         │
│ Exercise Types:                         │
│ [✓] Fill in the blank                   │
│ [✓] Translation                         │
│ [✓] Sentence completion                 │
│ [✓] Multiple choice                     │
│ [ ] Essay                               │
│ [ ] Conversation                        │
│                                         │
│         [Generate Homework]             │
└─────────────────────────────────────────┘

Then the backend would create:

    assignment = HomeworkAssignment(
        cefr_level=CEFRLevel(form_data['level']),
        quantity=int(form_data['quantity']),
        grammar_focus=GrammarFocus(form_data['grammar']) if form_data['grammar'] else None,
        topic=form_data['topic'] if form_data['topic'] else None,
        exercise_types=[ExerciseType(t) for t in form_data['exercise_types']]
    )

    # Pass to homework generator with the topic as additional context
    context = assignment.to_prompt_context()
    # Use context in exercise generation prompts
    """
    )

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("=" * 80)
    print("\n💡 Benefits of manual input:")
    print("   • No AI parsing errors")
    print("   • Faster (no LLM call for parsing)")
    print("   • Clear UI/UX for teachers")
    print("   • Easy validation")
    print("   • Topic goes directly to homework generator as context")


if __name__ == "__main__":
    main()
