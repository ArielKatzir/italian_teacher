# Teacher Homework Assignment System

Simple, reliable manual input system for creating homework assignments.

## Overview

This module provides a straightforward way for teachers to create homework assignments by directly specifying parameters through a UI, rather than using AI-based natural language parsing.

## Why Manual Input?

We initially tried AI-based command parsing with guided generation (Outlines) and template completion, but found:
- ❌ Outlines had caching issues causing incorrect outputs for sequential calls
- ❌ Template completion had ~50% accuracy on parameter extraction
- ❌ Both methods were slow (40-80s per parse)
- ❌ Unpredictable errors with natural language ambiguity

Manual input is:
- ✅ 100% reliable (no parsing errors)
- ✅ Instant (no LLM call needed)
- ✅ Better UX (teachers know exactly what they're setting)
- ✅ Easy to validate
- ✅ Simple to integrate with UI frameworks

## Usage

### Basic Example

```python
from src.educational.teacher import (
    HomeworkAssignment,
    CEFRLevel,
    GrammarFocus
)

# Create an assignment
assignment = HomeworkAssignment(
    cefr_level=CEFRLevel.A2,
    quantity=5,
    grammar_focus=GrammarFocus.PAST_TENSE,
    topic="history of Milan"
)

# Get context for homework generator
context = assignment.to_prompt_context()
# Output:
# CEFR Level: A2
# Grammar Focus: Past Tense
# Topic: history of Milan
# Number of Exercises: 5
```

### UI Integration

In a web form, teachers select:
- **CEFR Level**: Dropdown (A1, A2, B1, B2, C1, C2)
- **Quantity**: Number input (1-20)
- **Grammar Focus**: Optional dropdown (present_tense, past_tense, etc.)
- **Topic**: Optional text input (e.g., "Italian food", "history of Milan")
- **Exercise Types**: Checkboxes (fill_in_blank, translation, etc.)

Backend creates the assignment object from form data:

```python
assignment = HomeworkAssignment(
    cefr_level=CEFRLevel(form_data['level']),
    quantity=int(form_data['quantity']),
    grammar_focus=GrammarFocus(form_data['grammar']) if form_data['grammar'] else None,
    topic=form_data['topic'] if form_data['topic'] else None,
    exercise_types=[ExerciseType(t) for t in form_data['exercise_types']]
)
```

### Passing to Homework Generator

The `to_prompt_context()` method formats the assignment as context for exercise generation:

```python
assignment = HomeworkAssignment(
    cefr_level=CEFRLevel.B1,
    quantity=10,
    grammar_focus=GrammarFocus.SUBJUNCTIVE,
    topic="Italian cinema"
)

context = assignment.to_prompt_context()

# Use in homework generator prompt:
prompt = f"""
Generate Italian language exercises.

{context}

Create exercises that match these requirements...
"""
```

## API Reference

### HomeworkAssignment

**Required Parameters:**
- `cefr_level: CEFRLevel` - Student proficiency level (A1-C2)
- `quantity: int` - Number of exercises (1-20, default: 5)

**Optional Parameters:**
- `grammar_focus: Optional[GrammarFocus]` - Specific grammar topic
- `topic: Optional[str]` - Custom topic/theme
- `student_groups: List[str]` - Target groups (default: ["all"])
- `exercise_types: List[ExerciseType]` - Types of exercises to include
- `difficulty_scaling: bool` - Adjust difficulty within level (default: True)

**Methods:**
- `to_prompt_context() -> str` - Format assignment as context for prompts
- `__repr__() -> str` - Human-readable representation

### Enums

**CEFRLevel**: A1, A2, B1, B2, C1, C2

**GrammarFocus**: PRESENT_TENSE, PAST_TENSE, FUTURE_TENSE, SUBJUNCTIVE, CONDITIONAL, IMPERATIVE, PASSIVE_VOICE, PRONOUNS, ARTICLES, PREPOSITIONS, CONJUNCTIONS

**ExerciseType**: FILL_IN_BLANK, TRANSLATION, SENTENCE_COMPLETION, MULTIPLE_CHOICE, ESSAY, CONVERSATION

## Components

### 1. HomeworkAssignment (`assignment.py`)

Manual assignment configuration with validation and prompt context generation.

### 2. HomeworkGenerator (`homework_generator.py`)

**Exercise Generation Engine**
- Uses the Marco model to create level-appropriate exercises
- Supports multiple exercise types
- Generates topic-specific and grammar-focused content
- Scales difficulty within CEFR levels

### 3. AssignmentManager (`assignment_manager.py`)

**Distribution and Tracking**
- Manages student profiles and assignment distribution
- Tracks assignment progress and completion
- Provides statistics and reporting

## Demo

Run the demo to see examples:

```bash
python demos/teacher_manual_demo.py
```

## Future Enhancements

For production, consider:
- **GPT-4/Claude API** for optional natural language input (as convenience feature, not primary)
- **Template library** for common assignment types
- **Assignment history** and reuse functionality
- **Student progress tracking** integration
