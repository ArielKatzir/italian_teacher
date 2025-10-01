# Staging Environment - Teacher to Student Flow

## Overview

Complete end-to-end staging environment for testing the teacher-to-student homework flow.

## What It Tests

The staging environment simulates the complete workflow:

1. **Teacher creates a class** → Add students with CEFR levels
2. **Teacher creates assignment** → Using HomeworkAssignment dataclass
3. **Homework is generated** → Exercises matching specifications
4. **Homework is distributed** → All specified students receive it
5. **Verification** → Homework matches assignment parameters

## Quick Start

### Run Staging Demo
```bash
python demos/staging_teacher_flow.py
```

This runs two scenarios:
- **Scenario 1**: A2 class, past tense, history of Milan (3 students)
- **Scenario 2**: Mixed levels, present tense, general topic (3 students)

###Run Integration Tests
```bash
make test-teacher-flow
```

All 9 tests should pass ✅

## Test Coverage

### Integration Tests (9 tests, all passing)

1. ✅ **Complete flow simple** - End-to-end with basic assignment
2. ✅ **Grammar focus matching** - Homework exercises match grammar
3. ✅ **Topic inclusion** - Topic appears in exercises
4. ✅ **Multiple students** - Same homework to multiple students
5. ✅ **Homework verification** - Validation passes for correct homework
6. ✅ **Quantity matching** - Exercise count matches assignment
7. ✅ **Different CEFR levels** - Works for A1, A2, B1, B2
8. ✅ **Status tracking** - Assignment status tracked correctly
9. ✅ **Real-world scenario** - A2 past tense Milan scenario

## Staging Environment API

### Setup Class

```python
from demos.staging_teacher_flow import StagingEnvironment
from educational.teacher.assignment import HomeworkAssignment, CEFRLevel, GrammarFocus

env = StagingEnvironment()

# Add students
env.add_student("s001", "Maria Rossi", "A2")
env.add_student("s002", "Giovanni Bianchi", "A2")
```

### Create Assignment

```python
assignment = HomeworkAssignment(
    cefr_level=CEFRLevel.A2,
    quantity=5,
    grammar_focus=GrammarFocus.PAST_TENSE,
    topic="history of Milan"
)
```

### Generate Homework

```python
homework_set = env.create_homework(assignment)
# Returns HomeworkSet with exercises matching assignment
```

### Verify Correctness

```python
env.verify_homework_correctness(assignment, homework_set)
# Returns True if homework matches all specifications
```

### Distribute to Students

```python
distributions = env.distribute_homework(homework_set)
# All students receive the homework
```

### Check Student Assignments

```python
assignments = env.get_student_assignments("s001")
# Get all assignments for a student
```

### Export Data

```python
env.export_assignments("data/staging/my_scenario.json")
# Export all students and assignments to JSON
```

## Exported Data Structure

```json
{
  "students": {
    "s001": {
      "student_id": "s001",
      "name": "Maria Rossi",
      "cefr_level": "A2"
    }
  },
  "assignments": {
    "s001_hw_20251001120146": {
      "student_id": "s001",
      "assignment_id": "hw_20251001120146",
      "cefr_level": "A2",
      "grammar_focus": "past_tense",
      "topic": "history of Milan",
      "num_exercises": 5,
      "assigned_at": "2025-10-01T12:01:46.996230",
      "status": "assigned",
      "exercises": [
        {
          "type": "fill_in_blank",
          "question": "Complete with past tense: Ieri io _____ (andare) a Milano. [history of Milan]",
          "answer": "sono andato"
        }
      ]
    }
  }
}
```

## Verification

The staging environment verifies that generated homework matches assignment specifications:

- ✅ **CEFR Level** matches
- ✅ **Grammar Focus** matches (if specified)
- ✅ **Topic** matches (if specified)
- ✅ **Quantity** matches (number of exercises)

Example output:
```
🔍 VERIFICATION
================================================================================
✅ CEFR Level: A2
✅ Grammar Focus: past_tense
✅ Topic: history of Milan
✅ Quantity: 5 exercises

✅ VERIFICATION PASSED: All homework matches assignment specifications!
```

## Current Status

### ✅ Working

- Teacher creates assignments (manual input)
- Assignment parameters defined (level, grammar, topic, quantity)
- Mock homework generation (placeholder exercises)
- Homework distribution to students
- Verification that homework matches specifications
- Student assignment tracking
- JSON export for inspection

### 🚧 Next Steps

1. **Replace mock generation with real HomeworkGenerator**
   - Integrate with MarcoInference
   - Use assignment.to_prompt_context() in prompts
   - Generate diverse, topic-specific exercises

2. **Add student UI**
   - Student login/authentication
   - View assigned homework
   - Complete exercises
   - Submit answers

3. **Add grading system**
   - Automatic grading for fill-in-blank/multiple choice
   - Manual grading for essays
   - Track scores and progress

4. **Add notifications**
   - Email/SMS when homework assigned
   - Reminders for due dates
   - Completion notifications

5. **Add progress tracking**
   - Student progress dashboard
   - Teacher overview of class progress
   - Individual student reports

## Running Tests

```bash
# Run all teacher tests
make test-teacher        # Unit tests (18 tests)
make test-teacher-flow   # Integration tests (9 tests)

# Run all tests
make test-integration    # All integration tests
make test                # Full test suite
```

## Files

```
demos/
└── staging_teacher_flow.py         # Staging environment & scenarios

tests/integration/
└── test_teacher_student_flow.py    # 9 integration tests

data/staging/
├── scenario_1.json                 # Exported data from scenario 1
└── scenario_2.json                 # Exported data from scenario 2
```

## Example Scenarios

### Scenario 1: A2 Past Tense - History of Milan
- **Students**: 3 A2 level students
- **Assignment**: 5 exercises, past tense, history of Milan
- **Result**: All students receive 5 past tense exercises about Milan

### Scenario 2: Mixed Levels - Present Tense
- **Students**: A1, A2, B1 (mixed levels)
- **Assignment**: 3 exercises, A2 level, present tense
- **Result**: All students receive the homework (in production, would filter by level)

## Integration with Production

When ready for production:

1. Replace `StagingEnvironment` with real `AssignmentManager`
2. Replace mock homework generation with `HomeworkGenerator`
3. Connect to real database instead of in-memory storage
4. Add student authentication
5. Add UI for teachers and students

The staging environment provides the blueprint for the production system.
