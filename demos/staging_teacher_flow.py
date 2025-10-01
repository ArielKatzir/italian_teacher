#!/usr/bin/env python3
"""
Staging Environment - Teacher to Student Flow

This demo simulates the complete flow:
1. Teacher creates a class and adds students
2. Teacher creates homework assignment
3. Homework is generated (mocked for now, will integrate with real generator)
4. Homework is distributed to students
5. Verify each student received the correct homework
"""

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from educational.teacher.assignment import CEFRLevel, GrammarFocus, HomeworkAssignment


# Simplified models for staging (don't require MarcoInference)
@dataclass
class Exercise:
    """Individual homework exercise."""

    exercise_type: str
    question: str
    answer: str = None
    choices: List[str] = None
    hint: str = None


@dataclass
class HomeworkSet:
    """Complete set of homework exercises."""

    assignment_id: str
    cefr_level: str
    grammar_focus: str = None
    topic: str = None
    exercises: List[Exercise] = None

    def __post_init__(self):
        if self.exercises is None:
            self.exercises = []


@dataclass
class Student:
    """Student profile."""

    student_id: str
    name: str
    cefr_level: str

    def __repr__(self):
        return f"Student({self.student_id}, {self.name}, {self.cefr_level})"


@dataclass
class StudentAssignment:
    """Assignment given to a student."""

    student_id: str
    homework_set: HomeworkSet
    assigned_at: datetime
    status: str = "assigned"

    def to_dict(self):
        return {
            "student_id": self.student_id,
            "assignment_id": self.homework_set.assignment_id,
            "cefr_level": self.homework_set.cefr_level,
            "grammar_focus": self.homework_set.grammar_focus,
            "topic": self.homework_set.topic,
            "num_exercises": len(self.homework_set.exercises),
            "assigned_at": self.assigned_at.isoformat(),
            "status": self.status,
            "exercises": [
                {"type": ex.exercise_type, "question": ex.question, "answer": ex.answer}
                for ex in self.homework_set.exercises
            ],
        }


class StagingEnvironment:
    """Staging environment for testing teacher-to-student flow."""

    def __init__(self):
        self.students: Dict[str, Student] = {}
        self.assignments: Dict[str, StudentAssignment] = {}

    def add_student(self, student_id: str, name: str, cefr_level: str):
        """Add a student to the class."""
        student = Student(student_id=student_id, name=name, cefr_level=cefr_level)
        self.students[student_id] = student
        print(f"✅ Added student: {student}")
        return student

    def create_homework(self, assignment: HomeworkAssignment) -> HomeworkSet:
        """
        Generate homework from assignment specification.

        For now, this is mocked. Later, this will call the real HomeworkGenerator.
        """
        print(f"\n📝 Generating homework for: {assignment}")
        print(f"   Context: {assignment.to_prompt_context()}")

        # Mock homework generation
        exercises = []

        # Generate exercises based on assignment
        for i in range(assignment.quantity):
            if GrammarFocus.PAST_TENSE == assignment.grammar_focus:
                exercise = Exercise(
                    exercise_type="fill_in_blank",
                    question=f"Complete with past tense: Ieri io _____ (andare) a Milano. [{assignment.topic if assignment.topic else 'general'}]",
                    answer="sono andato",
                    hint="Use passato prossimo",
                )
            elif GrammarFocus.PRESENT_TENSE == assignment.grammar_focus:
                exercise = Exercise(
                    exercise_type="fill_in_blank",
                    question=f"Complete with present tense: Oggi io _____ (mangiare) la pizza. [{assignment.topic if assignment.topic else 'general'}]",
                    answer="mangio",
                    hint="Use presente indicativo",
                )
            else:
                exercise = Exercise(
                    exercise_type="translation",
                    question=f"Translate to Italian: I study Italian. [{assignment.topic if assignment.topic else 'general'}]",
                    answer="Studio l'italiano",
                )

            exercises.append(exercise)

        homework_set = HomeworkSet(
            assignment_id=f"hw_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            cefr_level=assignment.cefr_level.value,
            grammar_focus=assignment.grammar_focus.value if assignment.grammar_focus else None,
            topic=assignment.topic,
            exercises=exercises,
        )

        print(f"✅ Generated {len(exercises)} exercises")
        for i, ex in enumerate(exercises[:2], 1):  # Show first 2
            print(f"   {i}. {ex.question}")
        if len(exercises) > 2:
            print(f"   ... and {len(exercises) - 2} more")

        return homework_set

    def distribute_homework(self, homework_set: HomeworkSet, student_ids: List[str] = None):
        """Distribute homework to students."""
        if student_ids is None:
            student_ids = list(self.students.keys())

        print(f"\n📤 Distributing homework to {len(student_ids)} student(s)...")

        distributions = []
        for student_id in student_ids:
            if student_id not in self.students:
                print(f"⚠️  Student {student_id} not found, skipping")
                continue

            assignment = StudentAssignment(
                student_id=student_id,
                homework_set=homework_set,
                assigned_at=datetime.now(),
                status="assigned",
            )

            self.assignments[f"{student_id}_{homework_set.assignment_id}"] = assignment
            distributions.append(assignment)

            student = self.students[student_id]
            print(f"✅ Assigned to {student.name} ({student_id})")

        return distributions

    def get_student_assignments(self, student_id: str) -> List[StudentAssignment]:
        """Get all assignments for a student."""
        return [
            assignment
            for key, assignment in self.assignments.items()
            if assignment.student_id == student_id
        ]

    def verify_homework_correctness(
        self, assignment: HomeworkAssignment, homework_set: HomeworkSet
    ):
        """Verify that generated homework matches assignment specifications."""
        print(f"\n🔍 VERIFICATION")
        print("=" * 80)

        errors = []

        # Check CEFR level
        if homework_set.cefr_level != assignment.cefr_level.value:
            errors.append(
                f"CEFR level mismatch: expected {assignment.cefr_level.value}, got {homework_set.cefr_level}"
            )
        else:
            print(f"✅ CEFR Level: {homework_set.cefr_level}")

        # Check grammar focus
        expected_grammar = assignment.grammar_focus.value if assignment.grammar_focus else None
        if homework_set.grammar_focus != expected_grammar:
            errors.append(
                f"Grammar focus mismatch: expected {expected_grammar}, got {homework_set.grammar_focus}"
            )
        else:
            print(f"✅ Grammar Focus: {homework_set.grammar_focus or 'None (general)'}")

        # Check topic
        if homework_set.topic != assignment.topic:
            errors.append(f"Topic mismatch: expected {assignment.topic}, got {homework_set.topic}")
        else:
            print(f"✅ Topic: {homework_set.topic or 'None (general)'}")

        # Check quantity
        if len(homework_set.exercises) != assignment.quantity:
            errors.append(
                f"Quantity mismatch: expected {assignment.quantity}, got {len(homework_set.exercises)}"
            )
        else:
            print(f"✅ Quantity: {len(homework_set.exercises)} exercises")

        if errors:
            print(f"\n❌ VERIFICATION FAILED:")
            for error in errors:
                print(f"   - {error}")
            return False
        else:
            print(f"\n✅ VERIFICATION PASSED: All homework matches assignment specifications!")
            return True

    def export_assignments(self, filepath: str):
        """Export all assignments to JSON."""
        data = {
            "students": {sid: asdict(student) for sid, student in self.students.items()},
            "assignments": {
                key: assignment.to_dict() for key, assignment in self.assignments.items()
            },
        }

        Path(filepath).write_text(json.dumps(data, indent=2))
        print(f"\n💾 Exported assignments to: {filepath}")


def demo_scenario_1():
    """Scenario 1: A2 students, past tense, history of Milan."""
    print("=" * 80)
    print("📚 SCENARIO 1: A2 Past Tense - History of Milan")
    print("=" * 80)

    env = StagingEnvironment()

    # Step 1: Create class and add students
    print("\n1️⃣  SETTING UP CLASS")
    print("-" * 80)
    env.add_student("s001", "Maria Rossi", "A2")
    env.add_student("s002", "Giovanni Bianchi", "A2")
    env.add_student("s003", "Lucia Verde", "A2")

    # Step 2: Teacher creates assignment
    print("\n2️⃣  CREATING ASSIGNMENT")
    print("-" * 80)
    assignment = HomeworkAssignment(
        cefr_level=CEFRLevel.A2,
        quantity=5,
        grammar_focus=GrammarFocus.PAST_TENSE,
        topic="history of Milan",
    )
    print(f"Assignment: {assignment}")

    # Step 3: Generate homework
    print("\n3️⃣  GENERATING HOMEWORK")
    print("-" * 80)
    homework_set = env.create_homework(assignment)

    # Step 4: Verify homework correctness
    env.verify_homework_correctness(assignment, homework_set)

    # Step 5: Distribute to all students
    print("\n4️⃣  DISTRIBUTING TO STUDENTS")
    print("-" * 80)
    env.distribute_homework(homework_set)

    # Step 6: Verify students received homework
    print("\n5️⃣  STUDENT VERIFICATION")
    print("-" * 80)
    for student_id in env.students:
        assignments = env.get_student_assignments(student_id)
        student = env.students[student_id]
        print(f"{student.name}: {len(assignments)} assignment(s)")
        if assignments:
            latest = assignments[0]
            print(f"   - {len(latest.homework_set.exercises)} exercises")
            print(f"   - Level: {latest.homework_set.cefr_level}")
            print(f"   - Grammar: {latest.homework_set.grammar_focus}")

    # Export
    env.export_assignments("data/staging/scenario_1.json")

    return env


def demo_scenario_2():
    """Scenario 2: Mixed levels, present tense, no specific topic."""
    print("\n\n" + "=" * 80)
    print("📚 SCENARIO 2: Mixed Levels - Present Tense")
    print("=" * 80)

    env = StagingEnvironment()

    # Add students with different levels
    print("\n1️⃣  SETTING UP CLASS (Mixed Levels)")
    print("-" * 80)
    env.add_student("s001", "Beginner Student", "A1")
    env.add_student("s002", "Elementary Student", "A2")
    env.add_student("s003", "Intermediate Student", "B1")

    # Create assignment for A2 level
    print("\n2️⃣  CREATING ASSIGNMENT (A2 Level)")
    print("-" * 80)
    assignment = HomeworkAssignment(
        cefr_level=CEFRLevel.A2, quantity=3, grammar_focus=GrammarFocus.PRESENT_TENSE
    )
    print(f"Assignment: {assignment}")

    # Generate and distribute
    homework_set = env.create_homework(assignment)
    env.verify_homework_correctness(assignment, homework_set)

    # Distribute only to A2 students (but we'll send to all for demo)
    print("\n3️⃣  DISTRIBUTING TO ALL STUDENTS")
    print("-" * 80)
    print("Note: In production, would filter by level")
    env.distribute_homework(homework_set)

    return env


def main():
    """Run staging demos."""
    print("🎓 STAGING ENVIRONMENT - TEACHER TO STUDENT FLOW")
    print("=" * 80)
    print("This demo simulates the complete flow from teacher creating")
    print("an assignment to students receiving homework.")
    print()

    # Create staging directory
    staging_dir = Path("data/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # Run scenarios
    demo_scenario_1()
    demo_scenario_2()

    print("\n\n" + "=" * 80)
    print("✅ STAGING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Replace mock homework generation with real HomeworkGenerator")
    print("2. Add student UI to receive and complete homework")
    print("3. Add progress tracking and grading")
    print("4. Add notifications when homework is assigned")


if __name__ == "__main__":
    main()
