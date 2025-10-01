"""
Background service for generating homework using MarcoInference.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import select

from ..database import Assignment, AsyncSessionLocal, Homework


async def generate_homework_for_assignment(assignment_id: int):
    """
    Background task to generate homework for all students in an assignment.

    This function:
    1. Fetches the assignment and all associated students
    2. Generates exercises using MarcoInference for each student
    3. Updates homework records with generated exercises
    4. Updates assignment status to 'completed' or 'failed'
    """
    async with AsyncSessionLocal() as db:
        try:
            # Fetch assignment
            result = await db.execute(select(Assignment).where(Assignment.id == assignment_id))
            assignment = result.scalar_one_or_none()

            if not assignment:
                print(f"Assignment {assignment_id} not found")
                return

            # Update assignment status to 'generating'
            assignment.status = "generating"
            await db.commit()

            # Fetch all homework records for this assignment
            result = await db.execute(
                select(Homework).where(Homework.assignment_id == assignment_id)
            )
            homework_list = result.scalars().all()

            # Generate exercises for each student
            for homework in homework_list:
                try:
                    # Generate exercises using MarcoInference
                    exercises = await generate_exercises(
                        cefr_level=assignment.cefr_level,
                        grammar_focus=assignment.grammar_focus,
                        topic=assignment.topic,
                        quantity=assignment.quantity,
                        exercise_types=assignment.exercise_types,
                    )

                    # Update homework record
                    homework.exercises_json = exercises
                    homework.status = "available"
                    await db.commit()

                except Exception as e:
                    print(f"Error generating homework for student {homework.student_id}: {e}")
                    homework.status = "failed"
                    await db.commit()

            # Update assignment status to 'completed'
            assignment.status = "completed"
            assignment.completed_at = datetime.utcnow()
            await db.commit()

            print(f"Assignment {assignment_id} completed successfully")

        except Exception as e:
            print(f"Error generating homework for assignment {assignment_id}: {e}")

            # Update assignment status to 'failed'
            if assignment:
                assignment.status = "failed"
                assignment.error_message = str(e)
                await db.commit()


async def generate_exercises(
    cefr_level: str, grammar_focus: str, topic: str, quantity: int, exercise_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Generate exercises using MarcoInference.

    For now, this is a mock implementation. Replace with real MarcoInference integration.

    Args:
        cefr_level: CEFR level (A1-C2)
        grammar_focus: Grammar topic (e.g., "past_tense")
        topic: Content topic (e.g., "history of Milan")
        quantity: Number of exercises
        exercise_types: List of exercise types

    Returns:
        List of exercise dictionaries
    """
    # TODO: Replace with real MarcoInference integration
    # from src.fine_tuning.inference import MarcoInference

    # Mock implementation for testing
    exercises = []

    for i in range(quantity):
        exercise_type = exercise_types[i % len(exercise_types)]

        if exercise_type == "fill_in_blank":
            exercises.append(
                {
                    "type": "fill_in_blank",
                    "question": f"Io ___ a Roma ieri. (andare)",
                    "correct_answer": "sono andato" if grammar_focus == "past_tense" else "vado",
                    "explanation": f"Using {grammar_focus or 'present tense'} conjugation",
                }
            )

        elif exercise_type == "translation":
            exercises.append(
                {
                    "type": "translation",
                    "question": f"Translate: I went to {topic or 'Milan'} yesterday.",
                    "correct_answer": f"Sono andato a {topic or 'Milano'} ieri.",
                    "explanation": f"CEFR {cefr_level} translation exercise",
                }
            )

        elif exercise_type == "multiple_choice":
            exercises.append(
                {
                    "type": "multiple_choice",
                    "question": f"Which is the correct {grammar_focus or 'form'}?",
                    "correct_answer": "sono andato",
                    "options": ["vado", "sono andato", "andavo", "andrò"],
                    "explanation": f"Past tense of 'andare' at {cefr_level} level",
                }
            )

        else:
            exercises.append(
                {
                    "type": exercise_type,
                    "question": f"Exercise {i+1} for {cefr_level} level",
                    "correct_answer": "Sample answer",
                    "explanation": f"Generated for {grammar_focus or 'general'} practice",
                }
            )

    # Simulate generation delay
    await asyncio.sleep(2)

    return exercises
