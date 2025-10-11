"""
Background service for generating homework using italian_exercise_generator_lora model.

This service requires Colab GPU inference API:
- Connects to remote Colab GPU via HTTP (ngrok tunnel)
- Uses fine-tuned italian_exercise_generator_lora model
- Generates high-quality Italian language exercises
- INFERENCE_API_URL environment variable is required
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import aiohttp
from sqlalchemy import select

from ..database import Assignment, AsyncSessionLocal, Homework

# Colab Inference API Configuration
# Get URL from environment variable or use None for mock mode
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", None)
# Example: INFERENCE_API_URL = "https://abc123.ngrok.io"


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
    Generate exercises using italian_exercise_generator_lora via Colab GPU inference API.

    This function calls the Colab inference service to generate authentic Italian
    language exercises using the fine-tuned exercise generator model.

    Args:
        cefr_level: CEFR level (A1-C2)
        grammar_focus: Grammar topic (e.g., "past_tense")
        topic: Content topic (e.g., "history of Milan")
        quantity: Number of exercises
        exercise_types: List of exercise types

    Returns:
        List of exercise dictionaries

    Environment Variables:
        INFERENCE_API_URL: URL of Colab inference API (ngrok tunnel) - REQUIRED

    Raises:
        ValueError: If INFERENCE_API_URL is not set
    """
    if not INFERENCE_API_URL:
        error_msg = (
            "❌ INFERENCE_API_URL not set. GPU inference is required.\n"
            "   Please start your Colab notebook and set the environment variable:\n"
            '   export INFERENCE_API_URL="https://your-ngrok-url.ngrok.io"'
        )
        print(error_msg)
        raise ValueError(
            "INFERENCE_API_URL environment variable is required for exercise generation"
        )

    # Use Colab GPU inference
    return await _generate_exercises_remote(
        cefr_level, grammar_focus, topic, quantity, exercise_types
    )


async def _generate_exercises_remote(
    cefr_level: str, grammar_focus: str, topic: str, quantity: int, exercise_types: List[str]
) -> List[Dict[str, Any]]:
    """
    Generate exercises using remote Colab GPU inference API.

    Makes HTTP POST request to Colab FastAPI service running on GPU.

    Raises:
        Exception: If generation fails for any reason (no fallback)
    """
    request_payload = {
        "cefr_level": cefr_level,
        "grammar_focus": grammar_focus or "",  # Convert None to empty string
        "topic": topic or "",  # Convert None to empty string
        "quantity": quantity,
        "exercise_types": exercise_types,
        "max_tokens": 1500,
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{INFERENCE_API_URL}/generate",
            json=request_payload,
            timeout=aiohttp.ClientTimeout(total=180),  # 180 second timeout (3 minutes)
        ) as response:
            if response.status == 200:
                data = await response.json()
                exercises = data.get("exercises", [])

                print(
                    f"✅ Generated {len(exercises)} exercises via Colab GPU "
                    f"({data.get('inference_time', 0):.2f}s, "
                    f"{data.get('generated_tokens', 0)} tokens)"
                )

                return exercises
            else:
                error_text = await response.text()
                error_msg = f"Colab API error {response.status}: {error_text[:500]}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
