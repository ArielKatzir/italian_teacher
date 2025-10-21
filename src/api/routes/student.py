"""
Student API endpoints - for retrieving homework.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import Homework, Student, get_db
from ..schemas import ExerciseData, HomeworkListResponse, HomeworkResponse

router = APIRouter(prefix="/api/student", tags=["student"])


@router.get("/{student_id}/homework", response_model=HomeworkListResponse)
async def get_student_homework(
    student_id: int,
    status: str = "all",  # Filter by status: all, available, pending, in_progress, completed
    db: AsyncSession = Depends(get_db),
):
    """
    Get all homework for a specific student.

    Args:
        student_id: The student's ID
        status: Filter by homework status (default: "all")
    """
    # Verify student exists
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one_or_none()

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Get homework filtered by status
    query = select(Homework).where(Homework.student_id == student_id)

    if status and status != "all":
        query = query.where(Homework.status == status)

    result = await db.execute(query.order_by(Homework.created_at.desc()))
    homework_list = result.scalars().all()

    # Format response
    formatted_homework = []
    for hw in homework_list:
        if hw.exercises_json:
            exercises = [ExerciseData(**ex) for ex in hw.exercises_json]
        else:
            exercises = []

        formatted_homework.append(
            HomeworkResponse(
                id=hw.id,
                assignment_id=hw.assignment_id,
                student_id=hw.student_id,
                exercises=exercises,
                status=hw.status,
                created_at=hw.created_at,
                completed_at=hw.completed_at,
            )
        )

    return HomeworkListResponse(homework=formatted_homework, total=len(formatted_homework))


@router.get("/{student_id}/homework/{homework_id}", response_model=HomeworkResponse)
async def get_specific_homework(
    student_id: int, homework_id: int, db: AsyncSession = Depends(get_db)
):
    """Get a specific homework by ID."""
    result = await db.execute(
        select(Homework).where(Homework.id == homework_id, Homework.student_id == student_id)
    )
    homework = result.scalar_one_or_none()

    if not homework:
        raise HTTPException(status_code=404, detail="Homework not found")

    # Format exercises
    if homework.exercises_json:
        exercises = [ExerciseData(**ex) for ex in homework.exercises_json]
    else:
        exercises = []

    return HomeworkResponse(
        id=homework.id,
        assignment_id=homework.assignment_id,
        student_id=homework.student_id,
        exercises=exercises,
        status=homework.status,
        created_at=homework.created_at,
        completed_at=homework.completed_at,
    )
