"""
Teacher API endpoints - for creating students and assignments.
"""

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import Assignment, AssignmentStudent, Homework, Student, get_db
from ..schemas import AssignmentCreate, AssignmentResponse, StudentCreate, StudentResponse
from ..services.homework_service import generate_homework_for_assignment

router = APIRouter(prefix="/api/teacher", tags=["teacher"])


@router.post("/students", response_model=StudentResponse, status_code=201)
async def create_student(student: StudentCreate, db: AsyncSession = Depends(get_db)):
    """Create a new student."""
    # Check if email already exists
    result = await db.execute(select(Student).where(Student.email == student.email))
    existing_student = result.scalar_one_or_none()

    if existing_student:
        raise HTTPException(status_code=400, detail="Student with this email already exists")

    # Create new student
    db_student = Student(name=student.name, email=student.email)
    db.add(db_student)
    await db.commit()
    await db.refresh(db_student)

    return db_student


@router.get("/students", response_model=List[StudentResponse])
async def list_students(db: AsyncSession = Depends(get_db)):
    """List all students."""
    result = await db.execute(select(Student).order_by(Student.created_at.desc()))
    students = result.scalars().all()
    return students


@router.delete("/students/{student_id}", status_code=204)
async def delete_student(student_id: int, db: AsyncSession = Depends(get_db)):
    """
    Delete a student.

    This will also delete:
    - All homework records for this student
    - All assignment-student relationships

    Returns 204 No Content on success.
    """
    # Check if student exists
    result = await db.execute(select(Student).where(Student.id == student_id))
    student = result.scalar_one_or_none()

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    # Delete student (cascade will delete homework and assignment_students)
    await db.delete(student)
    await db.commit()

    return None


@router.post("/assignments", response_model=AssignmentResponse, status_code=201)
async def create_assignment(
    assignment: AssignmentCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new homework assignment.

    This will:
    1. Create the assignment record with status='pending'
    2. Trigger background homework generation for all specified students
    3. Return immediately with the assignment_id
    """
    # Validate that all students exist
    result = await db.execute(select(Student).where(Student.id.in_(assignment.student_ids)))
    students = result.scalars().all()

    if len(students) != len(assignment.student_ids):
        raise HTTPException(status_code=404, detail="One or more students not found")

    # Create assignment
    db_assignment = Assignment(
        cefr_level=assignment.cefr_level,
        grammar_focus=assignment.grammar_focus,
        topic=assignment.topic,
        quantity=assignment.quantity,
        exercise_types=assignment.exercise_types,
        status="pending",
    )
    db.add(db_assignment)
    await db.commit()
    await db.refresh(db_assignment)

    # Create assignment-student relationships
    for student_id in assignment.student_ids:
        assignment_student = AssignmentStudent(
            assignment_id=db_assignment.id, student_id=student_id
        )
        db.add(assignment_student)

    # Create homework records with status='pending'
    for student_id in assignment.student_ids:
        homework = Homework(assignment_id=db_assignment.id, student_id=student_id, status="pending")
        db.add(homework)

    await db.commit()

    # Schedule background homework generation
    background_tasks.add_task(generate_homework_for_assignment, assignment_id=db_assignment.id)

    # Return assignment with student_ids
    assignment_dict = {
        "id": db_assignment.id,
        "cefr_level": db_assignment.cefr_level,
        "grammar_focus": db_assignment.grammar_focus,
        "topic": db_assignment.topic,
        "quantity": db_assignment.quantity,
        "exercise_types": db_assignment.exercise_types,
        "status": db_assignment.status,
        "student_ids": assignment.student_ids,  # From request
        "created_at": db_assignment.created_at,
        "completed_at": db_assignment.completed_at,
        "error_message": db_assignment.error_message,
    }

    return assignment_dict


@router.get("/assignments/{assignment_id}", response_model=AssignmentResponse)
async def get_assignment(assignment_id: int, db: AsyncSession = Depends(get_db)):
    """Get assignment status and details."""
    result = await db.execute(select(Assignment).where(Assignment.id == assignment_id))
    assignment = result.scalar_one_or_none()

    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    # Get student IDs from junction table
    result = await db.execute(
        select(AssignmentStudent.student_id).where(AssignmentStudent.assignment_id == assignment_id)
    )
    student_ids = [row[0] for row in result.fetchall()]

    # Convert to dict and add student_ids
    assignment_dict = {
        "id": assignment.id,
        "cefr_level": assignment.cefr_level,
        "grammar_focus": assignment.grammar_focus,
        "topic": assignment.topic,
        "quantity": assignment.quantity,
        "exercise_types": assignment.exercise_types,
        "status": assignment.status,
        "student_ids": student_ids,
        "created_at": assignment.created_at,
        "completed_at": assignment.completed_at,
        "error_message": assignment.error_message,
    }

    return assignment_dict


@router.get("/assignments", response_model=List[AssignmentResponse])
async def list_assignments(db: AsyncSession = Depends(get_db)):
    """List all assignments."""
    result = await db.execute(select(Assignment).order_by(Assignment.created_at.desc()))
    assignments = result.scalars().all()

    # Get student IDs for each assignment
    assignment_list = []
    for assignment in assignments:
        result = await db.execute(
            select(AssignmentStudent.student_id).where(
                AssignmentStudent.assignment_id == assignment.id
            )
        )
        student_ids = [row[0] for row in result.fetchall()]

        assignment_dict = {
            "id": assignment.id,
            "cefr_level": assignment.cefr_level,
            "grammar_focus": assignment.grammar_focus,
            "topic": assignment.topic,
            "quantity": assignment.quantity,
            "exercise_types": assignment.exercise_types,
            "status": assignment.status,
            "student_ids": student_ids,
            "created_at": assignment.created_at,
            "completed_at": assignment.completed_at,
            "error_message": assignment.error_message,
        }
        assignment_list.append(assignment_dict)

    return assignment_list
