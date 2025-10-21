"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


# Student Schemas
class StudentCreate(BaseModel):
    """Request schema for creating a student."""

    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr


class StudentResponse(BaseModel):
    """Response schema for student data."""

    id: int
    name: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True


# Assignment Schemas
class AssignmentCreate(BaseModel):
    """Request schema for creating a homework assignment."""

    cefr_level: str = Field(..., pattern="^(A1|A2|B1|B2|C1|C2)$")
    grammar_focus: Optional[str] = None
    topic: Optional[str] = None
    quantity: int = Field(default=5, ge=1, le=20)
    exercise_types: List[str] = Field(
        default_factory=lambda: ["fill_in_blank", "translation", "multiple_choice"]
    )
    student_ids: List[int] = Field(..., min_items=1)  # List of student IDs to assign to


class AssignmentResponse(BaseModel):
    """Response schema for assignment data."""

    id: int
    cefr_level: str
    grammar_focus: Optional[str]
    topic: Optional[str]
    quantity: int
    exercise_types: List[str]
    status: str
    student_ids: List[int] = []  # List of student IDs assigned to this homework
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

    class Config:
        from_attributes = True


# Homework Schemas
class ExerciseData(BaseModel):
    """Schema for individual exercise."""

    type: str  # fill_in_blank, translation, multiple_choice, etc.
    question: str
    correct_answer: str
    options: Optional[List[str]] = None  # For multiple choice
    explanation: Optional[str] = None


class HomeworkResponse(BaseModel):
    """Response schema for student homework."""

    id: int
    assignment_id: int
    student_id: int
    exercises: List[ExerciseData]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class HomeworkListResponse(BaseModel):
    """Response schema for list of homework."""

    homework: List[HomeworkResponse]
    total: int
