"""
Database models and session management for Italian Teacher API.
"""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, relationship

# Database URL - SQLite for local development
DATABASE_URL = "sqlite+aiosqlite:///./data/italian_teacher.db"

# Create async engine
async_engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log SQL queries for debugging
    future=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


# Database Models
class Student(Base):
    """Student model - stores student information."""

    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    homework = relationship("Homework", back_populates="student")


class Assignment(Base):
    """Assignment model - teacher's homework specification."""

    __tablename__ = "assignments"

    id = Column(Integer, primary_key=True, index=True)

    # Assignment parameters (from HomeworkAssignment dataclass)
    cefr_level = Column(String, nullable=False)  # A1, A2, B1, B2, C1, C2
    grammar_focus = Column(String, nullable=True)  # past_tense, subjunctive, etc.
    topic = Column(String, nullable=True)  # "history of Milan", etc.
    quantity = Column(Integer, default=5)
    exercise_types = Column(JSON, nullable=False)  # List of exercise types

    # Status tracking
    status = Column(String, default="pending")  # pending, generating, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Relationships
    homework = relationship("Homework", back_populates="assignment")
    assignment_students = relationship("AssignmentStudent", back_populates="assignment")


class AssignmentStudent(Base):
    """Junction table for assignment-student many-to-many relationship."""

    __tablename__ = "assignment_students"

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(
        Integer, ForeignKey("assignments.id", ondelete="CASCADE"), nullable=False
    )
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    assignment = relationship("Assignment", back_populates="assignment_students")
    student = relationship("Student")


class Homework(Base):
    """Homework model - generated exercises for each student."""

    __tablename__ = "homework"

    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(
        Integer, ForeignKey("assignments.id", ondelete="CASCADE"), nullable=False
    )
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)

    # Generated exercises (JSON structure)
    exercises_json = Column(JSON, nullable=True)  # Will store list of exercises

    # Status tracking
    status = Column(String, default="pending")  # pending, available, in_progress, completed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    assignment = relationship("Assignment", back_populates="homework")
    student = relationship("Student", back_populates="homework")


# Database initialization
async def init_db():
    """Initialize database tables."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database sessions."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
