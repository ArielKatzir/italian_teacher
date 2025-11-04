"""
Italian Teacher API - FastAPI application.

This API provides endpoints for:
- Teachers to create students and homework assignments
- Students to retrieve their homework
- Background homework generation using GPU inference
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routes import student, teacher


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    print("Initializing database...")
    await init_db()
    print("Database initialized successfully")
    yield
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Italian Teacher API",
    description="API for managing Italian language homework assignments",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(teacher.router)
app.include_router(student.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Italian Teacher API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "teacher": {
                "create_student": "POST /api/teacher/students",
                "list_students": "GET /api/teacher/students",
                "create_assignment": "POST /api/teacher/assignments",
                "get_assignment": "GET /api/teacher/assignments/{assignment_id}",
                "list_assignments": "GET /api/teacher/assignments",
            },
            "student": {
                "get_homework": "GET /api/student/{student_id}/homework",
                "get_specific_homework": "GET /api/student/{student_id}/homework/{homework_id}",
            },
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
