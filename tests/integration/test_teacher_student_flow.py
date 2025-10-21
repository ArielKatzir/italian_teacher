"""
Integration test for teacher-to-student homework flow via API.

Tests the complete flow:
1. Teacher creates students via API
2. Teacher creates homework assignments
3. Homework is generated (mock or real via Colab)
4. Students retrieve their homework
5. Students submit answers

NOTE: These tests are currently skipped as the API has changed significantly.
They need to be rewritten to match the new async FastAPI implementation.
"""

import pytest

pytest.skip("API tests need to be rewritten for new async implementation", allow_module_level=True)

from httpx import ASGITransport, AsyncClient

from api.database import Base, async_engine
from api.main import app


@pytest.fixture(autouse=True)
async def reset_database():
    """Reset database before each test."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client():
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.integration
class TestTeacherStudentFlow:
    """Integration tests for teacher-to-student homework flow via API."""

    async def test_create_student(self, client):
        """Test creating a student."""
        response = await client.post(
            "/teacher/students", json={"name": "Mario Rossi", "cefr_level": "A2"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Mario Rossi"
        assert data["cefr_level"] == "A2"
        assert "id" in data

    async def test_create_homework_assignment(self, client):
        """Test creating a homework assignment."""
        # First create a student
        student_response = await client.post(
            "/teacher/students", json={"name": "Maria Bianchi", "cefr_level": "A2"}
        )
        student_id = student_response.json()["id"]

        # Create homework
        homework_response = await client.post(
            "/teacher/homework",
            json={
                "student_id": student_id,
                "cefr_level": "A2",
                "grammar_focus": "present_tense",
                "topic": "daily routines",
                "quantity": 5,
                "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
            },
        )
        assert homework_response.status_code == 200
        data = homework_response.json()
        assert data["student_id"] == student_id
        assert data["cefr_level"] == "A2"
        assert data["status"] == "pending"
        assert "id" in data

    async def test_student_get_homework(self, client):
        """Test student retrieving their homework."""
        # Create student
        student_response = await client.post(
            "/teacher/students", json={"name": "Lucia Verde", "cefr_level": "B1"}
        )
        student_id = student_response.json()["id"]

        # Create homework
        await client.post(
            "/teacher/homework",
            json={
                "student_id": student_id,
                "cefr_level": "B1",
                "grammar_focus": "past_tense",
                "topic": "vacation stories",
                "quantity": 3,
                "exercise_types": ["translation", "fill_in_blank"],
            },
        )

        # Student gets homework
        homework_response = await client.get(f"/student/{student_id}/homework?status=pending")
        assert homework_response.status_code == 200
        data = homework_response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["cefr_level"] == "B1"
        assert data[0]["status"] == "pending"

    async def test_complete_flow_simple(self, client):
        """Test complete flow with simple assignment."""
        # 1. Teacher creates student
        student_response = await client.post(
            "/teacher/students", json={"name": "Test Student", "cefr_level": "A2"}
        )
        student_id = student_response.json()["id"]

        # 2. Teacher creates homework
        homework_response = await client.post(
            "/teacher/homework",
            json={
                "student_id": student_id,
                "cefr_level": "A2",
                "grammar_focus": "present_tense",
                "topic": "daily routines",
                "quantity": 5,
                "exercise_types": ["fill_in_blank", "translation"],
            },
        )
        homework_id = homework_response.json()["id"]

        # 3. Student retrieves homework
        student_homework = await client.get(f"/student/{student_id}/homework")
        assert student_homework.status_code == 200
        assignments = student_homework.json()
        assert len(assignments) >= 1

        # Find the homework we just created
        assignment = next(a for a in assignments if a["id"] == homework_id)
        assert assignment["cefr_level"] == "A2"
        assert assignment["grammar_focus"] == "present_tense"

    async def test_multiple_students(self, client):
        """Test multiple students receiving homework."""
        # Create 3 students
        students = []
        for i in range(3):
            response = await client.post(
                "/teacher/students", json={"name": f"Student {i+1}", "cefr_level": "A2"}
            )
            students.append(response.json()["id"])

        # Create homework for each
        for student_id in students:
            response = await client.post(
                "/teacher/homework",
                json={
                    "student_id": student_id,
                    "cefr_level": "A2",
                    "grammar_focus": "present_tense",
                    "topic": "family",
                    "quantity": 5,
                    "exercise_types": ["fill_in_blank"],
                },
            )
            assert response.status_code == 200

        # Verify each student has homework
        for student_id in students:
            response = await client.get(f"/student/{student_id}/homework")
            assert response.status_code == 200
            assignments = response.json()
            assert len(assignments) >= 1

    async def test_different_cefr_levels(self, client):
        """Test homework for different CEFR levels."""
        levels = ["A1", "A2", "B1", "B2"]

        for level in levels:
            # Create student
            student_response = await client.post(
                "/teacher/students", json={"name": f"Student {level}", "cefr_level": level}
            )
            student_id = student_response.json()["id"]

            # Create homework
            homework_response = await client.post(
                "/teacher/homework",
                json={
                    "student_id": student_id,
                    "cefr_level": level,
                    "grammar_focus": "present_tense",
                    "topic": "greetings",
                    "quantity": 3,
                    "exercise_types": ["fill_in_blank"],
                },
            )
            assert homework_response.status_code == 200
            data = homework_response.json()
            assert data["cefr_level"] == level

    async def test_list_students(self, client):
        """Test listing all students."""
        # Create students
        for i in range(3):
            await client.post(
                "/teacher/students", json={"name": f"Student {i+1}", "cefr_level": "A2"}
            )

        # List students
        response = await client.get("/teacher/students")
        assert response.status_code == 200
        students = response.json()
        assert len(students) >= 3

    async def test_list_teacher_homework(self, client):
        """Test teacher listing homework assignments."""
        # Create student
        student_response = await client.post(
            "/teacher/students", json={"name": "Test Student", "cefr_level": "A2"}
        )
        student_id = student_response.json()["id"]

        # Create multiple homework
        for i in range(3):
            await client.post(
                "/teacher/homework",
                json={
                    "student_id": student_id,
                    "cefr_level": "A2",
                    "grammar_focus": "present_tense",
                    "topic": f"topic_{i}",
                    "quantity": 5,
                    "exercise_types": ["fill_in_blank"],
                },
            )

        # List all homework
        response = await client.get("/teacher/homework")
        assert response.status_code == 200
        homework = response.json()
        assert len(homework) >= 3

    async def test_filter_homework_by_student(self, client):
        """Test filtering homework by student."""
        # Create 2 students
        student1_response = await client.post(
            "/teacher/students", json={"name": "Student 1", "cefr_level": "A2"}
        )
        student1_id = student1_response.json()["id"]

        student2_response = await client.post(
            "/teacher/students", json={"name": "Student 2", "cefr_level": "B1"}
        )
        student2_id = student2_response.json()["id"]

        # Create homework for both
        for student_id in [student1_id, student2_id]:
            await client.post(
                "/teacher/homework",
                json={
                    "student_id": student_id,
                    "cefr_level": "A2",
                    "grammar_focus": "present_tense",
                    "topic": "test",
                    "quantity": 5,
                    "exercise_types": ["fill_in_blank"],
                },
            )

        # Filter by student1
        response = await client.get(f"/teacher/homework?student_id={student1_id}")
        assert response.status_code == 200
        homework = response.json()
        assert all(h["student_id"] == student1_id for h in homework)

    async def test_real_world_scenario_a2_class(self, client):
        """
        Real-world scenario: A2 class, present tense, daily routines.
        """
        # Setup class
        students = []
        for name in ["Maria Rossi", "Giovanni Bianchi", "Lucia Verde"]:
            response = await client.post(
                "/teacher/students", json={"name": name, "cefr_level": "A2"}
            )
            students.append(response.json()["id"])

        # Create assignment for all students
        for student_id in students:
            response = await client.post(
                "/teacher/homework",
                json={
                    "student_id": student_id,
                    "cefr_level": "A2",
                    "grammar_focus": "present_tense",
                    "topic": "daily routines",
                    "quantity": 5,
                    "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
                },
            )
            assert response.status_code == 200

        # Verify all students received homework
        for student_id in students:
            response = await client.get(f"/student/{student_id}/homework")
            assert response.status_code == 200
            assignments = response.json()
            assert len(assignments) >= 1
            assert assignments[0]["cefr_level"] == "A2"
            assert assignments[0]["grammar_focus"] == "present_tense"
            assert assignments[0]["topic"] == "daily routines"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
