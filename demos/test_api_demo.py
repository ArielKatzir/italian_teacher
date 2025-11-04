#!/usr/bin/env python3
"""
Italian Teacher API Demo Script

This script demonstrates the complete workflow:
1. Create students
2. Create homework assignment
3. Wait for background generation
4. Retrieve student homework

Run this script with: python test_api_demo.py
Make sure the API server is running first: ./run_api.sh
"""
import sys
import time

import requests

BASE_URL = "http://localhost:8000"


def check_server():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        return False
    return False


def main():
    print("=" * 60)
    print("🇮🇹 Italian Teacher API Demo")
    print("=" * 60)

    # Check server
    print("\n📡 Checking if server is running...")
    if not check_server():
        print("❌ Server is not running!")
        print("\nPlease start the server first:")
        print("   ./run_api.sh")
        print("\nOr visit: http://localhost:8000/docs")
        sys.exit(1)
    print("✅ Server is running!\n")

    # 1. Create students
    print("=" * 60)
    print("STEP 1: Creating Students")
    print("=" * 60)

    students = []
    student_data = [
        ("Mario Rossi", "mario.rossi@example.com"),
        ("Giulia Bianchi", "giulia.bianchi@example.com"),
        ("Luca Verdi", "luca.verdi@example.com"),
    ]

    for name, email in student_data:
        try:
            response = requests.post(
                f"{BASE_URL}/api/teacher/students", json={"name": name, "email": email}
            )
            if response.status_code == 201:
                student = response.json()
                students.append(student)
                print(f"✅ Created: {student['name']} (ID: {student['id']})")
            elif response.status_code == 400:
                # Student already exists, get from list
                response = requests.get(f"{BASE_URL}/api/teacher/students")
                all_students = response.json()
                student = next((s for s in all_students if s["email"] == email), None)
                if student:
                    students.append(student)
                    print(f"ℹ️  Exists: {student['name']} (ID: {student['id']})")
        except Exception as e:
            print(f"❌ Error creating {name}: {e}")

    if not students:
        print("❌ No students created!")
        sys.exit(1)

    # 2. Create assignment
    print("\n" + "=" * 60)
    print("STEP 2: Creating Homework Assignment")
    print("=" * 60)

    print(f"\n📝 Assignment Details:")
    print(f"   CEFR Level: B1")
    print(f"   Grammar Focus: past_tense")
    print(f"   Topic: Italian Renaissance art")
    print(f"   Quantity: 5 exercises")
    print(f"   Students: {', '.join([s['name'] for s in students])}")

    try:
        response = requests.post(
            f"{BASE_URL}/api/teacher/assignments",
            json={
                "cefr_level": "B1",
                "grammar_focus": "past_tense",
                "topic": "Italian Renaissance art",
                "quantity": 5,
                "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
                "student_ids": [s["id"] for s in students],
            },
        )
        assignment = response.json()
        print(f"\n✅ Assignment created!")
        print(f"   Assignment ID: {assignment['id']}")
        print(f"   Status: {assignment['status']}")
    except Exception as e:
        print(f"❌ Error creating assignment: {e}")
        sys.exit(1)

    # 3. Wait for generation
    print("\n" + "=" * 60)
    print("STEP 3: Waiting for Homework Generation")
    print("=" * 60)
    print("\n⏳ Background task is generating exercises...")

    max_wait = 10
    for i in range(max_wait):
        time.sleep(1)

        # Check status
        try:
            response = requests.get(f"{BASE_URL}/api/teacher/assignments/{assignment['id']}")
            assignment = response.json()

            if assignment["status"] == "completed":
                print(f"\n✅ Generation complete! (took {i+1} seconds)")
                break
            elif assignment["status"] == "failed":
                print(f"\n❌ Generation failed!")
                print(f"   Error: {assignment.get('error_message', 'Unknown error')}")
                sys.exit(1)
            else:
                print(f"   [{i+1}/{max_wait}s] Status: {assignment['status']}", end="\r")
        except Exception as e:
            print(f"\n❌ Error checking status: {e}")

    if assignment["status"] != "completed":
        print(f"\n⚠️  Still generating after {max_wait} seconds...")
        print("   Check manually: http://localhost:8000/docs")

    # 4. Retrieve homework
    print("\n" + "=" * 60)
    print("STEP 4: Retrieving Student Homework")
    print("=" * 60)

    for student in students:
        try:
            response = requests.get(
                f"{BASE_URL}/api/student/{student['id']}/homework", params={"status": "available"}
            )
            homework = response.json()

            print(f"\n👤 {student['name']}:")
            if homework["total"] > 0:
                hw = homework["homework"][0]
                print(f"   ✅ Has {len(hw['exercises'])} exercises")
                print(f"   📋 First exercise:")
                first_ex = hw["exercises"][0]
                print(f"      Type: {first_ex['type']}")
                print(f"      Question: {first_ex['question']}")
                if first_ex.get("options"):
                    print(f"      Options: {', '.join(first_ex['options'])}")
                print(f"      Answer: {first_ex['correct_answer']}")
            else:
                print(f"   ⚠️  No homework available yet")
        except Exception as e:
            print(f"   ❌ Error retrieving homework: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   Students created: {len(students)}")
    print(f"   Assignment ID: {assignment['id']}")
    print(f"   Status: {assignment['status']}")
    print(f"\n🌐 View in browser:")
    print(f"   Interactive docs: http://localhost:8000/docs")
    print(
        f"   Assignment details: http://localhost:8000/api/teacher/assignments/{assignment['id']}"
    )
    print(f"\n📚 Database location:")
    print(f"   data/italian_teacher.db")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
