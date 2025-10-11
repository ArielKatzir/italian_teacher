# Italian Teacher API - Demo Guide

Complete guide to starting, using, and stopping the API server.

---

## 🚀 Quick Start

### 0. Set Up GPU Inference (REQUIRED)

**GPU inference is required** - the API will not work without it.

1. **Open your Colab GPU notebook** ([demos/colab_inference_api.ipynb](demos/colab_inference_api.ipynb))
2. **Run all cells** to start the inference server
3. **Copy the ngrok URL** from the output (e.g., `https://abc123.ngrok.io`)
4. **Set the environment variable:**
   ```bash
   export INFERENCE_API_URL="https://your-ngrok-url.ngrok.io"
   ```

**Without this setup**, exercise generation will fail with an error.

### 1. Start the Server

```bash
cd "/Users/arielkatzir/Library/CloudStorage/GoogleDrive-ari.katzir@gmail.com/My Drive/Colab Notebooks/italian_teacher"
./run_api.sh
```

**You should see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
Initializing database...
Database initialized successfully
INFO:     Application startup complete.
```

✅ **Server is now running at http://localhost:8000**

### 2. Open Interactive API Docs

**Option A: Browser (Recommended)**
```
http://localhost:8000/docs
```

This opens **Swagger UI** - a beautiful interface where you can test all endpoints by clicking buttons!

**Option B: Alternative Docs**
```
http://localhost:8000/redoc
```

ReDoc format (read-only, better for documentation).

---

## 📚 Complete Workflow Demo

### Step 1: Create Students

**Using Browser (http://localhost:8000/docs):**
1. Click on `POST /api/teacher/students`
2. Click "Try it out"
3. Enter JSON:
   ```json
   {
     "name": "Marco Rossi",
     "email": "marco@example.com"
   }
   ```
4. Click "Execute"

**Using Command Line:**
```bash
# Create Student 1
curl -X POST "http://localhost:8000/api/teacher/students" \
  -H "Content-Type: application/json" \
  -d '{"name": "Marco Rossi", "email": "marco@example.com"}'

# Create Student 2
curl -X POST "http://localhost:8000/api/teacher/students" \
  -H "Content-Type: application/json" \
  -d '{"name": "Giulia Bianchi", "email": "giulia@example.com"}'
```

**Expected Response:**
```json
{
  "id": 1,
  "name": "Marco Rossi",
  "email": "marco@example.com",
  "created_at": "2025-10-01T12:00:00.123456"
}
```

---

### Step 2: List All Students

**Browser:** Click `GET /api/teacher/students` → "Try it out" → "Execute"

**Command Line:**
```bash
curl "http://localhost:8000/api/teacher/students"
```

**Expected Response:**
```json
[
  {
    "id": 1,
    "name": "Marco Rossi",
    "email": "marco@example.com",
    "created_at": "2025-10-01T12:00:00.123456"
  },
  {
    "id": 2,
    "name": "Giulia Bianchi",
    "email": "giulia@example.com",
    "created_at": "2025-10-01T12:00:05.789012"
  }
]
```

---

### Step 3: Create Homework Assignment

**Browser:**
1. Click `POST /api/teacher/assignments`
2. Click "Try it out"
3. Enter JSON:
   ```json
   {
     "cefr_level": "A2",
     "grammar_focus": "past_tense",
     "topic": "history of Milan",
     "quantity": 5,
     "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
     "student_ids": [1, 2]
   }
   ```
4. Click "Execute"

**Command Line:**
```bash
curl -X POST "http://localhost:8000/api/teacher/assignments" \
  -H "Content-Type: application/json" \
  -d '{
    "cefr_level": "A2",
    "grammar_focus": "past_tense",
    "topic": "history of Milan",
    "quantity": 5,
    "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
    "student_ids": [1, 2]
  }'
```

**Expected Response:**
```json
{
  "id": 1,
  "cefr_level": "A2",
  "grammar_focus": "past_tense",
  "topic": "history of Milan",
  "quantity": 5,
  "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
  "status": "pending",  ⬅️ Will change to "completed" in ~2 seconds
  "created_at": "2025-10-01T12:00:00.123456",
  "completed_at": null,
  "error_message": null
}
```

**⚠️ Important:** The assignment is created immediately, but homework generation happens **in the background** using GPU.
- Wait ~90 seconds for 5 exercises (varies by GPU: L4, V100, A100)
- If GPU is not available, generation will fail

---

### Step 4: Check Assignment Status

**Wait for generation to complete** (~90 seconds on GPU), then check status:

**Browser:** Click `GET /api/teacher/assignments/{assignment_id}` → Enter `1` → "Execute"

**Command Line:**
```bash
curl "http://localhost:8000/api/teacher/assignments/1"
```

**Expected Response:**
```json
{
  "id": 1,
  "cefr_level": "A2",
  "grammar_focus": "past_tense",
  "topic": "history of Milan",
  "quantity": 5,
  "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
  "status": "completed",  ⬅️ Changed from "pending"!
  "created_at": "2025-10-01T12:00:00.123456",
  "completed_at": "2025-10-01T12:00:03.456789",  ⬅️ Now has timestamp
  "error_message": null
}
```

**Status meanings:**
- `pending` - Assignment created, homework generation not started yet
- `generating` - Currently generating exercises
- `completed` - All homework generated successfully
- `failed` - Generation failed (check `error_message`)

---

### Step 5: Student Retrieves Homework

Now students can see their homework!

**Browser:** Click `GET /api/student/{student_id}/homework` → Enter `1` → "Execute"

**Command Line:**
```bash
# Get homework for student 1 (Marco)
curl "http://localhost:8000/api/student/1/homework?status=available"

# Get homework for student 2 (Giulia)
curl "http://localhost:8000/api/student/2/homework?status=available"
```

**Expected Response:**
```json
{
  "homework": [
    {
      "id": 1,
      "assignment_id": 1,
      "student_id": 1,
      "exercises": [
        {
          "type": "fill_in_blank",
          "question": "Io ___ a Roma ieri. (andare)",
          "correct_answer": "sono andato",
          "options": null,
          "explanation": "Using past_tense conjugation"
        },
        {
          "type": "translation",
          "question": "Translate: I went to history of Milan yesterday.",
          "correct_answer": "Sono andato a history of Milan ieri.",
          "options": null,
          "explanation": "CEFR A2 translation exercise"
        },
        {
          "type": "multiple_choice",
          "question": "Which is the correct past_tense?",
          "correct_answer": "sono andato",
          "options": ["vado", "sono andato", "andavo", "andrò"],
          "explanation": "Past tense of 'andare' at A2 level"
        },
        {
          "type": "fill_in_blank",
          "question": "Io ___ a Roma ieri. (andare)",
          "correct_answer": "sono andato",
          "options": null,
          "explanation": "Using past_tense conjugation"
        },
        {
          "type": "translation",
          "question": "Translate: I went to history of Milan yesterday.",
          "correct_answer": "Sono andato a history of Milan ieri.",
          "options": null,
          "explanation": "CEFR A2 translation exercise"
        }
      ],
      "status": "available",
      "created_at": "2025-10-01T12:00:00.123456",
      "completed_at": null
    }
  ],
  "total": 1
}
```

---

## 🎯 Advanced Usage

### Filter Homework by Status

```bash
# Get available homework (default)
curl "http://localhost:8000/api/student/1/homework?status=available"

# Get homework in progress
curl "http://localhost:8000/api/student/1/homework?status=in_progress"

# Get completed homework
curl "http://localhost:8000/api/student/1/homework?status=completed"

# Get all homework (no filter)
curl "http://localhost:8000/api/student/1/homework"
```

### Get Specific Homework

```bash
# Get homework by ID
curl "http://localhost:8000/api/student/1/homework/1"
```

### List All Assignments

```bash
curl "http://localhost:8000/api/teacher/assignments"
```

---

## 🧪 Testing with Python

Create a test script:

```python
# test_api_demo.py
import requests
import time

BASE_URL = "http://localhost:8000"

print("=== Italian Teacher API Demo ===\n")

# 1. Create students
print("1. Creating students...")
students = []
for name, email in [("Marco Rossi", "marco@test.com"), ("Giulia Bianchi", "giulia@test.com")]:
    response = requests.post(
        f"{BASE_URL}/api/teacher/students",
        json={"name": name, "email": email}
    )
    student = response.json()
    students.append(student)
    print(f"   ✅ Created: {student['name']} (ID: {student['id']})")

# 2. Create assignment
print("\n2. Creating homework assignment...")
response = requests.post(
    f"{BASE_URL}/api/teacher/assignments",
    json={
        "cefr_level": "B1",
        "grammar_focus": "subjunctive",
        "topic": "Italian culture",
        "quantity": 3,
        "exercise_types": ["fill_in_blank", "translation"],
        "student_ids": [s['id'] for s in students]
    }
)
assignment = response.json()
print(f"   ✅ Assignment created (ID: {assignment['id']})")
print(f"   📊 Status: {assignment['status']}")

# 3. Wait for generation
print("\n3. Waiting for homework generation...")
for i in range(3):
    time.sleep(1)
    print(f"   ⏳ {i+1} seconds...")

# 4. Check assignment status
response = requests.get(f"{BASE_URL}/api/teacher/assignments/{assignment['id']}")
assignment = response.json()
print(f"   📊 Status: {assignment['status']}")

# 5. Get student homework
print("\n4. Retrieving student homework...")
for student in students:
    response = requests.get(
        f"{BASE_URL}/api/student/{student['id']}/homework",
        params={"status": "available"}
    )
    homework = response.json()
    if homework['total'] > 0:
        num_exercises = len(homework['homework'][0]['exercises'])
        print(f"   ✅ {student['name']}: {num_exercises} exercises")
        print(f"      First question: {homework['homework'][0]['exercises'][0]['question']}")
    else:
        print(f"   ⚠️  {student['name']}: No homework yet")

print("\n=== Demo Complete! ===")
```

Run it:
```bash
python test_api_demo.py
```

---

## 🛑 Stop the Server

**Press `Ctrl+C` in the terminal where the server is running**

You'll see:
```
INFO:     Shutting down
INFO:     Waiting for application shutdown.
Shutting down...
INFO:     Application shutdown complete.
INFO:     Finished server process
```

---

## 📂 Where is Data Stored?

All data is stored in a **SQLite database file:**

```
data/italian_teacher.db
```

**To view the database:**
```bash
# Install sqlite3 if needed: brew install sqlite

# Open database
sqlite3 data/italian_teacher.db

# View tables
.tables

# Query students
SELECT * FROM students;

# Query assignments
SELECT * FROM assignments;

# Query homework
SELECT * FROM homework;

# Exit
.quit
```

**To reset the database (delete all data):**
```bash
rm data/italian_teacher.db
# Next time you start the server, it will create a fresh empty database
```

---

## 🐛 Troubleshooting

### Server won't start - Port already in use

```bash
# Find process using port 8000
lsof -ti:8000

# Kill it
kill -9 $(lsof -ti:8000)

# Or use a different port
/usr/local/opt/python@3.9/bin/python3.9 -m uvicorn src.api.main:app --reload --port 8001
```

### Database locked error

```bash
# Someone else is using the database file
# Stop all running servers and try again
```

### Import errors

```bash
# Make sure dependencies are installed
pip install -r requirements.txt
```

---

## 📊 API Endpoints Summary

### Teacher Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/teacher/students` | Create a student |
| GET | `/api/teacher/students` | List all students |
| POST | `/api/teacher/assignments` | Create homework assignment (generates in background) |
| GET | `/api/teacher/assignments/{id}` | Get assignment status |
| GET | `/api/teacher/assignments` | List all assignments |

### Student Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/student/{student_id}/homework` | Get student's homework (filter by status) |
| GET | `/api/student/{student_id}/homework/{homework_id}` | Get specific homework by ID |

### Utility Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and endpoint list |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/redoc` | Alternative API documentation (ReDoc) |

---

## 🔮 Next Steps

**Current Status:**
- ✅ API working with GPU exercise generation via `italian_exercise_generator_lora`
- ✅ Database storing students, assignments, homework
- ✅ Background task queue for async generation
- ✅ Colab GPU inference API integration (via ngrok tunnel)

**How Exercise Generation Works:**
1. Set `INFERENCE_API_URL` environment variable to your Colab ngrok URL
2. Uses fine-tuned `models/italian_exercise_generator_lora` model
3. Generates high-quality, level-appropriate Italian exercises
4. ~90 seconds for 5 exercises on GPU (L4, V100, or A100)

**⚠️ GPU is required** - generation will fail without `INFERENCE_API_URL` set

**TODO:**
1. Add student answer submission endpoint
2. Add homework grading system
3. Add teacher analytics dashboard

**Generation Service Location:**
[src/api/services/homework_service.py](src/api/services/homework_service.py:94) - `generate_exercises()` function

---

## 📞 Quick Reference Card

```bash
# Start server
./run_api.sh

# Stop server
Ctrl+C

# Open docs
http://localhost:8000/docs

# Health check
curl http://localhost:8000/health

# View database
sqlite3 data/italian_teacher.db

# Reset database
rm data/italian_teacher.db
```

Happy coding! 🇮🇹 📚
