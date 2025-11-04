# Italian Teacher API

REST API for managing Italian language homework assignments with GPU-powered exercise generation.

## Architecture Overview

This system consists of **TWO API services** that work together:

```
┌─────────────────────────────┐
│  Service 1: Main API        │
│  (Local Machine)            │
│  Port: 8000                 │
│  File: src/api/main.py      │
│  ├─ Teacher endpoints       │
│  ├─ Student endpoints       │
│  └─ SQLite database         │
└───────────┬─────────────────┘
            │ HTTP POST /generate
            ↓
┌─────────────────────────────┐
│  Service 2: GPU Inference   │
│  (Google Colab)             │
│  Port: 8001 (via ngrok)     │
│  File: src/api/inference/   │
│        colab_api.py         │
│  ├─ vLLM inference engine   │
│  ├─ Fine-tuned model        │
│  └─ Exercise generation     │
└─────────────────────────────┘
```

**Both services must be running** for exercise generation to work!

## Quick Reference

**Three ways to use the API:**
1. 🖥️  **CLI Tools** (easiest) - `python -m src.cli.teacher_cli` or `python -m src.cli.student_cli`
2. 🌐 **HTTP API** (programmatic) - Direct REST API calls
3. 📖 **Interactive Docs** (exploration) - http://localhost:8000/docs

**Both services must be running:**
- Service 1 (Local): `./run_api.sh` → http://localhost:8000
- Service 2 (Colab GPU): Run Colab notebook → Get ngrok URL

## Quick Start

### Step 1: Start the GPU Inference Service (Service 2)

This service runs on Google Colab and handles the actual AI model inference.

1. Open [demos/colab_inference_api.ipynb](../../demos/colab_inference_api.ipynb) in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells to start the inference server
4. **Copy the ngrok URL** from the output (e.g., `https://abc123.ngrok-free.dev`)

The Colab service will start on port 8001 and expose these endpoints via ngrok:
- `GET /` - Health check
- `GET /health` - Detailed GPU status
- `POST /generate` - Generate exercises

### Step 2: Start the Main API Service (Service 1)

This service runs on your local machine and provides teacher/student endpoints.

```bash
# Export the Colab API URL (REQUIRED)
export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"

# Start the local API
./run_api.sh
```

The local API will start at http://localhost:8000

### Step 3: Verify Both Services

**Check Service 1 (Local API):**
```bash
curl http://localhost:8000/health
# Should return: {"status": "healthy"}
```

**Check Service 2 (Colab GPU):**
```bash
curl https://your-ngrok-url.ngrok-free.dev/health
# Should return: {"status": "healthy", "gpu_available": true, ...}
```

### Interactive API Documentation

Visit http://localhost:8000/docs for Swagger UI with interactive testing.

## Usage Options

You can interact with the API in three ways:

1. **Web API** (this guide) - HTTP requests via curl, Postman, or code
2. **CLI Tools** - Command-line interface for teachers and students
3. **Interactive Docs** - Swagger UI at http://localhost:8000/docs

### CLI Tools (Recommended for Quick Testing)

The project includes CLI tools for easier interaction:

**Teacher CLI** - Manage students and assignments
```bash
# Create a student
python -m src.cli.teacher_cli student create --name "Mario Rossi" --email "mario@example.com"

# List all students
python -m src.cli.teacher_cli student list

# Create assignment for students
python -m src.cli.teacher_cli assignment create \
  --student-ids 1,2 \
  --level A2 \
  --topic "daily routines" \
  --quantity 5

# Check assignment status
python -m src.cli.teacher_cli assignment status --id 1
```

**Student CLI** - View homework
```bash
# List homework for a student
python -m src.cli.student_cli homework list --student-id 1

# View specific homework with all exercises
python -m src.cli.student_cli homework view --student-id 1 --homework-id 1
```

**Installation:**
```bash
pip install typer rich httpx
```

**Full CLI Guide:** See [docs/CLI_GUIDE.md](../../docs/CLI_GUIDE.md) for complete CLI documentation.

## API Endpoints

### Teacher Endpoints

**Create Student**
```bash
POST /api/teacher/students
Content-Type: application/json

{
  "name": "Mario Rossi",
  "email": "mario@example.com"
}
```

**List Students**
```bash
GET /api/teacher/students
```

**Create Homework Assignment**
```bash
POST /api/teacher/assignments
Content-Type: application/json

{
  "cefr_level": "A2",
  "grammar_focus": "past_tense",
  "topic": "history of Milan",
  "quantity": 5,
  "exercise_types": ["fill_in_blank", "translation", "multiple_choice"],
  "student_ids": [1, 2]
}
```

**Get Assignment Status**
```bash
GET /api/teacher/assignments/{assignment_id}
```

**List All Assignments**
```bash
GET /api/teacher/assignments
```

### Student Endpoints

**Get Student Homework**
```bash
GET /api/student/{student_id}/homework?status=available
```

**Get Specific Homework**
```bash
GET /api/student/{student_id}/homework/{homework_id}
```

### Utility Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and endpoint list |
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation (Swagger UI) |
| GET | `/redoc` | Alternative API documentation (ReDoc) |

## Detailed Architecture

### Two-Service Design

```
┌──────────────────────────────────────────────────────────┐
│  SERVICE 1: Main API (Local Machine)                     │
│  File: src/api/main.py                                   │
│  Port: 8000                                              │
│                                                          │
│  Teacher Endpoints:                                      │
│  • POST /api/teacher/students                           │
│  • POST /api/teacher/assignments                        │
│  • GET  /api/teacher/assignments/{id}                   │
│                                                          │
│  Student Endpoints:                                      │
│  • GET  /api/student/{id}/homework                      │
│                                                          │
│  Background Service:                                     │
│  • src/api/services/homework_service.py                 │
│    - Calls Service 2 via HTTP                           │
│    - INFERENCE_API_URL environment variable             │
└──────────────────────────────────────────────────────────┘
                          │
                          │ HTTP POST to /generate
                          │ {"cefr_level": "A2", ...}
                          ↓
┌──────────────────────────────────────────────────────────┐
│  SERVICE 2: GPU Inference API (Google Colab)             │
│  File: src/api/inference/colab_api.py                    │
│  Port: 8001 (exposed via ngrok tunnel)                   │
│                                                          │
│  Inference Endpoints:                                    │
│  • GET  /         - Health check                        │
│  • GET  /health   - GPU status                          │
│  • POST /generate - Generate exercises                   │
│                                                          │
│  Model Stack:                                            │
│  • vLLM inference engine (4.4x speedup)                 │
│  • Fine-tuned Italian exercise generator                │
│  • GPU: NVIDIA L4/T4/V100/A100                          │
│  • Response: JSON array of exercises                     │
└──────────────────────────────────────────────────────────┘
```

### Request Flow

1. **Teacher creates assignment** → Service 1 (local API)
2. **Background task starts** → Service 1 calls Service 2
3. **GPU generates exercises** → Service 2 (Colab) processes request
4. **Exercises saved to DB** → Service 1 stores results
5. **Student retrieves homework** → Service 1 serves from DB

## Complete Workflow Examples

### Option A: Using CLI (Recommended)

The CLI provides a simpler, more user-friendly interface:

**1. Create Students**
```bash
python -m src.cli.teacher_cli student create --name "Mario Rossi" --email "mario@example.com"
python -m src.cli.teacher_cli student create --name "Giulia Bianchi" --email "giulia@example.com"
```

**2. Create Assignment**
```bash
python -m src.cli.teacher_cli assignment create \
  --student-ids 1,2 \
  --level A2 \
  --topic "history of Milan" \
  --grammar "past_tense" \
  --quantity 5
```

**3. Check Status**
```bash
python -m src.cli.teacher_cli assignment status --id 1
```

**4. Student Views Homework**
```bash
python -m src.cli.student_cli homework list --student-id 1
python -m src.cli.student_cli homework view --student-id 1 --homework-id 1
```

### Option B: Using HTTP API

Direct API calls for programmatic access:

**1. Create Students**

```bash
curl -X POST "http://localhost:8000/api/teacher/students" \
  -H "Content-Type: application/json" \
  -d '{"name": "Mario Rossi", "email": "mario@example.com"}'

curl -X POST "http://localhost:8000/api/teacher/students" \
  -H "Content-Type: application/json" \
  -d '{"name": "Giulia Bianchi", "email": "giulia@example.com"}'
```

### 2. Create Assignment

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

Assignment is created with `status: "pending"`. Exercise generation happens in the background (~90 seconds for 5 exercises on GPU).

### 3. Check Assignment Status

```bash
curl "http://localhost:8000/api/teacher/assignments/1"
```

Wait until `status` changes from `"pending"` to `"completed"`.

**Status meanings:**
- `pending` - Assignment created, generation not started
- `generating` - Currently generating exercises
- `completed` - All homework generated successfully
- `failed` - Generation failed (check `error_message`)

### 4. Student Retrieves Homework

```bash
curl "http://localhost:8000/api/student/1/homework?status=available"
```

Response includes all exercises with questions, correct answers, and explanations.

## Configuration

### Environment Variables (Service 1)

```bash
# REQUIRED - URL of Service 2 (Colab GPU inference)
export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"

# Optional
export PORT=8000                # API port (default: 8000)
export LOG_LEVEL=info          # Logging level
```

### Service Configuration Files

**Service 1 (Local API):**
- Main application: `src/api/main.py`
- Homework service: `src/api/services/homework_service.py`
- Database models: `src/api/database.py`
- Configuration: Environment variables

**Service 2 (GPU Inference):**
- Inference API: `src/api/inference/colab_api.py`
- Colab notebook: `demos/colab_inference_api.ipynb`
- Model location: Colab GPU environment
- Configuration: Defined in notebook cells

### Model Configuration (Service 2)

The GPU inference service uses:
- **Model**: TeacherPet_italian_grpo
- **Training Method**: GRPO (Group Relative Policy Optimization) - reinforcement learning
- **Advantages**: Better grammar accuracy, tense consistency, reduced hallucination
- **Inference Engine**: vLLM for 4.4x speed improvement
- **Temperature**: 0.4 (optimal for structured output)
- **Max Tokens**: 2500 per request
- **Location**: `models/TeacherPet_italian_grpo/` (loaded in Colab GPU memory)

## Performance

### Generation Speed
- **3 exercises**: ~10-15 seconds (180-250 tokens)
- **5 exercises**: ~90-100 seconds (400-600 tokens)

### Quality Metrics
- **Quality Score**: 100/100
- **Token Generation**: ~1670 tokens for 5 exercises
- **Success Rate**: 100% (all exercises complete)

### Optimization
- **vLLM**: 4.4x speed improvement over standard transformers
- **Temperature**: 0.4 (optimal for structured output)
- **Max Tokens**: 2500 (enough for 5 complete exercises)

## Testing

### Using Python

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# Create student
response = requests.post(
    f"{BASE_URL}/api/teacher/students",
    json={"name": "Test Student", "email": "test@example.com"}
)
student = response.json()
print(f"Created student {student['id']}")

# Create assignment
response = requests.post(
    f"{BASE_URL}/api/teacher/assignments",
    json={
        "cefr_level": "B1",
        "grammar_focus": "subjunctive",
        "topic": "Italian culture",
        "quantity": 3,
        "exercise_types": ["fill_in_blank", "translation"],
        "student_ids": [student['id']]
    }
)
assignment = response.json()
print(f"Created assignment {assignment['id']}")

# Wait for generation
print("Waiting for generation...")
time.sleep(3)

# Check status
response = requests.get(f"{BASE_URL}/api/teacher/assignments/{assignment['id']}")
print(f"Status: {response.json()['status']}")

# Get homework
response = requests.get(
    f"{BASE_URL}/api/student/{student['id']}/homework",
    params={"status": "available"}
)
homework = response.json()
print(f"Found {homework['total']} homework assignments")
```

### Using Browser

Visit http://localhost:8000/docs for interactive testing with Swagger UI.

## Database

All data is stored in SQLite:

```bash
# Location
data/italian_teacher.db

# View database
sqlite3 data/italian_teacher.db

# View tables
.tables

# Query students
SELECT * FROM students;

# Query assignments
SELECT * FROM assignments;

# Exit
.quit

# Reset database
rm data/italian_teacher.db
```

## Troubleshooting

### Service 1 (Local API) Issues

**Port 8000 Already in Use**
```bash
# Find and kill process
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn src.api.main:app --reload --port 8001
```

**INFERENCE_API_URL Not Set**
```bash
# Error: "INFERENCE_API_URL environment variable is required"
# Solution: Start Service 2 (Colab) first and set the URL
export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"
```

**Database Locked**
```bash
# Stop all running servers
# Delete and restart
rm data/italian_teacher.db
./run_api.sh
```

### Service 2 (Colab GPU) Issues

**Colab Service Not Running**
- Check the Colab notebook is still active
- Colab free tier disconnects after ~12 hours
- Re-run all cells in the notebook to restart

**Connection Timeout to Colab**
```bash
# Test if Service 2 is reachable
curl https://your-ngrok-url.ngrok-free.dev/health

# If timeout:
# 1. Check Colab notebook is running
# 2. Verify ngrok URL hasn't changed
# 3. Restart the ngrok cell in Colab
```

**ngrok URL Changed**
```bash
# ngrok URLs change each time you restart the Colab cell
# Get new URL from Colab output, then:
export INFERENCE_API_URL="https://NEW-URL.ngrok-free.dev"
# Restart Service 1
./run_api.sh
```

**GPU Out of Memory**
```bash
# In Colab, restart runtime:
# Runtime → Restart runtime
# Then re-run all cells
```

### Exercise Generation Issues

**Generation Fails with HTTP 500**
- Check Service 2 logs in Colab for errors
- Model may not be loaded correctly
- Try restarting Colab runtime

**Low Quality Exercises**
- Check Colab output for parsing strategy
- Should be "strategy1_direct_array" for best quality
- Verify temperature is 0.4 (optimal)

**Generation Takes Too Long (>2 minutes)**
- Normal for 5 exercises: ~90-100 seconds
- Check GPU type in Colab (L4 is fastest)
- Ensure vLLM is being used (not regular transformers)

## Cost Analysis

### Free Tier
- **Colab**: 12-15 hours/week GPU time (FREE)
- **ngrok**: Unlimited bandwidth, 1 tunnel (FREE)
- **Total**: $0/month

### Paid Tier (Optional)
- **Colab Pro**: $10/month (longer sessions, better GPUs)
- **ngrok Pro**: $8/month (static URLs, 3 tunnels)
- **Total**: $18/month for production use

The free tier is perfect for development and low-volume usage (<100 requests/day).

## Service Files Reference

### Service 1 (Local API) Files
| File | Purpose |
|------|---------|
| `src/api/main.py` | FastAPI application entry point |
| `src/api/routes/teacher.py` | Teacher endpoints (students, assignments) |
| `src/api/routes/student.py` | Student endpoints (homework retrieval) |
| `src/api/services/homework_service.py` | Background task for calling Service 2 |
| `src/api/database.py` | SQLite database models and connection |
| `src/api/models.py` | Pydantic request/response models |
| `run_api.sh` | Startup script for local API |

### Service 2 (GPU Inference) Files
| File | Purpose |
|------|---------|
| `src/api/inference/colab_api.py` | FastAPI inference application |
| `demos/colab_inference_api.ipynb` | Colab notebook to run Service 2 |

### How They Connect

```python
# In src/api/services/homework_service.py:
async def generate_exercises(...):
    # Service 1 calls Service 2 via HTTP
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{INFERENCE_API_URL}/generate",  # Service 2 endpoint
            json=request_payload
        ) as response:
            data = await response.json()
            return data['exercises']
```

## Additional Documentation

- **[CLI Guide](../../docs/CLI_GUIDE.md)** - Complete command-line interface documentation
- **[API Demo Guide](../../demos/API_DEMO_GUIDE.md)** - Comprehensive HTTP API usage examples
- **[Colab GPU Setup](../../docs/COLAB_GPU_SETUP.md)** - Complete Colab setup guide (Service 2)
- **[Main README](../../README.md)** - Project overview

## Support

### Service Health Checks
- **Service 1 (Local)**: http://localhost:8000/health
- **Service 2 (Colab)**: https://your-ngrok-url.ngrok-free.dev/health

### Documentation
- **Service 1 Interactive Docs**: http://localhost:8000/docs
- **Issues**: Open an issue on GitHub
