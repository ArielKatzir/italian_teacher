# Italian Teacher API - Quick Start Guide

## 🚀 Get Started in 3 Minutes

### Option 1: Local API Only (Mock Data)

```bash
# Start the API
./run_api.sh

# The API is now running at http://localhost:8000
# It will generate mock exercises (for testing)
```

Visit http://localhost:8000/docs to see the API documentation.

### Option 2: Local API + Colab GPU (Real Model)

**Step 1: Start Colab Inference Service** (one-time setup per session)

1. Open `demos/colab_inference_api.ipynb` in Google Colab
2. Enable GPU: `Runtime` → `Change runtime type` → `GPU`
3. Run all cells (takes ~1 minute)
4. Copy the ngrok URL from the output

**Step 2: Configure Local API**

```bash
# Export the Colab URL
export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"

# Start the local API
./run_api.sh
```

**Step 3: Test**

```bash
# Test exercise generation quality
python test_exercise_quality.py
```

Expected output:
```
✅ Success: True
📊 Quality Score: 100.0/100
🔢 Tokens Generated: 1670

✨ No issues found!
```

## 📚 Documentation

- **[Colab GPU Setup Guide](docs/COLAB_GPU_SETUP.md)** - Detailed setup instructions
- **[API Demo Guide](demos/API_DEMO_GUIDE.md)** - API usage examples
- **[Colab GPU Integration](docs/development/COLAB_GPU_INTEGRATION.md)** - Architecture details

## 🧪 Testing

```bash
# Unit tests
make test-unit

# Teacher API tests
make test-teacher

# Integration tests (requires running API)
make test-teacher-flow

# Exercise quality validation
python test_exercise_quality.py
```

## 🏗️ Project Structure

```
italian_teacher/
├── src/
│   └── api/
│       ├── main.py                    # FastAPI app
│       ├── services/
│       │   └── homework_service.py    # Homework generation logic
│       └── inference/
│           └── colab_api.py           # Colab GPU inference API
├── demos/
│   └── colab_inference_api.ipynb      # Colab notebook (8 cells)
├── docs/
│   └── COLAB_GPU_SETUP.md             # Complete setup guide
├── tests/
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests
├── utils/
│   └── exercise_validator.py          # Quality validation
└── test_exercise_quality.py           # Quick quality test
```

## 🎯 Common Tasks

### Using the CLI (Recommended)

**Teacher creates a student:**
```bash
./teacher student create --name "Mario Rossi" --email "mario@example.com"
```

**Teacher creates homework:**
```bash
./teacher assignment create \
  --student-ids 1 \
  --level A2 \
  --topic "daily routines" \
  --quantity 5
```

**Teacher checks status:**
```bash
./teacher assignment status --id 1
```

**Student views homework:**
```bash
./student homework list --student-id 1
./student homework view --student-id 1 --homework-id 1
```

See [CLI_GUIDE.md](CLI_GUIDE.md) for complete CLI documentation.

### Using the API Directly

**Create a Student:**
```bash
curl -X POST http://localhost:8000/api/teacher/students \
  -H "Content-Type: application/json" \
  -d '{"name": "Mario Rossi", "email": "mario@example.com"}'
```

**Create Assignment:**
```bash
curl -X POST http://localhost:8000/api/teacher/assignments \
  -H "Content-Type: application/json" \
  -d '{
    "student_ids": [1],
    "cefr_level": "A2",
    "grammar_focus": "present_tense",
    "topic": "daily routines",
    "quantity": 5,
    "exercise_types": ["fill_in_blank", "translation", "multiple_choice"]
  }'
```

**Get Student Homework:**
```bash
curl http://localhost:8000/api/student/1/homework?status=available
```

## 🔧 Environment Variables

```bash
# Required for Colab GPU integration
export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"

# Optional
export PORT=8000                # API port (default: 8000)
export LOG_LEVEL=info          # Logging level
```

## ⚡ Performance

- **Mock mode**: Instant (testing only)
- **Colab GPU mode**:
  - 3 exercises: ~10-15 seconds
  - 5 exercises: ~90-100 seconds
  - Quality: 95-100/100 score

## 🐛 Troubleshooting

### Local API won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill the process if needed
kill -9 <PID>
```

### Colab connection timeout
```bash
# Check Colab notebook is running
curl $INFERENCE_API_URL/health

# If no response, restart Cell 7 in Colab
```

### Low quality exercises
```bash
# Check the Colab notebook output for errors
# Look for "strategy4" or "strategy5" (fallback modes)
# Should be "strategy1_direct_array" for best quality
```

## 📖 Next Steps

1. ✅ Complete Quick Start above
2. 📚 Read [Colab GPU Setup Guide](docs/COLAB_GPU_SETUP.md)
3. 🧪 Run `python test_exercise_quality.py`
4. 🎨 Try the API at http://localhost:8000/docs
5. 🚀 Build your application!

## 💡 Tips

- Keep the Colab notebook open while using the API
- ngrok URLs expire after ~2 hours (free tier)
- Use Colab Pro ($10/month) for longer sessions
- Check [ROADMAP.md](ROADMAP.md) for upcoming features

## 🆘 Need Help?

- Check [Troubleshooting](docs/COLAB_GPU_SETUP.md#troubleshooting) section
- Review [Architecture Docs](docs/development/COLAB_GPU_INTEGRATION.md)
- Run tests to verify setup: `make test-unit`
