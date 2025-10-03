# Italian Teacher AI - Marco Language Tutor

An AI-powered Italian language teaching system with a REST API for personalized homework generation. The system uses a fine-tuned Marco v3 model running on Google Colab GPU via ngrok tunnel for high-quality exercise generation.

## 🎯 Current Status: Phase 3 - Teacher API & Colab GPU Integration ✅

- ✅ **Teacher/Student API**: FastAPI backend with SQLite database for homework management
- ✅ **Marco v3 Model**: Successfully fine-tuned Minerva-7B with Italian teaching specialization
- ✅ **Colab GPU Integration**: Remote inference via ngrok tunnel (4.4x faster with vLLM)
- ✅ **Exercise Generation**: 100/100 quality score with 5 complete exercises per request
- 🎯 **Next**: Frontend UI and production deployment

## 🚀 Quick Start

### Option 1: Local API Only (Mock Data)

Perfect for testing the API structure without GPU:

```bash
# Start the API
./run_api.sh

# Visit http://localhost:8000/docs for interactive API documentation
```

### Option 2: Local API + Colab GPU (Real Model) ⭐ Recommended

Full setup with GPU-powered exercise generation:

**Step 1: Start Colab Inference Service**
1. Open `demos/colab_inference_api.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: `Runtime` → `Change runtime type` → `GPU`
3. Run all 8 cells (takes ~1 minute)
4. Copy the ngrok URL from the output

**Step 2: Configure & Start Local API**
```bash
# Export the Colab API URL
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
⏱️  Inference Time: 98.50s

✨ No issues found!
```

📚 **Detailed guides**:
- [QUICKSTART.md](QUICKSTART.md) - Complete quick start guide
- [docs/COLAB_GPU_SETUP.md](docs/COLAB_GPU_SETUP.md) - Comprehensive Colab setup
- [demos/API_DEMO_GUIDE.md](demos/API_DEMO_GUIDE.md) - API usage examples

## 🧑‍🏫 What is Marco?

**Marco** is an AI Italian teacher that provides:
- **Personalized Homework**: Generate exercises by CEFR level (A1-C2), grammar focus, and topic
- **Multiple Exercise Types**: Fill-in-blank, translation, multiple choice
- **Quality Validation**: 100/100 quality score with comprehensive validation
- **Fast Inference**: 4.4x speed improvement with vLLM optimization

## 📁 Project Structure

```
italian_teacher/
├── src/
│   └── api/
│       ├── main.py                    # FastAPI application
│       ├── models.py                  # Database models (Student, Homework)
│       ├── database.py                # SQLite database setup
│       ├── services/
│       │   └── homework_service.py    # Exercise generation logic
│       └── inference/
│           └── colab_api.py           # Colab GPU inference API
│
├── demos/
│   ├── colab_inference_api.ipynb      # Colab notebook (8 cells)
│   └── API_DEMO_GUIDE.md              # API usage examples
│
├── docs/
│   ├── COLAB_GPU_SETUP.md             # Complete Colab setup guide
│   └── development/
│       └── COLAB_GPU_INTEGRATION.md   # Architecture documentation
│
├── tests/
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests
│
├── utils/
│   └── exercise_validator.py          # Quality validation
│
├── models/
│   └── minerva_marco_v3_merged/       # Fine-tuned Marco v3 model
│
├── QUICKSTART.md                      # Quick start guide
├── test_exercise_quality.py           # Quality test script
└── run_api.sh                         # API startup script
```

## 🎯 API Features

### Teacher Endpoints

**Create Student**
```bash
POST /teacher/students
{
  "name": "Mario Rossi",
  "cefr_level": "A2"
}
```

**Create Homework Assignment**
```bash
POST /teacher/homework
{
  "student_id": 1,
  "cefr_level": "A2",
  "grammar_focus": "present_tense",
  "topic": "daily routines",
  "quantity": 5,
  "exercise_types": ["fill_in_blank", "translation", "multiple_choice"]
}
```

**List Assignments**
```bash
GET /teacher/homework?student_id=1&status=pending
```

### Student Endpoints

**Get Homework**
```bash
GET /student/{student_id}/homework?status=pending
```

**Submit Exercise**
```bash
POST /student/submit
{
  "student_id": 1,
  "homework_id": 1,
  "exercise_index": 0,
  "answer": "vado"
}
```

Visit http://localhost:8000/docs for full interactive API documentation.

## 🏗️ Architecture

```
┌─────────────────────┐
│   Local Mac         │
│   (Port 8000)       │
│   FastAPI Server    │
│   ├─ Teacher API    │
│   ├─ Student API    │
│   └─ SQLite DB      │
└──────────┬──────────┘
           │ HTTP Request
           ↓
┌──────────────────────┐
│   ngrok Tunnel       │
│   (Public HTTPS)     │
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│   Google Colab       │
│   (Port 8001)        │
│   GPU: NVIDIA L4/T4  │
│   ├─ vLLM Engine     │
│   ├─ Marco v3 Model  │
│   └─ FastAPI Service │
└──────────────────────┘
```

**Key Benefits**:
- 🚀 **4.4x faster inference** with vLLM on GPU
- 💰 **Free GPU** via Google Colab (12-15 hours/week)
- 🔒 **Secure tunnel** via ngrok HTTPS
- 🎯 **100/100 quality** validated exercise generation

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

## 🚀 Performance

### Generation Speed
- **3 exercises**: ~10-15 seconds (180-250 tokens)
- **5 exercises**: ~90-100 seconds (400-600 tokens)

### Quality Metrics
- **Quality Score**: 100/100
- **Parsing Strategy**: strategy1_direct_array (best quality)
- **Token Generation**: 1670 tokens for 5 exercises
- **Success Rate**: 100% (all 5 exercises complete)

### Optimization
- **vLLM**: 4.4x speed improvement over standard transformers
- **Temperature**: 0.4 (optimal for structured output)
- **Max Tokens**: 2500 (enough for 5 complete exercises)

## 🔧 Configuration

### Environment Variables

```bash
# Required for Colab GPU integration
export INFERENCE_API_URL="https://your-ngrok-url.ngrok-free.dev"

# Optional
export PORT=8000                # API port (default: 8000)
export LOG_LEVEL=info          # Logging level
```

### Model Configuration

The system uses Marco v3, a fine-tuned version of Minerva-7B specialized for Italian teaching:
- **Base Model**: Minerva-7B
- **Fine-tuning**: LoRA adapters on Italian teaching dataset
- **Location**: `models/minerva_marco_v3_merged/`
- **Inference Engine**: vLLM for optimal performance

## 🐛 Troubleshooting

### Common Issues

**"Model not found"**
- Verify model exists at `models/minerva_marco_v3_merged/`
- Update `MODEL_PATH` in Colab notebook Cell 3

**"Connection timeout"**
- Check Colab notebook is still running
- Verify ngrok URL is correct
- Restart Cell 7 in Colab if needed

**"Low quality exercises"**
- Check Colab output for parsing strategy
- Should be "strategy1_direct_array" for best quality
- Verify temperature is 0.4 (optimal)

**"Port 8000 already in use"**
```bash
# Find and kill process using port 8000
lsof -i :8000
kill -9 <PID>
```

See [docs/COLAB_GPU_SETUP.md](docs/COLAB_GPU_SETUP.md#troubleshooting) for comprehensive troubleshooting.

## 💰 Cost Analysis

### Free Tier
- **Colab**: 12-15 hours/week GPU time (FREE)
- **ngrok**: Unlimited bandwidth, 1 tunnel (FREE)
- **Total**: $0/month

### Paid Tier (Optional)
- **Colab Pro**: $10/month (longer sessions, better GPUs)
- **ngrok Pro**: $8/month (static URLs, 3 tunnels)
- **Total**: $18/month for production use

The free tier is perfect for development and low-volume usage (<100 requests/day).

## 📋 Roadmap

**Phase 3 (Current)**: ✅ Teacher API & Colab GPU Integration
- [x] FastAPI Backend with SQLite
- [x] Teacher/Student endpoints
- [x] Colab GPU integration via ngrok
- [x] 100/100 quality exercise generation
- [ ] Frontend UI (Next)

**Phase 4**: Advanced Features
- [ ] Batch exercise generation
- [ ] Exercise caching for performance
- [ ] More exercise types (audio, images)
- [ ] Student progress tracking

**Phase 5**: Production Deployment
- [ ] Deploy to AWS/GCP with GPU
- [ ] User authentication
- [ ] Payment integration
- [ ] Mobile app

See [ROADMAP.md](ROADMAP.md) for complete development plan.

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv ~/.venvs/py312
source ~/.venvs/py312/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
make lint
```

## 📖 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 3 minutes
- **[docs/COLAB_GPU_SETUP.md](docs/COLAB_GPU_SETUP.md)** - Complete Colab setup guide (450+ lines)
- **[docs/development/COLAB_GPU_INTEGRATION.md](docs/development/COLAB_GPU_INTEGRATION.md)** - Architecture deep dive
- **[demos/API_DEMO_GUIDE.md](demos/API_DEMO_GUIDE.md)** - API usage examples
- **[ROADMAP.md](ROADMAP.md)** - Development roadmap

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

- **Issues**: Open an issue on GitHub
- **Documentation**: Check [docs/](docs/) folder
- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- **Troubleshooting**: See [docs/COLAB_GPU_SETUP.md](docs/COLAB_GPU_SETUP.md#troubleshooting)

## ✨ Acknowledgments

- **Minerva-7B**: Base model for Italian language understanding
- **vLLM**: High-performance inference engine
- **Google Colab**: Free GPU access for development
- **FastAPI**: Modern web framework for API development
- **ngrok**: Secure tunneling service

---

Built with ❤️ for Italian language learners
