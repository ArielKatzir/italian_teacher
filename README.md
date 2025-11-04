# Italian Teacher AI

An AI-powered Italian language teaching system with a REST API for personalized homework generation. The system uses a fine-tuned Italian exercise generation model running on Google Colab GPU via ngrok tunnel for high-quality exercise generation.

## 🎯 Current Status: Phase 4 - GRPO Model Training Complete ✅

- ✅ **Teacher/Student API**: FastAPI backend with SQLite database for homework management
- ✅ **GRPO Fine-tuned Model**: TeacherPet_italian_grpo trained with reinforcement learning
- ✅ **Colab GPU Integration**: Remote inference via ngrok tunnel (4.4x faster with vLLM)
- ✅ **Exercise Generation**: High-quality, grammatically accurate exercises
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
- [src/api/README.md](src/api/README.md) - API documentation and usage
- [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) - Command-line interface guide
- [demos/API_DEMO_GUIDE.md](demos/API_DEMO_GUIDE.md) - API demo examples

## 🧑‍🏫 Features

The Italian Teacher AI provides:
- **Personalized Homework**: Generate exercises by CEFR level (A1-C2), grammar focus, and topic
- **Multiple Exercise Types**: Fill-in-blank, translation, multiple choice
- **RL-Optimized Generation**: GRPO training for improved grammatical accuracy and coherence
- **Quality Validation**: Comprehensive validation with grammar checking
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
│   └── TeacherPet_italian_grpo/       # GRPO fine-tuned exercise generation model
│
├── QUICKSTART.md                      # Quick start guide
├── test_exercise_quality.py           # Quality test script
└── run_api.sh                         # API startup script
```

## 🎯 API & CLI Overview

### Three Ways to Use the System

**1. Command-Line Interface (CLI)** - Easiest for quick testing
```bash
# Teacher CLI - Create students and assignments
python -m src.cli.teacher_cli student create --name "Mario" --email "mario@example.com"
python -m src.cli.teacher_cli assignment create --student-ids 1 --level A2 --topic "food"

# Student CLI - View homework
python -m src.cli.student_cli homework list --student-id 1
```

**2. REST API** - For programmatic access
```bash
curl -X POST http://localhost:8000/api/teacher/students \
  -H "Content-Type: application/json" \
  -d '{"name": "Mario", "email": "mario@example.com"}'
```

**3. Interactive API Docs** - For exploration and testing
- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Documentation

- **[CLI Guide](docs/CLI_GUIDE.md)** - Complete command-line interface documentation
- **[API README](src/api/README.md)** - Full API documentation with endpoints and examples
- **[API Demo Guide](demos/API_DEMO_GUIDE.md)** - HTTP API usage examples

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
│   ├─ Fine-tuned Model│
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

The system uses TeacherPet_italian_grpo, a reinforcement learning optimized model:
- **Training Method**: GRPO (Group Relative Policy Optimization) for improved quality
- **Base**: Fine-tuned on Italian teaching dataset with reward-based optimization
- **Location**: `models/TeacherPet_italian_grpo/`
- **Inference Engine**: vLLM for optimal performance
- **Key Advantages**: Better grammar accuracy, tense consistency, and reduced hallucination

## 🐛 Troubleshooting

### Common Issues

**"Model not found"**
- Verify model exists at `models/TeacherPet_italian_grpo/`
- Update `MODEL_PATH` in Colab notebook to point to the correct model directory

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

**Phase 4 (Current)**: ✅ GRPO Model Training Complete
- [x] FastAPI Backend with SQLite
- [x] Teacher/Student endpoints
- [x] Colab GPU integration via ngrok
- [x] GRPO reinforcement learning training
- [x] TeacherPet_italian_grpo model deployed
- [ ] Frontend UI (Next)

**Phase 5**: Advanced Features & UI
- [ ] Frontend UI for teachers and students
- [ ] Student progress analytics dashboard

**Phase 6**: Platform Features
- [ ] Batch exercise generation
- [ ] Exercise caching for performance
- [ ] More exercise types (audio, images)
- [ ] Student progress tracking

**Phase 7**: Production Deployment
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
- **[src/api/README.md](src/api/README.md)** - Complete API documentation
- **[docs/COLAB_GPU_SETUP.md](docs/COLAB_GPU_SETUP.md)** - Complete Colab setup guide
- **[docs/development/COLAB_GPU_INTEGRATION.md](docs/development/COLAB_GPU_INTEGRATION.md)** - Architecture deep dive
- **[demos/API_DEMO_GUIDE.md](demos/API_DEMO_GUIDE.md)** - API demo examples
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

- **vLLM**: High-performance inference engine
- **Google Colab**: Free GPU access for development
- **FastAPI**: Modern web framework for API development
- **ngrok**: Secure tunneling service
- **Hugging Face**: Model hosting and training infrastructure

---

Built with ❤️ for Italian language learners
