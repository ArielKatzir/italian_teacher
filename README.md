# Italian Teacher AI - Marco Language Tutor (Still in development stages)

An AI-powered Italian language teaching system focused on conversational learning and personalized education. Currently implementing a specialized fine-tuned model (Marco) for Italian language instruction.

## 🎯 Current Status: Phase 3 - Product Development (Ready for Production Focus!)

- ✅ **Marco v3 Model**: Successfully fine-tuned Minerva-7B with Italian teaching specialization
- ✅ **vLLM Optimization**: Achieved 4.4x speed improvement (4.21s → 0.95s) and 3.7x throughput increase
- ✅ **Model Merging Infrastructure**: Complete CLI tool for PEFT/LoRA adapter merging
- ✅ **Performance Optimization**: FlashAttention, KV caching, and continuous batching implemented
- 🎯 **Next**: Comprehensive Teaching Assistant Platform development

## 🧑‍🏫 Marco Agent

**Marco** is our encouraging Italian teacher agent that provides:
- Patient, supportive language instruction
- Grammar explanations with cultural context
- Personalized conversation practice
- Question generation by CEFR level and topic

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- GPU recommended for inference optimization (Colab Pro with L4/A100)
- vLLM for production-grade inference (4.4x speed improvement)

### Model Inference Setup

```bash
# Clone the repository
git clone <repository-url>
cd italian_teacher

# Create and activate virtual environment
python -m venv ~/.venvs/py312
source ~/.venvs/py312/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install vllm  # For optimized inference

# Model merging (if needed)
python src/fine_tuning/merge_models.py --base minerva --peft marco_v3
```

### Running the System

**Important**: Always activate your virtual environment first!
```bash
source ~/.venvs/py312/bin/activate  # Activate venv

# Quick start - Chat with Marco (Complete CLI)
python cli/simple_chat.py          # Full Marco agent with LLM integration

# Optimized inference with vLLM (4.4x faster!)
python src/inference/vllm_optimization_demo.ipynb  # Performance benchmarking

# Model management
python src/fine_tuning/merge_models.py --list-models  # List available models
python src/fine_tuning/merge_models.py --base minerva --peft marco_v3  # Merge models

# Development commands
make test             # Run all tests (160+ tests pass!)
make format           # Format code
make lint             # Run linting
```

### 🚀 **Colab Pro Usage (Recommended)**

Perfect for development with GPU acceleration:
```bash
# In Google Colab terminal
cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher

# Use optimized inference
pip install vllm
python cli/simple_chat.py  # Automatic GPU model selection with vLLM
```

See performance benchmarking notebook: `src/inference/vllm_optimization_demo.ipynb`

## 📁 Project Structure

```
italian_teacher/
├── src/
│   ├── italian_teacher/          # Main package
│   │   ├── agents/               # Individual agent implementations
│   │   ├── core/                 # Core framework (BaseAgent, Coordinator)
│   │   ├── data/                 # Data processing utilities
│   │   └── utils/                # Shared utilities
│   ├── fine_tuning/              # LoRA training pipeline
│   │   ├── lora_trainer.py       # Training implementation
│   │   ├── merge_models.py       # Model merging CLI tool
│   │   ├── config.py             # Training configuration
│   │   └── data_preprocessing.py # Data preparation
│   └── inference/                # Optimized inference
│       └── vllm_optimization_demo.ipynb # Performance benchmarking
├── models/                       # Trained models and adapters
│   ├── marco/v3/                # v3 LoRA adapter (production ready)
│   └── minerva_marco_v3_merged/ # Merged model for vLLM inference
├── data/                         # Training datasets
├── tests/                        # Test suites
├── configs/                      # Configuration files
├── docs/                         # Documentation
└── ROADMAP.md                    # Development roadmap
```

## 🛠️ Development

### Setup Development Environment

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

### Configuration

Edit `configs/development.yaml` to customize:
- Agent personalities and behaviors
- Model parameters and device settings
- Database and Redis connections
- Training hyperparameters

## 🧪 Testing

```bash
# Run all tests (160+ tests pass!)
make test

# Run specific test categories
pytest tests/unit/test_marco_agent.py    # Marco agent tests
pytest tests/unit/test_models.py         # Language model tests
pytest tests/unit/                       # All unit tests
pytest tests/integration/                # Integration tests
```

**Latest Status**: ✅ All 160 tests passing with complete Marco agent and model integration!

## 🚀 Performance & Optimization

The system achieves production-ready performance through advanced optimizations:

### vLLM Integration Results ✅
- **4.4x Speed Improvement**: 4.21s → 0.95s response time
- **3.7x Throughput Increase**: 23.8 → 88.2 tokens/second
- **Memory Efficient**: FlashAttention + KV caching
- **Production Ready**: Continuous batching for concurrent requests

### Model Training & Merging
1. **LoRA Fine-tuning**: Specialized Marco v3 model on Minerva-7B base
2. **Model Merging**: CLI tool for PEFT adapter integration
3. **Inference Optimization**: vLLM deployment for production speeds
4. **CEFR Conditioning**: Level-appropriate response generation (A1-C2)

See `ROADMAP.md` for complete development phases and `src/inference/vllm_optimization_demo.ipynb` for benchmarking.

## 🏗️ Architecture

### Multi-Agent Communication
- Agents communicate through a message-passing system
- Coordinator manages conversation flow and context switching
- Real-time collaboration allows seamless agent transitions

### Learning Features
- Progress tracking and personalized difficulty adjustment
- Conversation memory across sessions
- Cultural context integration
- Pronunciation feedback (with audio extensions)

## 📋 Roadmap

**Current Phase**: Phase 3 - Product Development
**Next Phase**: Phase 4 - Advanced Model Improvements (CEFR v4 training)
**After That**: Phase 5 - Market Validation & User Testing

See [ROADMAP.md](./roadmap.md) for the complete development plan with reorganized priorities focusing on product development first.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite and linters
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For questions and issues, please check the documentation or open an issue on GitHub.
