# Italian Teacher AI - Marco Language Tutor

An AI-powered Italian language teaching system focused on conversational learning and personalized education. Currently implementing a specialized fine-tuned model (Marco) for Italian language instruction.

## 🎯 Current Status: Phase 2.2 - LoRA Fine-tuning

- ✅ **Dataset Complete**: 10,130 high-quality Italian teaching examples with LLM-enhanced grammar explanations
- ✅ **Data Quality**: 92%+ success rate with Qwen2.5-3B grammar improvements
- 🔄 **In Progress**: LoRA fine-tuning infrastructure for Qwen2.5-7B-Instruct
- 🎯 **Next**: Specialized Marco teaching model deployment

## 🧑‍🏫 Marco Agent

**Marco** is our encouraging Italian teacher agent that provides:
- Patient, supportive language instruction
- Grammar explanations with cultural context
- Personalized conversation practice
- Question generation by CEFR level and topic

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- GPU recommended for fine-tuning (Colab Pro with T4/A100)

### Current Development Setup

```bash
# Clone the repository
git clone <repository-url>
cd italian_teacher

# Create and activate virtual environment
python -m venv ~/.venvs/py312
source ~/.venvs/py312/bin/activate

# Install dependencies
pip install -r requirements.txt
make install-dev
# Or manually: pip install -e ".[dev,training,audio]"
```

### Running the System

**Important**: Always activate your virtual environment first!
```bash
source ~/.venvs/py312/bin/activate  # Activate venv

# Quick start - Chat with Marco (Complete CLI)
python cli/simple_chat.py          # Full Marco agent with LLM integration

# Development commands
make test             # Run all tests (160+ tests pass!)
make format           # Format code
make lint             # Run linting

# Advanced usage
python scripts/train_agents.py --config configs/development.yaml
```

### 🚀 **Colab Pro Usage (Recommended)**

Perfect for development with free GPU access:
```bash
# In Google Colab terminal
cd /content/drive/MyDrive/Colab\ Notebooks/italian_teacher
python cli/simple_chat.py  # Automatic GPU model selection
```

See [Colab Pro Setup Guide](docs/COLAB_PRO_SETUP.md) for complete instructions.

## 📁 Project Structure

```
italian_teacher/
├── src/
│   ├── italian_teacher/          # Main package
│   │   ├── agents/               # Individual agent implementations
│   │   ├── core/                 # Core framework (BaseAgent, Coordinator)
│   │   ├── data/                 # Data processing utilities
│   │   └── utils/                # Shared utilities
│   └── fine_tuning/              # LoRA training pipeline
│       ├── lora_trainer.py       # Training implementation
│       ├── config.py             # Training configuration
│       ├── inference.py          # Model inference utilities
│       └── data_preprocessing.py # Data preparation
├── models/                       # Trained LoRA models
│   ├── marco_lora_v1/           # v1 model (deprecated - poor quality)
│   └── marco_lora_v2/           # v2 model (planned - high quality)
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

## 📊 Training

The system uses LoRA (Low-Rank Adaptation) fine-tuning to create specialized agent personalities:

1. **Data Collection**: Gather Italian conversation data, social media, literature
2. **Preprocessing**: Clean and format data for each agent personality
3. **Training**: Fine-tune base models with agent-specific LoRA adapters
4. **Evaluation**: Test conversation quality and learning effectiveness

See `italian-teacher-roadmap.md` for detailed training procedures.

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

See [italian-teacher-roadmap.md](./docs/italian-teacher-roadmap.md) for the complete 32-week development plan.

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
