# Italian Teacher Multi-Agent Framework - (Still in development)

A sophisticated multi-agent AI system for personalized Italian language learning, featuring distinct AI personalities that collaborate to provide immersive, contextual language education.

## 🎭 Meet the Agents

- **Marco** - Friendly conversationalist who encourages speaking practice
- **Professoressa Rossi** - Grammar expert who corrects mistakes and explains rules
- **Nonna Giulia** - Cultural storyteller sharing idioms, traditions, and regional expressions
- **Lorenzo** - Young Italian introducing slang, modern expressions, and pop culture
- **Coordinator** - Manages conversations, tracks progress, and orchestrates agent interactions

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- GPU recommended for training (CPU works for inference)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd italian_teacher

# Create and activate virtual environment
python -m venv ~/.venvs/py312  # or: python -m venv venv
source ~/.venvs/py312/bin/activate  # or: source venv/bin/activate

# Install dependencies
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
├── src/italian_teacher/          # Main package
│   ├── agents/                   # Individual agent implementations
│   ├── core/                     # Core framework (BaseAgent, Coordinator)
│   ├── data/                     # Data processing utilities
│   ├── models/                   # ML model handling (LoRA adapters)
│   └── utils/                    # Shared utilities
├── tests/                        # Test suites
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
└── italian-teacher-roadmap.md    # Development roadmap
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
