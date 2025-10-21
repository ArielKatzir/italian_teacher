"""
Shared pytest fixtures and configuration for the Italian Teacher project.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "app": {
            "name": "Italian Teacher Test",
            "version": "0.1.0",
            "debug": True,
            "log_level": "DEBUG",
        },
        "models": {
            "base_model": "microsoft/DialoGPT-small",  # Smaller for tests
            "device": "cpu",
            "max_length": 64,  # Shorter for faster tests
            "temperature": 0.7,
        },
        "agents": {
            "marco": {"personality": "friendly_conversationalist", "encouragement_frequency": 0.3},
            "professoressa_rossi": {"personality": "grammar_expert", "correction_threshold": 0.5},
            "nonna_giulia": {"personality": "cultural_storyteller", "story_frequency": 0.2},
            "lorenzo": {"personality": "modern_italian", "slang_probability": 0.4},
        },
        "coordinator": {
            "max_context_length": 5,  # Shorter for tests
            "agent_switch_threshold": 0.6,
            "difficulty_adjustment_rate": 0.1,
        },
    }


@pytest.fixture
def mock_agent():
    """Mock agent for testing."""
    agent = Mock()
    agent.agent_id = "test_agent"
    agent.personality = "test_personality"
    agent.generate_response = AsyncMock(return_value="Test response")
    agent.is_available = Mock(return_value=True)
    return agent


@pytest.fixture
def sample_conversation_history():
    """Sample conversation history for testing."""
    return [
        {"role": "user", "content": "Ciao! Come stai?", "timestamp": "2024-01-01T10:00:00Z"},
        {
            "role": "marco",
            "content": "Ciao! Sto bene, grazie! E tu?",
            "timestamp": "2024-01-01T10:00:05Z",
        },
        {"role": "user", "content": "Sto bene anch'io!", "timestamp": "2024-01-01T10:00:10Z"},
    ]


@pytest.fixture
def sample_training_data():
    """Sample training data for testing."""
    return [
        {"input": "Come si dice 'hello' in italiano?", "output": "Si dice 'ciao' o 'salve'."},
        {
            "input": "Che cosa significa 'arrivederci'?",
            "output": "Significa 'goodbye' o 'see you later'.",
        },
        {"input": "Come stai?", "output": "Sto bene, grazie! E tu?"},
    ]


@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db = Mock()
    db.execute = AsyncMock()
    db.fetch_all = AsyncMock(return_value=[])
    db.fetch_one = AsyncMock(return_value=None)
    db.close = AsyncMock()
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis connection for testing."""
    redis = Mock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.exists = AsyncMock(return_value=False)
    return redis


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    model = Mock()
    model.generate = Mock(return_value=["Generated text response"])
    model.tokenizer = Mock()
    model.tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
    model.tokenizer.decode = Mock(return_value="Decoded text")
    return model


# Test markers for different test categories
pytest_plugins = []


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "ml: mark test as requiring ML models")
    config.addinivalue_line("markers", "api: mark test as API test")


# Skip ML tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    if not config.getoption("--ml-tests"):
        skip_ml = pytest.mark.skip(reason="ML tests skipped (use --ml-tests to run)")
        for item in items:
            if "ml" in item.keywords:
                item.add_marker(skip_ml)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption("--ml-tests", action="store_true", default=False, help="run ML-related tests")
    parser.addoption("--slow-tests", action="store_true", default=False, help="run slow tests")
