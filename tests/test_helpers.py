"""
Test helper functions and utilities.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml


def create_test_config(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a test configuration with optional overrides."""
    base_config = {
        "app": {
            "name": "Italian Teacher Test",
            "version": "0.1.0",
            "debug": True,
        },
        "models": {
            "base_model": "microsoft/DialoGPT-small",
            "device": "cpu",
            "max_length": 32,
            "temperature": 0.8,
        },
        "agents": {
            "marco": {"personality": "friendly_conversationalist"},
            "professoressa_rossi": {"personality": "grammar_expert"},
        },
    }

    if overrides:
        base_config.update(overrides)

    return base_config


def create_temp_config_file(config: Dict[str, Any]) -> Path:
    """Create a temporary configuration file."""
    temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_file)
    temp_file.flush()
    return Path(temp_file.name)


def assert_italian_text(text: str) -> None:
    """Assert that text appears to be Italian."""
    assert isinstance(text, str)
    assert len(text) > 0

    # Basic checks for Italian characteristics
    italian_words = [
        "ciao",
        "come",
        "che",
        "molto",
        "bene",
        "grazie",
        "per",
        "con",
        "del",
        "la",
        "il",
    ]
    italian_patterns = ["zione", "zion", "ghi", "gli", "sci", "gn"]

    text_lower = text.lower()
    has_italian_word = any(word in text_lower for word in italian_words)
    has_italian_pattern = any(pattern in text_lower for pattern in italian_patterns)

    # Should have either common Italian words or patterns
    assert has_italian_word or has_italian_pattern, f"Text doesn't appear to be Italian: {text}"


def create_sample_conversation(length: int = 3) -> List[Dict[str, Any]]:
    """Create a sample conversation for testing."""
    base_messages = [
        {"role": "user", "content": "Ciao! Come stai?", "timestamp": "2024-01-01T10:00:00Z"},
        {
            "role": "marco",
            "content": "Ciao! Sto bene, grazie! E tu?",
            "timestamp": "2024-01-01T10:00:05Z",
        },
        {"role": "user", "content": "Sto bene anch'io!", "timestamp": "2024-01-01T10:00:10Z"},
        {
            "role": "professoressa_rossi",
            "content": "Perfetto! La vostra conversazione è molto naturale.",
            "timestamp": "2024-01-01T10:00:15Z",
        },
        {"role": "user", "content": "Grazie professoressa!", "timestamp": "2024-01-01T10:00:20Z"},
    ]

    return base_messages[:length]


def create_mock_response(agent_type: str = "marco") -> str:
    """Create a mock response based on agent type."""
    responses = {
        "marco": "Ciao! Sono Marco, molto piacere di conoscerti!",
        "professoressa_rossi": "Buongiorno. Iniziamo con la grammatica italiana.",
        "nonna_giulia": "Ciao caro! Ti racconto una storia della mia gioventù...",
        "lorenzo": "Ehi! Come va? Hai sentito l'ultima canzone di Måneskin?",
    }

    return responses.get(agent_type, "Ciao! Come posso aiutarti?")


def validate_agent_response(response: str, agent_type: str) -> bool:
    """Validate that a response matches expected agent personality."""
    response_lower = response.lower()

    personality_indicators = {
        "marco": ["ciao", "piacere", "bene", "grazie"],
        "professoressa_rossi": ["buongiorno", "grammatica", "corretto", "sbagliato"],
        "nonna_giulia": ["caro", "storia", "tradizione", "famiglia"],
        "lorenzo": ["ehi", "cool", "figata", "vai"],
    }

    indicators = personality_indicators.get(agent_type, [])
    return any(indicator in response_lower for indicator in indicators)


class MockAgent:
    """A simple mock agent for testing."""

    def __init__(self, agent_id: str, personality: str):
        self.agent_id = agent_id
        self.personality = personality
        self._available = True

    async def generate_response(self, input_text: str) -> str:
        """Generate a mock response."""
        return create_mock_response(self.agent_id)

    def is_available(self) -> bool:
        """Check if agent is available."""
        return self._available

    def set_availability(self, available: bool):
        """Set agent availability."""
        self._available = available


def create_test_agents() -> Dict[str, MockAgent]:
    """Create a set of test agents."""
    return {
        "marco": MockAgent("marco", "friendly_conversationalist"),
        "professoressa_rossi": MockAgent("professoressa_rossi", "grammar_expert"),
        "nonna_giulia": MockAgent("nonna_giulia", "cultural_storyteller"),
        "lorenzo": MockAgent("lorenzo", "modern_italian"),
    }


def assert_valid_timestamp(timestamp: str) -> None:
    """Assert that timestamp is in valid ISO format."""
    assert isinstance(timestamp, str)
    assert "T" in timestamp
    assert timestamp.endswith("Z")

    # Basic format check: YYYY-MM-DDTHH:MM:SSZ
    date_part, time_part = timestamp.split("T")
    assert len(date_part) == 10  # YYYY-MM-DD
    assert len(time_part) == 9  # HH:MM:SSZ
    assert date_part.count("-") == 2
    assert time_part.count(":") == 2
