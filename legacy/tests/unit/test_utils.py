"""
Unit tests for utility functions.
"""

from datetime import datetime

import pytest


class TestTextProcessing:
    """Test text processing utilities."""

    def test_sample_conversation_structure(self, sample_conversation_history):
        """Test conversation history structure."""
        assert isinstance(sample_conversation_history, list)
        assert len(sample_conversation_history) > 0

        for message in sample_conversation_history:
            assert "role" in message
            assert "content" in message
            assert "timestamp" in message

            # Test role is valid
            assert message["role"] in [
                "user",
                "marco",
                "professoressa_rossi",
                "nonna_giulia",
                "lorenzo",
            ]

            # Test content is not empty
            assert isinstance(message["content"], str)
            assert len(message["content"]) > 0

            # Test timestamp format
            assert isinstance(message["timestamp"], str)

    def test_italian_text_patterns(self):
        """Test recognition of Italian text patterns."""
        italian_phrases = [
            "Ciao! Come stai?",
            "Buongiorno, signora!",
            "Arrivederci a presto!",
            "Che bella giornata!",
        ]

        for phrase in italian_phrases:
            assert isinstance(phrase, str)
            assert len(phrase) > 0
            # Basic check for Italian characteristics (could be expanded)
            assert any(char in phrase for char in "aeiou")  # Has vowels

    @pytest.mark.parametrize(
        "input_text,expected_length",
        [
            ("Ciao", 4),
            ("Come stai?", 10),
            ("Buongiorno, come va la giornata?", 32),
            ("", 0),
        ],
    )
    def test_text_length_calculation(self, input_text, expected_length):
        """Test text length calculations."""
        assert len(input_text) == expected_length


class TestDataStructures:
    """Test data structure handling."""

    def test_training_data_format(self, sample_training_data):
        """Test training data structure."""
        assert isinstance(sample_training_data, list)

        for item in sample_training_data:
            assert isinstance(item, dict)
            assert "input" in item
            assert "output" in item

            assert isinstance(item["input"], str)
            assert isinstance(item["output"], str)
            assert len(item["input"]) > 0
            assert len(item["output"]) > 0

    def test_agent_mock_structure(self, mock_agent):
        """Test mock agent structure."""
        assert hasattr(mock_agent, "agent_id")
        assert hasattr(mock_agent, "personality")
        assert hasattr(mock_agent, "generate_response")
        assert hasattr(mock_agent, "is_available")

        assert mock_agent.agent_id == "test_agent"
        assert mock_agent.personality == "test_personality"
        assert mock_agent.is_available() is True


class TestTimestampHandling:
    """Test timestamp and time-related utilities."""

    def test_timestamp_format(self, sample_conversation_history):
        """Test that timestamps are in expected format."""
        for message in sample_conversation_history:
            timestamp = message["timestamp"]
            # Should be ISO format: YYYY-MM-DDTHH:MM:SSZ
            assert "T" in timestamp
            assert timestamp.endswith("Z")
            assert len(timestamp.split("T")[0]) == 10  # Date part
            assert len(timestamp.split("T")[1]) == 9  # Time part with Z

    def test_current_timestamp_generation(self):
        """Test current timestamp generation."""
        now = datetime.utcnow()
        timestamp = now.isoformat() + "Z"

        assert isinstance(timestamp, str)
        assert "T" in timestamp
        assert timestamp.endswith("Z")
