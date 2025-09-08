"""
Unit tests for configuration handling.
"""

import tempfile
from pathlib import Path

import pytest
import yaml


class TestConfig:
    """Test configuration loading and validation."""

    def test_sample_config_structure(self, sample_config):
        """Test that sample config has required structure."""
        assert "app" in sample_config
        assert "models" in sample_config
        assert "agents" in sample_config
        assert "coordinator" in sample_config

        # Test app config
        assert sample_config["app"]["name"] == "Italian Teacher Test"
        assert sample_config["app"]["version"] == "0.1.0"

        # Test all agents are present
        expected_agents = ["marco", "professoressa_rossi", "nonna_giulia", "lorenzo"]
        for agent in expected_agents:
            assert agent in sample_config["agents"]

    def test_config_loading_from_file(self, temp_dir, sample_config):
        """Test loading configuration from YAML file."""
        config_file = temp_dir / "test_config.yaml"

        # Write config to file
        with open(config_file, "w") as f:
            yaml.dump(sample_config, f)

        # Read it back
        with open(config_file, "r") as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config == sample_config

    def test_config_agent_personalities(self, sample_config):
        """Test agent personality configurations."""
        agents = sample_config["agents"]

        # Test Marco (friendly conversationalist)
        marco = agents["marco"]
        assert marco["personality"] == "friendly_conversationalist"
        assert 0 <= marco["encouragement_frequency"] <= 1

        # Test Professoressa Rossi (grammar expert)
        rossi = agents["professoressa_rossi"]
        assert rossi["personality"] == "grammar_expert"
        assert 0 <= rossi["correction_threshold"] <= 1

        # Test Nonna Giulia (cultural storyteller)
        giulia = agents["nonna_giulia"]
        assert giulia["personality"] == "cultural_storyteller"
        assert 0 <= giulia["story_frequency"] <= 1

        # Test Lorenzo (modern Italian)
        lorenzo = agents["lorenzo"]
        assert lorenzo["personality"] == "modern_italian"
        assert 0 <= lorenzo["slang_probability"] <= 1

    @pytest.mark.parametrize(
        "agent_name", ["marco", "professoressa_rossi", "nonna_giulia", "lorenzo"]
    )
    def test_all_agents_have_personality(self, sample_config, agent_name):
        """Test that all agents have personality defined."""
        agents = sample_config["agents"]
        assert "personality" in agents[agent_name]
        assert isinstance(agents[agent_name]["personality"], str)
        assert len(agents[agent_name]["personality"]) > 0
