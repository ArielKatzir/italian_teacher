"""
Agent configuration utilities for loading and managing agent personalities and settings.

This module provides utilities for loading agent configurations from files,
validating them, and creating AgentPersonality objects.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from .base_agent import AgentPersonality


class AgentConfigError(Exception):
    """Raised when there are issues with agent configuration."""


class AgentConfigLoader:
    """Loads and validates agent configurations from various sources."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.

        Args:
            config_dir: Directory containing agent configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to configs directory in project root
            # From src/core/agent_config.py -> italian_teacher/configs/agents
            self.config_dir = Path(__file__).parent.parent.parent / "configs" / "agents"

        self.config_cache: Dict[str, AgentPersonality] = {}

    def load_agent_personality(self, agent_name: str) -> AgentPersonality:
        """
        Load personality configuration for a specific agent.

        Args:
            agent_name: Name of the agent (marco, professoressa_rossi, etc.)

        Returns:
            AgentPersonality object configured for the agent

        Raises:
            AgentConfigError: If configuration cannot be loaded or is invalid
        """
        # Only allow predefined agents
        if agent_name not in self._get_default_agent_names():
            raise AgentConfigError(
                f"Unknown agent: {agent_name}. Available agents: {self._get_default_agent_names()}"
            )

        # Check cache first
        if agent_name in self.config_cache:
            return self.config_cache[agent_name]

        # Load from file (required)
        config_file = self._find_config_file(agent_name)
        if not config_file:
            raise AgentConfigError(f"Config file not found for {agent_name}")

        try:
            config_data = self._load_config_file(config_file)
            personality = self._validate_and_create_personality(config_data)

            # Cache the result
            self.config_cache[agent_name] = personality
            return personality

        except Exception as e:
            raise AgentConfigError(f"Failed to load config for {agent_name}: {e}")

    def _find_config_file(self, agent_name: str) -> Optional[Path]:
        """Find configuration file for an agent."""
        possible_extensions = [".yaml", ".yml", ".json"]

        for ext in possible_extensions:
            config_file = self.config_dir / f"{agent_name}{ext}"
            if config_file.exists():
                return config_file

        return None

    def _load_config_file(self, config_file: Path) -> Dict[str, Any]:
        """Load configuration from a file."""
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                if config_file.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                elif config_file.suffix == ".json":
                    return json.load(f)
                else:
                    raise AgentConfigError(f"Unsupported config format: {config_file.suffix}")
        except Exception as e:
            raise AgentConfigError(f"Error reading {config_file}: {e}")

    def _validate_and_create_personality(self, config_data: Dict[str, Any]) -> AgentPersonality:
        """Validate configuration data and create AgentPersonality object."""
        try:
            # Convert YAML structure to AgentPersonality format
            converted_data = self._convert_yaml_to_personality(config_data)

            # Pydantic handles all validation, defaults, and type checking
            return AgentPersonality.model_validate(converted_data)
        except ValidationError as e:
            # Convert Pydantic validation errors to our custom error type
            error_details = []
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                message = error["msg"]
                error_details.append(f"{field}: {message}")

            raise AgentConfigError(f"Invalid configuration: {'; '.join(error_details)}")

    def _convert_yaml_to_personality(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert YAML config structure to AgentPersonality format."""
        converted = config_data.copy()

        # Handle topic_focus conversion (nested dict to flat list)
        if "topic_focus" in config_data and isinstance(config_data["topic_focus"], dict):
            topics = []
            topic_focus = config_data["topic_focus"]

            # Add primary topics
            if "primary" in topic_focus:
                topics.extend(topic_focus["primary"])

            # Add secondary topics
            if "secondary" in topic_focus:
                topics.extend(topic_focus["secondary"])

            converted["topic_focus"] = topics

        # Extract simple values from nested structures
        # Handle response_patterns (keep as dict, AgentPersonality expects this)
        if "response_patterns" in config_data:
            converted["response_patterns"] = config_data["response_patterns"]

        # Handle cultural_knowledge (convert to dict if needed)
        if "cultural_knowledge" in config_data:
            cultural = config_data["cultural_knowledge"]
            if isinstance(cultural, dict):
                # Flatten cultural knowledge into a single dict
                flattened_cultural = {}
                for key, value in cultural.items():
                    if isinstance(value, list):
                        flattened_cultural[key] = value
                    else:
                        flattened_cultural[key] = str(value)
                converted["cultural_knowledge"] = flattened_cultural

        # Remove fields that AgentPersonality doesn't recognize
        fields_to_remove = ["conversation_style", "teaching_approach", "behavioral_settings"]
        for field in fields_to_remove:
            converted.pop(field, None)

        return converted

    def save_agent_personality(self, agent_name: str, personality: AgentPersonality) -> None:
        """
        Save an agent personality configuration to a file.

        Args:
            agent_name: Name of the agent
            personality: AgentPersonality object to save
        """
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        config_file = self.config_dir / f"{agent_name}.yaml"
        config_data = personality.model_dump()

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)

            # Update cache
            self.config_cache[agent_name] = personality

        except Exception as e:
            raise AgentConfigError(f"Failed to save config for {agent_name}: {e}")

    def list_available_agents(self) -> List[str]:
        """List all available agent configurations."""
        if not self.config_dir.exists():
            return list(self._get_default_agent_names())

        agents = set()

        # Add agents from config files
        for config_file in self.config_dir.glob("*.yaml"):
            agents.add(config_file.stem)
        for config_file in self.config_dir.glob("*.yml"):
            agents.add(config_file.stem)
        for config_file in self.config_dir.glob("*.json"):
            agents.add(config_file.stem)

        # Add default agents
        agents.update(self._get_default_agent_names())

        return sorted(list(agents))

    def _get_default_agent_names(self) -> List[str]:
        """Get list of default agent names."""
        return ["marco", "professoressa_rossi", "nonna_giulia", "lorenzo"]

    def reload_config(self, agent_name: Optional[str] = None) -> None:
        """
        Reload configuration from files.

        Args:
            agent_name: Specific agent to reload, or None to reload all
        """
        if agent_name:
            self.config_cache.pop(agent_name, None)
        else:
            self.config_cache.clear()


# Convenience functions


def load_agent_personality(
    agent_name: str, config_dir: Optional[Union[str, Path]] = None
) -> AgentPersonality:
    """
    Convenience function to load an agent personality.

    Args:
        agent_name: Name of the agent
        config_dir: Optional config directory

    Returns:
        AgentPersonality object
    """
    loader = AgentConfigLoader(config_dir)
    return loader.load_agent_personality(agent_name)


def create_agent_config_template(agent_name: str, output_file: Union[str, Path]) -> None:
    """
    Create a template configuration file for an agent.

    Args:
        agent_name: Name of the agent
        output_file: Path where to save the template
    """
    template = {
        "name": agent_name.replace("_", " ").title(),
        "role": "Italian Teacher",
        "speaking_style": "neutral",
        "personality_traits": ["helpful", "patient", "encouraging"],
        "expertise_areas": ["italian_language", "conversation"],
        "response_patterns": {
            "greeting": "Ciao! Come posso aiutarti oggi?",
            "encouragement": "Molto bene! Continua così!",
            "correction": "Non è esatto. La forma corretta è: {correction}",
        },
        "cultural_knowledge": {
            "regions": ["Lombardia", "Toscana", "Sicilia"],
            "traditions": ["pasta", "famiglia", "arte"],
        },
        "correction_style": "gentle",
        "enthusiasm_level": 5,
        "formality_level": 5,
        "correction_frequency": 5,
        "topic_focus": ["general_conversation"],
        "patience_level": 5,
        "encouragement_frequency": 5,
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(template, f, default_flow_style=False, allow_unicode=True)
