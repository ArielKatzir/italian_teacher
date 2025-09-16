"""
Model configuration utilities for loading and managing model settings.

This module provides utilities for loading model configurations from files,
validating them, and creating ModelConfig objects.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import ValidationError

from .base_model import ModelConfig


class ModelConfigError(Exception):
    """Raised when there are issues with model configuration."""


class ModelConfigLoader:
    """Loads and validates model configurations from various sources."""

    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the model configuration loader.

        Args:
            config_dir: Directory containing model configuration files
        """
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to configs/models directory in project root
            # From src/models/model_config.py -> italian_teacher/configs/models
            self.config_dir = Path(__file__).parent.parent.parent / "configs" / "models"

        self.config_cache: Dict[str, ModelConfig] = {}

    def load_model_config(self, model_name: str) -> ModelConfig:
        """
        Load configuration for a specific model.

        Args:
            model_name: Name of the model (llama3.1-8b, mistral-7b, etc.)

        Returns:
            ModelConfig object configured for the model

        Raises:
            ModelConfigError: If configuration cannot be loaded or is invalid
        """
        # Check cache first
        if model_name in self.config_cache:
            return self.config_cache[model_name]

        # Load from file
        config_file = self._find_config_file(model_name)
        if not config_file:
            raise ModelConfigError(f"Config file not found for model: {model_name}")

        try:
            config_data = self._load_config_file(config_file)
            model_config = self._validate_and_create_config(config_data)

            # Cache the result
            self.config_cache[model_name] = model_config
            return model_config

        except Exception as e:
            raise ModelConfigError(f"Failed to load config for {model_name}: {e}")

    def _find_config_file(self, model_name: str) -> Optional[Path]:
        """Find configuration file for a model."""
        possible_extensions = [".yaml", ".yml", ".json"]

        for ext in possible_extensions:
            config_file = self.config_dir / f"{model_name}{ext}"
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
                    raise ModelConfigError(f"Unsupported config format: {config_file.suffix}")
        except Exception as e:
            raise ModelConfigError(f"Error reading {config_file}: {e}")

    def _validate_and_create_config(self, config_data: Dict[str, Any]) -> ModelConfig:
        """Validate configuration data and create ModelConfig object."""
        try:
            # Remove non-ModelConfig fields that are just metadata
            cleaned_data = {
                k: v
                for k, v in config_data.items()
                if k
                not in [
                    "description",
                    "recommended_for",
                    "memory_requirements",
                    "performance_notes",
                ]
            }

            # Pydantic handles all validation, defaults, and type checking
            return ModelConfig.model_validate(cleaned_data)
        except ValidationError as e:
            # Convert Pydantic validation errors to our custom error type
            error_details = []
            for error in e.errors():
                field = ".".join(str(x) for x in error["loc"])
                message = error["msg"]
                error_details.append(f"{field}: {message}")

            raise ModelConfigError(f"Invalid configuration: {'; '.join(error_details)}")

    def save_model_config(self, model_name: str, config: ModelConfig) -> None:
        """
        Save a model configuration to a file.

        Args:
            model_name: Name of the model
            config: ModelConfig object to save
        """
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)

        config_file = self.config_dir / f"{model_name}.yaml"
        config_data = config.model_dump()

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, allow_unicode=True)

            # Update cache
            self.config_cache[model_name] = config

        except Exception as e:
            raise ModelConfigError(f"Failed to save config for {model_name}: {e}")

    def list_available_models(self) -> List[str]:
        """List all available model configurations."""
        if not self.config_dir.exists():
            return []

        models = set()

        # Add models from config files
        for config_file in self.config_dir.glob("*.yaml"):
            models.add(config_file.stem)
        for config_file in self.config_dir.glob("*.yml"):
            models.add(config_file.stem)
        for config_file in self.config_dir.glob("*.json"):
            models.add(config_file.stem)

        return sorted(list(models))

    def get_model_metadata(self, model_name: str) -> Dict[str, Any]:
        """
        Get metadata for a model (description, requirements, etc.).

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with metadata fields
        """
        config_file = self._find_config_file(model_name)
        if not config_file:
            return {}

        try:
            config_data = self._load_config_file(config_file)
            return {
                "description": config_data.get("description", "No description available"),
                "recommended_for": config_data.get("recommended_for", []),
                "memory_requirements": config_data.get("memory_requirements", "Unknown"),
                "performance_notes": config_data.get("performance_notes", "No notes available"),
            }
        except Exception:
            return {}

    def reload_config(self, model_name: Optional[str] = None) -> None:
        """
        Reload configuration from files.

        Args:
            model_name: Specific model to reload, or None to reload all
        """
        if model_name:
            self.config_cache.pop(model_name, None)
        else:
            self.config_cache.clear()


# Convenience functions


def load_model_config(
    model_name: str, config_dir: Optional[Union[str, Path]] = None
) -> ModelConfig:
    """
    Convenience function to load a model configuration.

    Args:
        model_name: Name of the model
        config_dir: Optional config directory

    Returns:
        ModelConfig object
    """
    loader = ModelConfigLoader(config_dir)
    return loader.load_model_config(model_name)


def get_available_models(config_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Convenience function to get available model configurations.

    Args:
        config_dir: Optional config directory

    Returns:
        List of available model names
    """
    loader = ModelConfigLoader(config_dir)
    return loader.list_available_models()
