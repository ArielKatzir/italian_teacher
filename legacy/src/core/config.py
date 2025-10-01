"""
Configuration management system for the Italian Teacher.

This module provides centralized configuration management for all system
components, replacing hardcoded values with configurable parameters.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class AgentDefaults:
    """Default values for agent capabilities."""

    max_concurrent_sessions: int = 5
    current_session_count: int = 0
    average_response_time: float = 1.0  # seconds
    success_rate: float = 0.95
    user_satisfaction: float = 0.85


@dataclass
class AgentLimits:
    """Agent operational limits."""

    max_errors: int = 5
    recent_messages_count: int = 10
    default_priority: int = 1


@dataclass
class AgentPersonalityDefaults:
    """Default personality trait values."""

    enthusiasm_level: int = 8
    formality_level: int = 3
    correction_frequency: int = 6
    patience_level: int = 8
    encouragement_frequency: int = 8


@dataclass
class RegistryConfig:
    """Agent registry configuration."""

    heartbeat_timeout: int = 300  # seconds
    cleanup_timeout: int = 300  # seconds
    max_results: int = 10


@dataclass
class ScoringWeights:
    """Agent scoring algorithm weights."""

    specialization: float = 0.4
    confidence: float = 0.3
    load: float = 0.2
    performance: float = 0.1


@dataclass
class ConfidenceThresholds:
    """Minimum confidence thresholds by user level."""

    beginner: float = 0.6
    intermediate: float = 0.7
    advanced: float = 0.8
    default: float = 0.7


@dataclass
class ComplexityThresholds:
    """Confidence thresholds by conversation complexity."""

    simple: float = 0.6
    medium: float = 0.7
    complex: float = 0.8
    default: float = 0.7


@dataclass
class DiscoveryConfig:
    """Agent discovery service configuration."""

    max_help_candidates: int = 3
    max_handoff_candidates: int = 3
    max_correction_candidates: int = 2
    confidence_thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    complexity_thresholds: ComplexityThresholds = field(default_factory=ComplexityThresholds)


@dataclass
class EventBusConfig:
    """Event bus configuration."""

    delivery_timeout: float = 5.0  # seconds


@dataclass
class RetentionConfig:
    """Data retention configuration."""

    immediate_cleanup_days: int = 0
    quick_cleanup_days: int = 1
    conversation_cleanup_days: int = 7
    monthly_cleanup_days: int = 30
    quarterly_cleanup_days: int = 90
    semi_annual_cleanup_days: int = 180
    annual_cleanup_days: int = 365
    long_term_cleanup_days: int = 1095  # 3 years
    final_cleanup_days: int = 2555  # 7 years


@dataclass
class SystemConfig:
    """Complete system configuration."""

    agent_defaults: AgentDefaults = field(default_factory=AgentDefaults)
    agent_limits: AgentLimits = field(default_factory=AgentLimits)
    personality_defaults: AgentPersonalityDefaults = field(default_factory=AgentPersonalityDefaults)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)


class ConfigManager:
    """Configuration manager with file loading and environment override support."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to YAML config file. If None, uses default locations.
        """
        self.config = SystemConfig()
        self._config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file and environment variables."""
        # Try to load from file
        config_data = self._load_from_file()
        if config_data:
            self._apply_config_data(config_data)

        # Apply environment overrides
        self._apply_env_overrides()

    def _load_from_file(self) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file."""
        config_paths = []

        if self._config_path:
            config_paths.append(Path(self._config_path))

        # Default config file locations (prioritize configs/ directory)
        config_paths.extend(
            [
                Path("configs/system.yaml"),  # Proper location for system config
                Path("configs/config.yaml"),  # Alternative in configs/
                Path("config.yaml"),  # Legacy fallback in root
                Path.home() / ".italian_teacher" / "system.yaml",  # User-specific system config
                Path.home() / ".italian_teacher" / "config.yaml",  # Legacy user config
            ]
        )

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        return yaml.safe_load(f)
                except Exception as e:
                    print(f"Warning: Failed to load config from {config_path}: {e}")
                    continue

        return None

    def _apply_config_data(self, config_data: Dict[str, Any]) -> None:
        """Apply configuration data from file."""
        # Apply system-level configurations if present
        system_config = config_data.get("system", {})

        if "agent_defaults" in system_config:
            self._update_dataclass(self.config.agent_defaults, system_config["agent_defaults"])

        if "agent_limits" in system_config:
            self._update_dataclass(self.config.agent_limits, system_config["agent_limits"])

        if "personality_defaults" in system_config:
            self._update_dataclass(
                self.config.personality_defaults, system_config["personality_defaults"]
            )

        if "registry" in system_config:
            self._update_dataclass(self.config.registry, system_config["registry"])

        if "scoring" in system_config:
            self._update_dataclass(self.config.scoring, system_config["scoring"])

        if "discovery" in system_config:
            discovery_config = system_config["discovery"]
            self._update_dataclass(self.config.discovery, discovery_config)

            # Ensure nested dataclasses are properly initialized
            if "confidence_thresholds" in discovery_config:
                if not hasattr(self.config.discovery, "confidence_thresholds"):
                    self.config.discovery.confidence_thresholds = ConfidenceThresholds()
                self._update_dataclass(
                    self.config.discovery.confidence_thresholds,
                    discovery_config["confidence_thresholds"],
                )

            if "complexity_thresholds" in discovery_config:
                if not hasattr(self.config.discovery, "complexity_thresholds"):
                    self.config.discovery.complexity_thresholds = ComplexityThresholds()
                self._update_dataclass(
                    self.config.discovery.complexity_thresholds,
                    discovery_config["complexity_thresholds"],
                )

        if "event_bus" in system_config:
            self._update_dataclass(self.config.event_bus, system_config["event_bus"])

        if "retention" in system_config:
            self._update_dataclass(self.config.retention, system_config["retention"])

    def _update_dataclass(self, instance: Any, data: Dict[str, Any]) -> None:
        """Update dataclass instance with dictionary data."""
        for key, value in data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Agent defaults
        self._env_override(
            "ITALIAN_TEACHER_MAX_CONCURRENT_SESSIONS", "agent_defaults.max_concurrent_sessions", int
        )
        self._env_override(
            "ITALIAN_TEACHER_AGENT_RESPONSE_TIME", "agent_defaults.average_response_time", float
        )
        self._env_override("ITALIAN_TEACHER_SUCCESS_RATE", "agent_defaults.success_rate", float)

        # Registry config
        self._env_override("ITALIAN_TEACHER_HEARTBEAT_TIMEOUT", "registry.heartbeat_timeout", int)
        self._env_override("ITALIAN_TEACHER_CLEANUP_TIMEOUT", "registry.cleanup_timeout", int)

        # Event bus
        self._env_override("ITALIAN_TEACHER_EVENT_TIMEOUT", "event_bus.delivery_timeout", float)

        # Discovery
        self._env_override("ITALIAN_TEACHER_MAX_CANDIDATES", "discovery.max_help_candidates", int)

    def _env_override(self, env_var: str, config_path: str, type_func) -> None:
        """Apply environment variable override to config path."""
        value = os.getenv(env_var)
        if value is not None:
            try:
                # Navigate to nested attribute
                parts = config_path.split(".")
                obj = self.config
                for part in parts[:-1]:
                    obj = getattr(obj, part)

                setattr(obj, parts[-1], type_func(value))
            except (ValueError, AttributeError) as e:
                print(f"Warning: Invalid environment override {env_var}={value}: {e}")

    def get_agent_defaults(self) -> AgentDefaults:
        """Get agent default configuration."""
        return self.config.agent_defaults

    def get_agent_limits(self) -> AgentLimits:
        """Get agent limits configuration."""
        return self.config.agent_limits

    def get_personality_defaults(self) -> AgentPersonalityDefaults:
        """Get personality defaults configuration."""
        return self.config.personality_defaults

    def get_registry_config(self) -> RegistryConfig:
        """Get registry configuration."""
        return self.config.registry

    def get_scoring_weights(self) -> ScoringWeights:
        """Get scoring weights configuration."""
        return self.config.scoring

    def get_discovery_config(self) -> DiscoveryConfig:
        """Get discovery configuration."""
        return self.config.discovery

    def get_event_bus_config(self) -> EventBusConfig:
        """Get event bus configuration."""
        return self.config.event_bus

    def get_retention_config(self) -> RetentionConfig:
        """Get retention configuration."""
        return self.config.retention


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_path: Optional[str] = None) -> ConfigManager:
    """Initialize the global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_path)
    return _config_manager
