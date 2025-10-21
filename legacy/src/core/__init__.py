"""
Core components for the Italian Teacher multi-agent system.

This module provides the base classes and utilities needed to create
and manage AI agents for Italian language learning.
"""

from .agent_config import (
    AgentConfigError,
    AgentConfigLoader,
    create_agent_config_template,
    load_agent_personality,
)
from .agent_discovery import (
    AgentDiscoveryService,
    default_discovery_service,
)
from .agent_registry import (
    AgentCapabilities,
    AgentMatch,
    AgentRegistration,
    AgentSpecialization,
    InMemoryAgentRegistry,
    SelectionCriteria,
    default_agent_registry,
)
from .base_agent import (
    AgentActivityEvent,
    AgentMessage,
    AgentPersonality,
    AgentStatus,
    BaseAgent,
    ConversationContext,
    MessageType,
)
from .coordinator import CoordinatorAgent  # Alias for backward compatibility
from .coordinator import (
    ConversationPhase,
    CoordinatorService,
    LearningGoal,
    SessionState,
)
from .event_bus import (
    AgentEventBus,
)
from .retention_policy import (
    DeletionStage,
    RetentionPolicy,
    RetentionPolicyManager,
    RetentionPreference,
    default_retention_manager,
)

__all__ = [
    # Base Agent Framework
    "BaseAgent",
    "AgentMessage",
    "AgentPersonality",
    "AgentActivityEvent",
    "ConversationContext",
    "MessageType",
    "AgentStatus",
    # Coordinator Service
    "CoordinatorService",
    "CoordinatorAgent",  # Backward compatibility alias
    "ConversationPhase",
    "LearningGoal",
    "SessionState",
    # Agent Registry & Discovery
    "AgentCapabilities",
    "AgentMatch",
    "AgentRegistration",
    "AgentSpecialization",
    "InMemoryAgentRegistry",
    "SelectionCriteria",
    "default_agent_registry",
    "AgentDiscoveryService",
    "default_discovery_service",
    # Event System
    "AgentEventBus",
    # Configuration
    "AgentConfigLoader",
    "AgentConfigError",
    "load_agent_personality",
    "create_agent_config_template",
    # Data Retention
    "RetentionPreference",
    "RetentionPolicy",
    "RetentionPolicyManager",
    "DeletionStage",
    "default_retention_manager",
]
