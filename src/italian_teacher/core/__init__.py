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
from .base_agent import (
    AgentActivityEvent,
    AgentMessage,
    AgentPersonality,
    AgentStatus,
    BaseAgent,
    ConversationContext,
    MessageType,
)
from .retention_policy import (
    DeletionStage,
    RetentionPolicy,
    RetentionPolicyManager,
    RetentionPreference,
    default_retention_manager,
)

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentPersonality",
    "AgentActivityEvent",
    "ConversationContext",
    "MessageType",
    "AgentStatus",
    "AgentConfigLoader",
    "AgentConfigError",
    "load_agent_personality",
    "create_agent_config_template",
    "RetentionPreference",
    "RetentionPolicy",
    "RetentionPolicyManager",
    "DeletionStage",
    "default_retention_manager",
]
