"""
Italian Teacher Core Interfaces

This module defines all the abstract interfaces that implementations
must satisfy. Use these for type hints and dependency injection.

Available Interfaces:
- AgentRegistry: Agent registration and discovery
- ConversationStateStore: Conversation persistence
- BaseAgent: Core agent behavior
- EventHandler: Event processing

Example Usage:
    from core.interfaces import AgentRegistry

    class MyService:
        def __init__(self, registry: AgentRegistry):
            self.registry = registry
"""

from .events import EventHandler
from .registry import AgentRegistry
from .storage import ConversationStateStore

__all__ = ["AgentRegistry", "ConversationStateStore", "EventHandler"]
