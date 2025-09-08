"""
Italian Teacher Multi-Agent Framework

A sophisticated multi-agent AI system for personalized Italian language learning.
"""

__version__ = "0.1.0"
__author__ = "Italian Teacher Team"

from .core.agent import BaseAgent
from .core.coordinator import Coordinator
from .core.conversation import ConversationManager

__all__ = [
    "BaseAgent",
    "Coordinator", 
    "ConversationManager",
]