"""
Italian Teacher Multi-Agent Framework

A sophisticated multi-agent AI system for personalized Italian language learning.
"""

__version__ = "0.1.0"
__author__ = "Italian Teacher Team"

from .core.agent_config import load_agent_personality
from .core.base_agent import AgentMessage, AgentPersonality, BaseAgent

__all__ = [
    "BaseAgent",
    "AgentMessage",
    "AgentPersonality",
    "load_agent_personality",
]
