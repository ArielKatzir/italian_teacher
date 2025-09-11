"""
Simple structured logging for agents.
"""

import logging
import sys

import structlog


def get_agent_logger(agent_id: str, agent_name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a logger with agent context automatically bound.

    Args:
        agent_id: Unique agent identifier
        agent_name: Agent display name

    Returns:
        Logger with agent context
    """
    return structlog.get_logger("agent").bind(agent_id=agent_id, agent_name=agent_name)


# Configure structured logging on import
logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
