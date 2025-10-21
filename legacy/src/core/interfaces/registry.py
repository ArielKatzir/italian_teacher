"""
Agent Registry Interface

Defines the contract for agent registration and discovery systems.
This interface enables different registry implementations (in-memory, Redis, database)
while maintaining consistent API for agent management.
"""

from abc import ABC, abstractmethod

# Import types using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..agent_registry import AgentAvailability, AgentMatch, AgentRegistration, SelectionCriteria


class AgentRegistry(ABC):
    """
    Abstract interface for agent registration and discovery.

    This interface defines the contract that all agent registry implementations
    must satisfy. It supports:
    - Agent lifecycle management (register/unregister)
    - Health monitoring (heartbeats, status updates)
    - Agent discovery with intelligent scoring
    - Stale agent cleanup

    Implementations:
    - InMemoryAgentRegistry: Simple in-memory storage
    - RedisAgentRegistry: Distributed Redis-based registry (future)
    - DatabaseAgentRegistry: Persistent database storage (future)
    """

    @abstractmethod
    async def register_agent(self, registration: "AgentRegistration") -> bool:
        """
        Register an agent with the registry.

        Args:
            registration: Complete agent registration information

        Returns:
            True if registration successful, False otherwise
        """

    @abstractmethod
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the registry.

        Args:
            agent_id: Unique identifier of agent to remove

        Returns:
            True if agent was found and removed, False otherwise
        """

    @abstractmethod
    async def update_agent_status(
        self,
        agent_id: str,
        availability: Optional["AgentAvailability"] = None,
        session_count: Optional[int] = None,
    ) -> bool:
        """
        Update an agent's status information.

        Args:
            agent_id: Agent to update
            availability: New availability status
            session_count: Current number of active sessions

        Returns:
            True if update successful, False if agent not found
        """

    @abstractmethod
    async def heartbeat(self, agent_id: str) -> bool:
        """
        Record a heartbeat from an agent to indicate it's still alive.

        Args:
            agent_id: Agent sending heartbeat

        Returns:
            True if heartbeat was recorded
        """

    @abstractmethod
    async def find_agents(
        self, criteria: "SelectionCriteria", max_results: Optional[int] = None
    ) -> List["AgentMatch"]:
        """
        Find agents matching the given criteria, ranked by suitability.

        Args:
            criteria: Selection criteria including specializations,
                     confidence requirements, and availability needs
            max_results: Maximum number of agents to return

        Returns:
            List of matching agents sorted by descending match score
        """

    @abstractmethod
    async def get_agent(self, agent_id: str) -> Optional["AgentRegistration"]:
        """
        Get specific agent registration by ID.

        Args:
            agent_id: Unique agent identifier

        Returns:
            Agent registration if found, None otherwise
        """

    @abstractmethod
    async def list_all_agents(self) -> List["AgentRegistration"]:
        """
        List all registered agents.

        Returns:
            List of all agent registrations
        """

    @abstractmethod
    async def cleanup_stale_agents(self, timeout: Optional[int] = None) -> int:
        """
        Remove agents that haven't sent heartbeats recently.

        Args:
            timeout: Seconds since last heartbeat to consider stale.
                    Uses registry configuration if not specified.

        Returns:
            Number of agents removed
        """
