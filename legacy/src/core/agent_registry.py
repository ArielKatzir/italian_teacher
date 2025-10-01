"""
Agent registration and discovery system implementation.

This module provides the infrastructure for agents to register their capabilities
and for the system to discover and select the most appropriate agent for requests.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from .config import get_config
from .interfaces.registry import AgentRegistry


class AgentSpecialization(Enum):
    """Types of specializations agents can have."""

    CONVERSATION = "conversation"
    GRAMMAR = "grammar"
    PRONUNCIATION = "pronunciation"
    CULTURAL_CONTEXT = "cultural_context"
    CORRECTIONS = "corrections"
    ENCOURAGEMENT = "encouragement"
    STORYTELLING = "storytelling"
    MODERN_SLANG = "modern_slang"
    FORMAL_LANGUAGE = "formal_language"
    CASUAL_LANGUAGE = "casual_language"


class AgentAvailability(Enum):
    """Agent availability states."""

    AVAILABLE = "available"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    UNAVAILABLE = "unavailable"


@dataclass
class AgentCapabilities:
    """Describes an agent's capabilities and current status."""

    # Core capabilities
    specializations: Set[AgentSpecialization] = field(default_factory=set)
    confidence_scores: Dict[AgentSpecialization, float] = field(default_factory=dict)

    # Resource limits (defaults will be applied at runtime)
    max_concurrent_sessions: int = 5  # Config override available
    current_session_count: int = 0

    # Performance metrics (defaults will be applied at runtime)
    average_response_time: float = 1.0  # Config override available
    success_rate: float = 0.95  # Config override available
    user_satisfaction: float = 0.85  # Config override available

    # Availability tracking
    availability: AgentAvailability = AgentAvailability.AVAILABLE
    last_heartbeat: datetime = field(default_factory=datetime.now)

    def get_load_factor(self) -> float:
        """Calculate current load as a percentage (0.0 = idle, 1.0 = at capacity)."""
        if self.max_concurrent_sessions == 0:
            return 0.0
        return min(1.0, self.current_session_count / self.max_concurrent_sessions)

    def is_available_for_new_session(self) -> bool:
        """Check if agent can accept a new session."""
        from datetime import datetime, timedelta

        # Check if heartbeat is stale (older than 5 minutes)
        heartbeat_timeout = timedelta(minutes=5)
        is_heartbeat_recent = (datetime.now() - self.last_heartbeat) < heartbeat_timeout

        return (
            self.availability == AgentAvailability.AVAILABLE
            and self.current_session_count < self.max_concurrent_sessions
            and is_heartbeat_recent
        )

    def get_confidence_for_specialization(self, specialization: AgentSpecialization) -> float:
        """Get confidence score for a specific specialization."""
        return self.confidence_scores.get(specialization, 0.0)


@dataclass
class AgentRegistration:
    """Complete agent registration information."""

    agent_id: str
    agent_name: str
    agent_type: str
    capabilities: AgentCapabilities
    registered_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    # Optional connection details for remote agents
    host: Optional[str] = None
    port: Optional[int] = None
    endpoint: Optional[str] = None

    def is_healthy(self, heartbeat_timeout: Optional[int] = None) -> bool:
        """Check if agent is considered healthy (recent heartbeat)."""
        if heartbeat_timeout is None:
            try:
                heartbeat_timeout = get_config().get_registry_config().heartbeat_timeout
            except (AttributeError, TypeError):
                heartbeat_timeout = 300  # Fallback
        return (datetime.now() - self.capabilities.last_heartbeat).seconds < heartbeat_timeout

    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""
        self.capabilities.last_heartbeat = datetime.now()
        self.last_updated = datetime.now()


class SelectionCriteria(BaseModel):
    """Criteria for selecting agents from the registry."""

    # Agent requirements
    required_specializations: Set[AgentSpecialization] = Field(default_factory=set)
    preferred_specializations: Set[AgentSpecialization] = Field(default_factory=set)
    minimum_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_load_factor: float = Field(default=0.8, ge=0.0, le=1.0)
    require_availability: bool = True

    # Scoring weights (Config override available)
    specialization_weight: float = 0.4
    confidence_weight: float = 0.3
    load_weight: float = 0.2
    performance_weight: float = 0.1


@dataclass
class AgentMatch:
    """Represents a matched agent with scoring details."""

    registration: AgentRegistration
    total_score: float
    specialization_score: float
    confidence_score: float
    load_score: float
    performance_score: float

    def __post_init__(self):
        """Ensure score is between 0.0 and 1.0."""
        self.total_score = max(0.0, min(1.0, self.total_score))


class InMemoryAgentRegistry(AgentRegistry):
    """In-memory implementation of agent registry."""

    def __init__(self):
        self._agents: Dict[str, AgentRegistration] = {}
        self._agents_by_type: Dict[str, List[str]] = {}  # agent_type -> [agent_ids]

    async def register_agent(self, registration: AgentRegistration) -> bool:
        """Register a new agent."""
        try:
            agent_id = registration.agent_id
            agent_type = registration.agent_type

            # Store registration
            self._agents[agent_id] = registration

            # Update type index
            if agent_type not in self._agents_by_type:
                self._agents_by_type[agent_type] = []

            if agent_id not in self._agents_by_type[agent_type]:
                self._agents_by_type[agent_type].append(agent_id)

            return True

        except Exception:
            return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Remove agent from registry."""
        if agent_id not in self._agents:
            return False

        try:
            registration = self._agents[agent_id]
            agent_type = registration.agent_type

            # Remove from main registry
            del self._agents[agent_id]

            # Update type index
            if agent_type in self._agents_by_type:
                if agent_id in self._agents_by_type[agent_type]:
                    self._agents_by_type[agent_type].remove(agent_id)

                # Clean up empty type lists
                if not self._agents_by_type[agent_type]:
                    del self._agents_by_type[agent_type]

            return True

        except Exception:
            return False

    async def update_agent_status(
        self,
        agent_id: str,
        availability: Optional[AgentAvailability] = None,
        session_count: Optional[int] = None,
    ) -> bool:
        """Update agent status."""
        if agent_id not in self._agents:
            return False

        try:
            registration = self._agents[agent_id]

            if availability is not None:
                registration.capabilities.availability = availability

            if session_count is not None:
                registration.capabilities.current_session_count = session_count

            registration.last_updated = datetime.now()
            return True

        except Exception:
            return False

    async def heartbeat(self, agent_id: str) -> bool:
        """Record agent heartbeat."""
        if agent_id not in self._agents:
            return False

        try:
            self._agents[agent_id].update_heartbeat()
            return True
        except Exception:
            return False

    async def find_agents(
        self, criteria: SelectionCriteria, max_results: Optional[int] = None
    ) -> List[AgentMatch]:
        """Find matching agents with scoring."""
        if max_results is None:
            try:
                max_results = get_config().get_registry_config().max_results
            except (AttributeError, TypeError):
                max_results = 5  # Fallback
        matches = []

        for registration in self._agents.values():
            # Skip unhealthy agents
            if not registration.is_healthy():
                continue

            # Skip unavailable agents if required
            if (
                criteria.require_availability
                and not registration.capabilities.is_available_for_new_session()
            ):
                continue

            # Score the agent
            match = self._score_agent_match(registration, criteria)

            # Apply minimum confidence filter
            if match.confidence_score >= criteria.minimum_confidence:
                matches.append(match)

        # Sort by total score (descending) and return top matches
        matches.sort(key=lambda m: m.total_score, reverse=True)
        return matches[:max_results]

    def _score_agent_match(
        self, registration: AgentRegistration, criteria: SelectionCriteria
    ) -> AgentMatch:
        """Calculate match score for an agent."""
        capabilities = registration.capabilities

        # Specialization score (now includes confidence weighting)
        specialization_score = self._calculate_specialization_score(
            capabilities, criteria.required_specializations, criteria.preferred_specializations
        )

        # Load score (lower load = higher score)
        load_score = 1.0 - capabilities.get_load_factor()

        # Performance score (combination of success rate, response time, satisfaction)
        performance_score = (
            capabilities.success_rate * 0.4
            + min(1.0, 2.0 / max(0.1, capabilities.average_response_time)) * 0.3
            + capabilities.user_satisfaction * 0.3
        )

        # Weighted total score (redistributing confidence weight to specialization)
        combined_specialization_weight = criteria.specialization_weight + criteria.confidence_weight
        total_score = (
            specialization_score * combined_specialization_weight
            + load_score * criteria.load_weight
            + performance_score * criteria.performance_weight
        )

        # Calculate separate confidence score for filtering
        all_specializations = criteria.required_specializations.union(
            criteria.preferred_specializations
        )
        if all_specializations:
            agent_relevant_specs = all_specializations.intersection(capabilities.specializations)
            if agent_relevant_specs:
                confidence_score = sum(
                    capabilities.get_confidence_for_specialization(spec)
                    for spec in agent_relevant_specs
                ) / len(agent_relevant_specs)
            else:
                confidence_score = 0.0
        else:
            confidence_score = 1.0  # No specific requirements

        return AgentMatch(
            registration=registration,
            total_score=total_score,
            specialization_score=specialization_score,
            confidence_score=confidence_score,
            load_score=load_score,
            performance_score=performance_score,
        )

    def _calculate_specialization_score(
        self,
        capabilities: AgentCapabilities,
        required: Set[AgentSpecialization],
        preferred: Set[AgentSpecialization],
    ) -> float:
        """Calculate specialization match score weighted by confidence."""
        agent_specializations = capabilities.specializations

        # Must have all required specializations
        if required and not required.issubset(agent_specializations):
            return 0.0

        # Score based on confidence-weighted overlap with preferred specializations
        if not preferred:
            # If no preferred specs, score based on required specs confidence
            if not required:
                return 1.0
            required_confidence = sum(
                capabilities.get_confidence_for_specialization(spec)
                for spec in required
                if spec in agent_specializations
            )
            return required_confidence / len(required)

        # Calculate confidence-weighted score for preferred specializations
        matching_preferred = preferred.intersection(agent_specializations)
        if not matching_preferred:
            return 0.0

        confidence_sum = sum(
            capabilities.get_confidence_for_specialization(spec) for spec in matching_preferred
        )
        return confidence_sum / len(preferred)

    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get specific agent registration."""
        return self._agents.get(agent_id)

    async def list_all_agents(self) -> List[AgentRegistration]:
        """List all registered agents."""
        return list(self._agents.values())

    async def cleanup_stale_agents(self, timeout: Optional[int] = None) -> int:
        """Remove stale agents."""
        if timeout is None:
            try:
                timeout = get_config().get_registry_config().cleanup_timeout
            except (AttributeError, TypeError):
                timeout = 300  # Fallback
        stale_agents = []
        cutoff_time = datetime.now() - timedelta(seconds=timeout)

        for agent_id, registration in self._agents.items():
            if registration.capabilities.last_heartbeat < cutoff_time:
                stale_agents.append(agent_id)

        # Remove stale agents
        removed_count = 0
        for agent_id in stale_agents:
            if await self.unregister_agent(agent_id):
                removed_count += 1

        return removed_count


# Default registry instance
default_agent_registry = InMemoryAgentRegistry()
