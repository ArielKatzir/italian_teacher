"""
Tests for agent registry and discovery system.
"""

from datetime import datetime, timedelta

import pytest

from core.agent_discovery import AgentDiscoveryService
from core.agent_events import EventType
from core.agent_registry import (
    AgentAvailability,
    AgentCapabilities,
    AgentRegistration,
    AgentSpecialization,
    InMemoryAgentRegistry,
    SelectionCriteria,
)


@pytest.fixture
def agent_registry():
    """Create an in-memory agent registry for testing."""
    return InMemoryAgentRegistry()


@pytest.fixture
def discovery_service(agent_registry):
    """Create a discovery service for testing."""
    return AgentDiscoveryService(agent_registry)


@pytest.fixture
def marco_capabilities():
    """Create capabilities for Marco (conversation specialist)."""
    return AgentCapabilities(
        specializations={AgentSpecialization.CONVERSATION, AgentSpecialization.ENCOURAGEMENT},
        confidence_scores={
            AgentSpecialization.CONVERSATION: 0.9,
            AgentSpecialization.ENCOURAGEMENT: 0.8,
        },
        max_concurrent_sessions=5,
        current_session_count=2,
        availability=AgentAvailability.AVAILABLE,
        average_response_time=1.2,
        success_rate=0.95,
        user_satisfaction=0.88,
        last_heartbeat=datetime.now(),
    )


@pytest.fixture
def professoressa_capabilities():
    """Create capabilities for Professoressa Rossi (grammar specialist)."""
    return AgentCapabilities(
        specializations={
            AgentSpecialization.GRAMMAR,
            AgentSpecialization.CORRECTIONS,
            AgentSpecialization.FORMAL_LANGUAGE,
        },
        confidence_scores={
            AgentSpecialization.GRAMMAR: 0.95,
            AgentSpecialization.CORRECTIONS: 0.92,
            AgentSpecialization.FORMAL_LANGUAGE: 0.85,
        },
        max_concurrent_sessions=3,
        current_session_count=1,
        availability=AgentAvailability.AVAILABLE,
        average_response_time=2.1,
        success_rate=0.98,
        user_satisfaction=0.82,
        last_heartbeat=datetime.now(),
    )


@pytest.fixture
def nonna_capabilities():
    """Create capabilities for Nonna Giulia (cultural specialist)."""
    return AgentCapabilities(
        specializations={AgentSpecialization.CULTURAL_CONTEXT, AgentSpecialization.STORYTELLING},
        confidence_scores={
            AgentSpecialization.CULTURAL_CONTEXT: 0.93,
            AgentSpecialization.STORYTELLING: 0.87,
        },
        max_concurrent_sessions=4,
        current_session_count=0,
        availability=AgentAvailability.AVAILABLE,
        average_response_time=1.8,
        success_rate=0.91,
        user_satisfaction=0.92,
        last_heartbeat=datetime.now(),
    )


@pytest.fixture
async def registered_agents(
    agent_registry, marco_capabilities, professoressa_capabilities, nonna_capabilities
):
    """Register sample agents in the registry."""
    agents = []

    # Register Marco
    marco = AgentRegistration(
        agent_id="marco_001",
        agent_name="Marco",
        agent_type="marco",
        capabilities=marco_capabilities,
    )
    await agent_registry.register_agent(marco)
    agents.append(marco)

    # Register Professoressa Rossi
    professoressa = AgentRegistration(
        agent_id="professoressa_001",
        agent_name="Professoressa Rossi",
        agent_type="professoressa_rossi",
        capabilities=professoressa_capabilities,
    )
    await agent_registry.register_agent(professoressa)
    agents.append(professoressa)

    # Register Nonna Giulia
    nonna = AgentRegistration(
        agent_id="nonna_001",
        agent_name="Nonna Giulia",
        agent_type="nonna_giulia",
        capabilities=nonna_capabilities,
    )
    await agent_registry.register_agent(nonna)
    agents.append(nonna)

    return agents


class TestAgentCapabilities:
    """Test the AgentCapabilities dataclass."""

    def test_load_factor_calculation(self):
        """Test load factor calculation."""
        capabilities = AgentCapabilities(max_concurrent_sessions=10, current_session_count=5)
        assert capabilities.get_load_factor() == 0.5

        # Test edge cases
        capabilities.current_session_count = 0
        assert capabilities.get_load_factor() == 0.0

        capabilities.current_session_count = 10
        assert capabilities.get_load_factor() == 1.0

        capabilities.current_session_count = 15  # Overloaded
        assert capabilities.get_load_factor() == 1.0  # Capped at 1.0

        capabilities.max_concurrent_sessions = 0  # Edge case
        assert capabilities.get_load_factor() == 0.0

    def test_availability_check(self):
        """Test availability checking."""
        capabilities = AgentCapabilities(
            max_concurrent_sessions=5,
            current_session_count=3,
            availability=AgentAvailability.AVAILABLE,
            last_heartbeat=datetime.now(),
        )

        assert capabilities.is_available_for_new_session() is True

        # Test overloaded
        capabilities.current_session_count = 5
        assert capabilities.is_available_for_new_session() is False

        # Test unavailable status
        capabilities.current_session_count = 3
        capabilities.availability = AgentAvailability.BUSY
        assert capabilities.is_available_for_new_session() is False

        # Test stale heartbeat
        capabilities.availability = AgentAvailability.AVAILABLE
        capabilities.last_heartbeat = datetime.now() - timedelta(minutes=10)
        assert capabilities.is_available_for_new_session() is False

    def test_confidence_scores(self):
        """Test confidence score retrieval."""
        capabilities = AgentCapabilities(
            confidence_scores={
                AgentSpecialization.GRAMMAR: 0.9,
                AgentSpecialization.CONVERSATION: 0.7,
            }
        )

        assert capabilities.get_confidence_for_specialization(AgentSpecialization.GRAMMAR) == 0.9
        assert (
            capabilities.get_confidence_for_specialization(AgentSpecialization.CONVERSATION) == 0.7
        )
        assert (
            capabilities.get_confidence_for_specialization(AgentSpecialization.PRONUNCIATION) == 0.0
        )


class TestAgentRegistration:
    """Test the AgentRegistration dataclass."""

    def test_health_check(self, marco_capabilities):
        """Test agent health checking."""
        registration = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="test",
            capabilities=marco_capabilities,
        )

        # Fresh registration should be healthy
        assert registration.is_healthy() is True

        # Old heartbeat should be unhealthy
        registration.capabilities.last_heartbeat = datetime.now() - timedelta(minutes=10)
        assert registration.is_healthy(heartbeat_timeout=300) is False
        assert registration.is_healthy(heartbeat_timeout=700) is True

    def test_heartbeat_update(self, marco_capabilities):
        """Test heartbeat updating."""
        registration = AgentRegistration(
            agent_id="test_agent",
            agent_name="Test Agent",
            agent_type="test",
            capabilities=marco_capabilities,
        )

        old_heartbeat = registration.capabilities.last_heartbeat
        old_updated = registration.last_updated

        # Small delay to ensure timestamp difference
        import time

        time.sleep(0.01)

        registration.update_heartbeat()

        assert registration.capabilities.last_heartbeat > old_heartbeat
        assert registration.last_updated > old_updated


class TestInMemoryAgentRegistry:
    """Test the in-memory agent registry implementation."""

    @pytest.mark.asyncio
    async def test_agent_registration(self, agent_registry, marco_capabilities):
        """Test agent registration and retrieval."""
        registration = AgentRegistration(
            agent_id="marco_001",
            agent_name="Marco",
            agent_type="marco",
            capabilities=marco_capabilities,
        )

        # Register agent
        success = await agent_registry.register_agent(registration)
        assert success is True

        # Retrieve agent
        retrieved = await agent_registry.get_agent("marco_001")
        assert retrieved is not None
        assert retrieved.agent_id == "marco_001"
        assert retrieved.agent_name == "Marco"
        assert retrieved.agent_type == "marco"

    @pytest.mark.asyncio
    async def test_agent_unregistration(self, agent_registry, marco_capabilities):
        """Test agent unregistration."""
        registration = AgentRegistration(
            agent_id="marco_001",
            agent_name="Marco",
            agent_type="marco",
            capabilities=marco_capabilities,
        )

        # Register then unregister
        await agent_registry.register_agent(registration)
        success = await agent_registry.unregister_agent("marco_001")
        assert success is True

        # Agent should be gone
        retrieved = await agent_registry.get_agent("marco_001")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_status_updates(self, agent_registry, marco_capabilities):
        """Test agent status updates."""
        registration = AgentRegistration(
            agent_id="marco_001",
            agent_name="Marco",
            agent_type="marco",
            capabilities=marco_capabilities,
        )

        await agent_registry.register_agent(registration)

        # Update status
        success = await agent_registry.update_agent_status(
            "marco_001", AgentAvailability.BUSY, session_count=5
        )
        assert success is True

        # Verify update
        retrieved = await agent_registry.get_agent("marco_001")
        assert retrieved.capabilities.availability == AgentAvailability.BUSY
        assert retrieved.capabilities.current_session_count == 5

    @pytest.mark.asyncio
    async def test_heartbeat(self, agent_registry, marco_capabilities):
        """Test heartbeat functionality."""
        registration = AgentRegistration(
            agent_id="marco_001",
            agent_name="Marco",
            agent_type="marco",
            capabilities=marco_capabilities,
        )

        await agent_registry.register_agent(registration)

        old_heartbeat = registration.capabilities.last_heartbeat

        # Send heartbeat
        import time

        time.sleep(0.01)
        success = await agent_registry.heartbeat("marco_001")
        assert success is True

        # Verify heartbeat was updated
        retrieved = await agent_registry.get_agent("marco_001")
        assert retrieved.capabilities.last_heartbeat > old_heartbeat

    @pytest.mark.asyncio
    async def test_find_agents_basic(self, agent_registry, registered_agents):
        """Test basic agent finding."""
        criteria = SelectionCriteria(
            preferred_specializations={AgentSpecialization.GRAMMAR}, minimum_confidence=0.5
        )

        matches = await agent_registry.find_agents(criteria, max_results=10)

        # Should find Professoressa Rossi
        assert len(matches) >= 1
        grammar_match = next(
            (
                match
                for match in matches
                if AgentSpecialization.GRAMMAR in match.registration.capabilities.specializations
            ),
            None,
        )
        assert grammar_match is not None
        assert grammar_match.registration.agent_type == "professoressa_rossi"

    @pytest.mark.asyncio
    async def test_find_agents_scoring(self, agent_registry, registered_agents):
        """Test agent scoring and ranking."""
        criteria = SelectionCriteria(
            preferred_specializations={AgentSpecialization.CONVERSATION},
            minimum_confidence=0.5,
            specialization_weight=1.0,  # Only care about specialization match
            confidence_weight=0.0,
            load_weight=0.0,
            performance_weight=0.0,
        )

        matches = await agent_registry.find_agents(criteria, max_results=10)

        # Marco should score highest for conversation
        assert len(matches) > 0
        top_match = matches[0]
        assert top_match.registration.agent_type == "marco"
        assert top_match.specialization_score > 0

    @pytest.mark.asyncio
    async def test_cleanup_stale_agents(self, agent_registry):
        """Test cleanup of stale agents."""
        # Create agent with old heartbeat
        old_capabilities = AgentCapabilities(last_heartbeat=datetime.now() - timedelta(minutes=10))
        old_registration = AgentRegistration(
            agent_id="old_agent",
            agent_name="Old Agent",
            agent_type="test",
            capabilities=old_capabilities,
        )

        # Create fresh agent
        fresh_capabilities = AgentCapabilities(last_heartbeat=datetime.now())
        fresh_registration = AgentRegistration(
            agent_id="fresh_agent",
            agent_name="Fresh Agent",
            agent_type="test",
            capabilities=fresh_capabilities,
        )

        # Register both
        await agent_registry.register_agent(old_registration)
        await agent_registry.register_agent(fresh_registration)

        # Cleanup stale agents (5 minute timeout)
        removed_count = await agent_registry.cleanup_stale_agents(timeout=300)
        assert removed_count == 1

        # Old agent should be gone, fresh should remain
        assert await agent_registry.get_agent("old_agent") is None
        assert await agent_registry.get_agent("fresh_agent") is not None

    @pytest.mark.asyncio
    async def test_list_all_agents(self, agent_registry, registered_agents):
        """Test listing all registered agents."""
        all_agents = await agent_registry.list_all_agents()

        assert len(all_agents) == 3
        agent_types = {agent.agent_type for agent in all_agents}
        assert agent_types == {"marco", "professoressa_rossi", "nonna_giulia"}


class TestAgentDiscoveryService:
    """Test the agent discovery service."""

    @pytest.mark.asyncio
    async def test_find_agent_for_help_request(self, discovery_service, registered_agents):
        """Test finding agents for help requests."""
        # Test grammar help
        matches = await discovery_service.find_agent_for_help_request(
            help_type="grammar", user_language_level="intermediate"
        )

        assert len(matches) >= 1
        # Should prefer Professoressa Rossi for grammar
        assert matches[0].registration.agent_type == "professoressa_rossi"

        # Test conversation help
        matches = await discovery_service.find_agent_for_help_request(
            help_type="conversation", user_language_level="beginner"
        )

        assert len(matches) >= 1
        # Should prefer Marco for conversation
        assert matches[0].registration.agent_type == "marco"

    @pytest.mark.asyncio
    async def test_find_agent_for_handoff(self, discovery_service, registered_agents):
        """Test finding agents for handoffs."""
        matches = await discovery_service.find_agent_for_handoff(
            reason="grammar_needed", current_agent_type="marco", conversation_complexity="medium"
        )

        assert len(matches) >= 1
        # Should find grammar specialist, but not Marco
        grammar_agents = [
            match
            for match in matches
            if AgentSpecialization.GRAMMAR in match.registration.capabilities.specializations
        ]
        assert len(grammar_agents) >= 1

        # Should not include current agent type
        marco_matches = [match for match in matches if match.registration.agent_type == "marco"]
        assert len(marco_matches) == 0

    @pytest.mark.asyncio
    async def test_find_agent_for_correction_review(self, discovery_service, registered_agents):
        """Test finding agents for correction reviews."""
        matches = await discovery_service.find_agent_for_correction_review(
            correction_type="verb_conjugation"
        )

        assert len(matches) >= 1
        # Should strongly prefer grammar specialists
        assert matches[0].registration.agent_type == "professoressa_rossi"
        assert matches[0].total_score > 0.7  # High confidence required

    @pytest.mark.asyncio
    async def test_find_best_agent(self, discovery_service, registered_agents):
        """Test finding the single best agent."""
        # Test help request
        best_agent = await discovery_service.find_best_agent(
            EventType.REQUEST_HELP, help_type="grammar", user_language_level="advanced"
        )

        assert best_agent is not None
        assert best_agent.agent_type == "professoressa_rossi"

        # Test handoff request
        best_agent = await discovery_service.find_best_agent(
            EventType.REQUEST_HANDOFF, reason="cultural_context", current_agent_type="marco"
        )

        assert best_agent is not None
        assert best_agent.agent_type == "nonna_giulia"

    @pytest.mark.asyncio
    async def test_get_agent_load_status(self, discovery_service, registered_agents):
        """Test getting agent load status."""
        load_status = await discovery_service.get_agent_load_status()

        # Should have status for all agent types
        assert "marco" in load_status
        assert "professoressa_rossi" in load_status
        assert "nonna_giulia" in load_status

        # Check Marco's status
        marco_status = load_status["marco"]
        assert marco_status["total_agents"] == 1
        assert marco_status["total_sessions"] == 2  # From fixture
        assert marco_status["total_capacity"] == 5
        assert marco_status["average_load"] == 0.4  # 2/5

    @pytest.mark.asyncio
    async def test_no_suitable_agent(self, discovery_service, registered_agents):
        """Test behavior when no suitable agent is found."""
        # Request with impossible requirements
        matches = await discovery_service.find_agent_for_help_request(
            help_type="nonexistent_skill", user_language_level="expert"
        )

        # Should still return some agents (fallback to general conversation)
        assert len(matches) >= 1

    @pytest.mark.asyncio
    async def test_agent_filtering_by_availability(
        self, discovery_service, agent_registry, registered_agents
    ):
        """Test that unavailable agents are filtered out."""
        # Make Marco unavailable
        await agent_registry.update_agent_status("marco_001", AgentAvailability.UNAVAILABLE)

        matches = await discovery_service.find_agent_for_help_request(
            help_type="conversation", user_language_level="beginner"
        )

        # Should not include unavailable Marco
        marco_matches = [match for match in matches if match.registration.agent_type == "marco"]
        assert len(marco_matches) == 0


class TestSelectionCriteria:
    """Test the selection criteria model."""

    def test_criteria_validation(self):
        """Test criteria validation."""
        # Valid criteria
        criteria = SelectionCriteria(
            minimum_confidence=0.8, max_load_factor=0.9, specialization_weight=0.4
        )
        assert criteria.minimum_confidence == 0.8
        assert criteria.max_load_factor == 0.9

    def test_criteria_defaults(self):
        """Test default values."""
        criteria = SelectionCriteria()
        assert criteria.minimum_confidence == 0.5
        assert criteria.max_load_factor == 0.8
        assert criteria.require_availability is True


class TestIntegration:
    """Integration tests combining registry, discovery, and scoring."""

    @pytest.mark.asyncio
    async def test_full_discovery_workflow(self, agent_registry, discovery_service):
        """Test complete discovery workflow."""
        # Register agents with different capabilities
        agents_data = [
            ("marco_1", "marco", {AgentSpecialization.CONVERSATION: 0.9}, 2, 5),
            ("marco_2", "marco", {AgentSpecialization.CONVERSATION: 0.8}, 1, 5),
            ("prof_1", "professoressa_rossi", {AgentSpecialization.GRAMMAR: 0.95}, 0, 3),
        ]

        for agent_id, agent_type, scores, sessions, max_sessions in agents_data:
            capabilities = AgentCapabilities(
                specializations=set(scores.keys()),
                confidence_scores=scores,
                current_session_count=sessions,
                max_concurrent_sessions=max_sessions,
                availability=AgentAvailability.AVAILABLE,
            )

            registration = AgentRegistration(
                agent_id=agent_id,
                agent_name=agent_id.replace("_", " ").title(),
                agent_type=agent_type,
                capabilities=capabilities,
            )

            await agent_registry.register_agent(registration)

        # Test load balancing - should prefer less loaded Marco
        best_agent = await discovery_service.find_best_agent(
            EventType.REQUEST_HELP, help_type="conversation", user_language_level="beginner"
        )

        assert best_agent is not None
        assert best_agent.agent_id == "marco_2"  # Less loaded

        # Test specialization matching - should prefer grammar expert
        best_agent = await discovery_service.find_best_agent(
            EventType.REQUEST_HELP, help_type="grammar", user_language_level="intermediate"
        )

        assert best_agent is not None
        assert best_agent.agent_id == "prof_1"  # Grammar specialist
