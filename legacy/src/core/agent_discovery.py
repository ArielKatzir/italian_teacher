"""
Agent discovery service for intelligent agent selection.

This module provides high-level discovery methods for finding the best agents
to handle specific types of requests, with built-in fallback strategies.
"""

import logging
from typing import Dict, List, Optional, Set

from .agent_events import EventType
from .agent_registry import (
    AgentMatch,
    AgentRegistration,
    AgentRegistry,
    AgentSpecialization,
    SelectionCriteria,
    default_agent_registry,
)
from .config import get_config


class AgentDiscoveryService:
    """Service for discovering and selecting agents based on request requirements."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.logger = logging.getLogger(__name__)

        # Define event type to specialization mappings
        self._event_specialization_map = {
            EventType.REQUEST_HELP: {
                "grammar": {AgentSpecialization.GRAMMAR, AgentSpecialization.CORRECTIONS},
                "pronunciation": {AgentSpecialization.PRONUNCIATION},
                "conversation": {
                    AgentSpecialization.CONVERSATION,
                    AgentSpecialization.CASUAL_LANGUAGE,
                },
                "cultural": {
                    AgentSpecialization.CULTURAL_CONTEXT,
                    AgentSpecialization.STORYTELLING,
                },
                "slang": {AgentSpecialization.MODERN_SLANG, AgentSpecialization.CASUAL_LANGUAGE},
                "formal": {AgentSpecialization.FORMAL_LANGUAGE, AgentSpecialization.GRAMMAR},
                "encouragement": {AgentSpecialization.ENCOURAGEMENT},
                "general": {AgentSpecialization.CONVERSATION},
            },
            EventType.REQUEST_CORRECTION_REVIEW: {
                AgentSpecialization.GRAMMAR,
                AgentSpecialization.CORRECTIONS,
                AgentSpecialization.FORMAL_LANGUAGE,
            },
            EventType.REQUEST_HANDOFF: {
                # Handoffs can go to any available agent, but prefer conversation specialists
                AgentSpecialization.CONVERSATION
            },
        }

    async def find_agent_for_help_request(
        self,
        help_type: str = "general",
        user_language_level: str = "beginner",
        max_candidates: Optional[int] = None,
    ) -> List[AgentMatch]:
        """
        Find agents suitable for a help request.

        Args:
            help_type: Type of help needed (grammar, pronunciation, etc.)
            user_language_level: User's language proficiency level
            max_candidates: Maximum number of candidates to return

        Returns:
            List of suitable agents, ranked by match score
        """
        if max_candidates is None:
            try:
                max_candidates = get_config().get_discovery_config().max_help_candidates
            except (AttributeError, TypeError):
                max_candidates = 3  # Fallback

        # Map help type to required/preferred specializations
        help_specializations = self._event_specialization_map[EventType.REQUEST_HELP].get(
            help_type, {AgentSpecialization.CONVERSATION}
        )

        # All help specializations are preferred (not required)
        required_specs = set()
        preferred_specs = help_specializations

        # Adjust criteria based on user language level
        min_confidence = self._get_min_confidence_for_level(user_language_level)

        criteria = SelectionCriteria(
            required_specializations=required_specs,
            preferred_specializations=preferred_specs,
            minimum_confidence=min_confidence,
            max_load_factor=0.8,
            require_availability=True,
        )

        matches = await self.registry.find_agents(criteria, max_candidates)

        self.logger.info(
            f"Found {len(matches)} agents for help_type='{help_type}', level='{user_language_level}'"
        )

        return matches

    async def find_agent_for_handoff(
        self,
        reason: str,
        current_agent_type: Optional[str] = None,
        conversation_complexity: str = "medium",
        max_candidates: Optional[int] = None,
    ) -> List[AgentMatch]:
        """
        Find agents suitable for conversation handoff.

        Args:
            reason: Reason for handoff
            current_agent_type: Type of current agent (to avoid same-type handoff)
            conversation_complexity: Complexity level of conversation
            max_candidates: Maximum number of candidates to return

        Returns:
            List of suitable agents for handoff
        """
        if max_candidates is None:
            try:
                max_candidates = get_config().get_discovery_config().max_handoff_candidates
            except (AttributeError, TypeError):
                max_candidates = 3  # Fallback

        # Determine required specializations based on handoff reason
        required_specs, preferred_specs = self._get_handoff_specializations(reason)

        # Adjust confidence requirements based on complexity
        try:
            complexity_thresholds = get_config().get_discovery_config().complexity_thresholds
            min_confidence = {
                "simple": complexity_thresholds.simple,
                "medium": complexity_thresholds.medium,
                "complex": complexity_thresholds.complex,
            }.get(conversation_complexity, complexity_thresholds.default)
        except (AttributeError, TypeError):
            # Fallback to hardcoded values if config fails
            min_confidence = {"simple": 0.6, "medium": 0.7, "complex": 0.8}.get(
                conversation_complexity, 0.7
            )

        criteria = SelectionCriteria(
            required_specializations=required_specs,
            preferred_specializations=preferred_specs,
            minimum_confidence=min_confidence,
            max_load_factor=0.9,  # Allow slightly higher load for handoffs
            require_availability=True,
        )

        matches = await self.registry.find_agents(criteria, max_candidates * 2)

        # Filter out agents of the same type as current agent (if specified)
        if current_agent_type:
            matches = [
                match for match in matches if match.registration.agent_type != current_agent_type
            ]

        result = matches[:max_candidates]

        self.logger.info(
            f"Found {len(result)} agents for handoff reason='{reason}', "
            f"excluding type='{current_agent_type}'"
        )

        return result

    async def find_agent_for_correction_review(
        self, correction_type: str = "general", max_candidates: Optional[int] = None
    ) -> List[AgentMatch]:
        """
        Find agents suitable for reviewing corrections.

        Args:
            correction_type: Type of correction to review
            max_candidates: Maximum number of candidates to return

        Returns:
            List of suitable agents for correction review
        """
        if max_candidates is None:
            try:
                max_candidates = get_config().get_discovery_config().max_correction_candidates
            except (AttributeError, TypeError):
                max_candidates = 2  # Fallback

        required_specs = self._event_specialization_map[EventType.REQUEST_CORRECTION_REVIEW]

        criteria = SelectionCriteria(
            required_specializations=required_specs,
            preferred_specializations=set(),
            minimum_confidence=0.8,  # High confidence required for corrections
            max_load_factor=0.7,  # Prefer less loaded agents for careful review
            require_availability=True,
        )

        matches = await self.registry.find_agents(criteria, max_candidates)

        self.logger.info(
            f"Found {len(matches)} agents for correction review type='{correction_type}'"
        )

        return matches

    async def find_best_agent(self, event_type: EventType, **kwargs) -> Optional[AgentRegistration]:
        """
        Find the single best agent for a request.

        Args:
            event_type: Type of event/request
            **kwargs: Additional parameters specific to event type

        Returns:
            Best matching agent registration, or None if no suitable agent found
        """
        matches = []

        if event_type == EventType.REQUEST_HELP:
            help_type = kwargs.get("help_type", "general")
            user_level = kwargs.get("user_language_level", "beginner")
            matches = await self.find_agent_for_help_request(
                help_type, user_level, max_candidates=1
            )

        elif event_type == EventType.REQUEST_HANDOFF:
            reason = kwargs.get("reason", "general")
            current_type = kwargs.get("current_agent_type")
            complexity = kwargs.get("conversation_complexity", "medium")
            matches = await self.find_agent_for_handoff(
                reason, current_type, complexity, max_candidates=1
            )

        elif event_type == EventType.REQUEST_CORRECTION_REVIEW:
            correction_type = kwargs.get("correction_type", "general")
            matches = await self.find_agent_for_correction_review(correction_type, max_candidates=1)

        if matches:
            self.logger.info(
                f"Selected agent '{matches[0].registration.agent_id}' "
                f"(score: {matches[0].total_score:.3f}) for {event_type.value}"
            )
            return matches[0].registration

        self.logger.warning(f"No suitable agent found for {event_type.value}")
        return None

    async def get_agent_load_status(self) -> Dict[str, Dict[str, any]]:
        """
        Get load status summary for all registered agents.

        Returns:
            Dictionary mapping agent_type to load statistics
        """
        all_agents = await self.registry.list_all_agents()

        status_by_type = {}
        for agent in all_agents:
            agent_type = agent.agent_type
            if agent_type not in status_by_type:
                status_by_type[agent_type] = {
                    "total_agents": 0,
                    "available_agents": 0,
                    "total_sessions": 0,
                    "total_capacity": 0,
                    "average_load": 0.0,
                    "agents": [],
                }

            type_stats = status_by_type[agent_type]
            type_stats["total_agents"] += 1
            type_stats["total_sessions"] += agent.capabilities.current_session_count
            type_stats["total_capacity"] += agent.capabilities.max_concurrent_sessions

            if agent.capabilities.is_available_for_new_session():
                type_stats["available_agents"] += 1

            type_stats["agents"].append(
                {
                    "agent_id": agent.agent_id,
                    "load_factor": agent.capabilities.get_load_factor(),
                    "availability": agent.capabilities.availability.value,
                    "session_count": agent.capabilities.current_session_count,
                    "max_sessions": agent.capabilities.max_concurrent_sessions,
                }
            )

        # Calculate average load for each type
        for type_stats in status_by_type.values():
            if type_stats["total_capacity"] > 0:
                type_stats["average_load"] = (
                    type_stats["total_sessions"] / type_stats["total_capacity"]
                )

        return status_by_type

    def _get_min_confidence_for_level(self, user_language_level: str) -> float:
        """Get minimum confidence threshold based on user level."""
        try:
            confidence_thresholds = get_config().get_discovery_config().confidence_thresholds
            level_thresholds = {
                "beginner": confidence_thresholds.beginner,
                "intermediate": confidence_thresholds.intermediate,
                "advanced": confidence_thresholds.advanced,
            }
            return level_thresholds.get(user_language_level, confidence_thresholds.default)
        except (AttributeError, TypeError):
            # Fallback to hardcoded values if config fails
            level_thresholds = {
                "beginner": 0.6,
                "intermediate": 0.7,
                "advanced": 0.8,
            }
            return level_thresholds.get(user_language_level, 0.7)

    def _get_handoff_specializations(
        self, reason: str
    ) -> tuple[Set[AgentSpecialization], Set[AgentSpecialization]]:
        """
        Determine required and preferred specializations for handoff.

        Returns:
            Tuple of (required_specializations, preferred_specializations)
        """
        handoff_specialization_map = {
            "grammar_needed": (
                {AgentSpecialization.GRAMMAR},
                {AgentSpecialization.CORRECTIONS, AgentSpecialization.FORMAL_LANGUAGE},
            ),
            "cultural_context": (
                {AgentSpecialization.CULTURAL_CONTEXT},
                {AgentSpecialization.STORYTELLING},
            ),
            "pronunciation_help": (
                {AgentSpecialization.PRONUNCIATION},
                {AgentSpecialization.CONVERSATION},
            ),
            "encouragement_needed": (
                {AgentSpecialization.ENCOURAGEMENT},
                {AgentSpecialization.CONVERSATION},
            ),
            "complexity": (
                set(),  # No specific requirements
                {AgentSpecialization.CONVERSATION, AgentSpecialization.FORMAL_LANGUAGE},
            ),
            "user_frustrated": (
                {AgentSpecialization.ENCOURAGEMENT},
                {AgentSpecialization.CONVERSATION},
            ),
        }

        return handoff_specialization_map.get(
            reason, (set(), {AgentSpecialization.CONVERSATION})  # Default fallback
        )


# Default discovery service instance
default_discovery_service = AgentDiscoveryService(default_agent_registry)
