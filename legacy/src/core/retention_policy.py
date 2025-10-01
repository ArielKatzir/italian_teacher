"""
Data retention policy management for the Italian Teacher system.

This module handles user preferences for data retention, deletion schedules,
and compliance with privacy regulations.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional


class RetentionPreference(Enum):
    """User preference levels for data retention."""

    PRIVACY_FOCUSED = "privacy_focused"
    BALANCED = "balanced"
    LEARNING_FOCUSED = "learning_focused"


@dataclass
class DeletionStage:
    """Defines when different types of data should be deleted."""

    name: str
    days_after_deactivation: int
    data_types: List[str]
    description: str


@dataclass
class RetentionPolicy:
    """Complete retention policy configuration."""

    preference: RetentionPreference
    deletion_stages: List[DeletionStage]
    allow_analytics: bool = True
    allow_research: bool = False
    compliance_region: str = "global"

    def get_ttl_for_data_type(self, data_type: str) -> Optional[int]:
        """Get TTL (time to live) in days for a specific data type."""
        for stage in self.deletion_stages:
            if data_type in stage.data_types:
                return stage.days_after_deactivation
        return None

    def get_deletion_schedule(self) -> Dict[int, List[str]]:
        """Get a schedule of what data to delete on which days."""
        schedule = {}
        for stage in self.deletion_stages:
            day = stage.days_after_deactivation
            if day not in schedule:
                schedule[day] = []
            schedule[day].extend(stage.data_types)
        return schedule


class RetentionPolicyManager:
    """Manages retention policies and provides user preference options."""

    def __init__(self):
        self._policies = self._create_default_policies()

    def get_policy(self, preference: RetentionPreference) -> RetentionPolicy:
        """Get retention policy for a user preference."""
        return self._policies[preference]

    def get_policy_summary(self, preference: RetentionPreference) -> Dict[str, Any]:
        """Get human-readable summary of a retention policy."""
        policy = self._policies[preference]

        return {
            "preference": preference.value,
            "conversation_retention": self._get_conversation_retention_summary(policy),
            "analytics_allowed": policy.allow_analytics,
            "research_allowed": policy.allow_research,
            "total_stages": len(policy.deletion_stages),
            "shortest_retention": min(
                stage.days_after_deactivation for stage in policy.deletion_stages
            ),
            "longest_retention": max(
                stage.days_after_deactivation for stage in policy.deletion_stages
            ),
            "description": self._get_policy_description(preference),
        }

    def _get_conversation_retention_summary(self, policy: RetentionPolicy) -> str:
        """Get summary of conversation data retention."""
        conv_ttl = policy.get_ttl_for_data_type("conversation_content")
        if conv_ttl is None:
            return "Permanent"
        elif conv_ttl == 0:
            return "Immediate deletion"
        elif conv_ttl == 1:
            return "1 day"
        else:
            return f"{conv_ttl} days"

    def _get_policy_description(self, preference: RetentionPreference) -> str:
        """Get description of what each policy means."""
        descriptions = {
            RetentionPreference.PRIVACY_FOCUSED: "Minimal data retention. Conversations deleted quickly, no long-term analytics.",
            RetentionPreference.BALANCED: "Moderate retention for functionality while respecting privacy. Good for most users.",
            RetentionPreference.LEARNING_FOCUSED: "Longer retention to improve learning experience and AI quality. Helps personalization.",
        }
        return descriptions[preference]

    def _create_default_policies(self) -> Dict[RetentionPreference, RetentionPolicy]:
        """Create the default retention policies for each preference level."""

        # PRIVACY_FOCUSED: Delete almost everything quickly
        privacy_stages = [
            DeletionStage(
                name="immediate_cleanup",
                days_after_deactivation=0,
                data_types=["ip_addresses", "device_fingerprints", "session_tokens"],
                description="Immediate deletion of identifying data",
            ),
            DeletionStage(
                name="quick_cleanup",
                days_after_deactivation=1,
                data_types=["raw_audio", "typing_patterns", "detailed_corrections"],
                description="Quick deletion of detailed interaction data",
            ),
            DeletionStage(
                name="conversation_cleanup",
                days_after_deactivation=7,
                data_types=["conversation_content", "agent_responses", "user_messages"],
                description="Delete conversation content after 1 week",
            ),
            DeletionStage(
                name="final_cleanup",
                days_after_deactivation=30,
                data_types=["learning_progress", "user_preferences", "error_logs"],
                description="Delete remaining personal data",
            ),
        ]

        # BALANCED: Reasonable retention for functionality
        balanced_stages = [
            DeletionStage(
                name="immediate_cleanup",
                days_after_deactivation=0,
                data_types=["ip_addresses", "device_fingerprints"],
                description="Remove identifying technical data",
            ),
            DeletionStage(
                name="detailed_data_cleanup",
                days_after_deactivation=7,
                data_types=["raw_audio", "typing_patterns"],
                description="Remove detailed behavioral data",
            ),
            DeletionStage(
                name="conversation_cleanup",
                days_after_deactivation=30,
                data_types=["conversation_content", "detailed_corrections"],
                description="Remove conversation details after 1 month",
            ),
            DeletionStage(
                name="learning_data_cleanup",
                days_after_deactivation=90,
                data_types=["learning_progress", "agent_responses"],
                description="Remove learning history after 3 months",
            ),
            DeletionStage(
                name="preference_cleanup",
                days_after_deactivation=365,
                data_types=["user_preferences", "error_logs"],
                description="Remove preferences and logs after 1 year",
            ),
        ]

        # LEARNING_FOCUSED: Longer retention for better experience
        learning_stages = [
            DeletionStage(
                name="security_cleanup",
                days_after_deactivation=0,
                data_types=["session_tokens", "device_fingerprints"],
                description="Remove security-sensitive data immediately",
            ),
            DeletionStage(
                name="behavioral_cleanup",
                days_after_deactivation=30,
                data_types=["raw_audio", "typing_patterns"],
                description="Remove detailed behavioral patterns",
            ),
            DeletionStage(
                name="conversation_cleanup",
                days_after_deactivation=180,
                data_types=["conversation_content"],
                description="Keep conversations for 6 months for context",
            ),
            DeletionStage(
                name="learning_cleanup",
                days_after_deactivation=365,
                data_types=["detailed_corrections", "agent_responses"],
                description="Keep learning data for 1 year for personalization",
            ),
            DeletionStage(
                name="long_term_cleanup",
                days_after_deactivation=1095,  # 3 years
                data_types=["learning_progress", "user_preferences"],
                description="Keep long-term learning progress for continuity",
            ),
            DeletionStage(
                name="final_cleanup",
                days_after_deactivation=2555,  # 7 years
                data_types=["error_logs", "ip_addresses"],
                description="Final cleanup of system logs",
            ),
        ]

        return {
            RetentionPreference.PRIVACY_FOCUSED: RetentionPolicy(
                preference=RetentionPreference.PRIVACY_FOCUSED,
                deletion_stages=privacy_stages,
                allow_analytics=False,
                allow_research=False,
            ),
            RetentionPreference.BALANCED: RetentionPolicy(
                preference=RetentionPreference.BALANCED,
                deletion_stages=balanced_stages,
                allow_analytics=True,
                allow_research=False,
            ),
            RetentionPreference.LEARNING_FOCUSED: RetentionPolicy(
                preference=RetentionPreference.LEARNING_FOCUSED,
                deletion_stages=learning_stages,
                allow_analytics=True,
                allow_research=True,
            ),
        }

    def create_custom_policy(
        self, base_preference: RetentionPreference, custom_overrides: Dict[str, int]
    ) -> RetentionPolicy:
        """Create a custom policy based on a base preference with overrides."""
        base_policy = self.get_policy(base_preference)

        # Clone the base policy
        custom_stages = []
        for stage in base_policy.deletion_stages:
            new_stage = DeletionStage(
                name=stage.name,
                days_after_deactivation=stage.days_after_deactivation,
                data_types=stage.data_types.copy(),
                description=stage.description,
            )

            # Apply overrides
            for data_type in stage.data_types:
                if data_type in custom_overrides:
                    new_stage.days_after_deactivation = custom_overrides[data_type]

            custom_stages.append(new_stage)

        return RetentionPolicy(
            preference=base_preference,
            deletion_stages=custom_stages,
            allow_analytics=base_policy.allow_analytics,
            allow_research=base_policy.allow_research,
        )

    def calculate_cleanup_date(
        self, preference: RetentionPreference, data_type: str, deactivation_date: datetime
    ) -> Optional[datetime]:
        """Calculate when a specific data type should be deleted."""
        policy = self.get_policy(preference)
        ttl_days = policy.get_ttl_for_data_type(data_type)

        if ttl_days is None:
            return None  # Never delete

        return deactivation_date + timedelta(days=ttl_days)

    def get_next_cleanup_date(
        self, preference: RetentionPreference, deactivation_date: datetime
    ) -> Optional[datetime]:
        """Get the next scheduled cleanup date for any data type."""
        policy = self.get_policy(preference)

        if not policy.deletion_stages:
            return None

        # Find the earliest cleanup date
        earliest_days = min(stage.days_after_deactivation for stage in policy.deletion_stages)
        return deactivation_date + timedelta(days=earliest_days)

    def get_data_types_for_cleanup(
        self, preference: RetentionPreference, cleanup_date: datetime, deactivation_date: datetime
    ) -> List[str]:
        """Get list of data types that should be cleaned up on a specific date."""
        policy = self.get_policy(preference)
        days_since_deactivation = (cleanup_date - deactivation_date).days

        data_types_to_cleanup = []
        for stage in policy.deletion_stages:
            if stage.days_after_deactivation == days_since_deactivation:
                data_types_to_cleanup.extend(stage.data_types)

        return data_types_to_cleanup


# Default instance for easy importing
default_retention_manager = RetentionPolicyManager()
