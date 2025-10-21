"""
Conversation State Storage Interface

Defines the contract for conversation state persistence systems.
This interface enables different storage implementations (in-memory, database, Redis)
while maintaining consistent API for conversation state management.
"""

from abc import ABC, abstractmethod
from datetime import datetime

# Import types using TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..base_agent import ConversationContext


class ConversationStateStore(ABC):
    """
    Abstract interface for conversation state persistence.

    This interface defines the contract that all conversation storage implementations
    must satisfy. It supports:
    - Context saving and loading
    - Session management per user
    - Cleanup of expired sessions
    - Scalable storage backends

    Implementations:
    - InMemoryConversationStore: Simple in-memory storage
    - DatabaseConversationStore: Persistent database storage (future)
    - RedisConversationStore: Distributed Redis-based storage (future)
    """

    @abstractmethod
    async def save_context(self, context: "ConversationContext") -> bool:
        """
        Save conversation context to persistent storage.

        Args:
            context: The conversation context to save

        Returns:
            True if save was successful, False otherwise
        """

    @abstractmethod
    async def load_context(self, session_id: str) -> Optional["ConversationContext"]:
        """
        Load conversation context from persistent storage.

        Args:
            session_id: The session ID to load

        Returns:
            ConversationContext if found, None otherwise
        """

    @abstractmethod
    async def delete_context(self, session_id: str) -> bool:
        """
        Delete conversation context from persistent storage.

        Args:
            session_id: The session ID to delete

        Returns:
            True if deletion was successful, False otherwise
        """

    @abstractmethod
    async def list_user_sessions(self, user_id: str) -> List[str]:
        """
        List all session IDs for a user.

        Args:
            user_id: The user ID to search for

        Returns:
            List of session IDs belonging to the user
        """

    @abstractmethod
    async def cleanup_expired_sessions(self, before_date: datetime) -> int:
        """
        Clean up sessions that expired before the given date.

        Args:
            before_date: Delete sessions last accessed before this date

        Returns:
            Number of sessions deleted
        """
