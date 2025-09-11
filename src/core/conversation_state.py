"""
Conversation state management system.

This module provides persistence and state management for conversation contexts,
enabling seamless handoffs between agents and session recovery.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_agent import ConversationContext


class ConversationStateStore(ABC):
    """Abstract interface for conversation state persistence."""

    @abstractmethod
    async def save_context(self, context: ConversationContext) -> bool:
        """
        Save conversation context to persistent storage.

        Args:
            context: The conversation context to save

        Returns:
            True if save was successful, False otherwise
        """

    @abstractmethod
    async def load_context(self, session_id: str) -> Optional[ConversationContext]:
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
            List of session IDs for the user
        """

    @abstractmethod
    async def cleanup_expired_sessions(self, before_date: datetime) -> int:
        """
        Clean up sessions that expired before the given date.

        Args:
            before_date: Delete sessions last active before this date

        Returns:
            Number of sessions deleted
        """


class InMemoryConversationStore(ConversationStateStore):
    """In-memory implementation for development and testing."""

    def __init__(self):
        self._contexts: Dict[str, ConversationContext] = {}

    async def save_context(self, context: ConversationContext) -> bool:
        """Save context to memory."""
        try:
            self._contexts[context.session_id] = context
            return True
        except Exception:
            return False

    async def load_context(self, session_id: str) -> Optional[ConversationContext]:
        """Load context from memory."""
        return self._contexts.get(session_id)

    async def delete_context(self, session_id: str) -> bool:
        """Delete context from memory."""
        try:
            if session_id in self._contexts:
                del self._contexts[session_id]
            return True
        except Exception:
            return False

    async def list_user_sessions(self, user_id: str) -> List[str]:
        """List all sessions for a user."""
        return [
            session_id
            for session_id, context in self._contexts.items()
            if context.user_id == user_id
        ]

    async def cleanup_expired_sessions(self, before_date: datetime) -> int:
        """Clean up expired sessions."""
        expired_sessions = [
            session_id
            for session_id, context in self._contexts.items()
            if context.last_interaction and context.last_interaction < before_date
        ]

        for session_id in expired_sessions:
            del self._contexts[session_id]

        return len(expired_sessions)


class ConversationStateManager:
    """
    High-level manager for conversation state operations.

    Provides a unified interface for state management with caching,
    error handling, and transaction support.
    """

    def __init__(self, store: ConversationStateStore):
        self.store = store
        self._cache: Dict[str, ConversationContext] = {}
        self._dirty_sessions: set = set()

    async def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get conversation context, checking cache first.

        Args:
            session_id: Session ID to retrieve

        Returns:
            ConversationContext if found, None otherwise
        """
        # Check cache first
        if session_id in self._cache:
            return self._cache[session_id]

        # Load from persistent store
        context = await self.store.load_context(session_id)
        if context:
            self._cache[session_id] = context

        return context

    async def save_context(self, context: ConversationContext, force: bool = False) -> bool:
        """
        Save conversation context with caching.

        Args:
            context: Context to save
            force: Force save even if not dirty

        Returns:
            True if save was successful
        """
        session_id = context.session_id

        # Update cache
        self._cache[session_id] = context

        # Mark as dirty for batch saving
        self._dirty_sessions.add(session_id)

        # Save immediately if forced or if it's a new session
        if force or session_id not in self._cache:
            success = await self.store.save_context(context)
            if success:
                self._dirty_sessions.discard(session_id)
            return success

        return True  # Cached for later batch save

    async def flush_dirty_sessions(self) -> Dict[str, bool]:
        """
        Save all dirty sessions to persistent storage.

        Returns:
            Dict mapping session_id to save success status
        """
        results = {}

        for session_id in list(self._dirty_sessions):
            if session_id in self._cache:
                context = self._cache[session_id]
                success = await self.store.save_context(context)
                results[session_id] = success

                if success:
                    self._dirty_sessions.discard(session_id)

        return results

    async def delete_context(self, session_id: str) -> bool:
        """
        Delete conversation context from both cache and storage.

        Args:
            session_id: Session to delete

        Returns:
            True if deletion was successful
        """
        # Remove from cache
        self._cache.pop(session_id, None)
        self._dirty_sessions.discard(session_id)

        # Delete from persistent storage
        return await self.store.delete_context(session_id)

    async def transfer_context(
        self,
        session_id: str,
        from_agent: str,
        to_agent: str,
        handoff_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Handle context transfer during agent handoffs.

        Args:
            session_id: Session being transferred
            from_agent: Agent handing off the conversation
            to_agent: Agent receiving the conversation
            handoff_metadata: Additional handoff information

        Returns:
            True if transfer was successful
        """
        context = await self.get_context(session_id)
        if not context:
            return False

        # Add handoff metadata to conversation history
        from .base_agent import AgentMessage, MessageType

        handoff_message = AgentMessage(
            sender_id="system",
            recipient_id=to_agent,
            message_type=MessageType.SYSTEM,
            content=f"Conversation handed off from {from_agent} to {to_agent}",
            metadata={
                "handoff": True,
                "from_agent": from_agent,
                "to_agent": to_agent,
                "handoff_metadata": handoff_metadata or {},
            },
        )

        context.conversation_history.append(handoff_message)
        context.last_interaction = datetime.now()

        # Save updated context
        return await self.save_context(context, force=True)

    async def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all session IDs for a user."""
        return await self.store.list_user_sessions(user_id)

    async def cleanup_expired_sessions(self, before_date: datetime) -> int:
        """Clean up expired sessions from storage and cache."""
        # Clean up from persistent storage
        deleted_count = await self.store.cleanup_expired_sessions(before_date)

        # Clean up from cache
        expired_cache_sessions = [
            session_id
            for session_id, context in self._cache.items()
            if context.last_interaction and context.last_interaction < before_date
        ]

        for session_id in expired_cache_sessions:
            self._cache.pop(session_id, None)
            self._dirty_sessions.discard(session_id)

        return deleted_count


# Default instance for use throughout the application
default_conversation_manager = ConversationStateManager(InMemoryConversationStore())
