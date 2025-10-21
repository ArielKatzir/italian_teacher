"""
Tests for conversation state management system.
"""

from datetime import datetime, timedelta

import pytest
from core.base_agent import (
    AgentMessage,
    ConversationContext,
    MessageType,
    RetentionPreference,
)
from core.conversation_state import (
    ConversationStateManager,
    InMemoryConversationStore,
)


@pytest.fixture
def sample_context():
    """Create a sample conversation context for testing."""
    return ConversationContext(
        user_id="test_user_123",
        session_id="test_session_456",
        conversation_history=[
            AgentMessage(
                sender_id="marco",
                message_type=MessageType.CONVERSATION,
                content="Ciao! Come stai?",
            ),
            AgentMessage(
                sender_id="test_user_123",
                message_type=MessageType.CONVERSATION,
                content="Bene, grazie!",
            ),
        ],
        current_topic="greetings",
        user_language_level="beginner",
        learning_goals=["conversation", "pronunciation"],
        retention_preference=RetentionPreference.BALANCED,
    )


@pytest.fixture
def memory_store():
    """Create an in-memory conversation store."""
    return InMemoryConversationStore()


@pytest.fixture
def state_manager(memory_store):
    """Create a conversation state manager with in-memory store."""
    return ConversationStateManager(memory_store)


class TestInMemoryConversationStore:
    """Test the in-memory conversation store implementation."""

    @pytest.mark.asyncio
    async def test_save_and_load_context(self, memory_store, sample_context):
        """Test saving and loading conversation context."""
        # Save context
        result = await memory_store.save_context(sample_context)
        assert result is True

        # Load context
        loaded_context = await memory_store.load_context(sample_context.session_id)
        assert loaded_context is not None
        assert loaded_context.user_id == sample_context.user_id
        assert loaded_context.session_id == sample_context.session_id
        assert len(loaded_context.conversation_history) == 2
        assert loaded_context.current_topic == "greetings"

    @pytest.mark.asyncio
    async def test_load_nonexistent_context(self, memory_store):
        """Test loading a context that doesn't exist."""
        loaded_context = await memory_store.load_context("nonexistent_session")
        assert loaded_context is None

    @pytest.mark.asyncio
    async def test_delete_context(self, memory_store, sample_context):
        """Test deleting conversation context."""
        # Save context first
        await memory_store.save_context(sample_context)

        # Verify it exists
        loaded_context = await memory_store.load_context(sample_context.session_id)
        assert loaded_context is not None

        # Delete context
        result = await memory_store.delete_context(sample_context.session_id)
        assert result is True

        # Verify it's gone
        loaded_context = await memory_store.load_context(sample_context.session_id)
        assert loaded_context is None

    @pytest.mark.asyncio
    async def test_list_user_sessions(self, memory_store):
        """Test listing all sessions for a user."""
        user_id = "test_user_123"

        # Create multiple sessions for the user
        contexts = []
        for i in range(3):
            context = ConversationContext(
                user_id=user_id,
                session_id=f"session_{i}",
            )
            contexts.append(context)
            await memory_store.save_context(context)

        # Create session for different user
        other_context = ConversationContext(
            user_id="other_user",
            session_id="other_session",
        )
        await memory_store.save_context(other_context)

        # List sessions for target user
        user_sessions = await memory_store.list_user_sessions(user_id)
        assert len(user_sessions) == 3
        assert all(session_id.startswith("session_") for session_id in user_sessions)
        assert "other_session" not in user_sessions

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, memory_store):
        """Test cleaning up expired sessions."""
        now = datetime.now()
        cutoff_date = now - timedelta(hours=24)

        # Create old context (should be cleaned up)
        old_context = ConversationContext(
            user_id="user1",
            session_id="old_session",
            last_interaction=cutoff_date - timedelta(hours=1),
        )
        await memory_store.save_context(old_context)

        # Create recent context (should be kept)
        recent_context = ConversationContext(
            user_id="user1",
            session_id="recent_session",
            last_interaction=now - timedelta(hours=1),
        )
        await memory_store.save_context(recent_context)

        # Create context with no last_interaction (should be kept)
        no_interaction_context = ConversationContext(
            user_id="user1",
            session_id="no_interaction_session",
        )
        await memory_store.save_context(no_interaction_context)

        # Clean up expired sessions
        deleted_count = await memory_store.cleanup_expired_sessions(cutoff_date)
        assert deleted_count == 1

        # Verify old session is gone
        old_loaded = await memory_store.load_context("old_session")
        assert old_loaded is None

        # Verify recent session is still there
        recent_loaded = await memory_store.load_context("recent_session")
        assert recent_loaded is not None

        # Verify no_interaction session is still there
        no_interaction_loaded = await memory_store.load_context("no_interaction_session")
        assert no_interaction_loaded is not None


class TestConversationStateManager:
    """Test the conversation state manager."""

    @pytest.mark.asyncio
    async def test_get_context_cache_hit(self, state_manager, sample_context):
        """Test getting context when it's already in cache."""
        # Put context in cache first
        await state_manager.save_context(sample_context)

        # Get context should hit cache
        loaded_context = await state_manager.get_context(sample_context.session_id)
        assert loaded_context is not None
        assert loaded_context.session_id == sample_context.session_id

        # Verify it's the same object from cache
        assert loaded_context is sample_context

    @pytest.mark.asyncio
    async def test_get_context_cache_miss(self, state_manager, sample_context):
        """Test getting context when it's not in cache (loads from store)."""
        # Save to store directly (bypassing cache)
        await state_manager.store.save_context(sample_context)

        # Get context should miss cache and load from store
        loaded_context = await state_manager.get_context(sample_context.session_id)
        assert loaded_context is not None
        assert loaded_context.session_id == sample_context.session_id

        # Verify it's now cached
        cached_context = await state_manager.get_context(sample_context.session_id)
        assert cached_context is loaded_context

    @pytest.mark.asyncio
    async def test_save_context_caching(self, state_manager, sample_context):
        """Test that save_context updates cache and marks dirty."""
        # Save context
        result = await state_manager.save_context(sample_context)
        assert result is True

        # Verify it's in cache
        assert sample_context.session_id in state_manager._cache
        assert state_manager._cache[sample_context.session_id] is sample_context

        # Verify it's marked dirty
        assert sample_context.session_id in state_manager._dirty_sessions

    @pytest.mark.asyncio
    async def test_flush_dirty_sessions(self, state_manager, sample_context):
        """Test flushing dirty sessions to persistent storage."""
        # Save context (marks as dirty)
        await state_manager.save_context(sample_context)

        # Verify it's dirty
        assert sample_context.session_id in state_manager._dirty_sessions

        # Flush dirty sessions
        results = await state_manager.flush_dirty_sessions()
        assert len(results) == 1
        assert results[sample_context.session_id] is True

        # Verify it's no longer dirty
        assert sample_context.session_id not in state_manager._dirty_sessions

        # Verify it's in persistent storage
        loaded_from_store = await state_manager.store.load_context(sample_context.session_id)
        assert loaded_from_store is not None

    @pytest.mark.asyncio
    async def test_delete_context(self, state_manager, sample_context):
        """Test deleting context from both cache and storage."""
        # Save context
        await state_manager.save_context(sample_context)
        await state_manager.flush_dirty_sessions()

        # Verify it exists
        assert await state_manager.get_context(sample_context.session_id) is not None

        # Delete context
        result = await state_manager.delete_context(sample_context.session_id)
        assert result is True

        # Verify it's gone from cache
        assert sample_context.session_id not in state_manager._cache
        assert sample_context.session_id not in state_manager._dirty_sessions

        # Verify it's gone from storage
        loaded_context = await state_manager.get_context(sample_context.session_id)
        assert loaded_context is None

    @pytest.mark.asyncio
    async def test_transfer_context(self, state_manager, sample_context):
        """Test transferring context during agent handoffs."""
        # Save initial context
        await state_manager.save_context(sample_context)

        # Record initial message count
        initial_message_count = len(sample_context.conversation_history)

        # Transfer context
        result = await state_manager.transfer_context(
            session_id=sample_context.session_id,
            from_agent="marco",
            to_agent="professoressa_rossi",
            handoff_metadata={"reason": "grammar_help", "urgency": "high"},
        )
        assert result is True

        # Verify handoff message was added
        updated_context = await state_manager.get_context(sample_context.session_id)
        assert len(updated_context.conversation_history) == initial_message_count + 1

        # Verify handoff message details
        handoff_message = updated_context.conversation_history[-1]
        assert handoff_message.sender_id == "system"
        assert handoff_message.message_type == MessageType.SYSTEM
        assert "marco" in handoff_message.content
        assert "professoressa_rossi" in handoff_message.content
        assert handoff_message.metadata["handoff"] is True
        assert handoff_message.metadata["from_agent"] == "marco"
        assert handoff_message.metadata["to_agent"] == "professoressa_rossi"
        assert handoff_message.metadata["handoff_metadata"]["reason"] == "grammar_help"

        # Verify last_interaction was updated
        assert updated_context.last_interaction is not None

    @pytest.mark.asyncio
    async def test_transfer_nonexistent_context(self, state_manager):
        """Test transferring context that doesn't exist."""
        result = await state_manager.transfer_context(
            session_id="nonexistent_session",
            from_agent="marco",
            to_agent="professoressa_rossi",
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, state_manager):
        """Test getting all sessions for a user."""
        user_id = "test_user"

        # Create multiple sessions
        for i in range(3):
            context = ConversationContext(
                user_id=user_id,
                session_id=f"session_{i}",
            )
            await state_manager.save_context(context)

        await state_manager.flush_dirty_sessions()

        # Get user sessions
        sessions = await state_manager.get_user_sessions(user_id)
        assert len(sessions) == 3
        assert all(session.startswith("session_") for session in sessions)

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, state_manager):
        """Test cleaning up expired sessions from both cache and storage."""
        now = datetime.now()
        cutoff_date = now - timedelta(hours=24)

        # Create old context
        old_context = ConversationContext(
            user_id="user1",
            session_id="old_session",
            last_interaction=cutoff_date - timedelta(hours=1),
        )
        await state_manager.save_context(old_context)
        await state_manager.flush_dirty_sessions()

        # Create recent context
        recent_context = ConversationContext(
            user_id="user1",
            session_id="recent_session",
            last_interaction=now - timedelta(hours=1),
        )
        await state_manager.save_context(recent_context)

        # Clean up expired sessions
        deleted_count = await state_manager.cleanup_expired_sessions(cutoff_date)
        assert deleted_count == 1

        # Verify old session is gone from cache
        assert "old_session" not in state_manager._cache

        # Verify recent session is still in cache
        assert "recent_session" in state_manager._cache


class TestConversationStateIntegration:
    """Integration tests for conversation state management."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, state_manager):
        """Test full lifecycle of conversation state management."""
        # Create context
        context = ConversationContext(
            user_id="test_user",
            session_id="test_session",
            current_topic="food",
            user_language_level="intermediate",
        )

        # Save context
        await state_manager.save_context(context)

        # Add some conversation history
        context.conversation_history.append(
            AgentMessage(
                sender_id="marco",
                content="Cosa ti piace mangiare?",
                message_type=MessageType.CONVERSATION,
            )
        )

        # Update context
        await state_manager.save_context(context)

        # Flush to storage
        await state_manager.flush_dirty_sessions()

        # Clear cache to force reload
        state_manager._cache.clear()

        # Load from storage
        loaded_context = await state_manager.get_context("test_session")
        assert loaded_context is not None
        assert loaded_context.current_topic == "food"
        assert len(loaded_context.conversation_history) == 1

        # Transfer to another agent
        await state_manager.transfer_context(
            session_id="test_session",
            from_agent="marco",
            to_agent="nonna_giulia",
            handoff_metadata={"reason": "cultural_context"},
        )

        # Verify transfer was recorded
        final_context = await state_manager.get_context("test_session")
        assert len(final_context.conversation_history) == 2
        assert final_context.conversation_history[-1].sender_id == "system"

        # Clean up
        await state_manager.delete_context("test_session")
        deleted_context = await state_manager.get_context("test_session")
        assert deleted_context is None
