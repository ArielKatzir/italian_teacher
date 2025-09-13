"""
Demonstration of the Coordinator Agent in action.

This example shows how the Coordinator Agent orchestrates conversations,
manages agent selection, and tracks learning progress.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core import (
    AgentCapabilities,
    AgentRegistration,
    AgentSpecialization,
    CoordinatorAgent,
    LearningGoal,
    default_agent_registry,
    default_discovery_service,
)


async def setup_sample_agents():
    """Set up sample agents in the registry."""
    print("🤖 Setting up sample agents...")

    # Marco - Conversation specialist
    marco = AgentRegistration(
        agent_id="marco_001",
        agent_name="Marco",
        agent_type="marco",
        capabilities=AgentCapabilities(
            specializations={AgentSpecialization.CONVERSATION, AgentSpecialization.ENCOURAGEMENT},
            confidence_scores={
                AgentSpecialization.CONVERSATION: 0.9,
                AgentSpecialization.ENCOURAGEMENT: 0.8,
            },
            max_concurrent_sessions=5,
            current_session_count=1,
        ),
    )

    # Professoressa Rossi - Grammar expert
    professoressa = AgentRegistration(
        agent_id="professoressa_001",
        agent_name="Professoressa Rossi",
        agent_type="professoressa_rossi",
        capabilities=AgentCapabilities(
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
            current_session_count=0,
        ),
    )

    # Nonna Giulia - Cultural expert
    nonna = AgentRegistration(
        agent_id="nonna_001",
        agent_name="Nonna Giulia",
        agent_type="nonna_giulia",
        capabilities=AgentCapabilities(
            specializations={
                AgentSpecialization.CULTURAL_CONTEXT,
                AgentSpecialization.STORYTELLING,
            },
            confidence_scores={
                AgentSpecialization.CULTURAL_CONTEXT: 0.93,
                AgentSpecialization.STORYTELLING: 0.87,
            },
            max_concurrent_sessions=4,
            current_session_count=0,
        ),
    )

    # Register all agents
    await default_agent_registry.register_agent(marco)
    await default_agent_registry.register_agent(professoressa)
    await default_agent_registry.register_agent(nonna)

    print(f"✅ Registered {len(await default_agent_registry.list_all_agents())} agents")


async def demo_basic_session():
    """Demonstrate a basic learning session."""
    print("\n🎯 Demo: Basic Learning Session")
    print("=" * 50)

    # Create coordinator
    coordinator = CoordinatorAgent(
        discovery_service=default_discovery_service,
    )

    # Define learning goals
    goals = [
        LearningGoal(
            goal_type="conversation",
            target_level="beginner",
            specific_topics=["greetings", "introductions"],
            estimated_duration=20,
        ),
        LearningGoal(
            goal_type="grammar",
            target_level="beginner",
            specific_topics=["present tense", "articles"],
            estimated_duration=30,
        ),
    ]

    # Start session
    session_id = await coordinator.create_session(
        user_id="demo_user",
        initial_agent_id="marco",
        learning_goals=goals,
    )

    print(f"📋 Started session: {session_id}")

    # Simulate conversation
    messages = [
        "How do I say 'hello' in Italian?",
        "What about 'good morning'?",
        "I think I made a grammar mistake - can you correct me?",
        "Tell me about Italian food culture",
        "Thank you for the lesson!",
    ]

    for i, message in enumerate(messages, 1):
        print(f"\n👤 User: {message}")

        # Simulate agent processing and updating session progress
        await coordinator.update_session_progress(
            session_id=session_id, message_content=message, agent_id="marco"
        )

        # Show session status after each message
        status = await coordinator.get_session_status(session_id)
        if status:
            print(
                f"📊 Status: {status['messages_exchanged']} messages, "
                f"Current agent: {status['current_agent']}, "
                f"Topics: {', '.join(status['topics_covered']) if status['topics_covered'] else 'None'}"
            )
        else:
            print("📊 Status: Session not found")

    # End session and show summary
    summary = await coordinator.end_session(session_id)
    print(f"\n📈 Session Summary:")
    print(f"   Duration: {summary['duration_minutes']:.1f} minutes")
    print(f"   Messages: {summary['messages_exchanged']}")
    print(f"   Topics covered: {', '.join(summary['topics_covered'])}")
    print(f"   Engagement score: {summary['engagement_score']:.2f}")


async def demo_context_switching():
    """Demonstrate intelligent context switching."""
    print("\n🔄 Demo: Context Switching")
    print("=" * 50)

    coordinator = CoordinatorAgent(discovery_service=default_discovery_service)

    session_id = await coordinator.create_session(
        user_id="demo_user2",
        initial_agent_id="marco",
    )

    print(f"📋 Started session: {session_id}")

    # Messages that should trigger different agents
    context_switch_messages = [
        ("Let's chat about my day", "conversation"),
        ("Wait, I made a grammar error in 'io mangio pasta'", "grammar"),
        ("Can you tell me about Italian traditions?", "cultural"),
        ("This is getting hard, I'm frustrated", "encouragement"),
        ("Back to practicing conversation", "conversation"),
    ]

    for message, expected_context in context_switch_messages:
        print(f"\n👤 User: {message}")

        # Simulate updating session progress (real agents would do this)
        await coordinator.update_session_progress(
            session_id=session_id, message_content=message, agent_id="marco"
        )

        # Show current session state
        status = await coordinator.get_session_status(session_id)
        if status:
            print(
                f"📊 Messages: {status['messages_exchanged']}, "
                f"Agent: {status['current_agent']}, "
                f"Topics: {', '.join(status['topics_covered']) if status['topics_covered'] else 'None'}"
            )

        print(
            f"💭 Expected context: {expected_context} (in real system, this would trigger agent handoffs)"
        )

    # Show final session status
    status = await coordinator.get_session_status(session_id)
    print(f"\n📊 Final Status:")
    print(f"   Messages exchanged: {status['messages_exchanged']}")
    print(f"   Topics covered: {', '.join(status['topics_covered'])}")
    print(f"   Current phase: {status['current_phase']}")

    await coordinator.end_session(session_id)


async def demo_progress_tracking():
    """Demonstrate progress tracking capabilities."""
    print("\n📈 Demo: Progress Tracking")
    print("=" * 50)

    coordinator = CoordinatorAgent(discovery_service=default_discovery_service)

    # Create session with specific goals
    goals = [
        LearningGoal(
            goal_type="grammar",
            target_level="intermediate",
            specific_topics=["verb conjugation", "past tense"],
            priority=1,
        ),
        LearningGoal(
            goal_type="conversation",
            target_level="beginner",
            specific_topics=["family", "food"],
            priority=2,
        ),
    ]

    session_id = await coordinator.create_session(
        user_id="demo_user3",
        initial_agent_id="marco",
        learning_goals=goals,
    )

    print(f"📋 Session goals:")
    for i, goal in enumerate(goals, 1):
        print(f"   {i}. {goal.goal_type} - {goal.target_level} level")
        print(f"      Topics: {', '.join(goal.specific_topics)}")

    # Simulate learning progression
    learning_messages = [
        "How do I conjugate 'essere' in past tense?",
        "My famiglia lives in Roma",
        "I was confused about 'ho mangiato' vs 'mangiavo'",
        "Tell me about typical Italian colazione",
        "Can you correct my sentence: 'Ieri io andato al ristorante'?",
        "My nonna makes the best pasta",
    ]

    print(f"\n🎓 Learning progression:")
    for i, message in enumerate(learning_messages, 1):
        print(f"\n{i}. 👤 User: {message}")

        # Extract topics before processing
        session_state = coordinator.active_sessions[session_id]
        topics_before = len(session_state.topics_covered)

        await coordinator.update_session_progress(
            session_id=session_id, message_content=message, agent_id="marco"
        )

        # Show progress update
        status = await coordinator.get_session_status(session_id)
        topics_after = len(status["topics_covered"])

        print(f"   📊 Progress update:")
        print(f"      Messages: {status['messages_exchanged']}")
        print(f"      Engagement: {status['engagement_score']:.2f}")
        print(f"      New topics: {topics_after - topics_before}")
        if status["topics_covered"]:
            print(f"      All topics: {', '.join(status['topics_covered'])}")

    # Final summary
    summary = await coordinator.end_session(session_id)
    print(f"\n🏆 Learning Session Complete!")
    print(f"   Total duration: {summary['duration_minutes']:.1f} minutes")
    print(f"   Messages exchanged: {summary['messages_exchanged']}")
    print(f"   Topics mastered: {len(summary['topics_covered'])}")
    print(f"   Final engagement: {summary['engagement_score']:.2f}")


async def demo_agent_load_balancing():
    """Demonstrate agent load balancing."""
    print("\n⚖️  Demo: Agent Load Balancing")
    print("=" * 50)

    coordinator = CoordinatorAgent(discovery_service=default_discovery_service)

    # Get current agent load status
    load_status = await default_discovery_service.get_agent_load_status()

    print("📊 Current agent load status:")
    for agent_type, status in load_status.items():
        print(f"   {agent_type}:")
        print(f"      Available agents: {status['available_agents']}/{status['total_agents']}")
        print(f"      Current load: {status['average_load']:.1%}")
        print(f"      Total capacity: {status['total_capacity']} sessions")

    # Start multiple sessions to show load balancing
    print(f"\n🚀 Starting multiple conversation sessions...")

    session_ids = []
    for i in range(3):
        session_id = await coordinator.create_session(
            user_id=f"user_{i+1}",
            initial_agent_id=f"Hello! This is user {i+1} wanting to practice conversation.",
        )
        session_ids.append(session_id)
        print(f"   Started session {i+1}: {session_id[:8]}...")

    # Check updated load status
    print(f"\n📊 Updated load status after starting sessions:")
    load_status = await default_discovery_service.get_agent_load_status()

    for agent_type, status in load_status.items():
        print(f"   {agent_type}: {status['average_load']:.1%} load")

    # Clean up sessions
    for session_id in session_ids:
        await coordinator.end_session(session_id)

    print(f"✅ Cleaned up {len(session_ids)} sessions")


async def main():
    """Run all coordinator demonstrations."""
    print("🎭 Italian Teacher - Coordinator Agent Demo")
    print("=" * 60)

    # Setup
    await setup_sample_agents()

    # Run demonstrations
    await demo_basic_session()
    await demo_context_switching()
    await demo_progress_tracking()
    await demo_agent_load_balancing()

    print(f"\n🎉 All demos completed successfully!")
    print(f"The Coordinator Agent is ready for Phase 1.3 integration! 🚀")


if __name__ == "__main__":
    asyncio.run(main())
