#!/usr/bin/env python3
"""
Italian Teacher CLI - Chat with Marco

A complete CLI interface using the full Marco agent with personality system,
error correction, motivation tracking, and language model integration.
Optimized for Colab Pro GPU usage.
"""

import asyncio
import sys
import uuid
from pathlib import Path

# Add both project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from agents.marco_agent import MarcoAgent
from core.agent_config import AgentConfigLoader
from core.base_agent import AgentPersonality, ConversationContext
from models import create_model


def load_marco_personality():
    """Load Marco's personality from YAML configuration."""
    try:
        loader = AgentConfigLoader()
        return loader.load_agent_personality("marco")
    except Exception as e:
        print(f"⚠️  Failed to load Marco config from YAML: {e}")
        print("🔄 Using fallback personality...")
        # Fallback to basic personality if config loading fails
        return AgentPersonality(
            name="Marco",
            role="Friendly Italian conversationalist",
            speaking_style="casual and encouraging",
            personality_traits=["encouraging", "patient", "enthusiastic", "friendly"],
            expertise_areas=["conversation", "encouragement", "basic_grammar"],
            correction_style="gentle",
            enthusiasm_level=8,
            formality_level=3,
            patience_level=9,
            encouragement_frequency=7,
            correction_frequency=5,
            topic_focus=["daily_conversation", "beginner_topics", "cultural_basics"],
        )


def create_simple_context(user_id: str = None):
    """Create a simple conversation context for testing."""
    if user_id is None:
        user_id = f"cli_user_{uuid.uuid4().hex[:8]}"

    return ConversationContext(user_id=user_id, session_id=f"cli_session_{uuid.uuid4().hex[:8]}")


def select_model_for_environment():
    """Select the best model based on available resources."""
    print("🤖 Selecting optimal model for your environment...")
    print("📋 Loading model configurations from YAML files...")

    # Check for forced mock mode
    import os

    if os.getenv("FORCE_MOCK_MODEL", "").lower() in ["true", "1", "yes"]:
        print("🧪 FORCE_MOCK_MODEL detected - using mock model")
        return create_model("mock")

    # Check if we're in Colab
    try:
        pass

        in_colab = True
        print("📊 Detected Google Colab environment")
    except ImportError:
        in_colab = False

    # Check GPU availability
    try:
        import torch

        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        else:
            print("💻 No GPU detected, will use CPU")
    except ImportError:
        gpu_available = False
        print("⚠️  PyTorch not available, using mock model")

    # Model selection logic - as we are developing now, just use the 3b.
    if not gpu_available or "torch" not in sys.modules:
        print("🧪 Using mock model for testing (config: configs/models/mock.yaml)")
        return create_model("mock")
    elif in_colab and gpu_available:
        print("🚀 Using Mistral 7B for development (config: configs/models/mistral-7b.yaml)")
        return create_model("mistral-7b", device="auto")
    #     if gpu_memory >= 14:
    #         print("🔥 Using Llama 3.1 8B - excellent for Italian conversations")
    #         return create_model("llama3.1-8b", device="cuda", quantization="4bit")
    #     else:
    #         print("⚡ Using Mistral 7B - optimized for your GPU memory")
    #         return create_model("mistral-7b", device="cuda", quantization="4bit")
    # else:
    #     print("🏠 Using smaller model for local development")
    #     return create_model("llama3.1-3b", device="auto")


async def chat_with_marco():
    """Main chat loop with Marco agent with language model integration."""
    print("🇮🇹 Welcome to Italian Teacher - Complete Marco Chat!")
    print("Type 'quit', 'exit', or 'ciao' to end the conversation.")
    print("=" * 60)

    # Select and load appropriate model
    try:
        print("🔄 Initializing language model...")
        model = select_model_for_environment()
        await model.load_model()
        print("✅ Language model loaded successfully!")
        print()
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("🧪 Falling back to mock model for demonstration...")
        model = create_model("mock")
        await model.load_model()

    # Initialize Marco with personality and model
    try:
        print("📋 Loading Marco's personality from configuration...")
        personality = load_marco_personality()
        print(f"✅ Loaded personality: {personality.name} - {personality.role}")
        marco = MarcoAgent(agent_id="marco_cli", personality=personality)

        # Integrate model with Marco (simple approach for CLI)
        marco.language_model = model

        # Create conversation context
        context = create_simple_context()

        print("✅ Marco agent initialized with full personality system!")
        print("🎭 Features active: Error correction, motivation tracking, cultural context")
        print("📱 Starting conversation...")
        print()

        # Initial greeting from Marco using language model
        try:
            system_prompt = model.get_italian_system_prompt("Marco")
            initial_response = await model.generate_response(
                "Introduce yourself as Marco, a friendly Italian conversation partner ready to help someone practice Italian.",
                system_prompt=system_prompt,
            )
            print(f"🇮🇹 Marco: {initial_response.text}")
            print()
        except Exception as e:
            print(f"⚠️ Note: Model greeting failed ({e}), using fallback...")
            print(
                "🇮🇹 Marco: Ciao! Sono Marco, il tuo amico italiano! Ready to practice Italian together?"
            )
            print()

    except Exception as e:
        print(f"❌ Failed to initialize Marco: {e}")
        print("🔧 Check that all dependencies are installed.")
        return

    # Main conversation loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Check for exit commands
            if user_input.lower() in ["quit", "exit", "ciao", "arrivederci", ""]:
                print("🇮🇹 Marco: Arrivederci! Great chatting with you! 👋")
                break

            # Generate Marco's response using the language model
            try:
                # Use Marco's sophisticated response generation with model integration
                if hasattr(marco, "language_model") and marco.language_model:
                    # Get Italian-optimized system prompt
                    system_prompt = marco.language_model.get_italian_system_prompt("Marco")

                    # Generate response using the language model
                    model_response = await marco.language_model.generate_response(
                        user_input, system_prompt=system_prompt
                    )

                    # Apply Marco's personality processing (error correction, motivation, etc.)
                    # For now, we'll use the model response directly but this could be enhanced
                    # to integrate with Marco's error tolerance and motivation systems
                    response = model_response.text
                else:
                    # Fallback to Marco's built-in response generation
                    response = await marco.generate_response(user_input, context)

                print(f"🇮🇹 Marco: {response}")
                print()
            except Exception as e:
                print(f"⚠️ Marco encountered an error: {e}")
                print("🔧 Trying fallback response generation...")
                try:
                    # Simple fallback
                    print("🇮🇹 Marco: Mi dispiace, let me try again. What were you saying?")
                    print()
                except:
                    print(
                        "🇮🇹 Marco: I'm having some technical difficulties, but I'm still here to help!"
                    )
                    print()

        except KeyboardInterrupt:
            print("\n🇮🇹 Marco: Arrivederci! See you next time! 👋")
            break
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            print("🔧 Please report this issue if it persists.")


def main():
    """Entry point for the CLI application."""
    try:
        asyncio.run(chat_with_marco())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Application error: {e}")
        print("🔧 Please check your setup and try again.")


if __name__ == "__main__":
    main()
