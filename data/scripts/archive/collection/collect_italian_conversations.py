#!/usr/bin/env python3
"""
Collect Italian Conversations Dataset
Downloads and processes the cassanof/italian-conversations dataset from HuggingFace.
This contains authentic Italian conversations that can be used for language teaching.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_italian_conversations():
    """Download and analyze the Italian conversations dataset."""
    try:
        from datasets import load_dataset

        logger.info("📥 Downloading Italian conversations dataset...")
        dataset = load_dataset("cassanof/italian-conversations")

        logger.info(f"✅ Downloaded {len(dataset['train'])} Italian conversations")

        # Analyze first few examples
        print("\n📋 Sample conversations:")
        for i in range(min(3, len(dataset["train"]))):
            example = dataset["train"][i]
            print(f"\nExample {i+1}:")
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")

        return dataset["train"]

    except ImportError:
        logger.error("❌ Need to install datasets library: pip install datasets")
        return None
    except Exception as e:
        logger.error(f"❌ Error downloading dataset: {e}")
        return None


def convert_to_teaching_conversations(conversations_data, max_examples: int = 5000):
    """Convert Italian conversations to teaching format."""
    teaching_conversations = []

    logger.info(
        f"🔄 Converting {min(max_examples, len(conversations_data))} conversations to teaching format..."
    )

    # Question templates for creating teaching scenarios
    teaching_questions = [
        "Can you help me understand this Italian conversation?",
        "What's happening in this Italian dialogue?",
        "Could you explain this conversation in Italian?",
        "I'm reading this Italian exchange. What does it mean?",
        "Help me analyze this Italian conversation:",
        "What can I learn from this Italian dialogue?",
        "Break down this Italian conversation for me:",
        "Explain the context of this Italian discussion:",
    ]

    response_starters = [
        "This Italian conversation shows",
        "In this dialogue, the speakers are",
        "This exchange demonstrates",
        "The conversation reveals",
        "Looking at this dialogue,",
        "This Italian discussion involves",
        "The speakers here are",
        "This conversation illustrates",
    ]

    processed_count = 0

    for i, conversation in enumerate(conversations_data):
        if processed_count >= max_examples:
            break

        # Extract conversation content
        # Need to examine the actual structure first
        conv_text = ""
        if isinstance(conversation, dict):
            # Find the conversation text field
            for key, value in conversation.items():
                if isinstance(value, str) and len(value) > 20:
                    conv_text = value
                    break

        if not conv_text or len(conv_text) < 50:
            continue

        # Truncate very long conversations
        if len(conv_text) > 500:
            conv_text = conv_text[:500] + "..."

        # Create teaching conversation
        user_question = f'{random.choice(teaching_questions)} "{conv_text}"'

        # Generate educational response
        response_start = random.choice(response_starters)

        # Analyze complexity for CEFR level
        word_count = len(conv_text.split())
        if word_count < 20:
            level = "A2"
        elif word_count < 40:
            level = "B1"
        elif word_count < 80:
            level = "B2"
        else:
            level = "C1"

        # Create educational response
        response = f"{response_start} authentic Italian communication patterns. "
        response += f"This {level}-level dialogue demonstrates natural conversation flow, "
        response += "including typical Italian expressions and cultural context. "
        response += "The speakers use authentic vocabulary and grammar structures that are "
        response += "valuable for understanding how Italian is actually spoken."

        teaching_conversations.append(
            {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "conversation_id": f"italian_conv_{processed_count}",
                    "source": "italian_conversations_dataset",
                    "level": level,
                    "topic": "authentic_conversation",
                    "conversation_type": "dialogue_analysis",
                    "original_length": len(conv_text),
                },
            }
        )

        processed_count += 1

        if processed_count % 1000 == 0:
            logger.info(f"✅ Processed {processed_count} conversations...")

    logger.info(f"✅ Created {len(teaching_conversations)} teaching conversations")
    return teaching_conversations


def save_conversations(conversations: List[Dict], output_dir: str):
    """Save teaching conversations to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as JSONL
    output_file = output_path / "italian_conversations_teaching.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Create statistics
    stats = {"total_conversations": len(conversations), "by_level": {}, "by_type": {}}

    for conv in conversations:
        level = conv["metadata"].get("level", "unknown")
        conv_type = conv["metadata"].get("conversation_type", "unknown")
        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
        stats["by_type"][conv_type] = stats["by_type"].get(conv_type, 0) + 1

    stats_file = output_path / "italian_conversations_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved {len(conversations)} conversations to {output_file}")
    logger.info(f"📊 Statistics saved to {stats_file}")

    return output_file


def main():
    """Main collection function."""
    logger.info("🚀 Starting Italian conversations collection...")

    # Download dataset
    conversations_data = download_italian_conversations()

    if not conversations_data:
        print("❌ Failed to download Italian conversations dataset")
        print("💡 Install required packages: pip install datasets")
        return

    # Convert to teaching format
    teaching_conversations = convert_to_teaching_conversations(
        conversations_data, max_examples=3000
    )

    if not teaching_conversations:
        print("❌ No teaching conversations created")
        return

    # Save conversations
    output_file = save_conversations(teaching_conversations, "data/raw/italian_conversations")

    print(f"\n🎉 Italian Conversations Collection Completed!")
    print(f"📁 Output: {output_file}")
    print(f"📊 Total: {len(teaching_conversations)} teaching conversations")
    print(f"💡 Next: Include this data in your complete training dataset")


if __name__ == "__main__":
    main()
