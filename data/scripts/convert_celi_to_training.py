#!/usr/bin/env python3
"""
Convert CELI Authentic Conversations to Training Format
Converts the authentic CELI corpus conversations into the format expected by the training pipeline.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_celi_conversations(file_path: Path) -> List[Dict]:
    """Load CELI authentic conversations."""
    conversations = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    conversations.append(conv)

        logger.info(f"✅ Loaded {len(conversations)} authentic CELI conversations")
        return conversations

    except Exception as e:
        logger.error(f"❌ Error loading CELI conversations: {e}")
        return []


def convert_to_training_format(conversations: List[Dict]) -> List[Dict]:
    """Convert CELI conversations to training pipeline format."""
    training_examples = []

    for conv in conversations:
        # Extract metadata
        metadata = conv.get("metadata", {})
        level = metadata.get("level", "B1")
        topic = metadata.get("topic", "general")
        metadata.get("conversation_type", "explanation")

        # Create training example in expected format
        training_example = {
            "conversation_id": metadata.get("conversation_id", f"celi_{len(training_examples)}"),
            "source": "celi_corpus_authentic",
            "level": level,
            "topic": topic,
            "conversation": conv["messages"],
        }

        training_examples.append(training_example)

    logger.info(f"✅ Converted {len(training_examples)} conversations to training format")
    return training_examples


def create_final_dataset(examples: List[Dict], output_dir: Path) -> None:
    """Create train/validation/test splits and save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle examples
    random.shuffle(examples)

    # Calculate splits (80/15/5)
    total = len(examples)
    train_size = int(total * 0.8)
    val_size = int(total * 0.15)

    train_examples = examples[:train_size]
    val_examples = examples[train_size : train_size + val_size]
    test_examples = examples[train_size + val_size :]

    # Analyze distribution
    level_distribution = {}
    for example in examples:
        level = example.get("level", "B1")
        level_distribution[level] = level_distribution.get(level, 0) + 1

    # Save datasets in HuggingFace chat format
    def save_chat_dataset(examples: List[Dict], output_file: Path) -> None:
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                chat_example = {
                    "messages": example["conversation"],
                    "metadata": {
                        "conversation_id": example["conversation_id"],
                        "source": example["source"],
                        "level": example["level"],
                        "topic": example["topic"],
                    },
                }
                f.write(json.dumps(chat_example, ensure_ascii=False) + "\n")

    # Save splits
    save_chat_dataset(train_examples, output_dir / "train.jsonl")
    save_chat_dataset(val_examples, output_dir / "validation.jsonl")
    save_chat_dataset(test_examples, output_dir / "test.jsonl")

    # Save metadata
    metadata = {
        "total_examples": total,
        "train_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "test_examples": len(test_examples),
        "cefr_distribution": level_distribution,
        "data_source": "celi_corpus_authentic",
        "description": "Authentic Italian learner language from CELI corpus",
    }

    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n🎉 Authentic CELI Training Dataset Created:")
    print(f"   📁 Output: {output_dir}")
    print(f"   📊 Train: {len(train_examples)} examples")
    print(f"   📊 Validation: {len(val_examples)} examples")
    print(f"   📊 Test: {len(test_examples)} examples")
    print(f"   📊 Total: {total} examples")
    print(f"   🎯 CEFR Distribution: {level_distribution}")

    # Calculate B1/B2 percentage
    b1_b2_count = level_distribution.get("B1", 0) + level_distribution.get("B2", 0)
    b1_b2_percentage = (b1_b2_count / total * 100) if total > 0 else 0
    print(f"   🎯 B1/B2 Percentage: {b1_b2_percentage:.1f}%")

    print(f"\n✅ Ready for LoRA training with authentic learner data!")
    print(f"   📋 Next: Update config.py paths to point to this dataset")


def main():
    """Main conversion function."""
    # Input and output paths
    celi_conversations_file = Path(
        "data/processed/celi_authentic/celi_authentic_conversations.jsonl"
    )
    output_dir = Path("data/processed/celi_training_ready")

    # Check input file exists
    if not celi_conversations_file.exists():
        print(f"❌ CELI conversations file not found: {celi_conversations_file}")
        print("   Run process_celi_corpus.py first to generate authentic conversations")
        return

    # Load and convert conversations
    print("🚀 Converting CELI authentic conversations to training format...")
    conversations = load_celi_conversations(celi_conversations_file)

    if not conversations:
        print("❌ No conversations loaded")
        return

    # Convert to training format
    training_examples = convert_to_training_format(conversations)

    if not training_examples:
        print("❌ No training examples created")
        return

    # Create final dataset
    create_final_dataset(training_examples, output_dir)


if __name__ == "__main__":
    main()
