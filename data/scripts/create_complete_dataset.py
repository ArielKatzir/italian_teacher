#!/usr/bin/env python3
"""
Create Complete A1-C2 Italian Teaching Dataset
Combines authentic CELI corpus data (B1-C2) with textbook content (A1-A2) for complete coverage.
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
    """Load CELI authentic conversations (B1-C2)."""
    conversations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    conversations.append(conv)
        logger.info(f"✅ Loaded {len(conversations)} CELI conversations (B1-C2)")
        return conversations
    except Exception as e:
        logger.error(f"❌ Error loading CELI conversations: {e}")
        return []


def load_textbook_conversations(file_path: Path) -> List[Dict]:
    """Load textbook conversations (A1-A2)."""
    conversations = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    # Convert to training format
                    training_conv = {"messages": conv["messages"], "metadata": conv["metadata"]}
                    conversations.append(training_conv)
        logger.info(f"✅ Loaded {len(conversations)} textbook conversations (A1-A2)")
        return conversations
    except Exception as e:
        logger.error(f"❌ Error loading textbook conversations: {e}")
        return []


def create_synthetic_a1_a2_content(target_count: int) -> List[Dict]:
    """Create additional A1/A2 content to balance the dataset."""
    synthetic_conversations = []

    # A1 level patterns
    a1_patterns = [
        {
            "italian": "Ciao! Come ti chiami?",
            "english": "Hi! What is your name?",
            "grammar": "present tense of chiamarsi (to be called)",
        },
        {
            "italian": "Io sono di Roma. E tu?",
            "english": "I am from Rome. And you?",
            "grammar": "present tense of essere (to be) with city names",
        },
        {
            "italian": "Mi piace la pizza italiana.",
            "english": "I like Italian pizza.",
            "grammar": "verb piacere (to like) construction",
        },
        {
            "italian": "Dove abiti?",
            "english": "Where do you live?",
            "grammar": "present tense of abitare (to live) in questions",
        },
        {
            "italian": "Ho fame. Andiamo al ristorante?",
            "english": "I am hungry. Shall we go to the restaurant?",
            "grammar": "avere fame (to be hungry) and suggestion with andiamo",
        },
    ]

    # A2 level patterns
    a2_patterns = [
        {
            "italian": "Ieri ho mangiato al ristorante con i miei amici.",
            "english": "Yesterday I ate at the restaurant with my friends.",
            "grammar": "passato prossimo with avere auxiliary",
        },
        {
            "italian": "Quando ero bambino, andavo sempre dal nonno.",
            "english": "When I was a child, I always went to my grandfather.",
            "grammar": "imperfetto for past habits and descriptions",
        },
        {
            "italian": "Vorrei imparare a cucinare la pasta.",
            "english": "I would like to learn to cook pasta.",
            "grammar": "conditional mood for polite requests",
        },
        {
            "italian": "Domani andrò al cinema se non piove.",
            "english": "Tomorrow I will go to the cinema if it does not rain.",
            "grammar": "futuro semplice with conditional clause",
        },
        {
            "italian": "Mi sembra che lui sia molto gentile.",
            "english": "It seems to me that he is very kind.",
            "grammar": "subjunctive mood with expressions of opinion",
        },
    ]

    # Question templates for variety
    question_templates = [
        "What does '{italian}' mean in Italian?",
        "Can you explain this Italian sentence: '{italian}'?",
        "Help me understand: '{italian}'",
        "I'm learning Italian. What does '{italian}' say?",
        "Could you translate '{italian}' for me?",
        "What's the grammar in '{italian}'?",
    ]

    # Response templates
    response_templates = [
        "That means '{english}'. This shows {grammar}.",
        "It translates to '{english}'. The grammar here demonstrates {grammar}.",
        "'{english}' - this is a great example of {grammar}.",
        "The meaning is '{english}'. Notice the {grammar} being used.",
        "This says '{english}'. It's perfect for learning {grammar}.",
    ]

    # Generate A1 content (60% of synthetic)
    a1_count = int(target_count * 0.6)
    for i in range(a1_count):
        pattern = random.choice(a1_patterns)
        user_question = random.choice(question_templates).format(italian=pattern["italian"])
        assistant_response = random.choice(response_templates).format(
            english=pattern["english"], grammar=pattern["grammar"]
        )

        synthetic_conversations.append(
            {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_response},
                ],
                "metadata": {
                    "conversation_id": f"synthetic_a1_{i}",
                    "source": "synthetic_textbook",
                    "level": "A1",
                    "topic": "basic_grammar",
                    "conversation_type": "explanation",
                },
            }
        )

    # Generate A2 content (40% of synthetic)
    a2_count = target_count - a1_count
    for i in range(a2_count):
        pattern = random.choice(a2_patterns)
        user_question = random.choice(question_templates).format(italian=pattern["italian"])
        assistant_response = random.choice(response_templates).format(
            english=pattern["english"], grammar=pattern["grammar"]
        )

        synthetic_conversations.append(
            {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": assistant_response},
                ],
                "metadata": {
                    "conversation_id": f"synthetic_a2_{i}",
                    "source": "synthetic_textbook",
                    "level": "A2",
                    "topic": "intermediate_grammar",
                    "conversation_type": "explanation",
                },
            }
        )

    logger.info(f"✅ Generated {len(synthetic_conversations)} synthetic A1/A2 conversations")
    return synthetic_conversations


def create_complete_dataset():
    """Create complete A1-C2 dataset combining all sources."""
    logger.info("🚀 Creating complete A1-C2 Italian teaching dataset...")

    # Load existing data
    celi_file = Path("data/processed/celi_training_ready/train.jsonl")
    textbook_file = Path("data/raw/textbook_content/textbook_conversations.jsonl")

    all_conversations = []

    # Load CELI data (B1-C2)
    if celi_file.exists():
        celi_conversations = load_celi_conversations(celi_file)
        all_conversations.extend(celi_conversations)
    else:
        logger.warning("CELI data not found")

    # Load textbook data (A1-A2)
    if textbook_file.exists():
        textbook_conversations = load_textbook_conversations(textbook_file)
        all_conversations.extend(textbook_conversations)
    else:
        logger.warning("Textbook data not found")

    # Count current A1/A2 levels
    current_a1_a2 = len(
        [c for c in all_conversations if c["metadata"].get("level") in ["A1", "A2"]]
    )

    # Add more A1/A2 content if needed
    target_a1_a2 = len(all_conversations) // 3  # Aim for ~33% A1/A2
    if current_a1_a2 < target_a1_a2:
        needed = target_a1_a2 - current_a1_a2
        synthetic_a1_a2 = create_synthetic_a1_a2_content(needed)
        all_conversations.extend(synthetic_a1_a2)

    # Analyze final distribution
    level_distribution = {}
    for conv in all_conversations:
        level = conv["metadata"].get("level", "unknown")
        level_distribution[level] = level_distribution.get(level, 0) + 1

    # Shuffle and create splits
    random.shuffle(all_conversations)
    total = len(all_conversations)
    train_size = int(total * 0.8)
    val_size = int(total * 0.15)

    train_conversations = all_conversations[:train_size]
    val_conversations = all_conversations[train_size : train_size + val_size]
    test_conversations = all_conversations[train_size + val_size :]

    # Save complete dataset
    output_dir = Path("data/processed/complete_a1_c2")
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_dataset(conversations: List[Dict], filename: str):
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    save_dataset(train_conversations, "train.jsonl")
    save_dataset(val_conversations, "validation.jsonl")
    save_dataset(test_conversations, "test.jsonl")

    # Save metadata
    metadata = {
        "total_examples": total,
        "train_examples": len(train_conversations),
        "validation_examples": len(val_conversations),
        "test_examples": len(test_conversations),
        "cefr_distribution": level_distribution,
        "data_sources": ["celi_corpus_authentic", "textbook_content", "synthetic_textbook"],
        "description": "Complete A1-C2 Italian teaching dataset with authentic CELI data",
    }

    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n🎉 Complete A1-C2 Italian Teaching Dataset Created:")
    print(f"   📁 Output: {output_dir}")
    print(f"   📊 Train: {len(train_conversations)} examples")
    print(f"   📊 Validation: {len(val_conversations)} examples")
    print(f"   📊 Test: {len(test_conversations)} examples")
    print(f"   📊 Total: {total} examples")
    print(f"   🎯 CEFR Distribution: {level_distribution}")

    # Calculate percentages
    for level, count in level_distribution.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"      {level}: {count} examples ({percentage:.1f}%)")

    print(f"\n✅ Complete dataset ready for Marco v2 training!")
    return output_dir


if __name__ == "__main__":
    create_complete_dataset()
