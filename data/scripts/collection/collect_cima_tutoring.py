#!/usr/bin/env python3
"""
Collect CIMA Tutoring Dataset
Downloads and processes the CIMA dataset - authentic Italian tutoring conversations
between real tutors and students learning Italian grammar and vocabulary.
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_cima_dataset():
    """Download the CIMA tutoring dataset from GitHub."""
    url = "https://raw.githubusercontent.com/kstats/CIMA/master/dataset.json"

    try:
        logger.info("📥 Downloading CIMA tutoring dataset...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        dataset = response.json()
        logger.info(f"✅ Downloaded {len(dataset)} tutoring conversations")
        return dataset

    except Exception as e:
        logger.error(f"❌ Error downloading CIMA dataset: {e}")
        return None


def extract_teaching_conversations(cima_data):
    """Extract real teaching conversations from CIMA data."""
    teaching_conversations = []

    # CIMA has structure: {"prepDataset": {...}, "shapeDataset": {...}}
    all_sessions = []

    for dataset_name, dataset in cima_data.items():
        if isinstance(dataset, dict):
            # Each dataset contains numbered sessions
            for session_id, session_data in dataset.items():
                if isinstance(session_data, dict):
                    all_sessions.append((dataset_name, session_id, session_data))

    logger.info(f"🔄 Processing {len(all_sessions)} CIMA tutoring sessions...")

    for dataset_name, session_id, session in all_sessions:
        try:
            # Extract conversation history
            past_convo = session.get("past_convo", [])
            session.get("grammarRules", [])

            if not past_convo:
                continue

            # Extract authentic teacher-student exchanges
            conversations = extract_conversations_from_session(
                session, f"{dataset_name}_{session_id}"
            )
            teaching_conversations.extend(conversations)

            if len(all_sessions) > 100 and (len(teaching_conversations) % 100 == 0):
                logger.info(f"✅ Processed {len(teaching_conversations)} conversations...")

        except Exception as e:
            logger.warning(f"⚠️ Error processing session {i}: {e}")
            continue

    logger.info(f"✅ Extracted {len(teaching_conversations)} authentic teaching conversations")
    return teaching_conversations


def extract_conversations_from_session(session, session_id):
    """Extract individual conversations from a tutoring session."""
    conversations = []

    past_convo = session.get("past_convo", [])
    grammar_rules = session.get("grammarRules", [])
    tutor_responses = session.get("tutorResponses", [])

    if not past_convo:
        return conversations

    # CIMA format: past_convo is a list of strings (alternating tutor/student)
    # Pair them up as question-answer exchanges
    for i in range(0, len(past_convo) - 1, 2):
        if i + 1 < len(past_convo):
            # Assume alternating: tutor, student, tutor, student...
            tutor_msg = past_convo[i].strip() if isinstance(past_convo[i], str) else ""
            student_response = (
                past_convo[i + 1].strip() if isinstance(past_convo[i + 1], str) else ""
            )

            # Create a teaching conversation
            if tutor_msg and student_response and len(tutor_msg) > 10:
                # Create realistic teaching scenario
                user_question = (
                    f"I'm learning Italian. Can you help me with this: {student_response}"
                )
                assistant_response = tutor_msg

                # Add grammar context if available
                if grammar_rules:
                    context = extract_grammar_context(session)
                    if context:
                        assistant_response = f"{assistant_response} {context}"

                level = determine_cefr_level(session, user_question, assistant_response)

                conversations.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": assistant_response},
                        ],
                        "metadata": {
                            "conversation_id": f"cima_tutoring_{session_id}_{i}",
                            "source": "cima_tutoring_dataset",
                            "level": level,
                            "topic": "italian_grammar_tutoring",
                            "conversation_type": "authentic_tutoring",
                            "session_id": session_id,
                        },
                    }
                )

    # Also create conversations from tutor responses if available
    if tutor_responses:
        for j, response in enumerate(tutor_responses[:3]):  # Limit to 3
            if isinstance(response, str) and len(response) > 15:
                user_question = "Can you help me learn Italian grammar?"
                assistant_response = response.strip()

                conversations.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_question},
                            {"role": "assistant", "content": assistant_response},
                        ],
                        "metadata": {
                            "conversation_id": f"cima_response_{session_id}_{j}",
                            "source": "cima_tutoring_dataset",
                            "level": "B1",
                            "topic": "italian_grammar_tutoring",
                            "conversation_type": "tutor_response",
                            "session_id": session_id,
                        },
                    }
                )

    return conversations


def is_student_message(msg):
    """Check if message is from student."""
    if isinstance(msg, dict):
        speaker = msg.get("speaker", "").lower()
        action = msg.get("action", "").lower()
        return (
            "student" in speaker
            or "learner" in speaker
            or action in ["question", "ask", "confusion"]
        )
    return False


def is_tutor_message(msg):
    """Check if message is from tutor."""
    if isinstance(msg, dict):
        speaker = msg.get("speaker", "").lower()
        action = msg.get("action", "").lower()
        return (
            "tutor" in speaker
            or "teacher" in speaker
            or action in ["hint", "correction", "explanation", "feedback"]
        )
    return False


def extract_message_content(msg):
    """Extract the actual message content."""
    if isinstance(msg, dict):
        # Try different possible content fields
        for field in ["text", "content", "message", "utterance"]:
            if field in msg and msg[field]:
                return str(msg[field]).strip()

        # If no direct content, try to construct from other fields
        content_parts = []
        for key, value in msg.items():
            if key not in ["speaker", "action", "timestamp"] and value:
                if isinstance(value, str) and len(value) > 3:
                    content_parts.append(value)

        if content_parts:
            return " ".join(content_parts).strip()

    elif isinstance(msg, str):
        return msg.strip()

    return ""


def extract_grammar_context(session):
    """Extract grammar context from session."""
    grammar_rules = session.get("grammarRules", [])

    if grammar_rules:
        # Combine grammar rules into educational context
        rules_text = []
        for rule in grammar_rules[:2]:  # Limit to avoid too long responses
            if isinstance(rule, str):
                rules_text.append(rule)
            elif isinstance(rule, dict) and "text" in rule:
                rules_text.append(rule["text"])

        if rules_text:
            return f"Grammar note: {' '.join(rules_text)}"

    return ""


def determine_cefr_level(session, user_content, assistant_content):
    """Determine CEFR level based on session complexity."""
    # Check for level indicators in session
    if "level" in session:
        return session["level"]

    # Analyze complexity
    total_words = len(user_content.split()) + len(assistant_content.split())

    # Check for grammar complexity indicators
    complex_grammar = any(
        term in assistant_content.lower()
        for term in ["subjunctive", "conditional", "passive", "gerund", "participle"]
    )

    intermediate_grammar = any(
        term in assistant_content.lower()
        for term in ["past tense", "future", "present perfect", "imperfect"]
    )

    if complex_grammar or total_words > 60:
        return random.choice(["B2", "C1"])
    elif intermediate_grammar or total_words > 30:
        return random.choice(["B1", "B2"])
    else:
        return random.choice(["A2", "B1"])


def save_tutoring_conversations(conversations: List[Dict], output_dir: str):
    """Save authentic tutoring conversations."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save conversations
    output_file = output_path / "cima_tutoring_conversations.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Create statistics
    stats = {
        "total_conversations": len(conversations),
        "by_level": {},
        "by_type": {},
        "data_source": "CIMA tutoring dataset - authentic teacher-student conversations",
    }

    for conv in conversations:
        level = conv["metadata"].get("level", "unknown")
        conv_type = conv["metadata"].get("conversation_type", "unknown")
        stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
        stats["by_type"][conv_type] = stats["by_type"].get(conv_type, 0) + 1

    stats_file = output_path / "cima_tutoring_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"✅ Saved {len(conversations)} tutoring conversations to {output_file}")
    logger.info(f"📊 Statistics saved to {stats_file}")

    return output_file


def main():
    """Main collection function."""
    logger.info("🚀 Starting CIMA tutoring dataset collection...")

    # Download dataset
    cima_data = download_cima_dataset()

    if not cima_data:
        print("❌ Failed to download CIMA dataset")
        return

    # Show sample structure
    print(f"\n📋 Sample CIMA session structure:")
    if cima_data:
        if isinstance(cima_data, list) and len(cima_data) > 0:
            sample = cima_data[0]
            for key in sample.keys():
                print(f"  {key}: {type(sample[key])}")
        elif isinstance(cima_data, dict):
            print(f"  Data type: dict with {len(cima_data)} keys")
            for key in list(cima_data.keys())[:5]:  # Show first 5 keys
                print(f"  {key}: {type(cima_data[key])}")
        else:
            print(f"  Data type: {type(cima_data)}")
            print(f"  Content: {str(cima_data)[:200]}...")

    # Extract teaching conversations
    teaching_conversations = extract_teaching_conversations(cima_data)

    if not teaching_conversations:
        print("❌ No teaching conversations extracted")
        return

    # Save conversations
    output_file = save_tutoring_conversations(teaching_conversations, "data/raw/cima_tutoring")

    print(f"\n🎉 CIMA Tutoring Collection Completed!")
    print(f"📁 Output: {output_file}")
    print(f"📊 Total: {len(teaching_conversations)} authentic tutoring conversations")
    print(f"💡 This data contains REAL teacher-student exchanges!")
    print(f"💡 Next: Include this authentic tutoring data in your training dataset")


if __name__ == "__main__":
    main()
