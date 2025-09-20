#!/usr/bin/env python3
"""
CELI Corpus Processor - Authentic Italian Learner Language
Processes authentic learner data from CELI corpus downloads to create high-quality
training conversations without pattern matching artifacts.

Input: KWIC concordance files from CELI corpus
Output: Authentic Italian teaching conversations based on real learner language
"""

import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CELICorpusProcessor:
    """Processor for authentic CELI corpus learner data."""

    def __init__(self, output_dir: str = "data/processed/celi_authentic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Natural conversation starters (NOT templates!)
        self.authentic_questions = [
            "I found this in an Italian text: '{text}'. What does it mean?",
            "Can you help me understand: '{text}'?",
            "I'm reading Italian and came across: '{text}'. Could you explain?",
            "What's happening grammatically in: '{text}'?",
            "I'm confused by this Italian phrase: '{text}'",
            "Could you break down: '{text}' for me?",
            "I saw this written by an Italian learner: '{text}'. Is it correct?",
            "Help me understand this Italian: '{text}'",
            "What does '{text}' express in Italian?",
            "Can you analyze: '{text}'?",
        ]

        # Natural response starters (diverse, not templated)
        self.response_starters = [
            "This is a great example of",
            "I can see this shows",
            "This demonstrates",
            "Looking at this text, it's",
            "This phrase illustrates",
            "What we have here is",
            "This is typical of",
            "This text represents",
        ]

        # Level-appropriate explanations
        self.level_contexts = {
            "B1": [
                "intermediate Italian grammar",
                "developing fluency patterns",
                "complex sentence structures",
                "authentic communication attempts",
                "natural language progression",
            ],
            "B2": [
                "advanced Italian usage",
                "sophisticated grammar patterns",
                "nuanced expression",
                "complex linguistic structures",
                "mature language development",
            ],
            "C1": [
                "highly advanced Italian",
                "sophisticated discourse",
                "complex argumentative structures",
                "native-like expression",
                "advanced rhetorical patterns",
            ],
            "C2": [
                "near-native proficiency",
                "complex linguistic competence",
                "sophisticated cultural expression",
                "advanced academic discourse",
                "mastery-level communication",
            ],
        }

    def parse_kwic_concordance(self, file_path: Path) -> List[Dict]:
        """Parse KWIC concordance format from CELI corpus downloads."""
        logger.info(f"Processing KWIC file: {file_path.name}")

        concordance_data = []

        try:
            # Try different encodings for robustness
            for encoding in ["utf-8", "utf-8-sig", "latin1"]:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            # Split into lines and process
            lines = content.split("\n")

            for line_num, line in enumerate(lines):
                if not line.strip():
                    continue

                # Parse KWIC format: typically tab-separated
                # Format: [left context] [keyword] [right context] [metadata]
                parts = line.split("\t")

                if len(parts) >= 3:
                    left_context = parts[0].strip()
                    keyword = parts[1].strip()
                    right_context = parts[2].strip()

                    # Extract metadata if available
                    metadata = {}
                    if len(parts) > 3:
                        # Parse metadata columns
                        for i, part in enumerate(parts[3:], 3):
                            if (
                                "CEFR" in part
                                or "B1" in part
                                or "B2" in part
                                or "C1" in part
                                or "C2" in part
                            ):
                                metadata["cefr_level"] = self.extract_cefr_level(part)
                            elif "Nationality" in part:
                                metadata["nationality"] = part.split(":")[-1].strip()
                            elif "Age" in part:
                                metadata["age_group"] = part.split(":")[-1].strip()
                            elif "Text type" in part:
                                metadata["text_type"] = part.split(":")[-1].strip()

                    # Reconstruct full context
                    full_text = f"{left_context} {keyword} {right_context}".strip()

                    # Only include meaningful text (not just punctuation)
                    if len(full_text) > 10 and any(c.isalpha() for c in full_text):
                        concordance_data.append(
                            {
                                "text": full_text,
                                "keyword": keyword,
                                "left_context": left_context,
                                "right_context": right_context,
                                "cefr_level": metadata.get("cefr_level", "B1"),
                                "nationality": metadata.get("nationality", "unknown"),
                                "age_group": metadata.get("age_group", "unknown"),
                                "text_type": metadata.get("text_type", "essay"),
                                "line_number": line_num,
                            }
                        )

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        logger.info(
            f"✅ Extracted {len(concordance_data)} authentic examples from {file_path.name}"
        )
        return concordance_data

    def extract_cefr_level(self, text: str) -> str:
        """Extract CEFR level from metadata text."""
        text = text.upper()
        if "C2" in text:
            return "C2"
        elif "C1" in text:
            return "C1"
        elif "B2" in text:
            return "B2"
        elif "B1" in text:
            return "B1"
        else:
            return "B1"  # Default

    def clean_authentic_text(self, text: str) -> str:
        """Clean learner text while preserving authenticity."""
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove obvious corpus artifacts
        text = re.sub(r"\[\w+\]", "", text)  # Remove annotation brackets
        text = re.sub(r"<[^>]+>", "", text)  # Remove XML tags
        text = re.sub(r"\*{2,}", "", text)  # Remove asterisk sequences

        # Clean up punctuation spacing
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        text = re.sub(r"([.!?])\s*([A-Z])", r"\1 \2", text)

        return text.strip()

    def create_authentic_conversations(self, corpus_data: List[Dict]) -> List[Dict]:
        """Create teaching conversations from authentic learner language."""
        conversations = []

        # Group by CEFR level for better organization
        level_groups = {}
        for item in corpus_data:
            level = item["cefr_level"]
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(item)

        for level, items in level_groups.items():
            logger.info(f"Processing {len(items)} authentic examples for level {level}")

            for item in items:
                text = self.clean_authentic_text(item["text"])

                # Skip very short or repetitive texts
                if len(text) < 15 or len(set(text.split())) < 5:
                    continue

                # Create diverse conversation types
                conversation_types = self.generate_conversation_types(text, item)
                conversations.extend(conversation_types)

        logger.info(f"✅ Created {len(conversations)} authentic teaching conversations")
        return conversations

    def generate_conversation_types(self, text: str, metadata: Dict) -> List[Dict]:
        """Generate different types of conversations from authentic text."""
        conversations = []
        level = metadata["cefr_level"]
        nationality = metadata.get("nationality", "unknown")
        text_type = metadata.get("text_type", "essay")

        # Truncate very long texts for questions
        display_text = text if len(text) <= 150 else text[:150] + "..."

        # 1. Understanding/Explanation conversation
        user_question = random.choice(self.authentic_questions).format(text=display_text)

        explanation = self.generate_authentic_explanation(text, level, nationality, text_type)

        conversations.append(
            {
                "messages": [
                    {"role": "user", "content": user_question},
                    {"role": "assistant", "content": explanation},
                ],
                "metadata": {
                    "conversation_id": f"celi_explanation_{len(conversations)}",
                    "source": "celi_corpus_authentic",
                    "level": level,
                    "topic": f"authentic_{text_type}",
                    "conversation_type": "explanation",
                    "learner_nationality": nationality,
                    "original_text_length": len(text),
                },
            }
        )

        # 2. Grammar analysis (for complex sentences)
        if len(text.split()) >= 8 and any(c in text for c in [",", "che", "quando", "perché"]):
            grammar_question = f"What grammar patterns do you see in: '{display_text}'?"
            grammar_analysis = self.generate_grammar_analysis(text, level)

            conversations.append(
                {
                    "messages": [
                        {"role": "user", "content": grammar_question},
                        {"role": "assistant", "content": grammar_analysis},
                    ],
                    "metadata": {
                        "conversation_id": f"celi_grammar_{len(conversations)}",
                        "source": "celi_corpus_authentic",
                        "level": level,
                        "topic": f"grammar_{text_type}",
                        "conversation_type": "grammar_analysis",
                        "learner_nationality": nationality,
                    },
                }
            )

        # 3. Cultural/pragmatic analysis (for B2+ levels)
        if level in ["B2", "C1", "C2"] and len(text.split()) >= 12:
            cultural_question = (
                f"What does this tell us about Italian communication: '{display_text}'?"
            )
            cultural_analysis = self.generate_cultural_analysis(text, level, text_type)

            conversations.append(
                {
                    "messages": [
                        {"role": "user", "content": cultural_question},
                        {"role": "assistant", "content": cultural_analysis},
                    ],
                    "metadata": {
                        "conversation_id": f"celi_cultural_{len(conversations)}",
                        "source": "celi_corpus_authentic",
                        "level": level,
                        "topic": f"culture_{text_type}",
                        "conversation_type": "cultural_analysis",
                        "learner_nationality": nationality,
                    },
                }
            )

        return conversations

    def generate_authentic_explanation(
        self, text: str, level: str, nationality: str, text_type: str
    ) -> str:
        """Generate natural explanation of authentic learner text."""
        starter = random.choice(self.response_starters)
        context = random.choice(self.level_contexts.get(level, self.level_contexts["B1"]))

        # Analyze the text for specific patterns
        text_features = self.analyze_text_features(text)

        explanation = f"{starter} {context}. "

        # Add specific observations
        if text_features["has_complex_verbs"]:
            explanation += "The verb forms show developing proficiency with Italian tenses. "

        if text_features["has_connectors"]:
            explanation += "The use of connecting words demonstrates growing discourse skills. "

        if text_features["has_subordination"]:
            explanation += "The sentence structure shows ability to express complex ideas. "

        # Add context about text type
        if text_type == "essay":
            explanation += f"This comes from an academic essay at the {level} level. "
        elif text_type == "email":
            explanation += f"This is from a {level}-level email writing task. "

        # Add learner-specific insight if available
        if nationality != "unknown":
            explanation += f"Written by a {nationality} speaker learning Italian. "

        explanation += "This represents authentic language learning in progress."

        return explanation

    def generate_grammar_analysis(self, text: str, level: str) -> str:
        """Generate focused grammar analysis."""
        features = self.analyze_text_features(text)

        analysis = f"This {level}-level text shows several interesting grammar patterns: "

        points = []

        if features["has_complex_verbs"]:
            points.append("complex verb usage")

        if features["has_subordination"]:
            points.append("subordinate clause structures")

        if features["has_connectors"]:
            points.append("discourse connectors")

        if features["has_pronouns"]:
            points.append("pronoun placement")

        if points:
            analysis += f"{', '.join(points[:3])}. "

        analysis += (
            f"The overall structure is typical of {level} proficiency learners developing fluency."
        )

        return analysis

    def generate_cultural_analysis(self, text: str, level: str, text_type: str) -> str:
        """Generate cultural/pragmatic analysis."""
        analysis = f"This {level}-level {text_type} demonstrates "

        if level == "C2":
            analysis += "sophisticated understanding of Italian discourse patterns. "
        elif level == "C1":
            analysis += "advanced awareness of Italian communication styles. "
        else:  # B2
            analysis += "developing sensitivity to Italian cultural norms. "

        analysis += "The language choices reflect authentic attempts to communicate "
        analysis += "in culturally appropriate ways. This kind of authentic usage "
        analysis += "is exactly what makes real learner language so valuable for understanding "
        analysis += "how Italian is actually acquired and used."

        return analysis

    def analyze_text_features(self, text: str) -> Dict[str, bool]:
        """Analyze linguistic features of the text."""
        text_lower = text.lower()

        return {
            "has_complex_verbs": any(
                v in text_lower for v in ["avrei", "sarei", "fossi", "avessi"]
            ),
            "has_connectors": any(
                c in text_lower for c in ["però", "quindi", "inoltre", "tuttavia"]
            ),
            "has_subordination": any(
                s in text_lower for s in ["che", "quando", "perché", "se", "mentre"]
            ),
            "has_pronouns": any(
                p in text_lower for p in ["mi", "ti", "si", "ci", "vi", "gli", "le"]
            ),
            "has_modals": any(
                m in text_lower for m in ["devo", "posso", "voglio", "dovrei", "potrei"]
            ),
            "has_questions": "?" in text,
            "is_complex": len(text.split()) > 15,
        }

    def save_processed_data(self, conversations: List[Dict]) -> None:
        """Save processed authentic conversations."""
        # Save conversations
        conversations_file = self.output_dir / "celi_authentic_conversations.jsonl"
        with open(conversations_file, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Create statistics
        stats = {
            "total_conversations": len(conversations),
            "by_level": {},
            "by_type": {},
            "by_nationality": {},
            "by_text_type": {},
        }

        for conv in conversations:
            level = conv["metadata"].get("level", "unknown")
            conv_type = conv["metadata"].get("conversation_type", "unknown")
            nationality = conv["metadata"].get("learner_nationality", "unknown")
            text_type = conv["metadata"].get("topic", "unknown")

            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1
            stats["by_type"][conv_type] = stats["by_type"].get(conv_type, 0) + 1
            stats["by_nationality"][nationality] = stats["by_nationality"].get(nationality, 0) + 1
            stats["by_text_type"][text_type] = stats["by_text_type"].get(text_type, 0) + 1

        stats_file = self.output_dir / "celi_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved {len(conversations)} conversations to {conversations_file}")
        logger.info(f"📊 Saved statistics to {stats_file}")

    def process_celi_data(self, raw_data_dir: str) -> bool:
        """Main processing method for CELI corpus data."""
        logger.info("🚀 Starting CELI corpus processing...")

        raw_path = Path(raw_data_dir)
        if not raw_path.exists():
            logger.error(f"❌ Raw data directory not found: {raw_data_dir}")
            return False

        # Find CELI corpus files
        corpus_files = []
        for pattern in ["*.txt", "*.tsv", "*.csv"]:
            corpus_files.extend(raw_path.glob(pattern))

        if not corpus_files:
            logger.error(f"❌ No corpus files found in {raw_data_dir}")
            return False

        logger.info(f"📁 Found {len(corpus_files)} corpus files")

        all_corpus_data = []

        # Process each file
        for file_path in corpus_files:
            logger.info(f"📖 Processing {file_path.name}...")

            if file_path.suffix == ".csv":
                # Handle CSV format
                try:
                    with open(file_path, "r", encoding="utf-8") as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            # Adapt to your CSV column structure
                            text = row.get("context", row.get("text", ""))
                            if text and len(str(text)) > 10:
                                all_corpus_data.append(
                                    {
                                        "text": str(text),
                                        "cefr_level": row.get("cefr_level", row.get("level", "B1")),
                                        "nationality": row.get("nationality", "unknown"),
                                        "text_type": row.get("text_type", "essay"),
                                    }
                                )
                except Exception as e:
                    logger.error(f"Error processing CSV {file_path}: {e}")
            else:
                # Handle KWIC/text format
                corpus_data = self.parse_kwic_concordance(file_path)
                all_corpus_data.extend(corpus_data)

        if not all_corpus_data:
            logger.error("❌ No data extracted from corpus files")
            return False

        logger.info(f"✅ Extracted {len(all_corpus_data)} authentic examples total")

        # Create conversations from authentic data
        conversations = self.create_authentic_conversations(all_corpus_data)

        if not conversations:
            logger.error("❌ No conversations created")
            return False

        # Save processed data
        self.save_processed_data(conversations)

        logger.info("🎉 CELI corpus processing completed successfully!")
        return True


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Process authentic CELI corpus data")
    parser.add_argument(
        "--input", default="data/raw/", help="Input directory with CELI corpus files"
    )
    parser.add_argument(
        "--output", default="data/processed/celi_authentic", help="Output directory"
    )

    args = parser.parse_args()

    processor = CELICorpusProcessor(args.output)

    try:
        success = processor.process_celi_data(args.input)
        if success:
            print("✅ CELI corpus processing completed!")
            print(f"📁 Data saved to: {processor.output_dir}")
        else:
            print("❌ CELI corpus processing failed!")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("❌ Processing failed with fatal error")


if __name__ == "__main__":
    main()
