#!/usr/bin/env python3
"""
Italian Textbook Content Collection Script
Collects authentic Italian language teaching content from popular textbooks.

Target textbooks:
- Nuovo Espresso series (A1-C2)
- Dieci series (A1-B2)
- UniversItalia series
- Bene! series
- Via del Corso series
"""

import json
import logging
import re

# import PyPDF2
# import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ItalianTextbookCollector:
    """Collector for Italian language textbook content."""

    def __init__(self, output_dir: str = "data/raw/textbook_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Textbook series information
        self.textbook_series = {
            "nuovo_espresso": {
                "levels": ["A1", "A2", "B1", "B2", "C1", "C2"],
                "publisher": "Alma Edizioni",
                "focus": "Comprehensive Italian course",
            },
            "dieci": {
                "levels": ["A1", "A2", "B1", "B2"],
                "publisher": "Alma Edizioni",
                "focus": "Modern approach to Italian",
            },
            "universitalia": {
                "levels": ["A1", "A2", "B1", "B2"],
                "publisher": "Alma Edizioni",
                "focus": "University students",
            },
            "via_del_corso": {
                "levels": ["A1", "A2", "B1", "B2"],
                "publisher": "Edilingua",
                "focus": "Communicative approach",
            },
        }

        # Grammar topics by CEFR level
        self.grammar_topics = {
            "A1": [
                "present tense",
                "articles",
                "nouns",
                "adjectives",
                "essere and avere",
                "regular verbs",
                "numbers",
                "prepositions",
                "family vocabulary",
            ],
            "A2": [
                "past tense (passato prossimo)",
                "imperfetto",
                "future tense",
                "modal verbs",
                "comparative",
                "pronouns",
                "reflexive verbs",
            ],
            "B1": [
                "subjunctive mood",
                "conditional",
                "relative pronouns",
                "passive voice",
                "gerund",
                "hypothetical clauses",
            ],
            "B2": [
                "advanced subjunctive",
                "reported speech",
                "complex syntax",
                "idiomatic expressions",
                "formal register",
            ],
        }

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        logger.warning(f"PDF extraction not available - install PyPDF2 and PyMuPDF for PDF support")
        return ""

    def identify_grammar_sections(self, text: str, level: str) -> List[Dict]:
        """Identify grammar explanation sections in textbook text."""
        grammar_sections = []
        topics = self.grammar_topics.get(level, [])

        for topic in topics:
            # Look for grammar explanations
            patterns = [
                rf"(.*{topic}.*?\n.*?\n.*?\n.*?\n.*?)\n\n",  # Multi-line explanations
                rf"(grammatica.*?{topic}.*?\n.*?\n.*?)\n",  # Grammar sections
                rf"({topic}:.*?\n.*?\n.*?)\n",  # Topic headers
                rf"(come si usa.*?{topic}.*?\n.*?\n.*?)\n",  # Usage explanations
            ]

            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for match in matches:
                    explanation = match.group(1).strip()
                    if len(explanation) > 50:  # Minimum length for useful content
                        grammar_sections.append(
                            {
                                "topic": topic,
                                "level": level,
                                "explanation": explanation,
                                "type": "grammar_explanation",
                            }
                        )

        return grammar_sections

    def extract_dialogues(self, text: str, level: str) -> List[Dict]:
        """Extract dialogue examples from textbook text."""
        dialogues = []

        # Look for dialogue patterns
        dialogue_patterns = [
            r"(A: .*?\nB: .*?(?:\nA: .*?\nB: .*?)*)",  # A/B dialogue format
            r"(\w+: .*?\n\w+: .*?(?:\n\w+: .*?)*)",  # Named speakers
            r"(- .*?\n- .*?(?:\n- .*?)*)",  # Dash format
        ]

        for pattern in dialogue_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                dialogue_text = match.group(1).strip()

                # Filter out very short or very long dialogues
                if 30 <= len(dialogue_text) <= 500:
                    dialogues.append(
                        {"dialogue": dialogue_text, "level": level, "type": "dialogue_example"}
                    )

        return dialogues

    def create_teaching_conversations(self, content_data: List[Dict]) -> List[Dict]:
        """Create teaching conversations from textbook content."""
        conversations = []

        for item in content_data:
            item_type = item.get("type")
            level = item.get("level", "A1")

            if item_type == "grammar_explanation":
                topic = item.get("topic")
                explanation = item.get("explanation")

                # Create Q&A about grammar topic
                conversations.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Can you explain {topic} in Italian grammar?",
                            },
                            {
                                "role": "assistant",
                                "content": f"Certainly! Let me explain {topic} at the {level} level. {explanation[:200]}...",
                            },
                        ],
                        "metadata": {
                            "conversation_id": f"textbook_grammar_{len(conversations)}",
                            "source": "textbook_content",
                            "level": level,
                            "topic": topic,
                            "conversation_type": "grammar_explanation",
                        },
                    }
                )

                # Create practice question
                conversations.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Can you give me some practice exercises for {topic}?",
                            },
                            {
                                "role": "assistant",
                                "content": f"Of course! Here are some {level}-level exercises for {topic}. Practice makes perfect!",
                            },
                        ],
                        "metadata": {
                            "conversation_id": f"textbook_practice_{len(conversations)}",
                            "source": "textbook_content",
                            "level": level,
                            "topic": topic,
                            "conversation_type": "practice_exercises",
                        },
                    }
                )

            elif item_type == "dialogue_example":
                dialogue = item.get("dialogue")

                # Create conversation about dialogue analysis
                conversations.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f'Can you help me understand this Italian dialogue: "{dialogue[:100]}..."',
                            },
                            {
                                "role": "assistant",
                                "content": f"Absolutely! This is a {level}-level dialogue. Let me break it down for you...",
                            },
                        ],
                        "metadata": {
                            "conversation_id": f"textbook_dialogue_{len(conversations)}",
                            "source": "textbook_content",
                            "level": level,
                            "topic": "dialogue_analysis",
                            "conversation_type": "dialogue_explanation",
                        },
                    }
                )

        logger.info(f"✅ Created {len(conversations)} teaching conversations from textbook content")
        return conversations

    def process_manual_content(self) -> List[Dict]:
        """Process manually collected textbook content."""
        content_data = []

        # Check for manually placed PDF files
        pdf_files = list(self.output_dir.glob("**/*.pdf"))

        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")

            # Extract level from filename if possible
            level = "A1"  # default
            filename_lower = pdf_file.name.lower()
            if "a1" in filename_lower:
                level = "A1"
            elif "a2" in filename_lower:
                level = "A2"
            elif "b1" in filename_lower:
                level = "B1"
            elif "b2" in filename_lower:
                level = "B2"

            # Extract text content
            text_content = self.extract_text_from_pdf(pdf_file)

            if text_content:
                # Extract grammar sections
                grammar_sections = self.identify_grammar_sections(text_content, level)
                content_data.extend(grammar_sections)

                # Extract dialogues
                dialogues = self.extract_dialogues(text_content, level)
                content_data.extend(dialogues)

                logger.info(
                    f"✅ Extracted {len(grammar_sections)} grammar sections and {len(dialogues)} dialogues from {pdf_file.name}"
                )

        # Add some manual high-quality examples if no PDFs found
        if not content_data:
            content_data = self.create_sample_textbook_content()

        return content_data

    def create_sample_textbook_content(self) -> List[Dict]:
        """Create sample textbook-style content for demonstration."""
        sample_content = [
            {
                "type": "grammar_explanation",
                "topic": "present tense",
                "level": "A1",
                "explanation": "Il presente indicativo si usa per esprimere azioni abituali o stati permanenti. I verbi regolari in -are seguono questo schema: io parlo, tu parli, lui/lei parla, noi parliamo, voi parlate, loro parlano.",
            },
            {
                "type": "dialogue_example",
                "level": "A1",
                "dialogue": "A: Ciao! Come ti chiami?\nB: Ciao! Io mi chiamo Marco. E tu?\nA: Io sono Sofia. Piacere di conoscerti!\nB: Piacere mio!",
            },
            {
                "type": "grammar_explanation",
                "topic": "passato prossimo",
                "level": "A2",
                "explanation": "Il passato prossimo si forma con gli ausiliari essere o avere + participio passato. Con essere: sono andato/a. Con avere: ho mangiato. La scelta dell'ausiliare dipende dal verbo.",
            },
            {
                "type": "dialogue_example",
                "level": "A2",
                "dialogue": "A: Che cosa hai fatto ieri sera?\nB: Ho guardato un film italiano. E tu?\nA: Io sono andata al cinema con i miei amici.\nB: Che bello! Che film avete visto?",
            },
        ]

        logger.info("✅ Created sample textbook content")
        return sample_content

    def save_processed_data(self, content_data: List[Dict], conversations: List[Dict]) -> None:
        """Save processed textbook data."""
        # Save raw content data
        content_file = self.output_dir / "extracted_textbook_content.json"
        with open(content_file, "w", encoding="utf-8") as f:
            json.dump(content_data, f, indent=2, ensure_ascii=False)

        # Save conversations
        conversations_file = self.output_dir / "textbook_conversations.jsonl"
        with open(conversations_file, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        # Create statistics
        stats = {
            "total_content_items": len(content_data),
            "total_conversations": len(conversations),
            "by_type": {},
            "by_level": {},
        }

        for item in content_data:
            item_type = item.get("type", "unknown")
            level = item.get("level", "unknown")
            stats["by_type"][item_type] = stats["by_type"].get(item_type, 0) + 1
            stats["by_level"][level] = stats["by_level"].get(level, 0) + 1

        stats_file = self.output_dir / "textbook_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(
            f"✅ Saved {len(content_data)} content items and {len(conversations)} conversations"
        )

    def collect(self) -> bool:
        """Main collection method."""
        logger.info("🚀 Starting textbook content collection...")

        # Create instruction file for manual content collection
        instructions_file = self.output_dir / "INSTRUCTIONS.md"
        if not instructions_file.exists():
            instructions = """# Textbook Content Collection Instructions

## How to Add Textbook PDFs

1. **Place PDF files** in this directory (`data/raw/textbook_content/`)

2. **Naming convention** (helps with automatic level detection):
   - `nuovo_espresso_a1.pdf`
   - `dieci_a2.pdf`
   - `universitalia_b1.pdf`
   - etc.

3. **Recommended textbooks**:
   - Nuovo Espresso series (A1-C2)
   - Dieci series (A1-B2)
   - UniversItalia series
   - Via del Corso series

4. **What gets extracted**:
   - Grammar explanations
   - Dialogue examples
   - Exercise patterns
   - Cultural context

5. **Run the script** with:
   ```bash
   python data/scripts/collection/collect_textbook_content.py
   ```

## Legal Note
Ensure you have proper rights to use any textbook content for educational purposes.
"""
            with open(instructions_file, "w", encoding="utf-8") as f:
                f.write(instructions)

        # Process any existing content
        content_data = self.process_manual_content()

        if not content_data:
            logger.warning("⚠️  No textbook content found. See INSTRUCTIONS.md for adding PDFs")
            return False

        # Create teaching conversations
        conversations = self.create_teaching_conversations(content_data)

        # Save all processed data
        self.save_processed_data(content_data, conversations)

        logger.info("🎉 Textbook content collection completed!")
        return True


def main():
    """Main execution function."""
    collector = ItalianTextbookCollector()

    try:
        success = collector.collect()
        if success:
            print("✅ Textbook content collection completed!")
            print(f"📁 Data saved to: {collector.output_dir}")
            print("📖 See INSTRUCTIONS.md for adding more textbook PDFs")
        else:
            print("⚠️  No textbook content found")
            print("📖 See data/raw/textbook_content/INSTRUCTIONS.md for adding PDFs")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("❌ Collection failed with fatal error")


if __name__ == "__main__":
    main()
