#!/usr/bin/env python3
"""
Tatoeba Italian Sentence Collector

Collects Italian sentences and translations from Tatoeba corpus.
Free example sentences with translations for language learning.

Usage:
    python collect_tatoeba.py --output ../raw/tatoeba_content/ --max-sentences 10000
"""

import argparse
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import requests


class TatoebaCollector:
    def __init__(self):
        """Initialize Tatoeba collector."""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self.base_url = "https://downloads.tatoeba.org"

        # Language codes
        self.italian_code = "ita"
        self.english_code = "eng"

    def download_sentences(self, language_code: str, output_dir: str) -> str:
        """Download sentences file for a specific language."""
        sentences_url = f"{self.base_url}/exports/sentences.tar.bz2"

        try:
            print(f"Downloading sentences database...")
            response = self.session.get(sentences_url, timeout=300)
            response.raise_for_status()

            # Save compressed file
            compressed_file = Path(output_dir) / "sentences.tar.bz2"
            with open(compressed_file, "wb") as f:
                f.write(response.content)

            print(f"✅ Downloaded sentences database ({len(response.content)} bytes)")
            return str(compressed_file)

        except Exception as e:
            print(f"❌ Error downloading sentences: {e}")
            return None

    def download_links(self, output_dir: str) -> str:
        """Download translation links file."""
        links_url = f"{self.base_url}/exports/links.tar.bz2"

        try:
            print(f"Downloading translation links...")
            response = self.session.get(links_url, timeout=300)
            response.raise_for_status()

            # Save compressed file
            compressed_file = Path(output_dir) / "links.tar.bz2"
            with open(compressed_file, "wb") as f:
                f.write(response.content)

            print(f"✅ Downloaded translation links ({len(response.content)} bytes)")
            return str(compressed_file)

        except Exception as e:
            print(f"❌ Error downloading links: {e}")
            return None

    def download_tab_delimited_pairs(self, output_dir: str) -> list:
        """Download pre-formatted Italian-English sentence pairs."""
        # Multiple sources for Italian-English pairs
        sources = [
            {
                "name": "manythings_italian_english",
                "url": "https://www.manythings.org/anki/ita-eng.zip",
                "description": "Italian-English sentence pairs from ManyThings.org",
            }
        ]

        downloaded_files = []

        for source in sources:
            try:
                print(f"Downloading {source['description']}...")
                response = self.session.get(source["url"], timeout=120)
                response.raise_for_status()

                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Handle ZIP files
                if source["url"].endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        for file_name in zip_file.namelist():
                            if file_name.endswith(".txt") and "ita" in file_name:
                                # Extract the Italian text file
                                content = zip_file.read(file_name).decode("utf-8")

                                # Save extracted content with unique filename
                                safe_filename = f"{source['name']}_{file_name.replace('/', '_')}"
                                output_file = output_path / safe_filename
                                with open(output_file, "w", encoding="utf-8") as f:
                                    f.write(content)

                                downloaded_files.append(
                                    {
                                        "source": source["name"],
                                        "file": str(output_file),
                                        "description": source["description"],
                                        "original_filename": file_name,
                                    }
                                )

                                print(f"✅ Extracted {file_name} to {output_file}")
                else:
                    # Direct text file
                    output_file = output_path / f"{source['name']}.txt"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(response.text)

                    downloaded_files.append(
                        {
                            "source": source["name"],
                            "file": str(output_file),
                            "description": source["description"],
                        }
                    )

                    print(f"✅ Downloaded to {output_file}")

            except Exception as e:
                print(f"❌ Error downloading from {source['url']}: {e}")
                continue

        return downloaded_files

    def parse_tab_delimited_pairs(self, file_path: str, max_sentences: int = None) -> list:
        """Parse tab-delimited sentence pairs."""
        sentence_pairs = []

        try:
            # Try different encodings
            encodings = ["utf-8", "latin-1", "utf-16"]
            content = None

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        content = f.read()
                        print(f"✅ Successfully read file with {encoding} encoding")
                        break
                except UnicodeDecodeError:
                    continue

            if not content:
                print(f"❌ Could not read {file_path} with any encoding")
                return []

            # Parse content line by line
            lines = content.strip().split("\n")

            for row_num, line in enumerate(lines):
                if max_sentences and row_num >= max_sentences:
                    break

                # Split on tab
                parts = line.split("\t")

                if len(parts) >= 2:
                    italian_sentence = parts[0].strip()
                    english_sentence = parts[1].strip()

                    if italian_sentence and english_sentence and len(italian_sentence) > 1:
                        sentence_pairs.append(
                            {
                                "italian": italian_sentence,
                                "english": english_sentence,
                                "source_line": row_num + 1,
                                "educational_value": self.assess_educational_value(
                                    italian_sentence, english_sentence
                                ),
                            }
                        )

            print(f"✅ Parsed {len(sentence_pairs)} sentence pairs from {file_path}")
            return sentence_pairs

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return []

    def assess_educational_value(self, italian_sentence: str, english_sentence: str) -> dict:
        """Assess the educational value of a sentence pair."""
        italian_words = italian_sentence.lower().split()
        english_words = english_sentence.lower().split()

        # Basic metrics for educational assessment
        metrics = {
            "italian_word_count": len(italian_words),
            "english_word_count": len(english_words),
            "sentence_length_category": (
                "short"
                if len(italian_words) <= 5
                else "medium" if len(italian_words) <= 12 else "long"
            ),
            "has_common_italian_words": any(
                word in italian_words
                for word in ["il", "la", "di", "che", "è", "sono", "ho", "hai"]
            ),
            "estimated_difficulty": (
                "beginner"
                if len(italian_words) <= 6
                else "intermediate" if len(italian_words) <= 12 else "advanced"
            ),
        }

        return metrics

    def collect_all_sentences(self, output_dir: str, max_sentences: int = None) -> None:
        """Collect all Tatoeba Italian sentences."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🇮🇹 Starting Tatoeba Italian sentence collection...")

        # Download pre-formatted sentence pairs (faster and easier)
        downloaded_files = self.download_tab_delimited_pairs(str(output_path))

        if not downloaded_files:
            print("❌ No sentence files downloaded")
            return

        all_sentence_pairs = []
        collection_stats = {
            "total_pairs": 0,
            "by_difficulty": {"beginner": 0, "intermediate": 0, "advanced": 0},
            "by_length": {"short": 0, "medium": 0, "long": 0},
            "sources": [],
        }

        for file_info in downloaded_files:
            print(f"\n📖 Processing {file_info['description']}...")

            sentence_pairs = self.parse_tab_delimited_pairs(file_info["file"], max_sentences)

            if sentence_pairs:
                # Add source metadata
                for pair in sentence_pairs:
                    pair["source"] = file_info["source"]
                    pair["source_description"] = file_info["description"]

                all_sentence_pairs.extend(sentence_pairs)

                # Update statistics
                collection_stats["sources"].append(
                    {
                        "name": file_info["source"],
                        "description": file_info["description"],
                        "pairs_count": len(sentence_pairs),
                    }
                )

                for pair in sentence_pairs:
                    difficulty = pair["educational_value"]["estimated_difficulty"]
                    length_cat = pair["educational_value"]["sentence_length_category"]

                    collection_stats["by_difficulty"][difficulty] += 1
                    collection_stats["by_length"][length_cat] += 1

        collection_stats["total_pairs"] = len(all_sentence_pairs)

        if all_sentence_pairs:
            # Sort by educational value (shorter sentences first for beginners)
            all_sentence_pairs.sort(
                key=lambda x: (
                    x["educational_value"]["estimated_difficulty"] == "advanced",
                    x["educational_value"]["italian_word_count"],
                )
            )

            # Save all sentence pairs
            output_file = output_path / "italian_sentences.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sentence_pairs": all_sentence_pairs,
                        "collection_metadata": {
                            "total_pairs": len(all_sentence_pairs),
                            "source": "tatoeba_corpus",
                            "language_pair": "italian_english",
                            "content_type": "sentence_pairs",
                            "educational_purpose": "vocabulary_and_grammar_examples",
                            "collection_date": datetime.now().isoformat(),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Save beginner-friendly subset
            beginner_pairs = [
                pair
                for pair in all_sentence_pairs
                if pair["educational_value"]["estimated_difficulty"] == "beginner"
            ][:1000]

            if beginner_pairs:
                beginner_file = output_path / "italian_sentences_beginner.json"
                with open(beginner_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "sentence_pairs": beginner_pairs,
                            "collection_metadata": {
                                "total_pairs": len(beginner_pairs),
                                "source": "tatoeba_corpus",
                                "difficulty_level": "beginner",
                                "language_pair": "italian_english",
                                "content_type": "sentence_pairs",
                                "educational_purpose": "basic_vocabulary_and_grammar",
                                "collection_date": datetime.now().isoformat(),
                            },
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

        # Save collection summary
        summary = {
            "collection_stats": collection_stats,
            "collection_date": datetime.now().isoformat(),
            "source": "tatoeba_sentence_corpus",
            "files_created": ["italian_sentences.json", "italian_sentences_beginner.json"],
        }

        with open(output_path / "tatoeba_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n🎉 Collection Summary:")
        print(f"   Total sentence pairs: {collection_stats['total_pairs']}")
        print(f"   By difficulty:")
        for difficulty, count in collection_stats["by_difficulty"].items():
            print(f"     {difficulty}: {count}")
        print(f"   By length:")
        for length, count in collection_stats["by_length"].items():
            print(f"     {length}: {count}")

        if collection_stats["total_pairs"] > 0:
            print(f"\n💾 Saved to: {output_path}")
            print(f"📝 Content includes:")
            print(f"   - Italian-English sentence pairs")
            print(f"   - Educational difficulty assessment")
            print(f"   - Progressive learning organization")
            print(f"   - Beginner-friendly subset")


def main():
    parser = argparse.ArgumentParser(description="Collect Italian sentences from Tatoeba corpus")
    parser.add_argument("--output", required=True, help="Output directory for sentence content")
    parser.add_argument("--max-sentences", type=int, help="Maximum number of sentences to collect")

    args = parser.parse_args()

    collector = TatoebaCollector()
    collector.collect_all_sentences(args.output, args.max_sentences)


if __name__ == "__main__":
    main()
