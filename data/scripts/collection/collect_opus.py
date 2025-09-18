#!/usr/bin/env python3
"""
OPUS Corpus Italian Collector

Collects Italian-English parallel texts from OPUS corpus.
Large collection of multilingual parallel corpora with tools and interfaces.

Usage:
    python collect_opus.py --output ../raw/opus_content/ --max-pairs 50000
"""

import argparse
import gzip
import json
import time
from datetime import datetime
from pathlib import Path

import requests


class OpusCollector:
    def __init__(self):
        """Initialize OPUS collector."""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self.base_url = "https://opus.nlpl.eu"

        # Available OPUS datasets with Italian-English pairs
        self.opus_datasets = [
            {
                "name": "OpenSubtitles",
                "description": "Movie subtitle translations",
                "url_pattern": "/OpenSubtitles/v2018/moses/en-it.txt.zip",
                "educational_value": "conversational_italian",
            },
            {
                "name": "Europarl",
                "description": "European Parliament proceedings",
                "url_pattern": "/Europarl/v8/moses/en-it.txt.zip",
                "educational_value": "formal_political_italian",
            },
            {
                "name": "TED2020",
                "description": "TED talk transcripts",
                "url_pattern": "/TED2020/v1/moses/en-it.txt.zip",
                "educational_value": "educational_presentations",
            },
            {
                "name": "WikiMatrix",
                "description": "Wikipedia article alignments",
                "url_pattern": "/WikiMatrix/v1/moses/en-it.txt.zip",
                "educational_value": "encyclopedic_content",
            },
            {
                "name": "News-Commentary",
                "description": "News commentary translations",
                "url_pattern": "/News-Commentary/v16/moses/en-it.txt.zip",
                "educational_value": "current_affairs_italian",
            },
        ]

    def download_opus_dataset(self, dataset: dict, output_dir: str) -> str:
        """Download a specific OPUS dataset."""
        dataset_url = f"{self.base_url}{dataset['url_pattern']}"

        try:
            print(f"Downloading {dataset['name']} ({dataset['description']})...")
            response = self.session.get(dataset_url, timeout=300)

            if response.status_code == 404:
                print(f"❌ Dataset {dataset['name']} not found at {dataset_url}")
                return None

            response.raise_for_status()

            # Save compressed file
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            compressed_file = output_path / f"{dataset['name'].lower()}_en_it.zip"
            with open(compressed_file, "wb") as f:
                f.write(response.content)

            print(f"✅ Downloaded {dataset['name']} ({len(response.content)} bytes)")
            return str(compressed_file)

        except Exception as e:
            print(f"❌ Error downloading {dataset['name']}: {e}")
            return None

    def get_opus_api_info(
        self, corpus: str, source_lang: str = "en", target_lang: str = "it"
    ) -> dict:
        """Get information about available OPUS corpus data via API."""
        api_url = f"{self.base_url}/opusapi/"

        try:
            # Get corpus info
            params = {
                "corpus": corpus,
                "source": source_lang,
                "target": target_lang,
                "preprocessing": "moses",
                "version": "latest",
            }

            response = self.session.get(api_url, params=params, timeout=60)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print(f"❌ Error getting API info for {corpus}: {e}")
            return {}

    def download_opus_via_api(self, corpus: str, output_dir: str, max_pairs: int = None) -> str:
        """Download OPUS corpus data using their API."""
        api_url = f"{self.base_url}/opusapi/"

        try:
            print(f"Downloading {corpus} via OPUS API...")

            # Get download URL
            params = {
                "corpus": corpus,
                "source": "en",
                "target": "it",
                "preprocessing": "moses",
                "version": "latest",
            }

            response = self.session.get(api_url, params=params, timeout=60)
            response.raise_for_status()

            api_data = response.json()

            if "corpora" in api_data and api_data["corpora"]:
                download_info = api_data["corpora"][0]
                download_url = download_info.get("url")

                if download_url:
                    # Download the actual corpus file
                    print(f"Downloading from: {download_url}")
                    corpus_response = self.session.get(download_url, timeout=300)
                    corpus_response.raise_for_status()

                    output_path = Path(output_dir)
                    output_path.mkdir(parents=True, exist_ok=True)

                    # Determine file extension from URL
                    if download_url.endswith(".gz"):
                        output_file = output_path / f"{corpus.lower()}_en_it.txt.gz"
                    elif download_url.endswith(".zip"):
                        output_file = output_path / f"{corpus.lower()}_en_it.zip"
                    else:
                        output_file = output_path / f"{corpus.lower()}_en_it.txt"

                    with open(output_file, "wb") as f:
                        f.write(corpus_response.content)

                    print(f"✅ Downloaded {corpus} to {output_file}")
                    return str(output_file)

            print(f"❌ No download URL found for {corpus}")
            return None

        except Exception as e:
            print(f"❌ Error downloading {corpus} via API: {e}")
            return None

    def parse_moses_format(self, file_path: str, max_pairs: int = None) -> list:
        """Parse Moses-format parallel text files."""
        sentence_pairs = []

        try:
            content = None

            # Handle ZIP files
            if file_path.endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(file_path, "r") as zip_file:
                    # Find the main text file
                    text_files = [f for f in zip_file.namelist() if f.endswith(".txt")]
                    if text_files:
                        main_file = text_files[0]  # Take the first text file
                        print(f"Extracting {main_file} from ZIP")
                        content = zip_file.read(main_file).decode("utf-8", errors="ignore")
                    else:
                        print(f"No .txt files found in {file_path}")
                        return []
            # Handle compressed files
            elif file_path.endswith(".gz"):
                with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            else:
                # Try different encodings for regular files
                encodings = ["utf-8", "latin-1", "cp1252"]
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                            print(f"Successfully read with {encoding} encoding")
                            break
                    except UnicodeDecodeError:
                        continue

            if not content:
                print(f"Could not read content from {file_path}")
                return []

            lines = content.strip().split("\n")

            # Moses format typically has source and target on same line separated by ' ||| '
            for line_num, line in enumerate(lines):
                if max_pairs and line_num >= max_pairs:
                    break

                if " ||| " in line:
                    parts = line.split(" ||| ")
                    if len(parts) >= 2:
                        english_sentence = parts[0].strip()
                        italian_sentence = parts[1].strip()

                        if english_sentence and italian_sentence:
                            sentence_pairs.append(
                                {
                                    "english": english_sentence,
                                    "italian": italian_sentence,
                                    "line_number": line_num + 1,
                                    "quality_score": self.assess_pair_quality(
                                        english_sentence, italian_sentence
                                    ),
                                }
                            )

            print(f"✅ Parsed {len(sentence_pairs)} sentence pairs from {file_path}")
            return sentence_pairs

        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
            return []

    def assess_pair_quality(self, english_sentence: str, italian_sentence: str) -> dict:
        """Assess the quality and educational value of a sentence pair."""
        en_words = english_sentence.split()
        it_words = italian_sentence.split()

        # Quality metrics
        length_ratio = len(it_words) / len(en_words) if en_words else 0

        quality_score = {
            "english_word_count": len(en_words),
            "italian_word_count": len(it_words),
            "length_ratio": length_ratio,
            "reasonable_length_ratio": 0.5 <= length_ratio <= 2.0,
            "has_punctuation": any(p in italian_sentence for p in ".!?"),
            "not_too_long": len(it_words) <= 50,
            "not_too_short": len(it_words) >= 3,
            "estimated_difficulty": self.estimate_difficulty(italian_sentence),
            "educational_category": self.categorize_content(english_sentence, italian_sentence),
        }

        # Overall quality score
        quality_score["overall_quality"] = (
            quality_score["reasonable_length_ratio"]
            + quality_score["not_too_long"]
            + quality_score["not_too_short"]
            + (1 if quality_score["estimated_difficulty"] in ["beginner", "intermediate"] else 0)
        ) / 4

        return quality_score

    def estimate_difficulty(self, italian_sentence: str) -> str:
        """Estimate difficulty level of Italian sentence."""
        words = italian_sentence.lower().split()
        word_count = len(words)

        # Basic difficulty estimation
        if word_count <= 6:
            return "beginner"
        elif word_count <= 15:
            return "intermediate"
        else:
            return "advanced"

    def categorize_content(self, english_sentence: str, italian_sentence: str) -> str:
        """Categorize the content type of the sentence pair."""
        combined_text = (english_sentence + " " + italian_sentence).lower()

        if any(word in combined_text for word in ["said", "says", "told", "asked", "replied"]):
            return "dialogue"
        elif any(word in combined_text for word in ["parliament", "government", "policy", "law"]):
            return "political"
        elif any(word in combined_text for word in ["study", "research", "according", "data"]):
            return "academic"
        elif any(word in combined_text for word in ["today", "yesterday", "news", "reported"]):
            return "news"
        else:
            return "general"

    def collect_all_opus_data(self, output_dir: str, max_pairs: int = None) -> None:
        """Collect all available OPUS Italian-English data."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🇮🇹 Starting OPUS corpus collection...")

        all_sentence_pairs = []
        collection_stats = {
            "total_pairs": 0,
            "by_source": {},
            "by_difficulty": {"beginner": 0, "intermediate": 0, "advanced": 0},
            "by_category": {},
            "high_quality_pairs": 0,
        }

        # Try to collect from known datasets
        successful_downloads = []

        for dataset in self.opus_datasets[:3]:  # Limit to first 3 datasets to avoid overwhelming
            print(f"\n📖 Attempting to collect {dataset['name']}...")

            # Try API first, then direct download
            downloaded_file = self.download_opus_via_api(
                dataset["name"], str(output_path), max_pairs
            )

            if not downloaded_file:
                downloaded_file = self.download_opus_dataset(dataset, str(output_path))

            if downloaded_file:
                successful_downloads.append({"dataset": dataset, "file": downloaded_file})

                # Parse the downloaded file
                sentence_pairs = self.parse_moses_format(downloaded_file, max_pairs)

                if sentence_pairs:
                    # Add source metadata
                    for pair in sentence_pairs:
                        pair["source"] = dataset["name"]
                        pair["source_description"] = dataset["description"]
                        pair["educational_value"] = dataset["educational_value"]

                    all_sentence_pairs.extend(sentence_pairs)

                    # Update statistics
                    collection_stats["by_source"][dataset["name"]] = len(sentence_pairs)

                    for pair in sentence_pairs:
                        difficulty = pair["quality_score"]["estimated_difficulty"]
                        category = pair["quality_score"]["educational_category"]

                        collection_stats["by_difficulty"][difficulty] += 1

                        if category not in collection_stats["by_category"]:
                            collection_stats["by_category"][category] = 0
                        collection_stats["by_category"][category] += 1

                        if pair["quality_score"]["overall_quality"] >= 0.75:
                            collection_stats["high_quality_pairs"] += 1

            time.sleep(2)  # Rate limiting

        collection_stats["total_pairs"] = len(all_sentence_pairs)

        if all_sentence_pairs:
            # Filter high-quality pairs for educational use
            high_quality_pairs = [
                pair
                for pair in all_sentence_pairs
                if pair["quality_score"]["overall_quality"] >= 0.75
            ]

            # Sort by quality and difficulty
            high_quality_pairs.sort(
                key=lambda x: (
                    x["quality_score"]["estimated_difficulty"] == "advanced",
                    -x["quality_score"]["overall_quality"],
                    x["quality_score"]["italian_word_count"],
                )
            )

            # Save all pairs
            output_file = output_path / "opus_italian_parallel.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "sentence_pairs": all_sentence_pairs,
                        "collection_metadata": {
                            "total_pairs": len(all_sentence_pairs),
                            "source": "opus_corpus",
                            "language_pair": "english_italian",
                            "content_type": "parallel_text",
                            "educational_purpose": "translation_examples",
                            "collection_date": datetime.now().isoformat(),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            # Save high-quality subset
            if high_quality_pairs:
                quality_file = output_path / "opus_italian_high_quality.json"
                with open(quality_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "sentence_pairs": high_quality_pairs[:10000],  # Limit to 10k best pairs
                            "collection_metadata": {
                                "total_pairs": len(high_quality_pairs[:10000]),
                                "source": "opus_corpus_filtered",
                                "quality_threshold": 0.75,
                                "language_pair": "english_italian",
                                "content_type": "parallel_text",
                                "educational_purpose": "high_quality_translation_examples",
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
            "successful_downloads": [d["dataset"]["name"] for d in successful_downloads],
            "collection_date": datetime.now().isoformat(),
            "source": "opus_parallel_corpus",
        }

        with open(output_path / "opus_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n🎉 OPUS Collection Summary:")
        print(f"   Total sentence pairs: {collection_stats['total_pairs']}")
        print(f"   High-quality pairs: {collection_stats['high_quality_pairs']}")
        print(f"   Successful datasets: {len(successful_downloads)}")
        print(f"   By difficulty:")
        for difficulty, count in collection_stats["by_difficulty"].items():
            print(f"     {difficulty}: {count}")

        if collection_stats["total_pairs"] > 0:
            print(f"\n💾 Saved to: {output_path}")
            print(f"📝 Content includes:")
            print(f"   - English-Italian parallel sentences")
            print(f"   - Quality assessment metrics")
            print(f"   - Multiple domain sources")
            print(f"   - Educational difficulty levels")


def main():
    parser = argparse.ArgumentParser(description="Collect Italian parallel texts from OPUS corpus")
    parser.add_argument("--output", required=True, help="Output directory for corpus content")
    parser.add_argument(
        "--max-pairs", type=int, help="Maximum number of sentence pairs per dataset"
    )

    args = parser.parse_args()

    collector = OpusCollector()
    collector.collect_all_opus_data(args.output, args.max_pairs)


if __name__ == "__main__":
    main()
