#!/usr/bin/env python3
"""
Spoken Italian Datasets Collection Script
Collects datasets from Marco Giordano's spoken Italian collection.

Sources:
- GitHub: https://github.com/marco-giordano/spoken-italian-datasets
- Zenodo: DOI: 10.5281/zenodo.14246196
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpokenItalianCollector:
    """Collector for spoken Italian datasets from Marco Giordano's collection."""

    def __init__(self, output_dir: str = "data/raw/spoken_italian"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dataset sources
        self.github_api_url = "https://api.github.com/repos/marco-giordano/spoken-italian-datasets"
        self.zenodo_url = "https://zenodo.org/record/14246196/files/spoken_italian_datasets.json"
        self.sheets_url = "https://docs.google.com/spreadsheets/d/1BT0z_Qa7nKz5zOLJY8z_qNVjVGz_JXzOj8-ZKjfP1J0/export?format=csv"

    def fetch_dataset_inventory(self) -> Optional[List[Dict]]:
        """Fetch the complete dataset inventory from available sources."""
        datasets = []

        # Try Zenodo first (most reliable)
        try:
            logger.info("Fetching dataset inventory from Zenodo...")
            response = requests.get(self.zenodo_url, timeout=30)
            response.raise_for_status()
            zenodo_data = response.json()

            if isinstance(zenodo_data, list):
                datasets.extend(zenodo_data)
            elif isinstance(zenodo_data, dict) and "datasets" in zenodo_data:
                datasets.extend(zenodo_data["datasets"])

            logger.info(f"✅ Found {len(datasets)} datasets from Zenodo")

        except Exception as e:
            logger.warning(f"Could not fetch from Zenodo: {e}")

        # Try GitHub API as backup
        if not datasets:
            try:
                logger.info("Fetching from GitHub API...")
                response = requests.get(f"{self.github_api_url}/contents/datasets.json", timeout=30)
                response.raise_for_status()

                github_data = response.json()
                if "download_url" in github_data:
                    file_response = requests.get(github_data["download_url"], timeout=30)
                    file_response.raise_for_status()
                    datasets = file_response.json()

                logger.info(f"✅ Found {len(datasets)} datasets from GitHub")

            except Exception as e:
                logger.warning(f"Could not fetch from GitHub: {e}")

        return datasets if datasets else None

    def filter_educational_datasets(self, datasets: List[Dict]) -> List[Dict]:
        """Filter datasets suitable for Italian language education."""
        educational_datasets = []

        for dataset in datasets:
            # Look for educational indicators
            name = dataset.get("name", "").lower()
            description = dataset.get("description", "").lower()
            dataset.get("source", "").lower()
            speech_type = dataset.get("speech_type", "").lower()

            # Educational relevance criteria
            educational_keywords = [
                "education",
                "teaching",
                "learning",
                "formal",
                "structured",
                "conversation",
                "dialogue",
                "interview",
                "lesson",
                "class",
            ]

            # Exclude non-educational content
            exclude_keywords = [
                "music",
                "song",
                "noise",
                "ambient",
                "background",
                "technical",
                "medical",
                "legal",
                "advertisement",
            ]

            is_educational = any(
                keyword in description or keyword in speech_type for keyword in educational_keywords
            )
            is_excluded = any(
                keyword in description or keyword in name for keyword in exclude_keywords
            )

            # Quality criteria
            has_transcription = dataset.get("has_transcription", False)
            duration_hours = dataset.get("duration_hours", 0)
            is_quality = (
                duration_hours > 0.5 and has_transcription
            )  # At least 30 minutes with transcription

            if is_educational and not is_excluded and is_quality:
                # Add educational score
                dataset["educational_score"] = self._calculate_educational_score(dataset)
                educational_datasets.append(dataset)

        # Sort by educational relevance
        educational_datasets.sort(key=lambda x: x["educational_score"], reverse=True)

        logger.info(f"✅ Filtered to {len(educational_datasets)} educational datasets")
        return educational_datasets

    def _calculate_educational_score(self, dataset: Dict) -> float:
        """Calculate educational relevance score for a dataset."""
        score = 0.0

        # Base score from content type
        speech_type = dataset.get("speech_type", "").lower()
        if "conversation" in speech_type or "dialogue" in speech_type:
            score += 3.0
        elif "interview" in speech_type:
            score += 2.5
        elif "formal" in speech_type:
            score += 2.0

        # Bonus for educational context
        source = dataset.get("source", "").lower()
        if "education" in source or "university" in source:
            score += 2.0

        # Quality indicators
        if dataset.get("has_transcription", False):
            score += 1.5
        if dataset.get("has_metadata", False):
            score += 1.0

        # Size bonus (but not too much)
        duration = dataset.get("duration_hours", 0)
        if 1 <= duration <= 20:  # Sweet spot for educational content
            score += min(duration * 0.1, 1.0)

        # Format bonus
        format_type = dataset.get("format", "").lower()
        if format_type in ["wav", "mp3", "flac"]:
            score += 0.5

        return score

    def create_teaching_conversations_from_metadata(self, datasets: List[Dict]) -> List[Dict]:
        """Create teaching conversations based on dataset metadata."""
        conversations = []

        for i, dataset in enumerate(datasets[:20]):  # Limit to top 20 datasets
            name = dataset.get("name", f"Dataset {i}")
            description = dataset.get("description", "Italian spoken content")
            speech_type = dataset.get("speech_type", "conversation")
            duration = dataset.get("duration_hours", 0)
            source = dataset.get("source", "unknown")

            # Create conversation about the dataset content
            conversations.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Can you tell me about Italian {speech_type}? I want to practice listening.",
                        },
                        {
                            "role": "assistant",
                            "content": f"Excellent! {speech_type.capitalize()} is great for practicing Italian. This type of content typically includes natural Italian speech patterns and authentic expressions. {description}",
                        },
                    ],
                    "metadata": {
                        "conversation_id": f"spoken_italian_{i}",
                        "source": "spoken_italian_datasets",
                        "level": "B1",  # Listening is typically intermediate
                        "topic": "listening_practice",
                        "conversation_type": "listening_guidance",
                        "dataset_info": {
                            "name": name,
                            "duration_hours": duration,
                            "source": source,
                            "speech_type": speech_type,
                        },
                    },
                }
            )

            # Create vocabulary conversation if description is rich
            if len(description) > 50:
                conversations.append(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": f"What vocabulary should I focus on for {speech_type} in Italian?",
                            },
                            {
                                "role": "assistant",
                                "content": f"For {speech_type}, you should focus on authentic Italian expressions and colloquial language. This will help you understand natural speech patterns.",
                            },
                        ],
                        "metadata": {
                            "conversation_id": f"spoken_vocab_{i}",
                            "source": "spoken_italian_datasets",
                            "level": "B2",
                            "topic": "vocabulary",
                            "conversation_type": "vocabulary_guidance",
                        },
                    }
                )

        logger.info(f"✅ Created {len(conversations)} teaching conversations from spoken datasets")
        return conversations

    def save_dataset_inventory(self, datasets: List[Dict]) -> None:
        """Save the filtered dataset inventory."""
        inventory_file = self.output_dir / "educational_datasets_inventory.json"

        with open(inventory_file, "w", encoding="utf-8") as f:
            json.dump(datasets, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ Saved {len(datasets)} educational datasets to {inventory_file}")

        # Create CSV summary
        csv_file = self.output_dir / "educational_datasets_summary.csv"
        summary_data = []

        for dataset in datasets:
            summary_data.append(
                {
                    "name": dataset.get("name", ""),
                    "duration_hours": dataset.get("duration_hours", 0),
                    "speech_type": dataset.get("speech_type", ""),
                    "source": dataset.get("source", ""),
                    "has_transcription": dataset.get("has_transcription", False),
                    "educational_score": dataset.get("educational_score", 0),
                    "description": (
                        dataset.get("description", "")[:100] + "..."
                        if len(dataset.get("description", "")) > 100
                        else dataset.get("description", "")
                    ),
                }
            )

        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)

        logger.info(f"✅ Saved dataset summary to {csv_file}")

    def save_conversations(self, conversations: List[Dict]) -> None:
        """Save generated conversations."""
        conversations_file = self.output_dir / "spoken_italian_conversations.jsonl"

        with open(conversations_file, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"✅ Saved {len(conversations)} conversations to {conversations_file}")

    def collect(self) -> bool:
        """Main collection method."""
        logger.info("🚀 Starting spoken Italian datasets collection...")

        # Fetch dataset inventory
        datasets = self.fetch_dataset_inventory()

        if not datasets:
            logger.error("❌ Could not fetch dataset inventory")
            return False

        # Filter for educational content
        educational_datasets = self.filter_educational_datasets(datasets)

        if not educational_datasets:
            logger.error("❌ No educational datasets found")
            return False

        # Save inventory
        self.save_dataset_inventory(educational_datasets)

        # Create teaching conversations
        conversations = self.create_teaching_conversations_from_metadata(educational_datasets)

        if conversations:
            self.save_conversations(conversations)

        logger.info("🎉 Spoken Italian datasets collection completed!")
        return True


def main():
    """Main execution function."""
    collector = SpokenItalianCollector()

    try:
        success = collector.collect()
        if success:
            print("✅ Spoken Italian datasets collection completed!")
            print(f"📁 Data saved to: {collector.output_dir}")
        else:
            print("❌ Collection failed!")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print("❌ Collection failed with fatal error")


if __name__ == "__main__":
    main()
