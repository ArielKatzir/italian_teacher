#!/usr/bin/env python3
"""
OnlineItalianClub Dialogue Collector

Collects Italian conversation dialogues from OnlineItalianClub.com.
Free educational content with 60+ dialogues organized by difficulty.

Usage:
    python collect_onlineitalianclub.py --output ../raw/onlineitalianclub_content/
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


class OnlineItalianClubCollector:
    def __init__(self):
        """Initialize OnlineItalianClub collector."""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self.base_url = "https://onlineitalianclub.com"

    def extract_dialogue_links(self) -> list:
        """Extract all dialogue links from the conversations index page."""
        conversations_url = f"{self.base_url}/italian-conversations/"

        try:
            response = self.session.get(conversations_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            dialogue_links = []

            # Look for numbered list of conversations
            # Find all links that contain "italian-conversation"
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "italian-conversation-" in href and href not in [conversations_url]:
                    full_url = urljoin(self.base_url, href)
                    title = link.get_text(strip=True)

                    # Extract conversation number from URL
                    match = re.search(r"italian-conversation-(\d+)", href)
                    conversation_num = int(match.group(1)) if match else 0

                    dialogue_links.append(
                        {"conversation_number": conversation_num, "title": title, "url": full_url}
                    )

            # Sort by conversation number
            dialogue_links.sort(key=lambda x: x["conversation_number"])

            print(f"Found {len(dialogue_links)} dialogue links")
            return dialogue_links

        except Exception as e:
            print(f"Error extracting dialogue links: {e}")
            return []

    def extract_dialogue_content(self, url: str) -> dict:
        """Extract dialogue content from a single conversation page."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract page title
            title_tag = soup.find("title")
            title = title_tag.text.strip() if title_tag else "Unknown Conversation"

            # Extract main content area
            content_area = (
                soup.find("div", class_="entry-content")
                or soup.find("main")
                or soup.find("article")
            )

            if not content_area:
                print(f"Could not find content area for {url}")
                return {}

            # Extract dialogue text - look for Italian text blocks
            dialogue_segments = []

            # Look for dialogue patterns (often in <p> tags or dialogue blocks)
            paragraphs = content_area.find_all(["p", "div"])

            for para in paragraphs:
                text = para.get_text(strip=True)
                if text and len(text) > 10:  # Filter out very short segments
                    # Check if this looks like Italian dialogue
                    if any(
                        italian_word in text.lower()
                        for italian_word in [
                            "ciao",
                            "come",
                            "che",
                            "cosa",
                            "dove",
                            "quando",
                            "perché",
                            "sono",
                            "hai",
                            "ho",
                        ]
                    ):
                        dialogue_segments.append(
                            {"text": text, "type": "dialogue" if ":" in text else "narrative"}
                        )

            # Extract any vocabulary or notes sections
            vocabulary = {}
            notes = []

            # Look for vocabulary lists or definitions
            for elem in content_area.find_all(["ul", "ol", "dl"]):
                items = elem.find_all("li") or elem.find_all("dt")
                for item in items:
                    item_text = item.get_text(strip=True)
                    if ":" in item_text or "=" in item_text:
                        parts = re.split("[=:]", item_text, 1)
                        if len(parts) == 2:
                            vocabulary[parts[0].strip()] = parts[1].strip()
                    elif len(item_text) > 5:
                        notes.append(item_text)

            # Extract audio references if available
            audio_urls = []
            for audio_tag in soup.find_all(["audio", "source"]):
                if audio_tag.get("src"):
                    audio_urls.append(urljoin(url, audio_tag["src"]))

            return {
                "title": title,
                "url": url,
                "dialogue_segments": dialogue_segments,
                "vocabulary": vocabulary,
                "notes": notes,
                "audio_urls": audio_urls,
                "total_segments": len(dialogue_segments),
                "collection_date": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return {}

    def collect_all_dialogues(self, output_dir: str) -> None:
        """Collect all OnlineItalianClub dialogues."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🇮🇹 Starting OnlineItalianClub dialogue collection...")

        # Get all dialogue links
        dialogue_links = self.extract_dialogue_links()

        if not dialogue_links:
            print("❌ No dialogue links found")
            return

        collected_count = 0
        failed_count = 0

        for dialogue in dialogue_links:
            print(
                f"📝 Collecting conversation {dialogue['conversation_number']}: {dialogue['title'][:50]}..."
            )

            try:
                content = self.extract_dialogue_content(dialogue["url"])

                if content and content.get("dialogue_segments"):
                    # Add metadata
                    content.update(
                        {
                            "conversation_number": dialogue["conversation_number"],
                            "source": "onlineitalianclub",
                            "language": "italian",
                            "content_type": "dialogue",
                            "educational_purpose": "conversation_practice",
                        }
                    )

                    # Save to file
                    filename = f"conversation_{dialogue['conversation_number']:02d}.json"
                    output_file = output_path / filename

                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(content, f, ensure_ascii=False, indent=2)

                    collected_count += 1
                    print(f"✅ Saved {len(content['dialogue_segments'])} segments")
                else:
                    print(f"❌ No content found for conversation {dialogue['conversation_number']}")
                    failed_count += 1

                # Rate limiting
                time.sleep(2)

            except Exception as e:
                print(f"❌ Error collecting conversation {dialogue['conversation_number']}: {e}")
                failed_count += 1
                continue

        # Save collection summary
        summary = {
            "total_dialogues_collected": collected_count,
            "total_dialogues_failed": failed_count,
            "total_dialogues_attempted": len(dialogue_links),
            "collection_date": datetime.now().isoformat(),
            "source": "onlineitalianclub_dialogues",
            "base_url": self.base_url,
        }

        with open(output_path / "onlineitalianclub_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n🎉 Collection Summary:")
        print(f"   Successfully collected: {collected_count} dialogues")
        print(f"   Failed: {failed_count} dialogues")
        print(f"   Total attempted: {len(dialogue_links)} dialogues")

        if collected_count > 0:
            print(f"\n💾 Saved to: {output_path}")
            print(f"📝 Content includes:")
            print(f"   - Italian conversation dialogues")
            print(f"   - Vocabulary definitions")
            print(f"   - Educational notes and explanations")
            print(f"   - Progressive difficulty levels")


def main():
    parser = argparse.ArgumentParser(description="Collect Italian dialogues from OnlineItalianClub")
    parser.add_argument("--output", required=True, help="Output directory for dialogue content")

    args = parser.parse_args()

    collector = OnlineItalianClubCollector()
    collector.collect_all_dialogues(args.output)


if __name__ == "__main__":
    main()
