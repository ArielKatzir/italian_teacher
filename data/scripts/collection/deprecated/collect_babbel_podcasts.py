#!/usr/bin/env python3
"""
Babbel Podcast Transcript Collector

Collects Italian podcast transcripts from Babbel's Timeline Notation player.
Based on URL patterns discovered for "Oggi nella storia" and "La bottega di Babbel".

Usage:
    python collect_babbel_podcasts.py --output ../raw/babbel_content/
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup


class BabbelPodcastCollector:
    def __init__(self):
        """Initialize Babbel podcast collector."""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def extract_transcript_from_html(self, html_content: str) -> dict:
        """Extract transcript segments from Babbel timeline HTML."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract basic episode info
        title_tag = soup.find("title")
        title = title_tag.text.strip() if title_tag else "Unknown Episode"

        # Find all transcript blocks
        transcript_blocks = soup.find_all("div", class_="tag_block")

        segments = []

        for block in transcript_blocks:
            try:
                # Extract timestamp
                time_element = block.find("div", class_="tag_time")
                if not time_element:
                    continue

                timestamp_text = time_element.get_text(strip=True)
                # Remove play icon text if present
                timestamp_text = re.sub(r"[^0-9:]", "", timestamp_text)

                # Extract transcript text
                transcript_body = block.find("div", class_="transcript_tag_body")
                if not transcript_body:
                    continue

                text = transcript_body.get_text(strip=True)
                if not text or len(text) < 10:  # Skip very short segments
                    continue

                # Extract data attributes for additional info
                time_data = block.get("data-time", 0)
                segment_id = block.get("id", "").replace("tag_", "")

                segments.append(
                    {
                        "timestamp": timestamp_text,
                        "time_ms": int(time_data) if time_data else 0,
                        "text": text,
                        "segment_id": segment_id,
                    }
                )

            except Exception as e:
                print(f"Error processing transcript block: {e}")
                continue

        # Extract episode notes if available
        notes_sections = soup.find_all("strong")
        episode_notes = {}

        for strong_tag in notes_sections:
            if "Note dell'episodio" in strong_tag.get_text():
                # Find vocabulary definitions after the notes header
                parent = strong_tag.parent
                if parent:
                    # Extract vocabulary definitions
                    vocab_text = parent.get_text()
                    vocab_lines = vocab_text.split("\n")
                    vocab_dict = {}

                    for line in vocab_lines:
                        if ":" in line and len(line.split(":")) == 2:
                            term, definition = line.split(":", 1)
                            vocab_dict[term.strip()] = definition.strip()

                    if vocab_dict:
                        episode_notes["vocabulary"] = vocab_dict

        return {
            "title": title,
            "segments": segments,
            "episode_notes": episode_notes,
            "total_segments": len(segments),
            "collection_date": datetime.now().isoformat(),
        }

    def collect_oggi_nella_storia(self, output_dir: Path) -> int:
        """Collect 'Oggi nella storia' podcast episodes."""
        print("🎧 Collecting 'Oggi nella storia' episodes...")

        podcast_dir = output_dir / "oggi_nella_storia"
        podcast_dir.mkdir(exist_ok=True)

        # URL patterns based on user's discovery
        episodes = []

        # Introduction episode
        episodes.append(
            {
                "name": "introduction",
                "url": "https://player.timelinenotation.com/ogginellastoria/23506",
            }
        )

        # July 1-15 episodes (sequential IDs)
        for day in range(1, 16):
            episode_id = 23506 + day  # Start from 23507 (July 1)
            episodes.append(
                {
                    "name": f"july{day:02d}",
                    "url": f"https://player.timelinenotation.com/ogginellastoria/{episode_id}",
                }
            )

        # July 16-31 episodes (different ID range)
        for day in range(16, 32):
            episode_id = 23860 + (day - 16)  # Start from 23860 (July 16)
            episodes.append(
                {
                    "name": f"july{day:02d}",
                    "url": f"https://player.timelinenotation.com/ogginellastoria/{episode_id}",
                }
            )

        collected_count = 0

        for episode in episodes:
            print(f"📺 Collecting episode: {episode['name']}")

            try:
                response = self.session.get(episode["url"], timeout=30)
                response.raise_for_status()

                # Extract transcript
                transcript_data = self.extract_transcript_from_html(response.text)

                if transcript_data["segments"]:
                    # Add episode metadata
                    transcript_data.update(
                        {
                            "episode_name": episode["name"],
                            "episode_url": episode["url"],
                            "podcast_series": "Oggi nella storia",
                            "cefr_level": "B1",
                            "language": "italian",
                        }
                    )

                    # Save episode data
                    output_file = podcast_dir / f"{episode['name']}.json"
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

                    collected_count += 1
                    print(
                        f"✅ Saved {len(transcript_data['segments'])} segments from {episode['name']}"
                    )
                else:
                    print(f"❌ No transcript content found for {episode['name']}")

                # Rate limiting
                time.sleep(2)

            except Exception as e:
                print(f"❌ Error collecting {episode['name']}: {e}")
                continue

        return collected_count

    def collect_la_bottega_di_babbel(self, output_dir: Path) -> int:
        """Collect 'La bottega di Babbel' podcast episodes."""
        print("\n🎧 Collecting 'La bottega di Babbel' episodes...")

        podcast_dir = output_dir / "la_bottega_di_babbel"
        podcast_dir.mkdir(exist_ok=True)

        # First, get the episode list page
        try:
            episodes_url = "https://player.timelinenotation.com/labottegadibabbel/episodes"
            response = self.session.get(episodes_url, timeout=30)
            response.raise_for_status()

            # Parse episode links from the episodes page
            soup = BeautifulSoup(response.text, "html.parser")

            # Look for episode links
            episode_links = []

            # Find links that match the bottega pattern
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "labottegadibabbel" in href and href != episodes_url:
                    # Extract episode ID from URL
                    episode_id = href.split("/")[-1]
                    if episode_id.isdigit():
                        episode_links.append(
                            {
                                "name": f"episode_{episode_id}",
                                "url": (
                                    href
                                    if href.startswith("http")
                                    else f"https://player.timelinenotation.com{href}"
                                ),
                            }
                        )

            print(f"Found {len(episode_links)} episodes")

            collected_count = 0

            for episode in episode_links[:10]:  # Limit to first 10 episodes
                print(f"📺 Collecting episode: {episode['name']}")

                try:
                    response = self.session.get(episode["url"], timeout=30)
                    response.raise_for_status()

                    # Extract transcript
                    transcript_data = self.extract_transcript_from_html(response.text)

                    if transcript_data["segments"]:
                        # Add episode metadata
                        transcript_data.update(
                            {
                                "episode_name": episode["name"],
                                "episode_url": episode["url"],
                                "podcast_series": "La bottega di Babbel",
                                "cefr_level": "unknown",  # To be determined from content
                                "language": "italian",
                            }
                        )

                        # Save episode data
                        output_file = podcast_dir / f"{episode['name']}.json"
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(transcript_data, f, ensure_ascii=False, indent=2)

                        collected_count += 1
                        print(
                            f"✅ Saved {len(transcript_data['segments'])} segments from {episode['name']}"
                        )
                    else:
                        print(f"❌ No transcript content found for {episode['name']}")

                    # Rate limiting
                    time.sleep(2)

                except Exception as e:
                    print(f"❌ Error collecting {episode['name']}: {e}")
                    continue

            return collected_count

        except Exception as e:
            print(f"❌ Error accessing La bottega di Babbel episodes: {e}")
            return 0

    def collect_all(self, output_dir: str) -> None:
        """Collect all Babbel podcast content."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🇮🇹 Starting Babbel podcast collection...")

        # Collect both podcast series
        oggi_count = self.collect_oggi_nella_storia(output_path)
        bottega_count = self.collect_la_bottega_di_babbel(output_path)

        total_collected = oggi_count + bottega_count

        print(f"\n🎉 Collection Summary:")
        print(f"   Oggi nella storia: {oggi_count} episodes")
        print(f"   La bottega di Babbel: {bottega_count} episodes")
        print(f"   Total episodes: {total_collected}")

        # Save collection summary
        summary = {
            "oggi_nella_storia_episodes": oggi_count,
            "la_bottega_di_babbel_episodes": bottega_count,
            "total_episodes": total_collected,
            "collection_date": datetime.now().isoformat(),
            "source": "babbel_podcasts_timeline_notation",
        }

        with open(output_path / "babbel_podcast_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        if total_collected > 0:
            print(
                f"\n💾 Saved collection summary to: {output_path / 'babbel_podcast_summary.json'}"
            )
            print(f"\n📝 Sample content collected:")
            print(f"   - Italian historical content (B1 level)")
            print(f"   - Timestamped transcripts with teaching explanations")
            print(f"   - Vocabulary definitions and cultural context")
            print(f"   - Professional Italian language learning content")
        else:
            print("\n❌ No episodes collected. Check network connection and URLs.")


def main():
    parser = argparse.ArgumentParser(description="Collect Italian podcast transcripts from Babbel")
    parser.add_argument("--output", required=True, help="Output directory for podcast content")

    args = parser.parse_args()

    collector = BabbelPodcastCollector()
    collector.collect_all(args.output)


if __name__ == "__main__":
    main()
