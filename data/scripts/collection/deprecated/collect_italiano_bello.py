#!/usr/bin/env python3
"""
Italiano Bello Course Collector

Collects Italian course content from Italiano-Bello.com.
Free online Italian courses organized by CEFR levels (A1-C2).

Usage:
    python collect_italiano_bello.py --output ../raw/italiano_bello_content/
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


class ItalianoBelloCollector:
    def __init__(self):
        """Initialize Italiano Bello collector."""
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )
        self.base_url = "https://italiano-bello.com"

    def discover_course_structure(self) -> dict:
        """Discover the course structure and available content."""
        courses_url = f"{self.base_url}/en/italian-courses/"

        try:
            response = self.session.get(courses_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            course_structure = {"levels": {}, "lessons": [], "dialogues": []}

            # Look for level-based organization (A1, A2, B1, B2, C1, C2)
            level_patterns = ["A1", "A2", "B1", "B2", "C1", "C2"]

            for level in level_patterns:
                # Find sections or links related to this level
                level_elements = soup.find_all(
                    ["a", "div", "section"], string=re.compile(level, re.IGNORECASE)
                )

                level_links = []
                for element in level_elements:
                    # If it's a link, add it directly
                    if element.name == "a" and element.get("href"):
                        href = element["href"]
                        full_url = urljoin(self.base_url, href)
                        level_links.append({"title": element.get_text(strip=True), "url": full_url})
                    else:
                        # Look for links within this element
                        links = element.find_all("a", href=True)
                        for link in links:
                            href = link["href"]
                            full_url = urljoin(self.base_url, href)
                            level_links.append(
                                {"title": link.get_text(strip=True), "url": full_url}
                            )

                if level_links:
                    course_structure["levels"][level] = level_links

            # Look for general lesson or course links
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                href = link["href"]
                link_text = link.get_text(strip=True).lower()

                # Skip navigation and footer links
                if any(
                    skip_word in href.lower()
                    for skip_word in ["#", "javascript:", "mailto:", "tel:"]
                ):
                    continue

                # Look for course/lesson related content
                if any(
                    course_word in link_text
                    for course_word in ["lesson", "lezione", "corso", "course", "unit", "unità"]
                ):
                    full_url = urljoin(self.base_url, href)
                    course_structure["lessons"].append(
                        {"title": link.get_text(strip=True), "url": full_url}
                    )

                # Look for dialogue content
                if any(
                    dialogue_word in link_text
                    for dialogue_word in ["dialogue", "dialogo", "conversation", "conversazione"]
                ):
                    full_url = urljoin(self.base_url, href)
                    course_structure["dialogues"].append(
                        {"title": link.get_text(strip=True), "url": full_url}
                    )

            print(f"Discovered course structure:")
            print(f"  Levels found: {list(course_structure['levels'].keys())}")
            print(f"  Total lessons: {len(course_structure['lessons'])}")
            print(f"  Total dialogues: {len(course_structure['dialogues'])}")

            return course_structure

        except Exception as e:
            print(f"Error discovering course structure: {e}")
            return {}

    def extract_lesson_content(self, url: str, lesson_type: str = "lesson") -> dict:
        """Extract content from a lesson or dialogue page."""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title_tag = soup.find("title") or soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "Unknown Lesson"

            # Extract main content
            content_selectors = [
                ".lesson-content",
                ".course-content",
                ".main-content",
                "article",
                ".content",
                "#content",
                ".entry-content",
            ]

            content_area = None
            for selector in content_selectors:
                content_area = soup.select_one(selector)
                if content_area:
                    break

            if not content_area:
                # Fallback to main tag or largest content block
                content_area = soup.find("main") or soup.find("div", id="main")

            if not content_area:
                print(f"Could not find content area for {url}")
                return {}

            # Extract Italian text content
            italian_content = []
            english_content = []

            for para in content_area.find_all(["p", "div", "span"]):
                text = para.get_text(strip=True)
                if text and len(text) > 10:
                    # Detect Italian vs English content
                    italian_indicators = [
                        "il",
                        "la",
                        "le",
                        "gli",
                        "di",
                        "da",
                        "in",
                        "con",
                        "per",
                        "sono",
                        "è",
                        "che",
                        "come",
                        "dove",
                        "quando",
                    ]
                    english_indicators = [
                        "the",
                        "and",
                        "or",
                        "but",
                        "this",
                        "that",
                        "with",
                        "for",
                        "from",
                        "they",
                        "have",
                        "will",
                    ]

                    italian_score = sum(
                        1 for word in italian_indicators if word in text.lower().split()
                    )
                    english_score = sum(
                        1 for word in english_indicators if word in text.lower().split()
                    )

                    if italian_score > english_score:
                        italian_content.append(text)
                    elif english_score > 0:
                        english_content.append(text)

            # Extract vocabulary
            vocabulary = {}
            vocab_sections = soup.find_all(
                ["div", "section"], class_=re.compile("vocab|dictionary|glossary", re.IGNORECASE)
            )
            for section in vocab_sections:
                for item in section.find_all(["li", "p", "tr"]):
                    item_text = item.get_text(strip=True)
                    if ":" in item_text or "=" in item_text:
                        parts = re.split("[=:]", item_text, 1)
                        if len(parts) == 2:
                            vocabulary[parts[0].strip()] = parts[1].strip()

            # Extract audio or media references
            media_urls = []
            for media_tag in soup.find_all(["audio", "video", "source"]):
                if media_tag.get("src"):
                    media_urls.append(urljoin(url, media_tag["src"]))

            return {
                "title": title,
                "url": url,
                "content_type": lesson_type,
                "italian_content": italian_content,
                "english_content": english_content,
                "vocabulary": vocabulary,
                "media_urls": media_urls,
                "total_italian_segments": len(italian_content),
                "total_english_segments": len(english_content),
                "collection_date": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return {}

    def collect_all_content(self, output_dir: str) -> None:
        """Collect all Italiano Bello course content."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("🇮🇹 Starting Italiano Bello content collection...")

        # Discover course structure
        course_structure = self.discover_course_structure()

        if not course_structure:
            print("❌ Could not discover course structure")
            return

        collected_counts = {"levels": 0, "lessons": 0, "dialogues": 0}

        # Collect level-based content
        for level, level_content in course_structure.get("levels", {}).items():
            if not level_content:
                continue

            print(f"\n📚 Collecting {level} level content...")

            level_dir = output_path / level.lower()
            level_dir.mkdir(exist_ok=True)

            for i, content_info in enumerate(level_content[:10], 1):  # Limit to 10 per level
                print(f"📖 Collecting {level} content {i}: {content_info['title'][:50]}...")

                try:
                    content = self.extract_lesson_content(content_info["url"], f"{level}_content")

                    if content and (content.get("italian_content") or content.get("vocabulary")):
                        # Add metadata
                        content.update(
                            {
                                "level": level,
                                "source": "italiano_bello",
                                "language": "italian",
                                "educational_purpose": "structured_learning",
                            }
                        )

                        # Save to file
                        filename = f"{level.lower()}_content_{i:02d}.json"
                        output_file = level_dir / filename

                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(content, f, ensure_ascii=False, indent=2)

                        collected_counts["levels"] += 1
                        print(f"✅ Saved {content['total_italian_segments']} Italian segments")
                    else:
                        print(f"❌ No content found")

                    time.sleep(2)

                except Exception as e:
                    print(f"❌ Error collecting {content_info['title']}: {e}")
                    continue

        # Collect general lessons
        if course_structure.get("lessons"):
            print(f"\n📝 Collecting general lessons...")

            lessons_dir = output_path / "lessons"
            lessons_dir.mkdir(exist_ok=True)

            for i, lesson_info in enumerate(
                course_structure["lessons"][:15], 1
            ):  # Limit to 15 lessons
                print(f"📝 Collecting lesson {i}: {lesson_info['title'][:50]}...")

                try:
                    content = self.extract_lesson_content(lesson_info["url"], "lesson")

                    if content and (content.get("italian_content") or content.get("vocabulary")):
                        content.update(
                            {
                                "source": "italiano_bello",
                                "language": "italian",
                                "educational_purpose": "lesson_content",
                            }
                        )

                        filename = f"lesson_{i:02d}.json"
                        output_file = lessons_dir / filename

                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(content, f, ensure_ascii=False, indent=2)

                        collected_counts["lessons"] += 1
                        print(f"✅ Saved lesson content")

                    time.sleep(2)

                except Exception as e:
                    print(f"❌ Error collecting lesson: {e}")
                    continue

        # Save collection summary
        summary = {
            "content_collected_by_type": collected_counts,
            "total_content_collected": sum(collected_counts.values()),
            "collection_date": datetime.now().isoformat(),
            "source": "italiano_bello_courses",
            "base_url": self.base_url,
        }

        with open(output_path / "italiano_bello_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        total_collected = sum(collected_counts.values())

        print(f"\n🎉 Collection Summary:")
        for content_type, count in collected_counts.items():
            print(f"   {content_type}: {count} items collected")
        print(f"   Total: {total_collected} items collected")

        if total_collected > 0:
            print(f"\n💾 Saved to: {output_path}")
            print(f"📚 Content includes:")
            print(f"   - Structured Italian course content")
            print(f"   - CEFR level-organized materials")
            print(f"   - Vocabulary and explanations")
            print(f"   - Progressive learning content")


def main():
    parser = argparse.ArgumentParser(description="Collect Italian courses from Italiano-Bello.com")
    parser.add_argument("--output", required=True, help="Output directory for course content")

    args = parser.parse_args()

    collector = ItalianoBelloCollector()
    collector.collect_all_content(args.output)


if __name__ == "__main__":
    main()
