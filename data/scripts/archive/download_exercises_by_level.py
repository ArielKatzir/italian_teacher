#!/usr/bin/env python3
"""
Download Italian exercises from onlineitalianclub.com for a specific CEFR level.
"""

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen, urlretrieve

LEVEL_URLS = {
    "A1": "https://onlineitalianclub.com/online-italian-course-beginner-level-a1/",
    "A2": "https://onlineitalianclub.com/online-italian-course-pre-intermediate-level-a2/",
    "B1": "https://onlineitalianclub.com/free-b1-intermediate-italian-exercises/",
    "B2": "https://onlineitalianclub.com/online-italian-course-upper-intermediate-b2/",
    "C1": "https://onlineitalianclub.com/online-italian-course-advanced-c1/",
    "C2": "https://onlineitalianclub.com/online-italian-course-proficient-c2/",
}


def extract_exercise_links(html_content, base_url):
    """Extract exercise links from the index page."""
    # Look for links to .html files in the free_italian_exercises directory
    pattern = r'href="([^"]*free_italian_exercises/[^"]+\.html)"'
    links = re.findall(pattern, html_content)

    # Also look for vocabulary exercise links
    vocab_pattern = r'href="([^"]*vocabulary-[^"]+/)"'
    vocab_links = re.findall(vocab_pattern, html_content)

    all_links = links + vocab_links

    # Convert to absolute URLs and remove duplicates
    absolute_links = []
    seen = set()
    for link in all_links:
        abs_url = urljoin(base_url, link)
        if abs_url not in seen:
            seen.add(abs_url)
            absolute_links.append(abs_url)

    return absolute_links


def download_exercises(level):
    """Download all exercises for a given level."""
    if level not in LEVEL_URLS:
        print(f"Error: Unknown level '{level}'. Valid levels: {', '.join(LEVEL_URLS.keys())}")
        sys.exit(1)

    # Setup paths
    base_dir = Path(__file__).parent.parent
    exercises_dir = base_dir / "raw" / f"onlineitalianclub_{level.lower()}_exercises"
    exercises_dir.mkdir(exist_ok=True, parents=True)

    # Fetch index page
    index_url = LEVEL_URLS[level]
    print(f"Fetching index page for {level}: {index_url}")

    try:
        with urlopen(index_url) as response:
            index_html = response.read().decode("utf-8")
    except Exception as e:
        print(f"Error fetching index page: {e}")
        sys.exit(1)

    # Extract exercise links
    exercise_links = extract_exercise_links(index_html, index_url)
    print(f"Found {len(exercise_links)} exercise links for {level}\n")

    # Download each exercise
    downloaded_files = []
    for i, url in enumerate(exercise_links, 1):
        # Determine filename
        filename = url.rstrip("/").split("/")[-1]
        if not filename.endswith(".html"):
            filename += ".html"

        filepath = exercises_dir / filename

        # Skip if already downloaded
        if filepath.exists():
            print(f"[{i}/{len(exercise_links)}] ⏭️  {filename} - already exists")
            downloaded_files.append(str(filepath))
            continue

        # Download with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(
                    f"[{i}/{len(exercise_links)}] ⬇️  Downloading {filename}...", end="", flush=True
                )
                urlretrieve(url, filepath)
                downloaded_files.append(str(filepath))
                print(f" ✅")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f" ⚠️  Failed (attempt {attempt+1}/{max_retries}), retrying...")
                    time.sleep(2)
                else:
                    print(f" ❌ Failed after {max_retries} attempts: {e}")

        # Be polite - wait 2 seconds between downloads
        if i < len(exercise_links):
            time.sleep(2)

    # Save manifest
    manifest = {
        "level": level,
        "index_url": index_url,
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_exercises": len(exercise_links),
        "files": downloaded_files,
    }

    manifest_file = exercises_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"✅ Download complete for {level}")
    print(f"📁 Directory: {exercises_dir}")
    print(f"📝 Downloaded: {len(downloaded_files)} / {len(exercise_links)} exercises")
    print(f"💾 Manifest: {manifest_file}")
    print(f"=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python download_exercises_by_level.py <level>")
        print(f"Valid levels: {', '.join(LEVEL_URLS.keys())}")
        sys.exit(1)

    level = sys.argv[1].upper()
    download_exercises(level)


if __name__ == "__main__":
    main()
