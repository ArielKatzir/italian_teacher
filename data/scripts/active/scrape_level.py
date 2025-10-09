#!/usr/bin/env python3
"""
Scrape exercises for a specific CEFR level from onlineitalianclub.com
Usage: python scrape_level.py <level> (e.g., A1, A2, B1, B2, C1, C2)
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def extract_exercise_links(html_content):
    """Extract exercise URLs from level index page."""
    # Pattern: href="url" where url contains free_italian_exercises
    pattern = r'href="(https://onlineitalianclub\.com/free_italian_exercises/[^"]+\.html)"'
    links = re.findall(pattern, html_content)

    # Remove duplicates while preserving order
    seen = set()
    unique_links = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)

    return unique_links


def download_html(url, output_dir, delay=2):
    """Download HTML file with polite delay."""
    filename = Path(urlparse(url).path).name
    output_path = output_dir / filename

    # Skip if already downloaded
    if output_path.exists():
        print(f"⏭️  Skip: {filename} (already exists)")
        return str(output_path)

    try:
        # Be polite - add delay and user agent
        time.sleep(delay)

        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0 (Educational Language Learning Bot)"}
        )

        with urllib.request.urlopen(req, timeout=10) as response:
            html_content = response.read()

        # Save to file
        with open(output_path, "wb") as f:
            f.write(html_content)

        print(f"✅ Downloaded: {filename}")
        return str(output_path)

    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python scrape_level.py <level> (e.g., A1, A2, B2, C1, C2)")
        sys.exit(1)

    level = sys.argv[1].lower()

    # Setup paths
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "raw"
    output_dir = raw_dir / f"onlineitalianclub_{level}_exercises"
    output_dir.mkdir(exist_ok=True)

    # Read index page
    index_file = raw_dir / f"onlineitalianclub_{level}_index.html"

    if not index_file.exists():
        print(f"Error: Index file not found: {index_file}")
        print(f"Please download it first from onlineitalianclub.com")
        sys.exit(1)

    # Try different encodings
    html_content = None
    for encoding in ["utf-8", "windows-1252", "latin-1"]:
        try:
            with open(index_file, "r", encoding=encoding) as f:
                html_content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if not html_content:
        print("Error: Could not read index file")
        sys.exit(1)

    # Extract exercise links
    links = extract_exercise_links(html_content)
    print(f"\n📋 Found {len(links)} exercise links for {level.upper()}\n")

    if len(links) == 0:
        print(f"⚠️  No exercises found in {index_file}")
        print(
            "This might mean the page structure is different or there are no JS exercises at this level"
        )
        sys.exit(0)

    # Download each exercise
    downloaded = []
    for i, url in enumerate(links, 1):
        print(f"[{i}/{len(links)}] ", end="")
        filepath = download_html(url, output_dir, delay=2)
        if filepath:
            downloaded.append(filepath)

    # Save list of downloaded files
    manifest = output_dir / "downloaded_exercises.json"
    with open(manifest, "w") as f:
        json.dump(
            {"level": level.upper(), "total": len(downloaded), "files": downloaded, "links": links},
            f,
            indent=2,
        )

    print(f"\n\n✅ Complete! Downloaded {len(downloaded)}/{len(links)} {level.upper()} exercises")
    print(f"📁 Saved to: {output_dir}")
    print(f"📄 Manifest: {manifest}")


if __name__ == "__main__":
    main()
