#!/usr/bin/env python3
"""
Batch extract exercises from downloaded HTML files.
"""

import json
import sys
from pathlib import Path

from parse_onlineitalianclub import JavaScriptExerciseParser, extract_metadata_from_filename


def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    exercises_dir = base_dir / "raw" / "onlineitalianclub_exercises"
    manifest_file = exercises_dir / "downloaded_exercises.json"

    if not manifest_file.exists():
        print(f"Error: Manifest not found: {manifest_file}")
        sys.exit(1)

    # Load manifest
    with open(manifest_file) as f:
        manifest = json.load(f)

    html_files = [Path(fp) for fp in manifest["files"]]
    print(f"📋 Processing {len(html_files)} exercise files...\n")

    # Parse all exercises
    parser = JavaScriptExerciseParser()
    all_exercises = []
    stats = {"success": 0, "failed": 0, "total_exercises": 0}

    for i, html_file in enumerate(html_files, 1):
        try:
            # Read HTML
            html_content = None
            for encoding in ["utf-8", "windows-1252", "latin-1"]:
                try:
                    with open(html_file, "r", encoding=encoding) as f:
                        html_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if not html_content:
                print(f"[{i}/{len(html_files)}] ❌ {html_file.name} - encoding error")
                stats["failed"] += 1
                continue

            # Check if it has JavaScript format
            if "var question" in html_content:
                exercises = parser.parse(html_content)

                if exercises:
                    metadata = extract_metadata_from_filename(html_file)
                    metadata["filename"] = html_file.name
                    metadata["cefr_level"] = "B1"
                    metadata["source"] = "onlineitalianclub.com"

                    # Detect exercise type
                    has_options = any("options" in ex for ex in exercises)
                    exercise_type = "multiple_choice" if has_options else "fill_in_blank"

                    all_exercises.append(
                        {
                            "metadata": metadata,
                            "exercise_type": exercise_type,
                            "exercises": exercises,
                        }
                    )

                    stats["success"] += 1
                    stats["total_exercises"] += len(exercises)
                    print(
                        f"[{i}/{len(html_files)}] ✅ {html_file.name} - {len(exercises)} exercises"
                    )
                else:
                    print(f"[{i}/{len(html_files)}] ⚠️  {html_file.name} - no exercises found")
                    stats["failed"] += 1
            else:
                print(
                    f"[{i}/{len(html_files)}] ⏭️  {html_file.name} - different format (skip for now)"
                )
                stats["failed"] += 1

        except Exception as e:
            print(f"[{i}/{len(html_files)}] ❌ {html_file.name} - {e}")
            stats["failed"] += 1

    # Save extracted exercises
    output_file = base_dir / "datasets" / "v4" / "onlineitalianclub_b1_extracted.json"
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_exercises, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n" + "=" * 60)
    print(f"📊 EXTRACTION SUMMARY")
    print(f"=" * 60)
    print(f"✅ Successfully extracted: {stats['success']} files")
    print(f"❌ Failed/Skipped: {stats['failed']} files")
    print(f"📝 Total exercises: {stats['total_exercises']}")
    print(f"💾 Saved to: {output_file}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
