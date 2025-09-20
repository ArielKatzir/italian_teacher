#!/usr/bin/env python3
"""
Setup script for data collection infrastructure.
"""

from pathlib import Path


def setup_collection_environment():
    """Set up the data collection environment."""
    print("Setting up data collection environment...")

    # Create necessary directories
    dirs = ["raw/babbel_content", "raw/tatoeba_content", "raw/italiano_bello_content"]

    for dir_name in dirs:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_name}")

    print("Setup complete!")


if __name__ == "__main__":
    setup_collection_environment()
