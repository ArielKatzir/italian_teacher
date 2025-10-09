#!/bin/bash

# Scrape all CEFR levels from onlineitalianclub.com
# Usage: bash scrape_all_levels.sh

LEVELS=("a1" "a2" "b2" "c1" "c2")
BASE_URL="https://onlineitalianclub.com"

echo "🌐 Downloading index pages for all CEFR levels..."

for level in "${LEVELS[@]}"; do
    case $level in
        "a1")
            url="${BASE_URL}/online-italian-course-beginner-level-a1/"
            ;;
        "a2")
            url="${BASE_URL}/online-italian-course-pre-intermediate-level-a2/"
            ;;
        "b2")
            url="${BASE_URL}/online-italian-course-upper-intermediate-b2/"
            ;;
        "c1")
            url="${BASE_URL}/online-italian-course-advanced-c1/"
            ;;
        "c2")
            url="${BASE_URL}/online-italian-course-proficient-c2/"
            ;;
    esac
    
    output="data/raw/onlineitalianclub_${level}_index.html"
    echo "📥 Downloading $level index..."
    curl -s "$url" -o "$output"
    sleep 2
done

echo "✅ Downloaded all index pages"
