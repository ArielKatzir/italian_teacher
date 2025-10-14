"""
Italian vocabulary lists with CEFR level tagging.

Loads comprehensive vocabulary from italian_vocabulary.json (16,887 words).
Source: https://github.com/vbvss199/Language-Learning-decks
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Cache for loaded vocabulary
_VOCABULARY_CACHE: Dict[str, Set[str]] = {}
_FULL_VOCABULARY: List[Dict] = []


def load_italian_vocabulary(force_reload: bool = False) -> List[Dict]:
    """
    Load full Italian vocabulary with CEFR tags.

    Returns:
        List of vocabulary entries with fields:
        - word: str
        - cefr_level: str (A1, A2, B1, B2, C1, C2)
        - english_translation: str
        - example_sentence_native: str
        - example_sentence_english: str
        - pos: str (noun, verb, adjective, adverb, etc.)
        - word_frequency: int
    """
    global _FULL_VOCABULARY

    if _FULL_VOCABULARY and not force_reload:
        return _FULL_VOCABULARY

    vocab_path = Path(__file__).parent / "italian_vocabulary.json"

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Italian vocabulary file not found at {vocab_path}\n"
            f"Download from: https://raw.githubusercontent.com/vbvss199/Language-Learning-decks/refs/heads/main/italian/italian.json"
        )

    with open(vocab_path, "r", encoding="utf-8") as f:
        _FULL_VOCABULARY = json.load(f)

    print(f"✅ Loaded {len(_FULL_VOCABULARY)} Italian words from vocabulary list")

    return _FULL_VOCABULARY


def get_vocabulary_by_cefr(
    level: str = None, cumulative: bool = True, pos_filter: str = None, force_reload: bool = False
) -> Set[str]:
    """
    Get Italian vocabulary words filtered by CEFR level.

    Args:
        level: CEFR level (A1, A2, B1, B2, C1, C2). If None, returns all words.
        cumulative: If True, includes all lower levels (e.g., B1 includes A1+A2+B1).
                   If False, only returns words from exact level specified.
        pos_filter: Filter by part of speech (noun, verb, adjective, adverb, etc.)
        force_reload: Force reload from JSON file

    Returns:
        Set of Italian words at specified level(s)

    Example:
        >>> vocab = get_vocabulary_by_cefr("B1", cumulative=True)
        >>> "casa" in vocab  # A1 word
        True
        >>> "esperienza" in vocab  # B1 word
        True
        >>> "paradigma" in vocab  # C2 word
        False
    """
    global _VOCABULARY_CACHE

    cache_key = f"{level}_{cumulative}_{pos_filter}"

    if cache_key in _VOCABULARY_CACHE and not force_reload:
        return _VOCABULARY_CACHE[cache_key]

    # Load full vocabulary
    full_vocab = load_italian_vocabulary(force_reload=force_reload)

    # CEFR level hierarchy
    level_hierarchy = ["A1", "A2", "B1", "B2", "C1", "C2"]

    # Determine which levels to include
    if level is None:
        included_levels = level_hierarchy
    elif cumulative:
        level_idx = level_hierarchy.index(level)
        included_levels = level_hierarchy[: level_idx + 1]
    else:
        included_levels = [level]

    # Filter vocabulary
    words = set()
    for entry in full_vocab:
        # Check CEFR level
        if entry["cefr_level"] not in included_levels:
            continue

        # Check POS filter
        if pos_filter and entry["pos"] != pos_filter:
            continue

        words.add(entry["word"].lower())

    # Cache result
    _VOCABULARY_CACHE[cache_key] = words

    return words


def get_vocabulary_stats() -> Dict:
    """
    Get statistics about the loaded vocabulary.

    Returns:
        Dict with keys:
        - total_words: int
        - by_cefr: Dict[str, int]
        - by_pos: Dict[str, int]
    """
    full_vocab = load_italian_vocabulary()

    stats = {
        "total_words": len(full_vocab),
        "by_cefr": defaultdict(int),
        "by_pos": defaultdict(int),
    }

    for entry in full_vocab:
        stats["by_cefr"][entry["cefr_level"]] += 1
        stats["by_pos"][entry["pos"]] += 1

    return dict(stats)


if __name__ == "__main__":
    # Test vocabulary loading
    print("Testing vocabulary loading...")

    stats = get_vocabulary_stats()
    print(f"\n📊 Vocabulary Statistics:")
    print(f"Total words: {stats['total_words']}")
    print(f"\nBy CEFR level:")
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        count = stats["by_cefr"].get(level, 0)
        print(f"  {level}: {count}")

    print(f"\nBy POS:")
    for pos, count in sorted(stats["by_pos"].items(), key=lambda x: -x[1]):
        print(f"  {pos}: {count}")

    # Test cumulative loading
    print("\n📚 Testing cumulative vocabulary:")
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        vocab = get_vocabulary_by_cefr(level, cumulative=True)
        print(f"  {level} (cumulative): {len(vocab)} words")

    # Test exact level loading
    print("\n📚 Testing exact level vocabulary:")
    for level in ["A1", "A2", "B1", "B2", "C1", "C2"]:
        vocab = get_vocabulary_by_cefr(level, cumulative=False)
        print(f"  {level} (exact): {len(vocab)} words")

    # Test POS filtering
    print("\n🔍 Testing POS filtering:")
    verbs_b1 = get_vocabulary_by_cefr("B1", cumulative=True, pos_filter="verb")
    nouns_a1 = get_vocabulary_by_cefr("A1", cumulative=True, pos_filter="noun")
    print(f"  B1 verbs: {len(verbs_b1)}")
    print(f"  A1 nouns: {len(nouns_a1)}")

    print("\n✅ All tests passed!")
