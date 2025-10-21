"""Data module for reward function."""

from .article_rules import ARTICLE_RULES
from .cefr_rules import CEFR_RULES
from .gender_exceptions import GENDER_EXCEPTIONS
from .invariant_adjectives import INVARIANT_ADJECTIVES
from .vocabulary_lists import get_vocabulary_by_cefr, get_vocabulary_stats, load_italian_vocabulary

__all__ = [
    "load_italian_vocabulary",
    "get_vocabulary_by_cefr",
    "get_vocabulary_stats",
    "GENDER_EXCEPTIONS",
    "ARTICLE_RULES",
    "INVARIANT_ADJECTIVES",
    "CEFR_RULES",
]
