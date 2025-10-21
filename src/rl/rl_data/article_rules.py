"""
Italian article rules.

Comprehensive rules for definite, indefinite, partitive, and elided articles.
"""

from typing import Any, Dict

ARTICLE_RULES: Dict[str, Any] = {
    # Definite articles
    "definite": {
        "Masc": {
            "Sing": {
                "il": ["default"],  # il gatto, il libro
                "lo": [
                    "s+consonant",
                    "z",
                    "gn",
                    "ps",
                    "x",
                    "y",
                ],  # lo studente, lo zaino
                "l'": ["vowel"],  # l'amico, l'uomo
            },
            "Plur": {
                "i": ["default"],  # i gatti
                "gli": [
                    "s+consonant",
                    "z",
                    "gn",
                    "ps",
                    "x",
                    "y",
                    "vowel",
                ],  # gli studenti, gli amici
            },
        },
        "Fem": {
            "Sing": {
                "la": ["default"],  # la casa
                "l'": ["vowel"],  # l'aquila, l'amica
            },
            "Plur": {
                "le": ["default", "vowel"],  # le case, le aquile
            },
        },
    },
    # Indefinite articles
    "indefinite": {
        "Masc": {
            "Sing": {
                "un": ["default", "vowel"],  # un gatto, un amico
                "uno": [
                    "s+consonant",
                    "z",
                    "gn",
                    "ps",
                    "x",
                    "y",
                ],  # uno studente
            },
        },
        "Fem": {
            "Sing": {
                "una": ["default"],  # una casa
                "un'": ["vowel"],  # un'amica
            },
        },
    },
    # Partitive articles (di + definite article)
    "partitive": {
        "Masc": {
            "Sing": ["del", "dello", "dell'"],
            "Plur": ["dei", "degli"],
        },
        "Fem": {
            "Sing": ["della", "dell'"],
            "Plur": ["delle"],
        },
    },
}
