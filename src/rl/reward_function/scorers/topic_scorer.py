"""
Topic adherence scorer.

Validates exercise relevance to requested topic (0-10 points).
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseScorer


class TopicScorer(BaseScorer):
    """
    Scores topic adherence (0-10 points).

    Uses semantic similarity to check if exercise is about the requested topic.
    """

    def __init__(self, nlp=None):
        super().__init__(nlp)

        # Load sentence transformer for semantic similarity
        print("Loading sentence transformer for topic similarity...")
        self.similarity_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
        print("✅ Sentence transformer loaded")

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score topic adherence using semantic similarity."""
        topic = request.get("topic")

        if not topic:
            return 10.0, []  # No topic specified

        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 0.0, ["No text found"]

        # Compute semantic similarity
        try:
            text_embedding = self.similarity_model.encode(text, convert_to_tensor=True)
            topic_embedding = self.similarity_model.encode(topic, convert_to_tensor=True)

            # Cosine similarity
            similarity = np.dot(text_embedding, topic_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(topic_embedding)
            )
            similarity = float(similarity)

            # Score based on similarity (adjusted for 10 points)
            if similarity > 0.7:
                score = 10.0
            elif similarity > 0.5:
                score = 7.0
            elif similarity > 0.3:
                score = 3.0
                errors.append(f"Low topic relevance: {similarity:.2f}")
            else:
                score = 0.0
                errors.append(f"Off-topic: {similarity:.2f}")

        except Exception as e:
            errors.append(f"Similarity error: {e}")
            score = 0.0

        return score, errors

    def _extract_italian_text(self, exercise: Dict[str, Any]) -> str:
        """
        Extract Italian text from exercise for analysis.

        Filters out English text (like "Translate:" prompts) to focus on Italian only.
        """
        import re

        parts = []

        if "question" in exercise:
            question = exercise["question"]
            # Remove common English prompts
            question = re.sub(
                r"^(Translate|Fill in the blank|Choose the correct answer):\s*",
                "",
                question,
                flags=re.IGNORECASE,
            )
            # Only include if it contains Italian-looking text (has Italian articles/words)
            italian_indicators = [
                "il",
                "la",
                "le",
                "gli",
                "lo",
                "un",
                "una",
                "è",
                "sono",
                "di",
                "a",
                "per",
                "che",
            ]
            if any(indicator in question.lower() for indicator in italian_indicators):
                parts.append(question)

        if "answer" in exercise:
            parts.append(exercise["answer"])

        return " ".join(parts)

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "topic_adherence"
