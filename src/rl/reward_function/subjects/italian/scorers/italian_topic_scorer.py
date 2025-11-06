"""
Topic adherence scorer.

Validates exercise relevance to requested topic (0-10 points).
Uses embedding similarity + optional LLM for contextual relevance.
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ....base import BaseScorer


class ItalianTopicScorer(BaseScorer):
    """
    Scores topic adherence (0-10 points).

    Uses semantic similarity to check if exercise is about the requested topic.
    Optional: LLM for contextual relevance (e.g., "shoes" → "tied", "run").
    """

    def __init__(self, nlp=None, device: str = "cpu"):
        super().__init__(nlp)

        # Load sentence transformer for semantic similarity
        print("Loading sentence transformer for topic similarity...")
        self.similarity_model = SentenceTransformer("./models/sentance_piece", device=device)
        print(f"✅ Sentence transformer loaded in {device}")

        # Check if OpenAI API is available for contextual relevance
        self.use_llm = os.environ.get("OPENAI_API_KEY") is not None
        self.daily_limit_hit = False  # Track if we've hit daily limit
        if self.use_llm:
            try:
                from openai import OpenAI

                self.client = OpenAI()
                print("  ✅ LLM topic checking enabled (OpenAI API)")
            except ImportError:
                self.use_llm = False
                print("  ⚠️ OpenAI not installed, using embedding similarity only")
        else:
            self.client = None

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

            # Convert to CPU and numpy for cosine similarity calculation
            text_emb_np = text_embedding.cpu().numpy()
            topic_emb_np = topic_embedding.cpu().numpy()

            # Cosine similarity
            similarity = np.dot(text_emb_np, topic_emb_np) / (
                np.linalg.norm(text_emb_np) * np.linalg.norm(topic_emb_np)
            )
            similarity = float(similarity)

            # Score based on similarity (adjusted for 10 points)
            # STRICTER thresholds to encourage better topic adherence
            if similarity > 0.7:
                score = 10.0  # Excellent match
            elif similarity > 0.55:
                score = 7.0  # Good match
            elif similarity > 0.4:
                score = 4.0  # Moderate match
                errors.append(f"Moderate topic relevance: {similarity:.2f}")
            elif similarity > 0.25:
                score = 2.0  # Weak match
                errors.append(f"Low topic relevance: {similarity:.2f}")
            else:
                score = 0.0  # Off-topic
                errors.append(f"Off-topic: {similarity:.2f}")

            # Optional: Use LLM for contextual relevance check (if score is borderline)
            # This helps catch cases where embedding similarity is low but context is good
            # e.g., "shoes" + "tied", "run" vs "shoes" + random words
            # Skip LLM check if we've hit daily limit
            if self.use_llm and not self.daily_limit_hit and 4.0 <= score <= 7.0:
                llm_score, llm_errors = self._check_topic_with_llm(text, exercise, request, topic)
                # LLM can adjust borderline scores up or down
                score = llm_score
                errors = llm_errors  # Replace with LLM errors

        except Exception as e:
            errors.append(f"Similarity error: {e}")
            score = 0.0

        return score, errors

    def _check_topic_with_llm(
        self, text: str, exercise: Dict[str, Any], request: Dict[str, Any], topic: str
    ) -> Tuple[float, List[str]]:
        """
        Use OpenAI API to check contextual topic relevance.

        This catches cases where:
        - Semantic similarity is borderline but context is appropriate
        - Words are topically related but embedding doesn't capture it
        - e.g., "shoes" → verbs like "tied", "run", "walk" should score high
        """
        try:
            prompt = f"""You are evaluating an Italian language exercise for topic relevance.

Exercise type: {exercise.get('type', 'unknown')}
Question: {exercise.get('question', '')}
Answer: {exercise.get('correct_answer', exercise.get('answer', ''))}

Requested topic: {topic}
CEFR level: {request.get('level', 'unknown')}

Rate the exercise on a scale of 0-10 for topic relevance:
10 = Clearly about the topic, uses related vocabulary and concepts
7-9 = Related to the topic with some relevant vocabulary
4-6 = Loosely related or mentions topic briefly
0-3 = Unrelated to the topic

Consider:
- Does the exercise use vocabulary related to the topic?
- Would a learner practicing this exercise learn words/concepts about the topic?
- Are verbs and adjectives contextually appropriate for the topic?
- Example: "shoes" → "tied", "run", "walk", "buy" are all relevant

Respond ONLY with a JSON object:
{{"score": <number 0-10>, "issue": "<brief explanation if score < 7, empty string otherwise>"}}"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )

            result_text = response.choices[0].message.content.strip()

            # Parse JSON response
            import json

            result = json.loads(result_text)

            llm_score = result.get("score", 7.0)
            issue = result.get("issue", "")

            errors = []
            if llm_score < 7 and issue:
                errors.append(f"Topic issue: {issue}")

            return llm_score, errors

        except Exception as e:
            # If LLM check fails, return original embedding-based score
            error_msg = str(e)

            # Check if this is a daily limit error
            if "requests per day" in error_msg.lower() or "rpd:" in error_msg.lower():
                if not self.daily_limit_hit:
                    print(f"  🚫 LLM topic check: daily limit reached, disabling for session")
                    self.daily_limit_hit = True
            # Only log non-rate-limit errors (rate limits are handled elsewhere)
            elif "429" not in error_msg and "rate limit" not in error_msg.lower():
                print(f"  ⚠️ LLM topic check failed: {error_msg[:100]}")

            return 7.0, []  # Neutral score for borderline cases

    def _extract_italian_text(self, exercise: Dict[str, Any]) -> str:
        """
        Extract Italian text from exercise for topic analysis.

        For translation exercises: ONLY uses the Italian answer (correct_answer).
        For other types: Uses the Italian question text.
        """
        import re

        exercise_type = exercise.get("type", "")

        # For translation exercises, ONLY analyze the Italian answer
        if exercise_type == "translation":
            return exercise.get("correct_answer", "").strip()

        # For fill-in-blank, include question with answer inserted
        if exercise_type == "fill_in_blank":
            question = exercise.get("question", "")
            answer = exercise.get("correct_answer", "")
            if "___" in question and answer:
                return question.replace("___", answer, 1).strip()

        # For other types (multiple_choice, etc.), use question if it's Italian
        if "question" in exercise:
            question = exercise["question"]
            # Remove common English prompts
            question_clean = re.sub(
                r"^(Translate|Fill in the blank|Choose the correct answer):\s*",
                "",
                question,
                flags=re.IGNORECASE,
            ).strip()

            # Check if it contains Italian indicators
            italian_indicators = [
                "il", "la", "le", "gli", "lo", "un", "una", "è", "sono",
                "di", "a", "per", "che", "ho", "hai", "ha", "abbiamo"
            ]
            if any(indicator in question_clean.lower().split() for indicator in italian_indicators):
                return question_clean

        return ""

    @property
    def max_score(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "topic_adherence"
