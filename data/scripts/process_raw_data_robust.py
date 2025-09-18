#!/usr/bin/env python3
"""
Robust Raw Data Processing Pipeline - Full Dataset Processing

Processes ALL available raw data to maximize B1/B2 content and scale to 5K-10K examples.
Handles large files, network timeouts, and focuses on higher CEFR levels.

NOTE: Grammar explanations are now improved via LLM processing in Colab notebooks.
This script creates the base dataset which is then enhanced with LLM-generated grammar.

Usage:
    python process_raw_data_robust.py --input ../raw/ --output ../processed/ --target-size 10000

Post-processing: Run Colab notebook with LLM to improve grammar explanations
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class RobustDataProcessor:
    def __init__(self):
        """Initialize robust data processor with B1/B2 focus."""
        # Context-appropriate encouragements
        self.question_encouragements = [
            "Great question! ",
            "Good question! ",
            "Interesting! ",
            "Let me help! ",
            "Sure! ",
            "Of course! ",
        ]

        self.general_encouragements = [
            "Bravissimo! ",
            "Perfetto! ",
            "Molto bene! ",
            "Che bravo/a! ",
            "Eccellente! ",
            "Fantastico! ",
            "Great job! ",
            "Well done! ",
            "You're doing great! ",
            "Keep it up! ",
        ]

        # Diverse conversation starters for translation
        self.translation_starters = [
            "How do you say '{english}' in Italian?",
            "What's the Italian for '{english}'?",
            "Can you translate '{english}' to Italian?",
            "How would you say '{english}' in Italian?",
            "What's '{english}' in Italian?",
            "I need to say '{english}' in Italian. How?",
            "Could you help me translate '{english}'?",
            "What would '{english}' be in Italian?",
        ]

        # Diverse responses for translations (more natural, less repetitive)
        self.translation_responses = [
            "In Italian, that's '{italian}'.",
            "You would say '{italian}'.",
            "The Italian is '{italian}'.",
            "That's '{italian}' in Italian.",
            "It's '{italian}'.",
            "You can say '{italian}'.",
            "The translation is '{italian}'.",
            "In Italian: '{italian}'.",
        ]

        # Varied grammar question starters
        self.grammar_starters = [
            "Can you analyze the grammar in this sentence: '{italian}'?",
            "What's the grammar structure of '{italian}'?",
            "Could you explain the grammar here: '{italian}'?",
            "Help me understand the grammar of '{italian}'?",
            "What grammatical patterns do you see in '{italian}'?",
            "Can you break down this sentence: '{italian}'?",
            "What's happening grammatically in '{italian}'?",
        ]

        # B1/B2 specific teaching patterns
        self.advanced_responses = [
            "This is perfect for intermediate learners! Let's explore the nuances...",
            "Now we're getting into more sophisticated Italian! Notice how...",
            "This B1 level content introduces some complex grammar patterns...",
            "At the B2 level, we can appreciate the cultural subtleties here...",
            "This advanced structure shows how Italian really works in context...",
        ]

        # B1/B2 grammar indicators
        self.b1_indicators = [
            "che",
            "quando",
            "perché",
            "mentre",
            "dopo che",
            "prima che",
            "benché",
            "sebbene",
            "purché",
            "a condizione che",
        ]

        self.b2_indicators = [
            "sia",
            "fosse",
            "avrebbe",
            "sarebbe",
            "vorrei",
            "dovrebbe",
            "qualora",
            "nel caso in cui",
            "a meno che",
            "chissà",
        ]

        # Semantic topic detection keywords
        self.topic_keywords = {
            "family": [
                "famiglia",
                "padre",
                "madre",
                "figlio",
                "figlia",
                "nonno",
                "nonna",
                "zio",
                "zia",
                "cugino",
                "fratello",
                "sorella",
                "marito",
                "moglie",
                "family",
                "father",
                "mother",
                "son",
                "daughter",
                "uncle",
                "aunt",
                "cousin",
                "brother",
                "sister",
            ],
            "food": [
                "mangiare",
                "cibo",
                "ristorante",
                "pizza",
                "pasta",
                "cena",
                "pranzo",
                "colazione",
                "pane",
                "vino",
                "cucina",
                "cucinare",
                "food",
                "eat",
                "restaurant",
                "dinner",
                "lunch",
                "breakfast",
                "cook",
                "wine",
            ],
            "time": [
                "tempo",
                "ora",
                "oggi",
                "ieri",
                "domani",
                "settimana",
                "mese",
                "anno",
                "mattina",
                "sera",
                "notte",
                "time",
                "hour",
                "today",
                "yesterday",
                "tomorrow",
                "week",
                "month",
                "year",
                "morning",
                "evening",
            ],
            "travel": [
                "viaggio",
                "viaggerei",
                "paese",
                "paesi",
                "città",
                "aeroporto",
                "treno",
                "macchina",
                "hotel",
                "vacanza",
                "travel",
                "trip",
                "country",
                "countries",
                "city",
                "airport",
                "train",
                "car",
                "vacation",
            ],
            "body": [
                "corpo",
                "testa",
                "mano",
                "piede",
                "occhi",
                "bocca",
                "naso",
                "orecchio",
                "braccio",
                "gamba",
                "body",
                "head",
                "hand",
                "foot",
                "eyes",
                "mouth",
                "nose",
                "ear",
                "arm",
                "leg",
            ],
            "colors": [
                "colore",
                "rosso",
                "blu",
                "verde",
                "giallo",
                "nero",
                "bianco",
                "color",
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
            ],
            "emotions": [
                "felice",
                "triste",
                "arrabbiato",
                "contento",
                "preoccupato",
                "stanco",
                "happy",
                "sad",
                "angry",
                "worried",
                "tired",
            ],
            "work": [
                "lavoro",
                "lavorare",
                "ufficio",
                "professore",
                "medico",
                "ingegnere",
                "impresa",
                "work",
                "job",
                "office",
                "teacher",
                "doctor",
                "engineer",
                "business",
            ],
            "weather": [
                "tempo",
                "sole",
                "pioggia",
                "neve",
                "caldo",
                "freddo",
                "nuvole",
                "weather",
                "sun",
                "rain",
                "snow",
                "hot",
                "cold",
                "clouds",
            ],
            "clothing": [
                "vestiti",
                "camicia",
                "pantaloni",
                "scarpe",
                "giacca",
                "vestito",
                "clothes",
                "shirt",
                "pants",
                "shoes",
                "jacket",
                "dress",
            ],
            "house": [
                "casa",
                "camera",
                "cucina",
                "bagno",
                "soggiorno",
                "letto",
                "tavolo",
                "house",
                "room",
                "kitchen",
                "bathroom",
                "living room",
                "bed",
                "table",
            ],
            "education": [
                "scuola",
                "università",
                "studiare",
                "imparare",
                "lezione",
                "esame",
                "school",
                "university",
                "study",
                "learn",
                "lesson",
                "exam",
            ],
            "greetings": [
                "ciao",
                "buongiorno",
                "buonasera",
                "arrivederci",
                "salve",
                "hello",
                "goodbye",
                "good morning",
                "good evening",
            ],
            "numbers": [
                "uno",
                "due",
                "tre",
                "quattro",
                "cinque",
                "numero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "number",
            ],
            "daily_activities": [
                "dormire",
                "mangiare",
                "bere",
                "camminare",
                "correre",
                "leggere",
                "scrivere",
                "sleep",
                "eat",
                "drink",
                "walk",
                "run",
                "read",
                "write",
            ],
        }

    def analyze_cefr_level(self, italian_text: str) -> str:
        """Analyze Italian text and assign accurate CEFR level with improved A2 detection."""
        words = italian_text.lower().split()
        word_count = len(words)
        text_lower = italian_text.lower()

        # A2 specific indicators - intermediate vocabulary and structures
        a2_indicators = [
            "perché",
            "dove",
            "quando",
            "come",
            "quanto",
            "quale",
            "chi",
            "prima",
            "dopo",
            "oggi",
            "ieri",
            "domani",
            "sempre",
            "mai",
            "spesso",
            "molto",
            "poco",
            "troppo",
            "abbastanza",
            "più",
            "meno",
            "voglio",
            "vorrei",
            "posso",
            "devo",
            "so",
            "conosco",
            "mi piace",
            "ti piace",
            "gli piace",
            "le piace",
            "questo",
            "quello",
            "questi",
            "quelli",
            "alcune",
            "tutti",
        ]

        # Check for advanced grammar patterns
        has_b2_grammar = any(indicator in text_lower for indicator in self.b2_indicators)
        has_b1_grammar = any(indicator in text_lower for indicator in self.b1_indicators)
        has_a2_patterns = any(indicator in text_lower for indicator in a2_indicators)

        # Italian verb patterns for different levels
        basic_verbs = ["sono", "sei", "è", "ho", "hai", "ha", "va", "faccio", "dici"]
        intermediate_verbs = ["ero", "era", "erano", "avevo", "aveva", "facevo", "dicevo"]
        complex_verbs = ["fossi", "fosse", "avessi", "avesse", "farei", "diresti"]

        has_basic_verbs = any(verb in text_lower for verb in basic_verbs)
        has_intermediate_verbs = any(verb in text_lower for verb in intermediate_verbs)
        has_complex_verbs = any(verb in text_lower for verb in complex_verbs)

        # Check for sentence complexity
        has_subordinate_clauses = italian_text.count(",") >= 1
        has_complex_tenses = has_intermediate_verbs or has_complex_verbs
        has_questions = any(q in text_lower for q in ["?", "come", "dove", "quando", "perché"])

        # Improved classification with better A2 detection
        if has_b2_grammar or has_complex_verbs or (word_count > 15 and has_complex_tenses):
            return "B2"
        elif has_b1_grammar or (word_count > 10 and has_subordinate_clauses) or (word_count > 14):
            return "B1"
        elif (
            has_a2_patterns
            or has_intermediate_verbs
            or has_questions
            or (word_count >= 6 and word_count <= 10)
            or (has_basic_verbs and word_count > 4)
        ):
            return "A2"
        else:
            return "A1"

    def detect_semantic_topic(self, italian_text: str, english_text: str) -> str:
        """Detect semantic topic based on content keywords."""
        combined_text = f"{italian_text.lower()} {english_text.lower()}"

        # Count matches for each topic
        topic_scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                topic_scores[topic] = score

        # Return the topic with highest score, or default
        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        else:
            return "general"

    def process_tatoeba_file_streaming(
        self, file_path: str, max_sentences: int = None
    ) -> List[Dict]:
        """Process Tatoeba file with streaming to handle large files."""
        print(f"📖 Processing Tatoeba file: {file_path}")

        try:
            # Handle both tab-delimited and JSON formats
            if file_path.endswith(".txt"):
                return self.process_tatoeba_txt_streaming(file_path, max_sentences)
            else:
                return self.process_tatoeba_json_streaming(file_path, max_sentences)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return []

    def process_tatoeba_txt_streaming(
        self, file_path: str, max_sentences: int = None
    ) -> List[Dict]:
        """Process large tab-delimited Tatoeba file in chunks."""
        sentence_pairs = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f):
                    if max_sentences and line_num >= max_sentences:
                        break

                    # Process every 5th line to get maximum data while managing memory
                    if line_num % 5 == 0:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            # Format is English-Italian-License, so swap the order
                            english_sentence = parts[0].strip()
                            italian_sentence = parts[1].strip()

                            if italian_sentence and english_sentence and len(italian_sentence) > 5:
                                cefr_level = self.analyze_cefr_level(italian_sentence)

                                # Prioritize B1/B2 content
                                if cefr_level in ["B1", "B2"]:
                                    priority_multiplier = 3  # Collect 3x more B1/B2 examples
                                else:
                                    priority_multiplier = 1

                                for _ in range(priority_multiplier):
                                    sentence_pairs.append(
                                        {
                                            "italian": italian_sentence,
                                            "english": english_sentence,
                                            "cefr_level": cefr_level,
                                            "source": "tatoeba",
                                            "priority": priority_multiplier,
                                        }
                                    )

            print(f"✅ Processed {len(sentence_pairs)} sentence pairs from Tatoeba")
            return sentence_pairs

        except Exception as e:
            print(f"❌ Error processing Tatoeba txt: {e}")
            return []

    def process_tatoeba_json_streaming(
        self, file_path: str, max_sentences: int = None
    ) -> List[Dict]:
        """Process Tatoeba JSON file with memory management."""
        sentence_pairs = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "sentence_pairs" in data:
                pairs = data["sentence_pairs"]
                total_pairs = min(len(pairs), max_sentences) if max_sentences else len(pairs)

                for i, pair in enumerate(pairs[:total_pairs]):
                    # IMPORTANT: JSON files have fields swapped! "italian" field contains English, "english" field contains Italian
                    english_sentence = pair.get(
                        "italian", ""
                    )  # This field actually contains English
                    italian_sentence = pair.get(
                        "english", ""
                    )  # This field actually contains Italian

                    if italian_sentence and english_sentence:
                        cefr_level = self.analyze_cefr_level(italian_sentence)

                        sentence_pairs.append(
                            {
                                "italian": italian_sentence,
                                "english": english_sentence,
                                "cefr_level": cefr_level,
                                "source": "tatoeba",
                                "original_difficulty": pair.get("educational_value", {}).get(
                                    "estimated_difficulty", "unknown"
                                ),
                            }
                        )

            print(f"✅ Processed {len(sentence_pairs)} pairs from Tatoeba JSON")
            return sentence_pairs

        except Exception as e:
            print(f"❌ Error processing Tatoeba JSON: {e}")
            return []

    def process_babbel_with_level_focus(self, babbel_dir: Path) -> List[Dict]:
        """Process Babbel data with focus on B1/B2 content."""
        training_examples = []

        babbel_files = list(babbel_dir.glob("**/*.json"))
        print(f"Processing {len(babbel_files)} Babbel files...")

        for file_path in babbel_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract level information
                episode_level = data.get("level", "A1")

                # Focus more on B1/B2 content
                if episode_level in ["B1", "B2"]:
                    processing_multiplier = 5  # Generate 5x more examples from B1/B2 content
                else:
                    processing_multiplier = 2

                # Extract Italian content
                italian_segments = []
                if "segments" in data:
                    for segment in data["segments"]:
                        text = segment.get("text", "").strip()
                        if len(text) > 10:  # Skip very short segments
                            italian_segments.append(text)
                elif "transcript_segments" in data:
                    for segment in data["transcript_segments"]:
                        text = segment.get("text", "").strip()
                        if len(text) > 10:  # Skip very short segments
                            italian_segments.append(text)

                # Generate training examples
                for segment in italian_segments[:10]:  # Limit per file
                    cefr_level = self.analyze_cefr_level(segment)

                    for i in range(processing_multiplier):
                        # Varied question starters for Babbel content
                        babbel_starters = [
                            f"What does this mean: {segment[:200]}?",
                            f"Can you explain this Italian: {segment[:200]}?",
                            f"Help me with this phrase: {segment[:200]}",
                            f"I heard this in a podcast: {segment[:200]}. What's it saying?",
                            f"Could you break down: {segment[:200]}?",
                            f"This Italian is confusing me: {segment[:200]}",
                        ]

                        user_question = random.choice(babbel_starters)
                        # Remove ellipsis if the segment is short enough
                        if len(segment) <= 200:
                            user_question = user_question.replace("...", "")

                        # Detect semantic topic for Babbel content
                        babbel_topic = self.detect_semantic_topic(segment, "")

                        training_examples.append(
                            {
                                "conversation_id": f"babbel_{file_path.stem}_{len(training_examples)}",
                                "source": "babbel_podcast",
                                "level": cefr_level,
                                "original_level": episode_level,
                                "topic": f"podcast_{babbel_topic}",
                                "conversation": [
                                    {"role": "user", "content": user_question},
                                    {
                                        "role": "assistant",
                                        "content": self.generate_practical_italian_explanation(
                                            segment, cefr_level
                                        ),
                                    },
                                ],
                            }
                        )

            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")
                continue

        print(f"Generated {len(training_examples)} examples from Babbel data")
        return training_examples

    def generate_practical_italian_explanation(self, italian_text: str, cefr_level: str) -> str:
        """Generate practical explanation for Italian podcast content."""
        # Sometimes add appropriate encouragement
        if random.random() < 0.3:
            encouragement = random.choice(self.question_encouragements).strip()
            response = f"{encouragement} "
        else:
            response = ""

        # Provide translation
        response += self.generate_approximate_translation(italian_text)

        # Add specific grammar points based on content
        grammar_points = []
        text_lower = italian_text.lower()

        # Identify specific grammar elements
        if "ora" in text_lower:
            grammar_points.append("'ora' means 'now'")
        if "stavolta" in text_lower:
            grammar_points.append("'stavolta' means 'this time'")
        if "prova" in text_lower:
            grammar_points.append("'prova' is the imperative form 'try!'")
        if "concentrarti" in text_lower:
            grammar_points.append("'concentrarti' uses the reflexive pronoun 'ti' with infinitive")
        if "di più" in text_lower:
            grammar_points.append("'di più' means 'more'")
        if "sulle" in text_lower:
            grammar_points.append("'sulle' = 'su' (on) + 'le' (the, feminine plural)")
        if "singole" in text_lower:
            grammar_points.append("'singole' means 'individual/single' (feminine plural)")

        # Add imperative detection
        if any(verb in text_lower for verb in ["prova", "ascolta", "guarda", "senti"]):
            grammar_points.append("uses imperative mood for commands/suggestions")

        # Add grammar insights
        if grammar_points:
            response += f" Key grammar: {', '.join(grammar_points[:3])}."  # Limit to 3 points

        return response

    def generate_approximate_translation(self, italian_text: str) -> str:
        """Generate approximate translation for common Italian phrases."""
        text_lower = italian_text.lower()

        # Common phrase translations
        if "e ora riascoltiamo" in text_lower:
            if "stavolta" in text_lower and "concentrarti" in text_lower:
                return "It says: 'And now let's listen again. This time, try to focus more on individual words and phrases.'"
            else:
                return "It means: 'And now let's listen again.'"
        elif "stavolta" in text_lower and "prova" in text_lower:
            return "It says: 'This time, try to focus more on individual words and phrases.'"
        elif "buongiorno" in text_lower:
            return "It means: 'Good morning.'"
        elif "come stai" in text_lower:
            return "It means: 'How are you?'"
        elif "grazie" in text_lower and "piacere" in text_lower:
            return (
                "It means something like: 'Thank you! It's a great pleasure to be here with you.'"
            )
        else:
            # For phrases we don't have specific translations for,
            # provide basic grammar analysis
            return self.generate_basic_grammar_explanation(italian_text)

    def generate_basic_grammar_explanation(self, italian_text: str) -> str:
        """Generate basic grammar explanation for Italian text."""
        text_lower = italian_text.lower()
        explanation_parts = []

        # Common question words
        question_words = {
            "cosa": "what",
            "chi": "who",
            "dove": "where",
            "quando": "when",
            "perché": "why",
            "come": "how",
            "quanto": "how much/many",
            "quale": "which",
        }

        # Common verbs with explanations
        common_verbs = {
            "è": "is (3rd person singular of essere)",
            "sono": "am/are (1st person singular or 3rd person plural of essere)",
            "accade": "happens (3rd person singular of accadere)",
            "ha": "has (3rd person singular of avere)",
            "abbiamo": "we have (1st person plural of avere)",
            "era": "was (3rd person singular imperfect of essere)",
            "erano": "were (3rd person plural imperfect of essere)",
            "aveva": "had (3rd person singular imperfect of avere)",
            "dice": "says (3rd person singular of dire)",
            "fa": "does/makes (3rd person singular of fare)",
        }

        # Discourse markers
        discourse_markers = {
            "dunque": "so/therefore (discourse marker)",
            "allora": "so/then (discourse marker)",
            "infatti": "in fact (discourse marker)",
            "però": "however (discourse marker)",
        }

        # Check for question structure
        if "?" in italian_text:
            explanation_parts.append("This is a question.")

        # Identify discourse markers
        for marker, meaning in discourse_markers.items():
            if marker in text_lower:
                explanation_parts.append(f"'{marker}' means {meaning}")

        # Identify question words
        for word, meaning in question_words.items():
            if word in text_lower:
                explanation_parts.append(f"'{word}' means '{meaning}'")

        # Identify common verbs
        for verb, meaning in common_verbs.items():
            if verb in text_lower:
                explanation_parts.append(f"'{verb}' means {meaning}")

        # Check for preposition + article contractions
        contractions = {
            "nella": "'nella' = 'in' + 'la' (in the, feminine singular)",
            "nella": "'nella' = 'in' + 'la' (in the, feminine singular)",
            "delle": "'delle' = 'di' + 'le' (of the, feminine plural)",
            "sulle": "'sulle' = 'su' + 'le' (on the, feminine plural)",
        }

        for contraction, meaning in contractions.items():
            if contraction in text_lower:
                explanation_parts.append(meaning)

        if explanation_parts:
            return f"This appears to mean something about the topic. Key grammar: {', '.join(explanation_parts[:3])}."
        else:
            # TODO: Future enhancement - use LLM API for sophisticated grammar analysis
            return self.use_llm_for_grammar_analysis(italian_text)

    def use_llm_for_grammar_analysis(self, italian_text: str) -> str:
        """
        Future enhancement: Use specialized LLM for advanced grammar analysis.

        This function could integrate with:
        - OpenAI GPT-4 with Italian grammar expertise
        - Anthropic Claude with linguistic analysis prompts
        - Local Italian language models

        For now, returns basic fallback.
        """
        # Placeholder for future LLM integration
        # Could use prompts like:
        # "Analyze this Italian sentence for a language learner: '{italian_text}'"
        # "Break down the grammar, vocabulary, and structure. Explain verb forms, pronouns, and syntax."

        return "This Italian text discusses practical language use."

    def generate_level_appropriate_response(self, italian_text: str, cefr_level: str) -> str:
        """Generate teaching response appropriate for CEFR level."""
        encouragement = random.choice(self.general_encouragements)

        if cefr_level in ["B1", "B2"]:
            advanced_intro = random.choice(self.advanced_responses)
            response = f"{encouragement}{advanced_intro} "
        else:
            response = f"{encouragement}This is {cefr_level} level Italian. "

        # Level-specific teaching approaches
        if cefr_level == "B2":
            response += f"At the B2 level, we're looking at sophisticated language use. This text: '{italian_text[:80]}...' "
            response += "demonstrates complex grammatical structures and cultural nuances. "
            response += "Notice the advanced vocabulary and sentence construction patterns. "

        elif cefr_level == "B1":
            response += f"This B1 intermediate text: '{italian_text[:80]}...' "
            response += "introduces more complex grammar and cultural context. "
            response += "You'll see subordinate clauses and varied tenses here. "

        elif cefr_level == "A2":
            response += f"This A2 pre-intermediate content: '{italian_text[:80]}...' "
            response += "builds on basic grammar with more vocabulary and sentence variety. "

        else:  # A1
            response += f"This A1 beginner text: '{italian_text[:80]}...' "
            response += "focuses on essential vocabulary and basic sentence structures. "

        response += "What specific aspect would you like me to explain further?"
        return response

    def create_diverse_conversation_types(self, sentence_pairs: List[Dict]) -> List[Dict]:
        """Create multiple conversation types from sentence pairs."""
        training_examples = []

        conversation_types = [
            "translation_request",
            "grammar_analysis",
            "cultural_context",
            "practice_exercise",
            "level_explanation",
        ]

        for pair in sentence_pairs:
            italian = pair["italian"]
            english = pair["english"]
            cefr_level = pair["cefr_level"]
            semantic_topic = self.detect_semantic_topic(italian, english)

            # Generate multiple conversation types per sentence pair
            for conv_type in conversation_types:
                if conv_type == "translation_request":
                    # Use varied conversation starters and responses
                    user_question = random.choice(self.translation_starters).format(english=english)
                    base_response = random.choice(self.translation_responses).format(
                        italian=italian
                    )

                    # Add occasional appropriate encouragement for translation
                    if random.random() < 0.3:  # 30% chance of encouragement
                        encouragement = random.choice(self.general_encouragements).strip()
                        response = f"{encouragement} {base_response}"
                    else:
                        response = base_response

                    training_examples.append(
                        {
                            "conversation_id": f"translation_{len(training_examples)}",
                            "source": pair["source"],
                            "level": cefr_level,
                            "topic": semantic_topic,
                            "conversation": [
                                {"role": "user", "content": user_question},
                                {"role": "assistant", "content": response},
                            ],
                        }
                    )

                elif conv_type == "grammar_analysis":
                    # Use varied grammar question starters
                    grammar_question = random.choice(self.grammar_starters).format(italian=italian)

                    training_examples.append(
                        {
                            "conversation_id": f"grammar_{len(training_examples)}",
                            "source": pair["source"],
                            "level": cefr_level,
                            "topic": f"grammar_{semantic_topic}",
                            "conversation": [
                                {"role": "user", "content": grammar_question},
                                {
                                    "role": "assistant",
                                    "content": self.generate_grammar_analysis(italian, cefr_level),
                                },
                            ],
                        }
                    )

                elif conv_type == "cultural_context" and cefr_level in ["B1", "B2"]:
                    training_examples.append(
                        {
                            "conversation_id": f"cultural_{len(training_examples)}",
                            "source": pair["source"],
                            "level": cefr_level,
                            "topic": "cultural_context",
                            "conversation": [
                                {
                                    "role": "user",
                                    "content": f"What's the cultural context behind: '{italian}'?",
                                },
                                {
                                    "role": "assistant",
                                    "content": self.generate_cultural_context(italian, cefr_level),
                                },
                            ],
                        }
                    )

        return training_examples

    def get_level_explanation(self, cefr_level: str, italian_text: str) -> str:
        """Get explanation appropriate for CEFR level."""
        if cefr_level == "B2":
            return "This advanced sentence demonstrates sophisticated Italian grammar and cultural awareness."
        elif cefr_level == "B1":
            return "This intermediate sentence shows complex grammar structures you're ready to master."
        elif cefr_level == "A2":
            return (
                "This pre-intermediate sentence introduces more varied vocabulary and structures."
            )
        else:
            return "This beginner sentence focuses on essential Italian patterns."

    def generate_grammar_analysis(self, italian_text: str, cefr_level: str) -> str:
        """Generate appropriate grammar analysis with proper detail for the question."""
        # Sometimes add encouragement appropriate for questions
        if random.random() < 0.4:  # 40% chance of encouragement
            encouragement = random.choice(self.question_encouragements).strip()
            response = f"{encouragement} "
        else:
            response = ""

        # Provide actual grammatical analysis appropriate to level
        if cefr_level in ["B1", "B2"]:
            # More detailed analysis for advanced levels
            if any(indicator in italian_text.lower() for indicator in self.b1_indicators):
                response += f"This sentence uses subordinate clauses with '{[ind for ind in self.b1_indicators if ind in italian_text.lower()][0] if any(ind in italian_text.lower() for ind in self.b1_indicators) else 'connecting words'}'. "

            if "sono" in italian_text.lower() or "è" in italian_text.lower():
                response += "The verb 'essere' (to be) appears here. "

            if any(
                tense in italian_text.lower()
                for tense in ["ho", "hai", "ha", "abbiamo", "avete", "hanno"]
            ):
                response += "This uses the auxiliary verb 'avere' for compound tenses. "

            # Add structural insights
            if "," in italian_text:
                response += "Notice the comma indicating a pause or clause separation. "

        else:  # A1/A2 levels
            # Basic but helpful analysis
            if "sono" in italian_text.lower():
                response += f"The verb 'sono' means 'I am' or 'they are'. "
            elif "è" in italian_text.lower():
                response += f"The verb 'è' means 'is' or 'he/she/it is'. "
            elif "ho" in italian_text.lower():
                response += f"'Ho' means 'I have'. "
            elif "hai" in italian_text.lower():
                response += f"'Hai' means 'you have'. "
            else:
                response += f"This shows basic Italian word order: subject + verb + object. "

        # Optional follow-up for grammar questions specifically
        if random.random() < 0.4:  # 40% chance for grammar questions
            response += "Would you like me to explain any specific part?"

        return response

    def generate_cultural_context(self, italian_text: str, cefr_level: str) -> str:
        """Generate cultural context explanation."""
        encouragement = random.choice(self.general_encouragements)

        response = f"{encouragement}This {cefr_level} level expression reveals interesting aspects of Italian culture. "
        response += (
            f"The phrase '{italian_text}' reflects how Italians communicate in social contexts. "
        )

        if cefr_level == "B2":
            response += "At this advanced level, you can appreciate the subtle cultural nuances and regional variations. "
        else:
            response += "Understanding the cultural background helps you use this phrase naturally in conversation. "

        response += (
            "This kind of cultural awareness is essential for authentic Italian communication!"
        )
        return response

    def generate_synthetic_b1_b2_content(self, target_count: int) -> List[Dict]:
        """Generate synthetic B1/B2 content to reach target distribution."""
        print(f"🔄 Generating {target_count} synthetic B1/B2 examples...")

        synthetic_examples = []

        # B1 level Italian patterns and vocabulary
        b1_patterns = [
            (
                "Quando ero bambino/a, mi piaceva giocare nel parco vicino casa mia.",
                "When I was a child, I liked to play in the park near my house.",
            ),
            (
                "Se avessi più tempo libero, viaggerei in tutti i paesi europei.",
                "If I had more free time, I would travel to all European countries.",
            ),
            (
                "Anche se piove, usciremo lo stesso per fare una passeggiata.",
                "Even if it rains, we'll go out anyway to take a walk.",
            ),
            (
                "Prima che tu arrivi, dovremo preparare tutto per la festa.",
                "Before you arrive, we'll have to prepare everything for the party.",
            ),
            (
                "Benché sia difficile, continuerò a studiare l'italiano ogni giorno.",
                "Although it's difficult, I'll continue studying Italian every day.",
            ),
        ]

        # A2 level Italian patterns - pre-intermediate content
        a2_patterns = [
            (
                "Dove vai di solito quando hai tempo libero?",
                "Where do you usually go when you have free time?",
            ),
            (
                "Mi piace molto leggere libri prima di dormire.",
                "I really like reading books before sleeping.",
            ),
            (
                "Ieri ho incontrato la mia amica al bar del centro.",
                "Yesterday I met my friend at the downtown bar.",
            ),
            (
                "Vorrei imparare a cucinare i piatti italiani tradizionali.",
                "I would like to learn to cook traditional Italian dishes.",
            ),
            (
                "Quando ero piccolo, andavo sempre dai miei nonni la domenica.",
                "When I was little, I always went to my grandparents' on Sunday.",
            ),
            (
                "Puoi aiutarmi a trovare la stazione degli autobus, per favore?",
                "Can you help me find the bus station, please?",
            ),
            (
                "La mia famiglia viene dall'Italia del sud, precisamente dalla Sicilia.",
                "My family comes from southern Italy, specifically from Sicily.",
            ),
            (
                "Ho studiato inglese per tre anni ma parlo ancora poco.",
                "I studied English for three years but I still speak little.",
            ),
            (
                "Domani devo andare dal medico perché non mi sento bene.",
                "Tomorrow I have to go to the doctor because I don't feel well.",
            ),
            (
                "Preferisco il tè al caffè, soprattutto quando fa freddo.",
                "I prefer tea to coffee, especially when it's cold.",
            ),
        ]

        # B2 level Italian patterns with complex grammar
        b2_patterns = [
            (
                "Qualora dovessi cambiare idea, fammi sapere il prima possibile.",
                "Should you change your mind, let me know as soon as possible.",
            ),
            (
                "Nonostante avesse studiato molto, non riuscì a superare l'esame.",
                "Despite having studied a lot, he didn't manage to pass the exam.",
            ),
            (
                "Era necessario che tutti fossero d'accordo prima di procedere.",
                "It was necessary that everyone agreed before proceeding.",
            ),
            (
                "Avrebbe preferito che gli avessero detto la verità fin dall'inizio.",
                "He would have preferred that they had told him the truth from the beginning.",
            ),
            (
                "Purché tu sia disposto a collaborare, riusciremo a completare il progetto.",
                "As long as you're willing to collaborate, we'll manage to complete the project.",
            ),
        ]

        # Teaching scenarios for different levels
        a2_scenarios = [
            "asking questions about daily routines",
            "expressing likes and dislikes",
            "describing past events with simple tenses",
            "making polite requests and offers",
            "talking about family and personal background",
        ]

        b1_scenarios = [
            "expressing past habits and experiences",
            "discussing hypothetical situations",
            "making comparisons and contrasts",
            "describing ongoing actions in the past",
            "expressing preferences and opinions",
        ]

        b2_scenarios = [
            "using advanced subjunctive forms",
            "expressing complex conditional thoughts",
            "discussing abstract concepts and ideas",
            "using sophisticated temporal relationships",
            "demonstrating cultural and linguistic nuances",
        ]

        # Generate content with A2 focus (A2: 30%, B1: 45%, B2: 25%)
        a2_count = int(target_count * 0.30)
        b1_count = int(target_count * 0.45)
        b2_count = target_count - a2_count - b1_count

        # Generate A2 content (30% of synthetic content)
        for i in range(a2_count):
            pattern = random.choice(a2_patterns)
            scenario = random.choice(a2_scenarios)

            # Varied A2 question formats
            a2_questions = [
                f"What does '{pattern[0]}' mean?",
                f"How do you understand '{pattern[0]}'?",
                f"Can you explain '{pattern[0]}'?",
                f"I'm confused by '{pattern[0]}'. Help?",
                f"What's happening in '{pattern[0]}'?",
                f"Could you translate '{pattern[0]}'?",
            ]

            user_question = random.choice(a2_questions)

            # More natural A2 responses
            a2_response_templates = [
                "That means '{pattern[1]}'. This is common for {scenario}.",
                "It's '{pattern[1]}'. You'll use this when {scenario}.",
                "The translation is '{pattern[1]}'. Perfect for {scenario}.",
                "'{pattern[1]}' - great example of {scenario} in Italian!",
            ]

            response_template = random.choice(a2_response_templates)
            response = response_template.format(pattern=pattern, scenario=scenario)

            # Detect semantic topic for A2 content
            a2_semantic_topic = self.detect_semantic_topic(pattern[0], pattern[1])

            synthetic_examples.append(
                {
                    "conversation_id": f"synthetic_a2_{i}",
                    "source": "synthetic_generation",
                    "level": "A2",
                    "topic": f"a2_{a2_semantic_topic}",
                    "conversation": [
                        {"role": "user", "content": user_question},
                        {"role": "assistant", "content": response},
                    ],
                }
            )

        # Generate B1 content (45% of synthetic content)
        for i in range(b1_count):
            pattern = random.choice(b1_patterns)
            scenario = random.choice(b1_scenarios)

            # Varied B1 question formats
            b1_questions = [
                f"What does this mean: '{pattern[0]}'?",
                f"Can you explain '{pattern[0]}'?",
                f"I'm working on this sentence: '{pattern[0]}'. Help?",
                f"This looks complex: '{pattern[0]}'. What's it saying?",
                f"Could you break down '{pattern[0]}'?",
                f"What's the meaning of '{pattern[0]}'?",
            ]

            user_question = random.choice(b1_questions)

            # More natural B1 responses (less verbose)
            if random.random() < 0.3:  # 30% chance of encouragement
                encouragement = random.choice(self.question_encouragements).strip()
                response = f"{encouragement} That means '{pattern[1]}'. "
            else:
                response = f"That means '{pattern[1]}'. "

            # Add brief explanation about the grammar concept
            response += f"This shows {scenario} in Italian."

            # Detect semantic topic for B1 content
            b1_semantic_topic = self.detect_semantic_topic(pattern[0], pattern[1])

            synthetic_examples.append(
                {
                    "conversation_id": f"synthetic_b1_{i}",
                    "source": "synthetic_generation",
                    "level": "B1",
                    "topic": f"b1_{b1_semantic_topic}",
                    "conversation": [
                        {"role": "user", "content": user_question},
                        {"role": "assistant", "content": response},
                    ],
                }
            )

        # Generate B2 content (25% of synthetic content)
        for i in range(b2_count):
            pattern = random.choice(b2_patterns)
            scenario = random.choice(b2_scenarios)

            # Varied B2 question formats
            b2_questions = [
                f"This is quite advanced: '{pattern[0]}'. What does it mean?",
                f"Can you help with this complex sentence: '{pattern[0]}'?",
                f"I'm struggling with '{pattern[0]}'. Could you explain?",
                f"What's the meaning behind '{pattern[0]}'?",
                f"This formal Italian is confusing: '{pattern[0]}'. Help?",
                f"Could you translate this advanced sentence: '{pattern[0]}'?",
            ]

            user_question = random.choice(b2_questions)

            # More concise B2 responses
            if random.random() < 0.4:  # 40% chance of encouragement for complex content
                encouragement = random.choice(self.question_encouragements).strip()
                response = f"{encouragement} It means '{pattern[1]}'. "
            else:
                response = f"It means '{pattern[1]}'. "

            # Brief explanation of advanced concept
            response += f"This demonstrates {scenario} - typical of formal or literary Italian."

            # Detect semantic topic for B2 content
            b2_semantic_topic = self.detect_semantic_topic(pattern[0], pattern[1])

            synthetic_examples.append(
                {
                    "conversation_id": f"synthetic_b2_{i}",
                    "source": "synthetic_generation",
                    "level": "B2",
                    "topic": f"b2_{b2_semantic_topic}",
                    "conversation": [
                        {"role": "user", "content": user_question},
                        {"role": "assistant", "content": response},
                    ],
                }
            )

        print(f"✅ Generated {len(synthetic_examples)} synthetic B1/B2 examples")
        return synthetic_examples

    def process_all_data_robust(
        self, input_dir: str, output_dir: str, target_size: int = 10000
    ) -> None:
        """Process all raw data with focus on B1/B2 content and target size."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        print("🇮🇹 Starting Robust Data Processing for B1/B2 Focus...")
        print(f"🎯 Target dataset size: {target_size} examples")

        all_sentence_pairs = []
        all_training_examples = []

        # Process Tatoeba (prioritize B1/B2)
        tatoeba_dir = input_path / "tatoeba_content"
        if tatoeba_dir.exists():
            print("\n📖 Processing Tatoeba data...")

            # Process large text file
            large_file = tatoeba_dir / "manythings_italian_english_ita.txt"
            if large_file.exists():
                pairs = self.process_tatoeba_file_streaming(str(large_file), target_size // 2)
                all_sentence_pairs.extend(pairs)

            # Process JSON files
            for json_file in tatoeba_dir.glob("*.json"):
                if "summary" not in json_file.name:
                    pairs = self.process_tatoeba_file_streaming(str(json_file), target_size // 4)
                    all_sentence_pairs.extend(pairs)

        # Process Babbel (focus on B1/B2)
        babbel_dir = input_path / "babbel_content"
        if babbel_dir.exists():
            print("\n📖 Processing Babbel data with B1/B2 focus...")
            babbel_examples = self.process_babbel_with_level_focus(babbel_dir)
            all_training_examples.extend(babbel_examples)

        # Create diverse conversation types from sentence pairs
        if all_sentence_pairs:
            print(
                f"\n🔄 Creating diverse conversation types from {len(all_sentence_pairs)} sentence pairs..."
            )
            # Prioritize B1/B2 pairs
            b1_b2_pairs = [p for p in all_sentence_pairs if p["cefr_level"] in ["B1", "B2"]]
            other_pairs = [p for p in all_sentence_pairs if p["cefr_level"] not in ["B1", "B2"]]

            print(f"Found {len(b1_b2_pairs)} B1/B2 pairs and {len(other_pairs)} other pairs")

            # Process more B1/B2 pairs
            selected_pairs = b1_b2_pairs + other_pairs[: target_size // 4]
            conversation_examples = self.create_diverse_conversation_types(
                selected_pairs[: target_size // 3]
            )
            all_training_examples.extend(conversation_examples)

        # Generate synthetic B1/B2 content if we don't have enough
        current_b1_b2_count = len(
            [ex for ex in all_training_examples if ex.get("level") in ["B1", "B2"]]
        )
        target_b1_b2_count = target_size // 3  # Aim for 33% B1/B2 content

        if current_b1_b2_count < target_b1_b2_count:
            print(f"\n🔄 Generating synthetic B1/B2 content...")
            synthetic_examples = self.generate_synthetic_b1_b2_content(
                target_b1_b2_count - current_b1_b2_count
            )
            all_training_examples.extend(synthetic_examples)

        # Balance dataset by CEFR level
        print(f"\n📊 Balancing dataset by CEFR level...")
        level_counts = {}
        for example in all_training_examples:
            level = example.get("level", "A1")
            level_counts[level] = level_counts.get(level, 0) + 1

        print(f"Current level distribution: {level_counts}")

        # Limit total to target size
        if len(all_training_examples) > target_size:
            # Prioritize B1/B2 content
            b1_b2_examples = [ex for ex in all_training_examples if ex.get("level") in ["B1", "B2"]]
            other_examples = [
                ex for ex in all_training_examples if ex.get("level") not in ["B1", "B2"]
            ]

            # Take all B1/B2 and fill remainder with others
            selected_examples = b1_b2_examples + other_examples[: target_size - len(b1_b2_examples)]
            all_training_examples = selected_examples[:target_size]

        # Create final dataset
        if all_training_examples:
            self.create_final_dataset(all_training_examples, output_path)
        else:
            print("❌ No training examples generated")

    def create_final_dataset(self, examples: List[Dict], output_dir: Path) -> None:
        """Create final training dataset with improved CEFR distribution."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle examples
        random.shuffle(examples)

        # Calculate splits
        total = len(examples)
        train_size = int(total * 0.8)
        val_size = int(total * 0.15)

        train_examples = examples[:train_size]
        val_examples = examples[train_size : train_size + val_size]
        test_examples = examples[train_size + val_size :]

        # Analyze final distribution
        final_distribution = {}
        for example in examples:
            level = example.get("level", "A1")
            final_distribution[level] = final_distribution.get(level, 0) + 1

        # Save datasets
        self.save_chat_dataset(train_examples, output_dir / "train.jsonl")
        self.save_chat_dataset(val_examples, output_dir / "validation.jsonl")
        self.save_chat_dataset(test_examples, output_dir / "test.jsonl")

        # Save metadata
        metadata = {
            "total_examples": total,
            "train_examples": len(train_examples),
            "validation_examples": len(val_examples),
            "test_examples": len(test_examples),
            "cefr_distribution": final_distribution,
            "creation_date": datetime.now().isoformat(),
            "processing_method": "robust_b1_b2_focused",
        }

        with open(output_dir / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n🎉 Base Dataset Created:")
        print(f"   Train: {len(train_examples)} examples")
        print(f"   Validation: {len(val_examples)} examples")
        print(f"   Test: {len(test_examples)} examples")
        print(f"   Total: {total} examples")
        print(f"   CEFR Distribution: {final_distribution}")
        print(
            f"   B1/B2 Percentage: {((final_distribution.get('B1', 0) + final_distribution.get('B2', 0)) / total * 100):.1f}%"
        )
        print(f"\n📋 Next Step: Run Colab LLM processing to improve grammar explanations")
        print(f"   Output will be saved to processed_llm_improved/ directory")

    def save_chat_dataset(self, examples: List[Dict], output_file: Path) -> None:
        """Save examples in Hugging Face chat format."""
        with open(output_file, "w", encoding="utf-8") as f:
            for example in examples:
                chat_example = {
                    "messages": example["conversation"],
                    "metadata": {
                        "conversation_id": example["conversation_id"],
                        "source": example["source"],
                        "level": example["level"],
                        "topic": example["topic"],
                    },
                }
                f.write(json.dumps(chat_example, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Robust processing for full Italian teaching dataset"
    )
    parser.add_argument("--input", required=True, help="Input directory containing raw data")
    parser.add_argument(
        "--output", required=True, help="Output directory for processed training data"
    )
    parser.add_argument("--target-size", type=int, default=10000, help="Target dataset size")

    args = parser.parse_args()

    processor = RobustDataProcessor()
    processor.process_all_data_robust(args.input, args.output, args.target_size)


if __name__ == "__main__":
    main()
