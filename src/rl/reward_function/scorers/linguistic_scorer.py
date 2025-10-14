"""
Linguistic quality scorer.

Comprehensive Italian grammar validation (0-35 points).
"""

from typing import Any, Dict, List, Optional, Tuple

import spacy

from ...rl_data import ARTICLE_RULES, GENDER_EXCEPTIONS, INVARIANT_ADJECTIVES
from .base import BaseScorer


class LinguisticScorer(BaseScorer):
    """
    Scores comprehensive Italian linguistic quality (0-35 points).

    Checks:
    - Article-noun gender agreement with elision, partitives (8 pts)
    - Number agreement (singular/plural consistency) (7 pts)
    - Adjective-noun agreement (gender and number) (7 pts)
    - Verb-subject agreement (person and number) (6 pts)
    - Preposition usage (common errors) (4 pts)
    - Pronoun agreement and positioning (3 pts)
    """

    def __init__(self, nlp: spacy.language.Language):
        super().__init__(nlp)
        self.gender_exceptions = GENDER_EXCEPTIONS
        self.article_rules = ARTICLE_RULES
        self.invariant_adjectives = INVARIANT_ADJECTIVES

    def score(self, exercise: Dict[str, Any], request: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Score comprehensive Italian linguistic quality."""
        errors = []
        text = self._extract_italian_text(exercise)

        if not text:
            return 0.0, ["No Italian text found"]

        # Parse with spaCy
        doc = self.nlp(text)

        # Component scores (total 35)
        article_score = 15.0
        number_score = 7.0
        adjective_score = 7.0
        verb_score = 6.0
        preposition_score = 4.0
        pronoun_score = 5.0

        # 1. Article-noun gender agreement (8 pts)
        article_errors = self._check_article_noun_agreement(doc)
        if article_errors:
            article_score = max(0, 8 - len(article_errors) * 3)
            errors.extend(article_errors)

        # 2. Number agreement (7 pts)
        number_errors = self._check_number_agreement(doc)
        if number_errors:
            number_score = max(0, 7 - len(number_errors) * 3.5)
            errors.extend(number_errors)

        # 3. Adjective-noun agreement (7 pts)
        adj_errors = self._check_adjective_noun_agreement(doc)
        if adj_errors:
            adjective_score = max(0, 7 - len(adj_errors) * 3.5)
            errors.extend(adj_errors)

        # 4. Verb-subject agreement (6 pts)
        verb_errors = self._check_verb_subject_agreement(doc)
        if verb_errors:
            verb_score = max(0, 6 - len(verb_errors) * 3)
            errors.extend(verb_errors)

        # 5. Preposition usage (4 pts)
        prep_errors = self._check_preposition_usage(doc, text)
        if prep_errors:
            preposition_score = max(0, 4 - len(prep_errors) * 2)
            errors.extend(prep_errors)

        # 6. Pronoun agreement (3 pts)
        pronoun_errors = self._check_pronoun_agreement(doc)
        if pronoun_errors:
            pronoun_score = max(0, 3 - len(pronoun_errors) * 1.5)
            errors.extend(pronoun_errors)

        total_linguistic_score = (
            article_score
            + number_score
            + adjective_score
            + verb_score
            + preposition_score
            + pronoun_score
        )

        return total_linguistic_score, errors

    def _check_article_noun_agreement(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Check article-noun gender agreement with comprehensive rules.

        Handles: definite, indefinite, partitive, and elided articles.
        """
        errors = []

        for token in doc:
            # Check if this is an article
            if token.pos_ != "DET":
                continue

            # Find the noun this article modifies
            noun = None
            if token.head.pos_ == "NOUN":
                noun = token.head
            elif token.dep_ == "det" and token.head.pos_ == "NOUN":
                noun = token.head

            if not noun:
                continue

            article = token.text.lower()
            noun_text = noun.text.lower()

            # Get noun gender (check exceptions first)
            noun_gender = self._get_noun_gender(noun_text, noun)
            if not noun_gender:
                continue  # Can't validate without gender info

            # Get noun number
            noun_number = noun.morph.get("Number")
            if not noun_number:
                noun_number = ["Sing"]  # Default assumption

            # Validate article matches noun
            expected_articles = self._get_valid_articles(noun_text, noun_gender, noun_number[0])

            if article not in expected_articles:
                errors.append(
                    f"Article-noun mismatch: '{article} {noun_text}' "
                    f"(expected: {'/'.join(expected_articles[:2])})"
                )

        return errors

    def _check_number_agreement(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Check singular/plural number agreement.

        Verifies articles, adjectives, and nouns agree in number.
        """
        errors = []

        for token in doc:
            if token.pos_ not in ["NOUN", "PROPN"]:
                continue

            noun_number = token.morph.get("Number")
            if not noun_number:
                continue

            # Check article-noun number agreement
            for child in token.children:
                if child.pos_ == "DET":
                    det_number = child.morph.get("Number")
                    if det_number and det_number != noun_number:
                        errors.append(
                            f"Number mismatch: '{child.text} {token.text}' "
                            f"(article: {det_number[0]}, noun: {noun_number[0]})"
                        )

        return errors

    def _check_adjective_noun_agreement(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Check adjective-noun gender and number agreement.

        Adjectives must match the noun they modify in both gender and number.
        """
        errors = []

        for token in doc:
            if token.pos_ != "ADJ":
                continue

            # Check if this is an invariant adjective
            if token.text.lower() in self.invariant_adjectives:
                continue  # These don't need to agree

            # Find the noun this adjective modifies
            noun = token.head if token.head.pos_ in ["NOUN", "PROPN"] else None

            if not noun:
                continue

            adj_gender = token.morph.get("Gender")
            adj_number = token.morph.get("Number")
            noun_gender = self._get_noun_gender(noun.text.lower(), noun)
            noun_number = noun.morph.get("Number")

            # Check gender agreement
            if adj_gender and noun_gender and adj_gender[0] != noun_gender:
                errors.append(
                    f"Adj-noun gender mismatch: '{token.text} {noun.text}' "
                    f"(adj: {adj_gender[0]}, noun: {noun_gender})"
                )

            # Check number agreement
            if adj_number and noun_number and adj_number != noun_number:
                errors.append(
                    f"Adj-noun number mismatch: '{token.text} {noun.text}' "
                    f"(adj: {adj_number[0]}, noun: {noun_number[0]})"
                )

        return errors

    def _check_verb_subject_agreement(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Check verb-subject agreement (person and number).

        Examples of errors:
        - "io vai" (should be "io vado")
        - "noi va" (should be "noi andiamo")
        """
        errors = []

        for token in doc:
            if token.pos_ != "VERB":
                continue

            # Find subject
            subject = None
            for child in token.children:
                if child.dep_ in ["nsubj", "nsubj:pass"]:
                    subject = child
                    break

            if not subject:
                continue

            # Get verb person and number
            verb_person = token.morph.get("Person")
            verb_number = token.morph.get("Number")

            # Get subject number
            subj_number = subject.morph.get("Number")

            # Check if subject is a pronoun (for person agreement)
            if subject.pos_ == "PRON":
                subj_person = subject.morph.get("Person")
                if subj_person and verb_person and subj_person != verb_person:
                    errors.append(
                        f"Verb-subject person mismatch: '{subject.text} {token.text}' "
                        f"(subj: {subj_person[0]}p, verb: {verb_person[0]}p)"
                    )

            # Check number agreement
            if subj_number and verb_number and subj_number != verb_number:
                errors.append(
                    f"Verb-subject number mismatch: '{subject.text} {token.text}' "
                    f"(subj: {subj_number[0]}, verb: {verb_number[0]})"
                )

        return errors

    def _check_preposition_usage(self, doc: spacy.tokens.Doc, text: str) -> List[str]:
        """
        Check common Italian preposition errors.

        Common mistakes:
        - "a" vs "in" (andare a Roma vs andare in Italia)
        - Missing prepositions with verbs (pensare di, parlare di)
        - Wrong preposition choice
        """
        errors = []

        # Common verb + preposition patterns
        verb_prep_patterns = {
            "andare": {"a": ["città", "scuola", "casa"], "in": ["italia", "francia", "centro"]},
            "pensare": {"di": True, "a": True},  # pensare di fare, pensare a qualcuno
            "parlare": {"di": True, "con": True},  # parlare di qualcosa, parlare con qualcuno
            "cercare": {"di": True},  # cercare di fare
            "smettere": {"di": True},  # smettere di fumare
            "continuare": {"a": True},  # continuare a lavorare
            "cominciare": {"a": True},  # cominciare a studiare
        }

        # Check for common errors
        text_lower = text.lower()

        # Check: "andare a Italia" (should be "in Italia")
        if "andare a italia" in text_lower or "andare a francia" in text_lower:
            errors.append(
                "Wrong preposition: 'andare a Italia' → should be 'andare in Italia' (country)"
            )

        # Check: "andare in Roma" (should be "a Roma")
        if "andare in roma" in text_lower or "andare in milano" in text_lower:
            errors.append("Wrong preposition: 'andare in Roma' → should be 'andare a Roma' (city)")

        # Check for missing prepositions with infinitives
        for token in doc:
            if token.pos_ == "VERB" and token.lemma_ in verb_prep_patterns:
                # Check if followed by infinitive without preposition
                for child in token.children:
                    if child.pos_ == "VERB" and "Inf" in child.morph.get("VerbForm", []):
                        # Check if there's a preposition between them
                        has_prep = False
                        for between_token in doc[token.i : child.i]:
                            if between_token.pos_ == "ADP":
                                has_prep = True
                                break

                        if not has_prep and verb_prep_patterns[token.lemma_] is not True:
                            errors.append(
                                f"Missing preposition: '{token.text} {child.text}' "
                                f"(may need preposition after '{token.lemma_}')"
                            )

        return errors

    def _check_pronoun_agreement(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Check pronoun gender and number agreement.

        Checks:
        - Subject pronouns match verb (already in verb-subject check)
        - Object pronouns match antecedent gender/number
        - Reflexive pronouns are correct
        """
        errors = []

        for token in doc:
            if token.pos_ == "PRON":
                # Check reflexive pronouns with reflexive verbs
                if "Reflex" in token.morph.get("PronType", []):
                    # Find the verb this pronoun is attached to
                    verb = token.head if token.head.pos_ == "VERB" else None
                    if verb:
                        # Check person/number agreement
                        pron_person = token.morph.get("Person")
                        verb_person = verb.morph.get("Person")
                        if pron_person and verb_person and pron_person != verb_person:
                            errors.append(
                                f"Reflexive pronoun mismatch: '{token.text}' with '{verb.text}' "
                                f"(pron: {pron_person[0]}p, verb: {verb_person[0]}p)"
                            )

        return errors

    def _get_noun_gender(self, noun_text: str, noun_token: spacy.tokens.Token) -> Optional[str]:
        """
        Get noun gender, checking exceptions first, then spaCy.
        """
        # Check exceptions dictionary first
        if noun_text in self.gender_exceptions:
            return self.gender_exceptions[noun_text]

        # Use spaCy morphology
        gender = noun_token.morph.get("Gender")
        if gender:
            return gender[0]

        return None

    def _get_valid_articles(self, noun_text: str, gender: str, number: str) -> List[str]:
        """
        Get list of valid articles for a noun based on gender, number, and phonetics.

        Returns list of acceptable articles (e.g., ["il", "lo"] for some masculine nouns).
        """
        articles = []

        # Determine phonetic context
        starts_with_vowel = noun_text[0] in "aeiouàèéìòù"
        starts_with_s_consonant = noun_text.startswith(
            ("sc", "sp", "st", "sb", "sf", "sg", "sl", "sm", "sn", "sr", "sv")
        )
        starts_with_z = noun_text.startswith("z")
        starts_with_gn = noun_text.startswith("gn")
        starts_with_ps = noun_text.startswith("ps")
        starts_with_x = noun_text.startswith("x")
        starts_with_y = noun_text.startswith("y")

        # Special phonetic contexts
        special_phonetic = (
            starts_with_s_consonant
            or starts_with_z
            or starts_with_gn
            or starts_with_ps
            or starts_with_x
            or starts_with_y
        )

        # Definite articles
        if gender == "Masc":
            if number == "Sing":
                if starts_with_vowel:
                    articles.append("l'")
                elif special_phonetic:
                    articles.append("lo")
                else:
                    articles.append("il")
            else:  # Plur
                if starts_with_vowel or special_phonetic:
                    articles.append("gli")
                else:
                    articles.append("i")
        else:  # Fem
            if number == "Sing":
                if starts_with_vowel:
                    articles.append("l'")
                else:
                    articles.append("la")
            else:  # Plur
                articles.append("le")

        # Indefinite articles (singular only)
        if number == "Sing":
            if gender == "Masc":
                if special_phonetic:
                    articles.append("uno")
                else:
                    articles.append("un")
            else:  # Fem
                if starts_with_vowel:
                    articles.append("un'")
                else:
                    articles.append("una")

        # Partitive articles
        if gender == "Masc":
            if number == "Sing":
                if starts_with_vowel:
                    articles.append("dell'")
                elif special_phonetic:
                    articles.append("dello")
                else:
                    articles.append("del")
            else:  # Plur
                if starts_with_vowel or special_phonetic:
                    articles.append("degli")
                else:
                    articles.append("dei")
        else:  # Fem
            if number == "Sing":
                if starts_with_vowel:
                    articles.append("dell'")
                else:
                    articles.append("della")
            else:  # Plur
                articles.append("delle")

        return articles

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
        return 35.0

    @property
    def name(self) -> str:
        return "linguistic_quality"
