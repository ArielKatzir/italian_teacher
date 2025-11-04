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
    Scores comprehensive Italian linguistic quality (re-weighted to 0-15 points).

    Checks:
    - Noun phrase agreement (article, adjective, noun) (3 pts)
    - Verb-subject agreement (person and number) (2 pts)
    - Past participle agreement (with essere/avere) (2 pts)
    - Pronoun agreement, placement, and redundancy (2 pts)
    - Sentence Fragments (e.g., missing main clause) (2 pts)
    - Adverb/Adjective usage errors (e.g., redundancy) (1 pt)
    - Preposition usage (common errors) (1 pt)
    - Repetition check for consecutive identical words (2 pts)
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

        # Component scores (total 15) - Re-weighted
        np_agreement_score = 3.0
        verb_score = 2.0
        participle_score = 2.0
        pronoun_score = 2.0
        fragment_score = 2.0
        adv_adj_score = 1.0
        preposition_score = 1.0
        repetition_score = 2.0

        # 1. Noun Phrase Agreement (articles, adjectives) - CONSOLIDATED
        np_errors = self._check_noun_phrase_agreement(doc)
        if np_errors:
            np_agreement_score = max(0, 3.0 - len(np_errors) * 1.5)
            errors.extend(np_errors)

        # 2. Verb-subject agreement (4 pts)
        verb_errors = self._check_verb_subject_agreement(doc)
        if verb_errors:
            verb_score = max(0, 2.0 - len(verb_errors) * 2)
            errors.extend(verb_errors)

        # 3. Past Participle Agreement (5 pts) - NEW
        participle_errors = self._check_past_participle_agreement(doc)
        if participle_errors:
            participle_score = max(0, 2.0 - len(participle_errors) * 1)
            errors.extend(participle_errors)

        # 4. Adverb/Adjective Usage Errors (2 pts) - NEW
        adv_adj_errors = self._check_adverb_adjective_errors(doc)
        if adv_adj_errors:
            adv_adj_score = 0.0 # Penalize heavily for these structural errors
            errors.extend(adv_adj_errors)

        # 5. Sentence Fragment Check (2 pts) - NEW
        fragment_errors = self._check_sentence_fragments(doc)
        if fragment_errors:
            fragment_score = 0.0 # This is a major error
            errors.extend(fragment_errors)

        # 6. Preposition usage (1 pt)
        prep_errors = self._check_preposition_usage(doc, text)
        if prep_errors:
            preposition_score = max(0, 1 - len(prep_errors) * 1)
            errors.extend(prep_errors)

        # 7. Pronoun agreement (2 pts)
        pronoun_errors = self._check_pronoun_agreement(doc)
        pronoun_errors.extend(self._check_subject_pronoun_redundancy(doc)) # Add new check
        if pronoun_errors:
            pronoun_score = max(0, 2.0 - len(pronoun_errors) * 1)
            errors.extend(pronoun_errors)

        # 8. Consecutive Repetition Check (2 pts)
        repetition_errors = self._check_consecutive_repetition(doc)
        if repetition_errors:
            repetition_score = 0.0 # This is a major error
            errors.extend(repetition_errors)


        total_linguistic_score = (
            np_agreement_score
            + verb_score
            + participle_score
            + fragment_score
            + adv_adj_score
            + preposition_score
            + pronoun_score
            + repetition_score
        )

        return total_linguistic_score, errors

    def _check_noun_phrase_agreement(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Consolidated check for noun phrase agreement.
        Verifies that articles and adjectives agree with the noun they modify
        in both gender and number.
        """
        errors = []

        for token in doc:
            if token.pos_ not in ["NOUN", "PROPN"]:
                continue

            noun = token
            noun_text = noun.text.lower()

            # Get noun gender (check exceptions first)
            noun_gender = self._get_noun_gender(noun_text, noun)
            if not noun_gender:
                continue  # Can't validate without gender info

            # Get noun number
            noun_number = noun.morph.get("Number")
            if not noun_number:
                noun_number = ["Sing"]  # Default assumption
            noun_number_val = noun_number[0]

            # New Check: Detect multiple determiners for the same noun
            determiners = [child for child in token.children if child.dep_ == "det"]
            if len(determiners) > 1:
                errors.append(f"Linguistic error: Multiple determiners ('{[d.text for d in determiners]}') for noun '{noun.text}'.")
                continue # Skip further checks for this broken phrase

            # New Check: Detect if a verb incorrectly splits a noun from its prepositional modifier
            # e.g., "compressione [VERB] del materiale"
            for child in token.children:
                if child.dep_ == "prep": # e.g., 'del' in "compressione del materiale"
                    # Check if there's a verb between the noun and its prepositional modifier
                    if any(t.pos_ in ["VERB", "AUX"] for t in doc[token.i + 1 : child.i]):
                        errors.append(f"Linguistic error: Verb incorrectly splits noun '{token.text}' from its modifier '{child.text}'.")

            # Check all dependent articles and adjectives
            for child in token.children:
                # Check Article Agreement
                if child.pos_ == "DET" and child.dep_ == "det":
                    article = child.text.lower()
                    expected_articles = self._get_valid_articles(noun_text, noun_gender, noun_number_val)
                    if article not in expected_articles:
                        errors.append(
                            f"Article-noun mismatch: '{article} {noun_text}' "
                            f"(expected one of: {', '.join(expected_articles)})"
                        )
                
                # Check Adjective Agreement
                elif child.pos_ == "ADJ" and child.dep_ == "amod":
                    adj = child
                    if adj.text.lower() in self.invariant_adjectives:
                        continue

                    adj_gender = adj.morph.get("Gender")
                    adj_number = adj.morph.get("Number")

                    if adj_gender and adj_gender[0] != noun_gender:
                        errors.append(
                            f"Adj-noun gender mismatch: '{adj.text} {noun.text}' "
                            f"(adj: {adj_gender[0]}, noun: {noun_gender})"
                        )
                    
                    if adj_number and adj_number[0] != noun_number_val:
                        errors.append(
                            f"Adj-noun number mismatch: '{adj.text} {noun.text}' "
                            f"(adj: {adj_number[0]}, noun: {noun_number_val})"
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

        # New Check: Detect multiple consecutive main verbs (e.g., "vai andiamo")
        for i in range(len(doc) - 1):
            token = doc[i]
            next_token = doc[i+1]
            # Check for two consecutive finite verbs (not auxiliaries or modals followed by infinitive)
            is_token_main_verb = token.pos_ == "VERB" and "Inf" not in token.morph.get("VerbForm", [])
            is_next_token_main_verb = next_token.pos_ == "VERB" and "Inf" not in next_token.morph.get("VerbForm", [])
            
            if is_token_main_verb and is_next_token_main_verb:
                # Check if they are not coordinated by 'e', 'o', etc.
                if next_token.dep_ != "conj":
                    errors.append(f"Linguistic error: Multiple consecutive main verbs '{token.text} {next_token.text}'.")
                    break # Penalize once for this severe error

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

    def _check_past_participle_agreement(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Checks past participle agreement in compound tenses.
        1. With 'essere', participle agrees with the subject.
        2. With 'avere', participle agrees with a preceding direct object pronoun (lo, la, li, le).
        """
        errors = []
        for token in doc:
            # Find a past participle
            if token.pos_ == "VERB" and "Part" in token.morph.get("VerbForm", []):
                participle = token
                aux = participle.head

                # Ensure the head is an auxiliary verb
                if aux.pos_ != "AUX":
                    continue

                participle_gender = participle.morph.get("Gender")
                participle_number = participle.morph.get("Number")

                # Case 1: Auxiliary is 'essere'
                if aux.lemma_ == "essere":
                    subject = next((child for child in aux.children if child.dep_ == "nsubj"), None)
                    if subject:
                        subj_gender = self._get_noun_gender(subject.text, subject)
                        subj_number = subject.morph.get("Number")

                        if participle_gender and subj_gender and participle_gender[0] != subj_gender:
                            errors.append(f"Participle agreement error (essere): '{participle.text}' should agree with subject '{subject.text}' (gender).")
                        if participle_number and subj_number and participle_number != subj_number:
                            errors.append(f"Participle agreement error (essere): '{participle.text}' should agree with subject '{subject.text}' (number).")

                # Case 2: Auxiliary is 'avere'
                elif aux.lemma_ == "avere":
                    # Find a preceding direct object pronoun (obj dependency on the auxiliary)
                    direct_obj_pron = next((child for child in aux.children if child.dep_ == "obj" and child.pos_ == "PRON"), None)
                    
                    if direct_obj_pron and direct_obj_pron.i < aux.i:
                        # Check if it's one of the agreeing pronouns
                        if direct_obj_pron.text.lower() in ["lo", "la", "li", "le", "l'"]:
                            pron_gender = direct_obj_pron.morph.get("Gender")
                            pron_number = direct_obj_pron.morph.get("Number")

                            # spaCy might not get gender for "l'", so we can't always check
                            if participle_gender and pron_gender and participle_gender != pron_gender:
                                errors.append(f"Participle agreement error (avere): '{participle.text}' should agree with pronoun '{direct_obj_pron.text}' (gender).")
                            if participle_number and pron_number and participle_number != pron_number:
                                errors.append(f"Participle agreement error (avere): '{participle.text}' should agree with pronoun '{direct_obj_pron.text}' (number).")

        return errors

    def _check_adverb_adjective_errors(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Checks for specific, common errors in adverb and adjective usage.
        - Redundant comparatives (e.g., "più è più efficiente")
        - Misplaced modifiers (e.g., "La dinamo più è...")
        """
        errors = []
        text_lower = doc.text.lower()

        # Check for redundant comparatives like "più è più"
        # This is a strong heuristic for a common model failure mode.
        if "più è più" in text_lower or "meno è meno" in text_lower:
            errors.append("Linguistic error: Redundant comparative structure (e.g., 'più è più').")

        # Check for misplaced modifiers, specifically 'più' acting as an adjective on a noun
        # when it's not a pronoun. e.g., "La dinamo più è..."
        for i in range(len(doc) - 2):
            # Pattern: NOUN + "più" + VERB/AUX
            if doc[i].pos_ == "NOUN" and doc[i+1].text.lower() == "più" and doc[i+2].pos_ in ["VERB", "AUX"]:
                # This is a strong indicator of the "La dinamo più è..." error.
                errors.append(f"Linguistic error: Misplaced modifier '{doc[i+1].text}' after noun '{doc[i].text}'.")
                break

        return errors

    def _check_sentence_fragments(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Checks for sentence fragments, like a relative clause without a main clause.
        e.g., "Il telo che è stato usato."

        IMPORTANT: Ignores fill-in-blank hint format like "(andare)" or exercise markers.
        """
        errors = []
        text_lower = doc.text.lower()

        # Skip fragment check if this looks like a fill-in-blank with hint
        # Pattern: (verb) indicates this is an exercise with a hint
        if '(' in text_lower and ')' in text_lower:
            # Likely a hint format, skip fragment checking
            return errors

        for sent in doc.sents:
            # Heuristic: If the root of the sentence is not the main verb (but part of a sub-clause),
            # it's likely a fragment.
            root = sent.root
            # A common fragment starts with a relative pronoun ('che', 'cui')
            has_subject = any(tok.dep_ in ("nsubj", "nsubj:pass") for tok in sent)

            if not has_subject and len(sent) > 2:
                # If there's no nominal subject, it's likely a fragment unless it's a command.
                if root.pos_ != "VERB" or "Imp" not in root.morph.get("Mood", []):
                    errors.append(f"Linguistic error: Sentence appears to be a fragment (missing subject): '{sent.text}'")
            if sent[0].pos_ in ["PRON", "SCONJ"] and sent[0].text.lower() == "che":
                 # If the root's head is itself (meaning it's the top of the tree) and it's not a main clause...
                 if root.head == root and root.dep_ != "ROOT":
                     errors.append(f"Linguistic error: Sentence appears to be a fragment (e.g., a relative clause without a main clause): '{sent.text}'")
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

                # Check for incorrect placement of object pronouns with auxiliary verbs
                # e.g., "La barchetta la è stata..." is wrong.
                if token.dep_ == "obj" and token.head.pos_ == "AUX":
                     # This is a strong heuristic. A direct object pronoun should not be a direct child of an auxiliary.
                     # It should be attached to the main verb or be part of a different clause.
                     errors.append(f"Pronoun placement error: Object pronoun '{token.text}' seems incorrectly attached to auxiliary '{token.head.text}'.")

        return errors

    def _check_subject_pronoun_redundancy(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Checks for redundant subject pronouns, which can sound unnatural.
        e.g., "Io vado al mercato." is correct but "Vado al mercato." is more common.
        """
        errors = []
        # These pronouns are often redundant if the verb form is unique
        redundant_pronouns = {"io", "tu", "noi", "voi"}

        for token in doc:
            # Find a subject pronoun
            if token.dep_ == "nsubj" and token.text.lower() in redundant_pronouns:
                verb = token.head
                # Check if it's a simple declarative sentence without contrast/emphasis
                # Heuristic: if the sentence is short and has no conjunctions, the pronoun is likely redundant.
                is_simple_clause = not any(c.dep_ == "conj" for c in verb.children)
                
                if verb.pos_ == "VERB" and is_simple_clause:
                    # This is a "soft" error, indicating unnatural phrasing rather than a grammatical mistake.
                    errors.append(f"Linguistic style: Subject pronoun '{token.text}' may be redundant.")
                    break # Penalize only once per sentence for this style issue.
        return errors

    def _check_consecutive_repetition(self, doc: spacy.tokens.Doc) -> List[str]:
        """
        Checks for identical consecutive words, which is almost always an error.
        e.g., "il camion trasporta trasporta merci"
        """
        errors = []
        for i in range(len(doc) - 1):
            token1 = doc[i]
            token2 = doc[i+1]

            # Check for identical text, ignoring case, for non-punctuation tokens
            if token1.is_alpha and token2.is_alpha and token1.text.lower() == token2.text.lower():
                errors.append(f"Linguistic error: Consecutive repetition of word '{token1.text}'.")
                break # Penalize once for this severe error

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
        For fill-in-the-blank, inserts the correct answer into the question.
        For translation exercises, ONLY uses the Italian answer (correct_answer).
        """
        import re

        exercise_type = exercise.get("type", "")

        # For translation exercises, ONLY analyze the Italian answer (correct_answer)
        # The question is in English, so we must not analyze it
        if exercise_type == "translation":
            if "correct_answer" in exercise:
                return exercise["correct_answer"].strip()
            return ""

        # For fill-in-the-blank, insert the answer into the question
        if exercise_type == "fill_in_blank":
            question = exercise.get("question", "")
            answer = exercise.get("correct_answer", "")
            if "___" in question and answer:
                return question.replace("___", answer, 1).strip()
            # Fallback if no blank found
            return f"{question} {answer}".strip()

        # For multiple_choice and other types, analyze the question
        # Remove common English prompts first
        question = exercise.get("question", "")
        question_clean = re.sub(
            r"^(Translate|Fill in the blank|Choose the correct answer):\s*",
            "",
            question,
            flags=re.IGNORECASE,
        ).strip()

        # Heuristic to check if the extracted text is likely Italian
        italian_indicators = [
            "il", "la", "le", "gli", "lo", "un", "una", "è", "sono", "di", "a", "per", "che",
            "ho", "hai", "ha", "abbiamo", "avete", "hanno", # common verb forms
            "io", "tu", "lui", "lei", "noi", "voi", "loro" # common pronouns
        ]

        if question_clean and any(indicator in question_clean.lower().split() for indicator in italian_indicators):
            return question_clean

        return "" # Return empty string if no Italian text detected


    @property
    def max_score(self) -> float:
        return 15.0

    @property
    def name(self) -> str:
        return "linguistic_quality"
