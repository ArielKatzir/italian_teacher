#!/usr/bin/env python3
"""
Parse Italian exercises from onlineitalianclub.com HTML files.
Handles two formats:
1. JavaScript embedded questions/answers
2. HTML with separate solution pages
"""

import json
import re
import sys
from pathlib import Path


class JavaScriptExerciseParser:
    """Parse exercises from JavaScript arrays in HTML."""

    def parse(self, html_content):
        """Extract exercises from JavaScript var question arrays."""
        # Try multiple choice format first (4 elements)
        exercises = self._parse_multiple_choice(html_content)

        if exercises:
            return exercises

        # Fall back to fill-in-blank format (3 elements)
        return self._parse_fill_in_blank(html_content)

    def _parse_multiple_choice(self, html_content):
        """Parse multiple choice format: Array(before, option1, option2, after)"""
        exercises = []

        # Pattern: var question1 = new Array("before","opt1","opt2","after");
        pattern = r'var\s+question\d+\s*=\s*new\s+Array\s*\(\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\s*\)'
        matches = re.findall(pattern, html_content)

        if not matches:
            return []

        # Get correct answers array: var correctArray = new Array(1,2,1,2,...);
        correct_pattern = r"var\s+correctArray\s*=\s*new\s+Array\s*\(([\d,\s]+)\)"
        correct_match = re.search(correct_pattern, html_content)

        if not correct_match:
            return []

        correct_answers = [int(x.strip()) for x in correct_match.group(1).split(",")]

        if len(matches) != len(correct_answers):
            return []

        for (before, opt1, opt2, after), correct_idx in zip(matches, correct_answers):
            # Clean text
            before = self._clean_text(before)
            opt1 = self._clean_text(opt1)
            opt2 = self._clean_text(opt2)
            after = self._clean_text(after)

            # Select correct answer (1-indexed in JavaScript)
            answer = opt1 if correct_idx == 1 else opt2

            # Build question with both options
            question = f"{before} [{opt1}/{opt2}] {after}".strip()

            exercises.append({"question": question, "answer": answer, "options": [opt1, opt2]})

        return exercises

    def _parse_fill_in_blank(self, html_content):
        """Parse fill-in-blank format: Array(before, answer, after, size)"""
        exercises = []

        # Pattern: var question1 = new Array("text before","answer", "text after", "size");
        pattern = r'var\s+question\d+\s*=\s*new\s+Array\s*\(\s*"([^"]*)",\s*"([^"]*)",\s*"([^"]*)",\s*"[^"]*"\s*\)'
        matches = re.findall(pattern, html_content)

        for before, answer, after in matches:
            # Clean up HTML entities and tags
            before = self._clean_text(before)
            after = self._clean_text(after)
            answer = self._clean_text(answer)

            # Combine question parts with blank indicator
            question = f"{before} ___ {after}".strip()

            exercises.append({"question": question, "answer": answer})

        return exercises

    def _clean_text(self, text):
        """Clean HTML entities and tags from text."""
        # Replace HTML entities
        text = text.replace("&egrave;", "è")
        text = text.replace("&Egrave;", "È")
        text = text.replace("&eacute;", "é")
        text = text.replace("&agrave;", "à")
        text = text.replace("&ograve;", "ò")
        text = text.replace("&ugrave;", "ù")
        text = text.replace("&igrave;", "ì")
        text = text.replace("&ccedil;", "ç")
        text = text.replace("&#39;", "'")
        text = text.replace("&quot;", '"')
        text = text.replace("&nbsp;", " ")

        # Remove <br> tags but keep space
        text = re.sub(r"<br\s*/?>", " ", text)

        # Remove other HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Clean up multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()


class HTMLExerciseParser:
    """Parse exercises from HTML with blanks (___) and separate solution page."""

    def parse_questions(self, html_content):
        """Extract questions with blanks from HTML."""
        questions = []

        # Find list items with blanks
        pattern = r"<li[^>]*>(.*?)</li>"
        items = re.findall(pattern, html_content, re.DOTALL)

        for item in items:
            # Clean HTML tags but keep text
            text = re.sub(r"<[^>]+>", " ", item)
            text = re.sub(r"\s+", " ", text).strip()

            # Only keep items with blanks
            if "___" in text:
                questions.append(text)

        return questions

    def parse_solutions(self, html_content):
        """Extract answers from solution HTML (bold/strong tags)."""
        answers = []

        # Find strong/bold text in list items
        pattern = r"<li[^>]*>.*?<strong>(.*?)</strong>.*?</li>"
        items = re.findall(pattern, html_content, re.DOTALL | re.IGNORECASE)

        for item in items:
            # Clean and normalize
            answer = re.sub(r"<[^>]+>", "", item).strip()
            if answer:
                answers.append(answer)

        return answers

    def combine(self, questions, answers):
        """Combine questions and answers into exercises."""
        exercises = []

        # Match questions with answers
        for q, a in zip(questions, answers):
            exercises.append({"question": q, "answer": a})

        return exercises


def extract_metadata_from_filename(filepath):
    """Extract metadata from HTML filename."""
    name = Path(filepath).stem

    # Try to extract grammar topic
    grammar_match = re.search(
        r"(Subjunctive|Conditional|Imperative|Pronoun|Preposition)", name, re.IGNORECASE
    )
    grammar = grammar_match.group(1).lower() if grammar_match else "grammar"

    return {"grammar_focus": grammar, "source": "onlineitalianclub.com"}


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_onlineitalianclub.py <html_file> [solution_file]")
        sys.exit(1)

    html_file = sys.argv[1]
    solution_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Read HTML content (try different encodings)
    encodings = ["utf-8", "windows-1252", "latin-1", "iso-8859-1"]
    html_content = None
    for encoding in encodings:
        try:
            with open(html_file, "r", encoding=encoding) as f:
                html_content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if html_content is None:
        print(f"Error: Could not decode {html_file} with any encoding")
        sys.exit(1)

    # Detect format and parse
    if "var question" in html_content:
        # JavaScript format
        parser = JavaScriptExerciseParser()
        exercises = parser.parse(html_content)
        exercise_type = "fill_in_blank"
    elif solution_file:
        # HTML with separate solution
        parser = HTMLExerciseParser()
        questions = parser.parse_questions(html_content)

        with open(solution_file, "r", encoding="utf-8") as f:
            solution_content = f.read()

        answers = parser.parse_solutions(solution_content)
        exercises = parser.combine(questions, answers)
        exercise_type = "fill_in_blank"
    else:
        print("Error: Could not determine format or missing solution file")
        sys.exit(1)

    # Extract metadata
    metadata = extract_metadata_from_filename(html_file)

    # Output result
    result = {"exercise_type": exercise_type, "metadata": metadata, "exercises": exercises}

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✅ Extracted {len(exercises)} exercises", file=sys.stderr)


if __name__ == "__main__":
    main()
