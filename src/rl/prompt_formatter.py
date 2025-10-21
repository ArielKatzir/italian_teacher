"""
Prompt formatting utilities for GRPO training.

Formats training requests into prompts with:
- Chat templates (Llama3 format)
- Grammar-specific instructions
- Few-shot examples for common issues
- Explicit constraints
"""

from typing import Any, Dict, List


def format_prompt_with_chat_template(
    request: Dict[str, Any], tokenizer, add_examples: bool = True
) -> str:
    """
    Format request using chat template + detailed instructions + optional few-shot examples.

    Args:
        request: Training request dict with level, grammar_focus, topic, etc.
        tokenizer: HuggingFace tokenizer with chat template support
        add_examples: If True, include few-shot examples for common grammar focuses

    Returns:
        Formatted prompt string ready for model input
    """
    topic = request.get("topic", "general Italian")
    grammar = request.get("grammar_focus", "general practice")
    level = request["level"]
    num_exercises = request["num_exercises"]

    # Build focus description
    topic_instruction = f"about '{topic}'"
    grammar_instruction = f"focusing on {grammar}"
    focus_text = f"{topic_instruction} {grammar_instruction}".strip()

    # Grammar-specific mandatory rules
    grammar_rule = _get_grammar_rule(grammar)

    # Create numbered placeholders to guide the model (like old version)
    exercise_numbers = ", ".join([f"#{i+1}" for i in range(num_exercises)])

    # SIMPLIFIED - back to old working format!
    # The old prompt was SHORT and CLEAR, the new one is TOO COMPLEX
    # NOTE: add_examples parameter is now ignored - examples were making prompt too complex
    user_message = f"""Create exactly {num_exercises} Italian language exercises ({exercise_numbers}) in JSON format {focus_text}.

REQUIREMENTS:
Level: {level}
Topic: {topic}
Grammar: {grammar}{grammar_rule}
Exercise types: {', '.join(request['exercise_types'])}

CRITICAL RULES:
1. TOPIC: Every exercise MUST be about "{topic}" - stay on topic throughout
2. REALISM: Use factual, natural scenarios appropriate for the topic
3. GRAMMAR: EVERY SINGLE exercise MUST test "{grammar}" at {level} level
4. MULTIPLE CHOICE: Provide 4 DIFFERENT grammatical forms as options
5. CONSISTENCY: Do not mix different topics or introduce unrelated subjects

OUTPUT FORMAT - JSON array with exercises testing {grammar}:
[
  {{"type": "fill_in_blank", "question": "[Italian sentence about {topic} with ___ blank for {grammar}]", "correct_answer": "[conjugated form in {grammar}]", "options": null, "explanation": "[grammar rule explanation]"}},
  {{"type": "translation", "question": "Translate: [English sentence about {topic} in {grammar}]", "correct_answer": "[Italian translation using {grammar}]", "options": null, "explanation": "[grammar note]"}},
  {{"type": "multiple_choice", "question": "[Italian sentence about {topic} with blank]", "correct_answer": "[correct form in {grammar}]", "options": ["[alt1]", "[alt2]", "[alt3]", "[alt4]"], "explanation": "[why this form is correct]"}}
]

NOW GENERATE {num_exercises} EXERCISES ABOUT "{topic}" TESTING "{grammar}" (remember: {grammar} ONLY!):
[
"""

    # Apply chat template
    messages = [
        {
            "role": "system",
            "content": "You are an expert Italian language teacher. Generate high-quality exercises based on the assignment specification. Output exercises in JSON format.",
        },
        {"role": "user", "content": user_message},
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return formatted_prompt


def _get_options_example(exercise_type: str) -> str:
    """Get correct options format for the exercise type."""
    if exercise_type == "multiple_choice":
        return '["option1", "option2", "option3", "option4"]'
    else:
        return "null"


def _get_grammar_rule(grammar: str) -> str:
    """Get mandatory grammar rule for specific focus areas."""
    grammar_lower = grammar.lower()

    if "past" in grammar_lower or "passato" in grammar_lower:
        return "\n⚠️ MANDATORY: Use ONLY past tense (passato prossimo like 'ho fatto', 'sono andato' OR imperfetto like 'facevo', 'andavo'). NO present tense!"
    elif "present" in grammar_lower or "presente" in grammar_lower:
        return "\n⚠️ MANDATORY: Use ONLY present tense (presente indicativo like 'faccio', 'vado'). NO past or future!"
    elif "future" in grammar_lower or "futuro" in grammar_lower:
        return "\n⚠️ MANDATORY: Use ONLY future tense (futuro semplice like 'farò', 'andrò'). NO present or past!"
    elif "conditional" in grammar_lower or "condizionale" in grammar_lower:
        return "\n⚠️ MANDATORY: Use ONLY conditional mood (condizionale like 'farei', 'andrei'). NO indicative!"
    elif "subjunctive" in grammar_lower or "congiuntivo" in grammar_lower:
        return "\n⚠️ MANDATORY: Use ONLY subjunctive mood (congiuntivo like 'che io faccia', 'che tu vada'). NO indicative!"

    return ""


def _get_few_shot_examples(grammar: str, topic: str, level: str, exercise_types: List[str]) -> str:
    """
    Get professional few-shot examples that match the requested exercise types.

    CRITICAL: Examples MUST match the requested type and include context clues.
    """
    grammar_lower = grammar.lower()
    examples = []

    # Generate examples for EACH requested type
    for ex_type in exercise_types[:2]:  # Show max 2 type examples to save tokens

        if ex_type == "fill_in_blank":
            # FILL-IN-BLANK: MUST include context clues!
            if "past" in grammar_lower:
                examples.append(
                    f"""✓ PERFECT fill_in_blank (past tense with base verb clue):
  {{
    "type": "fill_in_blank",
    "question": "Ieri (andare) ___ al cinema con Marco.",
    "correct_answer": "sono andato",
    "options": null,
    "explanation": "Passato prossimo di andare, 1st person singular"
  }}"""
                )
            elif "present" in grammar_lower:
                examples.append(
                    f"""✓ PERFECT fill_in_blank (present tense with base verb clue):
  {{
    "type": "fill_in_blank",
    "question": "Ogni mattina (fare) ___ colazione alle otto.",
    "correct_answer": "faccio",
    "options": null,
    "explanation": "Presente indicativo di fare, 1st person singular"
  }}"""
                )
            elif "article" in grammar_lower:
                examples.append(
                    f"""✓ PERFECT fill_in_blank (article with translation context):
  {{
    "type": "fill_in_blank",
    "question": "Translate: The book is interesting → ___ libro è interessante.",
    "correct_answer": "Il",
    "options": null,
    "explanation": "Definite article masculine singular"
  }}"""
                )
            else:
                # Generic example with context
                examples.append(
                    f"""✓ PERFECT fill_in_blank (with context clue):
  {{
    "type": "fill_in_blank",
    "question": "La mia famiglia (essere) ___ molto grande, ho tre fratelli.",
    "correct_answer": "è",
    "options": null,
    "explanation": "Shows family size as context for {grammar}"
  }}"""
                )

        elif ex_type == "multiple_choice":
            # MULTIPLE CHOICE: MUST have 4 options array
            if "past" in grammar_lower:
                examples.append(
                    f"""✓ PERFECT multiple_choice (past tense with 4 options):
  {{
    "type": "multiple_choice",
    "question": "Ieri Maria ___ al mercato.",
    "correct_answer": "è andata",
    "options": ["va", "è andata", "andrà", "andava"],
    "explanation": "Passato prossimo di andare, 3rd person singular feminine"
  }}"""
                )
            elif "present" in grammar_lower:
                examples.append(
                    f"""✓ PERFECT multiple_choice (present tense with 4 options):
  {{
    "type": "multiple_choice",
    "question": "Marco ___ la pizza ogni venerdì.",
    "correct_answer": "mangia",
    "options": ["mangiava", "mangia", "mangerà", "ha mangiato"],
    "explanation": "Presente indicativo di mangiare, 3rd person singular"
  }}"""
                )
            else:
                examples.append(
                    f"""✓ PERFECT multiple_choice (with 4 grammatical options):
  {{
    "type": "multiple_choice",
    "question": "I bambini ___ nel parco.",
    "correct_answer": "giocano",
    "options": ["gioca", "giocano", "giocate", "giocare"],
    "explanation": "Testing {grammar} with person/number variations"
  }}"""
                )

        elif ex_type == "translation":
            # TRANSLATION: English → Italian
            if "past" in grammar_lower:
                examples.append(
                    f"""✓ PERFECT translation (past tense):
  {{
    "type": "translation",
    "question": "Yesterday I went to the market.",
    "correct_answer": "Ieri sono andato al mercato.",
    "options": null,
    "explanation": "Passato prossimo translation"
  }}"""
                )
            else:
                examples.append(
                    f"""✓ PERFECT translation (English to Italian):
  {{
    "type": "translation",
    "question": "The students study every day.",
    "correct_answer": "Gli studenti studiano ogni giorno.",
    "options": null,
    "explanation": "Testing {grammar} in translation context"
  }}"""
                )

    if not examples:
        return ""

    # Add BAD example to show what NOT to do
    bad_example = ""
    if "fill_in_blank" in exercise_types:
        bad_example = f"""

✗ BAD fill_in_blank (NO CONTEXT - impossible to answer!):
  {{
    "type": "fill_in_blank",
    "question": "La collocazione ___.",
    "correct_answer": "è importante"
  }}
  ← WRONG! No clues! Use: "Translate: The placement is important → La collocazione ___." """

    return "\n\n".join(examples) + bad_example
