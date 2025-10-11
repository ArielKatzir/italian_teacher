"""
FastAPI inference service for Italian Exercise Generator model on Colab GPU.

This module provides a complete inference API that can be deployed on Google Colab
with GPU acceleration using vLLM. Uses the fine-tuned italian_exercise_generator_lora
model for generating high-quality Italian language exercises.
"""

import json
import re
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Request/Response models
class ExerciseRequest(BaseModel):
    cefr_level: str = Field(..., description="CEFR level (A1-C2)")
    grammar_focus: str = Field(..., description="Grammar topic")
    topic: str = Field(..., description="Content topic")
    quantity: int = Field(..., description="Number of exercises")
    exercise_types: List[str] = Field(..., description="Types of exercises")
    max_tokens: Optional[int] = Field(2500, description="Max tokens per generation")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")


class ExerciseResponse(BaseModel):
    exercises: List[dict]
    generated_tokens: int
    inference_time: float
    parsing_strategy: str


def create_inference_app(llm, port: int = 8001):
    """
    Create a FastAPI application for exercise generation.

    Args:
        llm: vLLM model instance
        port: Port number for the server

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Italian Exercise Generator API",
        description="Italian Teacher homework generation service using italian_exercise_generator_lora",
        version="2.0.0",
    )

    @app.get("/")
    async def root():
        """Health check endpoint"""
        import torch

        return {
            "status": "healthy",
            "service": "Italian Exercise Generator API",
            "model": "italian_exercise_generator_lora",
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "port": port,
            "version": "2.0.0",
        }

    @app.get("/health")
    async def health():
        """Detailed health check"""
        import torch

        return {
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "gpu_memory_allocated_gb": (
                torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            ),
            "model_loaded": llm is not None,
            "port": port,
        }

    @app.post("/generate", response_model=ExerciseResponse)
    async def generate_exercises(request: ExerciseRequest):
        """
        Generate Italian language exercises using italian_exercise_generator_lora model.

        Uses multiple parsing strategies to reliably extract exercises from model output.
        """
        try:
            import time

            start_time = time.time()

            # Create numbered placeholders to guide the model
            exercise_numbers = ", ".join([f"#{i+1}" for i in range(request.quantity)])

            topic = request.topic if request.topic else "general Italian"
            grammer = request.grammar_focus if request.grammar_focus else "general practice"

            # Build topic and grammar instructions
            topic_instruction = f"about '{topic}'"
            grammar_instruction = f"focusing on {grammer}"
            focus_text = f"{topic_instruction} {grammar_instruction}".strip()

            # Build grammar-specific instruction
            grammar_rule = ""
            if "past" in grammer.lower() or "passato" in grammer.lower():
                grammar_rule = "\n⚠️ MANDATORY: Use ONLY past tense (passato prossimo like 'ho fatto', 'sono andato' OR imperfetto like 'facevo', 'andavo'). NO present tense!"
            elif "present" in grammer.lower() or "presente" in grammer.lower():
                grammar_rule = "\n⚠️ MANDATORY: Use ONLY present tense (presente indicativo like 'faccio', 'vado'). NO past or future!"
            elif "future" in grammer.lower() or "futuro" in grammer.lower():
                grammar_rule = "\n⚠️ MANDATORY: Use ONLY future tense (futuro semplice like 'farò', 'andrò'). NO present or past!"
            elif "conditional" in grammer.lower() or "condizionale" in grammer.lower():
                grammar_rule = (
                    "\n⚠️ MANDATORY: Use ONLY conditional (condizionale like 'farei', 'andrei')."
                )
            elif "subjunctive" in grammer.lower() or "congiuntivo" in grammer.lower():
                grammar_rule = (
                    "\n⚠️ MANDATORY: Use ONLY subjunctive (congiuntivo like 'faccia', 'vada')."
                )

            # Optimized prompt that emphasizes the actual topic and diverse MC options
            prompt = f"""Create exactly {request.quantity} Italian language exercises ({exercise_numbers}) in JSON format {focus_text}.

REQUIREMENTS:
Level: {request.cefr_level}
Topic: {topic}
Grammar: {grammer}{grammar_rule}
Exercise types: {', '.join(request.exercise_types)}

CRITICAL RULES:
1. TOPIC: Every exercise MUST be about "{topic}" - stay on topic throughout
2. REALISM: Use factual, natural scenarios appropriate for the topic
3. GRAMMAR: EVERY SINGLE exercise MUST test "{grammer}" at {request.cefr_level} level
4. MULTIPLE CHOICE: Provide 4 DIFFERENT grammatical forms as options
5. CONSISTENCY: Do not mix different topics or introduce unrelated subjects

OUTPUT FORMAT - JSON array with exercises testing {grammer}:
[
  {{"type": "fill_in_blank", "question": "[Italian sentence about {topic} with ___ blank for {grammer}]", "correct_answer": "[conjugated form in {grammer}]", "options": null, "explanation": "[grammar rule explanation]"}},
  {{"type": "translation", "question": "Translate: [English sentence about {topic} in {grammer}]", "correct_answer": "[Italian translation using {grammer}]", "options": null, "explanation": "[grammar note]"}},
  {{"type": "multiple_choice", "question": "[Italian sentence about {topic} with blank]", "correct_answer": "[correct form in {grammer}]", "options": ["[alt1]", "[alt2]", "[alt3]", "[alt4]"], "explanation": "[why this form is correct]"}}
]

NOW GENERATE {request.quantity} EXERCISES ABOUT "{topic}" TESTING "{grammer}" (remember: {grammer} ONLY!):
["""

            # Configure sampling parameters
            from vllm import SamplingParams

            # Use optimal temperature (0.4) unless explicitly overridden
            actual_temp = request.temperature if request.temperature != 0.7 else 0.4

            sampling_params = SamplingParams(
                temperature=actual_temp,
                top_p=0.9,
                max_tokens=request.max_tokens,
                stop=["<|im_end|>", "<|endoftext|>"],
            )

            # Generate with vLLM
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()

            # Prepend opening bracket since prompt ends with it
            if not generated_text.startswith("["):
                generated_text = "[" + generated_text

            # Ensure array is closed
            if not generated_text.rstrip().endswith("]"):
                last_brace = generated_text.rfind("}")
                if last_brace > 0:
                    generated_text = generated_text[: last_brace + 1] + "\n]"

            # Parse exercises using multiple strategies
            exercises = _parse_exercises(generated_text, request)

            # Calculate metrics
            inference_time = time.time() - start_time
            generated_tokens = len(outputs[0].outputs[0].token_ids)

            parsing_strategy = exercises.get("strategy", "unknown")
            exercise_list = exercises.get("exercises", [])

            print(
                f"✅ Generated {len(exercise_list)} exercises in {inference_time:.2f}s ({generated_tokens} tokens) using {parsing_strategy}"
            )

            return ExerciseResponse(
                exercises=exercise_list,
                generated_tokens=generated_tokens,
                inference_time=inference_time,
                parsing_strategy=parsing_strategy,
            )

        except Exception as e:
            import traceback

            error_detail = traceback.format_exc()
            print(f"❌ Generation error: {e}")
            print(error_detail)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    return app


def _parse_exercises(generated_text: str, request: ExerciseRequest) -> dict:
    """
    Parse exercises from generated text using multiple fallback strategies.

    Args:
        generated_text: Raw model output
        request: Original request with configuration

    Returns:
        Dictionary with 'exercises' list and 'strategy' name
    """
    exercises = None
    parsing_strategy = "unknown"

    # Strategy 1: Direct JSON parsing
    if exercises is None:
        try:
            cleaned = generated_text
            if "```json" in cleaned:
                json_start = cleaned.find("```json") + 7
                json_end = cleaned.find("```", json_start)
                cleaned = cleaned[json_start:json_end].strip()
            elif "```" in cleaned:
                json_start = cleaned.find("```") + 3
                json_end = cleaned.find("```", json_start)
                cleaned = cleaned[json_start:json_end].strip()

            parsed = json.loads(cleaned)

            if isinstance(parsed, dict) and "exercises" in parsed:
                exercises = parsed["exercises"]
                parsing_strategy = "strategy1_dict_with_key"
            elif isinstance(parsed, list):
                exercises = parsed
                parsing_strategy = "strategy1_direct_array"
            else:
                exercises = [parsed]
                parsing_strategy = "strategy1_single_object"

            print(f"✅ Strategy 1 succeeded: {parsing_strategy}")

        except json.JSONDecodeError as e:
            print(f"⚠️  Strategy 1 failed: {e}")

    # Strategy 2: Regex extraction of JSON array
    if exercises is None:
        try:
            json_pattern = r"\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]"
            matches = re.findall(json_pattern, generated_text)

            if matches:
                longest_match = max(matches, key=len)
                exercises = json.loads(longest_match)
                parsing_strategy = "strategy2_regex_extraction"
                print(f"✅ Strategy 2 succeeded: found JSON array via regex")

        except Exception as e:
            print(f"⚠️  Strategy 2 failed: {e}")

    # Strategy 3: Find individual JSON objects
    if exercises is None:
        try:
            exercise_pattern = (
                r'\{[^{}]*"type"[^{}]*"question"[^{}]*"correct_answer"[^{}]*"explanation"[^{}]*\}'
            )
            exercise_matches = re.findall(exercise_pattern, generated_text, re.DOTALL)

            if exercise_matches:
                exercises = []
                for match in exercise_matches[: request.quantity]:
                    try:
                        ex = json.loads(match)
                        exercises.append(ex)
                    except:
                        continue

                if exercises:
                    parsing_strategy = "strategy3_individual_objects"
                    print(f"✅ Strategy 3 succeeded: extracted {len(exercises)} objects")

        except Exception as e:
            print(f"⚠️  Strategy 3 failed: {e}")

    # Strategy 4: Text parsing fallback
    if exercises is None:
        try:
            exercises = []
            chunks = re.split(r"\n(?=\d+\.|\*\*Exercise|\#)", generated_text)

            for chunk in chunks[: request.quantity]:
                if len(chunk.strip()) < 10:
                    continue

                exercise = {
                    "type": request.exercise_types[len(exercises) % len(request.exercise_types)],
                    "question": "",
                    "correct_answer": "",
                    "options": None,
                    "explanation": "",
                }

                lines = [l.strip() for l in chunk.split("\n") if l.strip()]
                if lines:
                    exercise["question"] = lines[0][:200]

                answer_match = re.search(
                    r"(?:answer|risposta|correct)[:\s]*([^\n]+)", chunk, re.IGNORECASE
                )
                if answer_match:
                    exercise["correct_answer"] = answer_match.group(1).strip()[:100]
                else:
                    exercise["correct_answer"] = "See explanation"

                exercise["explanation"] = chunk[:300]
                exercises.append(exercise)

            if exercises:
                parsing_strategy = "strategy4_text_extraction"
                print(f"✅ Strategy 4 succeeded: extracted {len(exercises)} from text")

        except Exception as e:
            print(f"⚠️  Strategy 4 failed: {e}")

    # Strategy 5: Ultimate fallback
    if exercises is None or len(exercises) == 0:
        print("⚠️  All parsing strategies failed, using fallback")
        exercises = []
        text_chunks = generated_text.split("\n\n")

        for i in range(min(request.quantity, max(len(text_chunks), 1))):
            chunk = text_chunks[i] if i < len(text_chunks) else generated_text[:500]

            exercises.append(
                {
                    "type": request.exercise_types[i % len(request.exercise_types)],
                    "question": f"Exercise {i+1}: {chunk[:200].strip()}",
                    "correct_answer": "See explanation",
                    "options": None,
                    "explanation": f"Generated content for {request.cefr_level} level {request.grammar_focus}",
                }
            )

        parsing_strategy = "strategy5_fallback"

    # Ensure correct quantity
    if len(exercises) < request.quantity:
        print(f"⚠️  Only got {len(exercises)}/{request.quantity} exercises, padding...")
        while len(exercises) < request.quantity:
            exercises.append(
                {
                    "type": request.exercise_types[len(exercises) % len(request.exercise_types)],
                    "question": f"Additional exercise {len(exercises) + 1}",
                    "correct_answer": "See explanation",
                    "options": None,
                    "explanation": f"{request.cefr_level} level {request.grammar_focus} practice",
                }
            )

    exercises = exercises[: request.quantity]

    # Validate and fix multiple choice exercises
    _validate_multiple_choice(exercises, request)

    # Validate Italian grammar and tense consistency (after JSON parsing!)
    _validate_italian_grammar(exercises, grammar_focus=request.grammar_focus)

    return {"exercises": exercises, "strategy": parsing_strategy}


def _validate_multiple_choice(exercises: List[dict], request: ExerciseRequest) -> None:
    """
    Validate multiple choice exercises and fix issues.

    Checks for:
    - Duplicate options (all options are the same)
    - Missing options
    - Correct answer not in options
    """
    for i, ex in enumerate(exercises):
        if ex.get("type") != "multiple_choice":
            continue

        options = ex.get("options", [])
        correct_answer = ex.get("correct_answer", "")

        # Check for duplicate options (ANY duplicates, not just all same)
        if options and len(set(options)) < len(options):
            unique_count = len(set(options))
            print(
                f"⚠️  Exercise {i+1}: Multiple choice has duplicate options ({unique_count}/4 unique). Regenerating options..."
            )

            # Generate diverse distractor options based on grammar focus
            ex["options"] = _generate_distractor_options(
                correct_answer=correct_answer,
                grammar_focus=request.grammar_focus,
                cefr_level=request.cefr_level,
            )

        # Check if correct answer is in options
        elif options and correct_answer not in options:
            print(f"⚠️  Exercise {i+1}: Correct answer not in options. Fixing...")
            ex["options"][0] = correct_answer

        # Check for missing or insufficient options
        elif not options or len(options) < 4:
            print(f"⚠️  Exercise {i+1}: Missing or insufficient options. Generating...")
            ex["options"] = _generate_distractor_options(
                correct_answer=correct_answer,
                grammar_focus=request.grammar_focus,
                cefr_level=request.cefr_level,
            )


def _generate_distractor_options(
    correct_answer: str, grammar_focus: str, cefr_level: str
) -> List[str]:
    """
    Generate plausible distractor options for multiple choice questions.

    This is a fallback when the model generates duplicate options.
    Creates grammatically related but incorrect alternatives.
    """
    options = [correct_answer]

    # Common verb conjugation patterns for Italian
    if any(
        keyword in grammar_focus.lower()
        for keyword in ["tense", "verb", "passato", "presente", "futuro"]
    ):
        # For verb tenses, generate common conjugation alternatives
        common_distractors = {
            # Present tense patterns
            "mangio": ["ho mangiato", "mangerò", "mangerei"],
            "vado": ["sono andato", "andrò", "andrei"],
            "sono": ["ero", "sarò", "sarei"],
            "ho": ["avevo", "avrò", "avrei"],
            # Past tense patterns
            "ho mangiato": ["mangio", "mangiavo", "mangiai"],
            "sono andato": ["vado", "andavo", "andai"],
            "era": ["è", "sarà", "sarebbe"],
            "aveva": ["ha", "avrà", "avrebbe"],
        }

        # Try to find similar pattern
        for pattern, distractors in common_distractors.items():
            if pattern in correct_answer.lower():
                options.extend(distractors[:3])
                break

    # If we still don't have 4 options, add generic distractors
    generic_distractors = ["altro", "diverso", "sbagliato"]
    while len(options) < 4:
        distractor = (
            generic_distractors[len(options) - 1]
            if len(options) <= 3
            else f"opzione {len(options)}"
        )
        options.append(distractor)

    return options[:4]


def _validate_italian_grammar(exercises: List[dict], grammar_focus: str = None) -> None:
    """
    Validate and fix Italian grammar errors in exercise text fields.

    Uses spaCy Italian NLP model to:
    1. Detect and fix article-noun gender agreement errors
    2. Validate tense consistency with grammar focus
    3. Report warnings for tense mismatches

    IMPORTANT: This runs AFTER JSON parsing, so it only modifies
    the text content, never the JSON structure.

    Args:
        exercises: List of exercise dicts
        grammar_focus: Grammar focus string (e.g., "past_tense", "present_tense")
    """
    try:
        pass

        import spacy

        # Load Italian language model (lazy load)
        if not hasattr(_validate_italian_grammar, "nlp"):
            try:
                _validate_italian_grammar.nlp = spacy.load("it_core_news_sm")
                print("✅ Italian NLP model loaded for grammar validation")
            except OSError:
                print("⚠️  Italian spaCy model not found. Skipping grammar validation.")
                print("   Install with: python -m spacy download it_core_news_sm")
                return

        nlp = _validate_italian_grammar.nlp

        for i, ex in enumerate(exercises):
            # Check each Italian text field
            fields_to_check = ["question", "answer", "explanation"]

            for field in fields_to_check:
                if field not in ex or not ex[field]:
                    continue

                text = ex[field]
                original_text = text

                # Skip if text is not Italian (e.g., English translation questions)
                if field == "question" and ex.get("type") == "translation":
                    if text.startswith("Translate:") or text.startswith("What is"):
                        continue

                # Parse with spaCy
                doc = nlp(text)

                # Validate tense consistency (warnings only)
                if grammar_focus and field in ["question", "answer"]:
                    _check_tense_consistency(doc, grammar_focus, i, field)

                # Find article-noun pairs and check gender agreement
                corrections = []
                for j, token in enumerate(doc):
                    # Check if this is an article followed by a noun
                    if token.pos_ == "DET" and j + 1 < len(doc):
                        next_token = doc[j + 1]

                        if next_token.pos_ == "NOUN":
                            article = token.text.lower()
                            next_token.text
                            noun_gender = next_token.morph.get("Gender")

                            if noun_gender:
                                correct_article = _get_correct_article(article, noun_gender[0])

                                if correct_article and correct_article != article:
                                    # Preserve original capitalization
                                    if token.text[0].isupper():
                                        correct_article = correct_article.capitalize()

                                    corrections.append(
                                        {
                                            "start": token.idx,
                                            "end": token.idx + len(token.text),
                                            "old": token.text,
                                            "new": correct_article,
                                        }
                                    )

                # Apply corrections (from end to start to preserve offsets)
                if corrections:
                    corrected_text = text
                    for correction in reversed(corrections):
                        corrected_text = (
                            corrected_text[: correction["start"]]
                            + correction["new"]
                            + corrected_text[correction["end"] :]
                        )

                    if corrected_text != original_text:
                        print(f"⚠️  Exercise {i+1} ({field}): Grammar corrected")
                        print(f"    Before: {original_text[:80]}")
                        print(f"    After:  {corrected_text[:80]}")
                        ex[field] = corrected_text

    except ImportError:
        print("⚠️  spaCy not installed. Skipping grammar validation.")
        print("   Install with: pip install spacy && python -m spacy download it_core_news_sm")


def _get_correct_article(article: str, gender: str) -> Optional[str]:
    """
    Get the correct Italian article based on gender.

    Args:
        article: Current article (un/una/il/la/lo/l'/gli/le/i)
        gender: Noun gender from spaCy ('Masc' or 'Fem')

    Returns:
        Correct article or None if no correction needed
    """
    article_lower = article.lower()

    # Masculine articles: il, lo, l', un, uno, i, gli
    masculine_articles = {"il", "lo", "l'", "un", "uno", "i", "gli"}
    # Feminine articles: la, l', una, le
    feminine_articles = {"la", "l'", "una", "le"}

    if gender == "Masc":
        # If using feminine article with masculine noun
        if article_lower in feminine_articles:
            # Map to masculine equivalent
            if article_lower == "una":
                return "un"
            elif article_lower == "la":
                return "il"
            elif article_lower == "le":
                return "i"  # or 'gli' depending on next letter

    elif gender == "Fem":
        # If using masculine article with feminine noun
        if article_lower in masculine_articles:
            # Map to feminine equivalent
            if article_lower in ["un", "uno"]:
                return "una"
            elif article_lower in ["il", "lo"]:
                return "la"
            elif article_lower in ["i", "gli"]:
                return "le"

    return None


def _check_tense_consistency(doc, grammar_focus: str, exercise_idx: int, field: str) -> None:
    """
    Check if the tense used in the text matches the grammar focus.

    Args:
        doc: spaCy Doc object
        grammar_focus: Grammar focus (e.g., "past_tense", "present_tense")
        exercise_idx: Exercise index for reporting
        field: Field name being checked
    """
    # Map grammar focus to expected verb tenses
    expected_tenses = {
        "past_tense": {"Past"},  # Passato prossimo, imperfetto
        "present_tense": {"Pres"},  # Presente
        "future_tense": {"Fut"},  # Futuro semplice
        "conditional": {"Cnd"},  # Condizionale
        "subjunctive": {"Sub"},  # Congiuntivo
    }

    if grammar_focus not in expected_tenses:
        return

    # Find all verbs in the sentence
    verbs = [token for token in doc if token.pos_ == "VERB" and token.dep_ != "aux"]

    if not verbs:
        return

    # Check if any verb matches expected tense
    found_tenses = set()
    for verb in verbs:
        tense = verb.morph.get("Tense")
        if tense:
            found_tenses.add(tense[0])

    expected = expected_tenses[grammar_focus]

    # If we found verbs but none match expected tense
    if found_tenses and not found_tenses.intersection(expected):
        print(f"🟠 Exercise {exercise_idx + 1} ({field}): Tense mismatch")
        print(f"    Expected: {grammar_focus} ({', '.join(expected)})")
        print(f"    Found: {', '.join(found_tenses)}")
        print(f"    Text: {doc.text[:80]}")
