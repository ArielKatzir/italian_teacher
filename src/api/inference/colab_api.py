"""
FastAPI inference service for Marco v3 model on Colab GPU.

This module provides a complete inference API that can be deployed on Google Colab
with GPU acceleration using vLLM.
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
        title="Marco v3 Inference API",
        description="Italian Teacher homework generation service",
        version="1.0.5",
    )

    @app.get("/")
    async def root():
        """Health check endpoint"""
        import torch

        return {
            "status": "healthy",
            "service": "Marco v3 Inference API",
            "model": "minerva_marco_v3_merged",
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            "port": port,
            "version": "1.0.5",
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
        Generate Italian language exercises using Marco v3 model.

        Uses multiple parsing strategies to reliably extract exercises from model output.
        """
        try:
            import time

            start_time = time.time()

            # Create numbered placeholders to guide the model
            exercise_numbers = ", ".join([f"#{i+1}" for i in range(request.quantity)])

            # Optimized prompt with 5 complete examples
            prompt = f"""Create exactly {request.quantity} Italian language exercises ({exercise_numbers}) in JSON format.

REQUIREMENTS:
Level: {request.cefr_level}
Grammar: {request.grammar_focus}
Topic: {request.topic}
Exercise types: {', '.join(request.exercise_types)}

OUTPUT FORMAT - JSON array with {request.quantity} complete exercises:
[
  {{"type": "fill_in_blank", "question": "Io ___ a casa ogni giorno.", "correct_answer": "vado", "options": null, "explanation": "Present tense of andare"}},
  {{"type": "translation", "question": "Translate: I wake up at 7am", "correct_answer": "Mi sveglio alle sette", "options": null, "explanation": "Reflexive verb svegliarsi"}},
  {{"type": "multiple_choice", "question": "Come si dice 'I sleep'?", "correct_answer": "Dormo", "options": ["Dormo", "Dorme", "Dormire", "Dormono"], "explanation": "First person of dormire"}},
  {{"type": "fill_in_blank", "question": "Loro ___ la colazione insieme.", "correct_answer": "fanno", "options": null, "explanation": "Third person plural of fare"}},
  {{"type": "translation", "question": "Translate: We study every day", "correct_answer": "Studiamo ogni giorno", "options": null, "explanation": "First person plural of studiare"}}
]

NOW GENERATE ALL {request.quantity} EXERCISES (do not stop until you complete all {request.quantity}):
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

    return {"exercises": exercises, "strategy": parsing_strategy}
