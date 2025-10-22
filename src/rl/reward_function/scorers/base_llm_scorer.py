"""
Base class for scorers that use a batched LLM API.
Handles batching, retries, and fallbacks for OpenAI calls.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Tuple
import random

import httpx
import json5  # Use the more lenient json5 parser

from .base import BaseScorer


class BaseLLMScorer(BaseScorer):
    """
    An abstract base class for scorers that use batched LLM calls.
    """

    def __init__(self, batch_size: int = 10):
        super().__init__(nlp=None)  # LLM scorers don't need spaCy
        self.batch_size = batch_size

        if not os.environ.get("OPENAI_API_KEY"):
            raise EnvironmentError(
                f"{self.__class__.__name__} requires an OpenAI API key."
            )
        try:
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(timeout=60.0)
            print(f"  ✅ LLM scoring enabled for {self.name} (batch size: {self.batch_size})")
        except ImportError:
            raise ImportError("OpenAI client not installed. Please run: pip install openai")

    def get_prompt(self, exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
        """
        Child classes must implement this to return the specific prompt
        for a batch of exercises.
        """
        raise NotImplementedError

    async def score_batch(
        self, exercises: List[Dict[str, Any]], request: Dict[str, Any], semaphore: asyncio.Semaphore = None
    ) -> List[Tuple[float, List[str]]]:
        """
        Scores a batch of exercises using the LLM.
        This is the main entry point for this scorer.
        """
        if not exercises:
            return []

        if semaphore:
            await semaphore.acquire()

        final_results = [(5.0, ["Scoring failed."])] * len(exercises)

        try:
            for attempt in range(2):  # Retry logic
                try:
                    user_prompt = self.get_prompt(exercises, request)
                    system_prompt = "You are an expert Italian language teacher and evaluator. Respond accurately in the requested JSON format."
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                    # Define the shared JSON schema for the output
                    json_output_schema = {
                        "type": "object",
                        "properties": {
                            "scores": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "score": {"type": "number"},
                                        "issue": {"type": "string"},
                                    },
                                    "required": ["id", "score", "issue"],
                                },
                            }
                        },
                        "required": ["scores"],
                    }
                    
                    response = None

                    # Use the model specified by the child class, or default to gpt-4o-mini.
                    model_to_use = getattr(self, "model", "gpt-4o-mini")

                    try:
                        api_kwargs = {
                            "model": model_to_use,
                            "messages": messages,
                            "timeout": 20.0,
                            "temperature": 0,
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "record_scores",
                                        "description": "Records the scores for a batch of exercises.",
                                        "parameters": json_output_schema,
                                    },
                                }
                            ],
                            "tool_choice": {"type": "function", "function": {"name": "record_scores"}},
                        }

                        response = await self.client.chat.completions.create(**api_kwargs)

                    except Exception as e:
                        print(f"  ⚠️ {model_to_use} failed ({e}). This batch will use neutral scores after retries.")
                
                    if response is None:
                        raise Exception(f"LLM model {model_to_use} failed to return a response.")

                    message = response.choices[0].message

                    if not message.tool_calls:
                        raise ValueError("LLM did not use the required 'record_scores' tool.")
                    tool_call = message.tool_calls[0]
                    result_text = tool_call.function.arguments

                    parsed_json = self._parse_llm_json(result_text)
                    scores_data = parsed_json.get("scores", [])

                    # Ensure we have a result for each exercise
                    if len(scores_data) != len(exercises):
                        raise ValueError(f"LLM returned {len(scores_data)} scores for {len(exercises)} exercises.")

                    for i, data in enumerate(scores_data):
                        score = float(data.get("score", 5.0))
                        issue = data.get("issue", "")
                        errors = [f"{self.name} issue ({model_to_use}): {issue}"] if score < 8 and issue else []
                        final_results[i] = (score, errors)

                    break  # Success, exit retry loop

                except (json.JSONDecodeError, ValueError, httpx.TimeoutException) as e:
                    # Add enhanced logging to see the problematic text
                    raw_output = result_text if 'result_text' in locals() and result_text else "Empty String"
                    sent_prompt = user_prompt if 'user_prompt' in locals() else "PROMPT NOT GENERATED" # type: ignore
                    print(f"  ⚠️ {self.name} scorer attempt {attempt + 1} failed: {e}")
                    print(f"     Raw {model_used} Output: '{raw_output}...'")
                    print(f"     Prompt Sent:\n---\n{sent_prompt}...\n---")
                    if attempt == 1:
                        print(f"  Exhausted retries for {self.name}. Using neutral scores.")
                        # final_results is already populated with default error scores

        except Exception as e:
            print(f"  ⚠️ Unrecoverable error in {self.name} scorer: {e}")

        finally:
            if semaphore:
                semaphore.release()

        return final_results

    async def score(
        self, exercise: Dict[str, Any], request: Dict[str, Any], semaphore: asyncio.Semaphore = None
    ) -> Tuple[float, List[str]]:
        """
        Scores a single exercise. This is for compatibility with the existing
        reward function structure but is inefficient. The `score_batch` method
        should be preferred.
        """
        results = await self.score_batch([exercise], request, semaphore)
        return results[0]

    def _parse_llm_json(self, text: str) -> Dict:
        """
        A robust, two-stage parser for LLM-generated JSON.
        1. Tries to parse the whole string with the lenient json5 library.
        2. If that fails, it finds the first '{' and last '}' to extract and parse the core JSON object.
        """
        if not text:
            raise ValueError("LLM returned an empty string.")
        try:
            # Stage 1: Try parsing the whole text with json5
            return json5.loads(text)
        except (json.JSONDecodeError, ValueError):
            # Stage 2: If it fails, find the JSON block manually
            start_brace = text.find("{")
            if start_brace == -1:
                raise json.JSONDecodeError("No JSON object start brace '{' found.", text, 0)

            end_brace = text.rfind("}")
            if end_brace == -1 or end_brace < start_brace:
                raise json.JSONDecodeError("No JSON object end brace '}' found after start.", text, 0)

            json_str = text[start_brace : end_brace + 1]
            return json5.loads(json_str)

    @property
    def name(self) -> str:
        raise NotImplementedError