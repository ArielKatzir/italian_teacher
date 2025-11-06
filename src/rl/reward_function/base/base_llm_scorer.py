"""
Base class for scorers that use a batched LLM API.
This class uses the LLMAPIHandler to abstract away the complexity of
handling multiple API providers, fallbacks, and retries.
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Tuple

import httpx
import json5  # Use the more lenient json5 parser
import random

from .base_scorer import BaseScorer
from .llm_api_handler import LLMAPIHandler

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# Centralized model configuration per scorer
# OPTIMIZED FOR BALANCED LOAD DISTRIBUTION
# Strategy: Each scorer starts with a DIFFERENT provider to spread load evenly
# Priority: Groq (fast) → DeepSeek (cheap) → OpenAI (reliable) → Anthropic → Gemini (backup)
SCORER_MODEL_CONFIG = {
    "grammar_correctness": {
        # Grammar: Start with Groq for speed
        "models": [
            "gemini-2.0-flash",
            "llama-3.3-70b-versatile",  
            "gpt-4.1-nano",              # OpenAI - reliable
            "llama-3.3-70b-versatile",   # Groq 70B - better quality
            "claude-3-haiku-20240307",   # Anthropic - backup
            "gemini-2.5-flash-lite",     # Gemini - last resort
            "gpt-4o-mini",               # OpenAI - backup
            "llama-3.1-70b-versatile",   # Groq - backup
        ],
        "default": "llama-3.1-8b-instant"
    },
    "cefr_alignment": {
        # CEFR: Start with DeepSeek for cost efficiency
        "models": [
            "deepseek-chat",             # DeepSeek - very cheap
            "llama-3.3-70b-versatile",   # Groq 70B - quality
            "gpt-4.1-nano",              # OpenAI - reliable
            "llama-3.1-8b-instant",      # Groq - fast
            "claude-3-5-haiku-20241022", # Anthropic - backup
            "gpt-4.1-mini",               # OpenAI - backup
            "gemini-2.0-flash",          # Gemini - backup
            "llama-3.1-70b-versatile",   # Groq - backup
        ],
        "default": "deepseek-chat"
    },
    "coherence": {
        # Coherence: Start with OpenAI for reliability
        "models": [
            "gpt-4.1-nano",              # OpenAI - reliable
            "llama-3.1-8b-instant",      # Groq - fast
            "deepseek-chat",             # DeepSeek - cheap
            "llama-3.3-70b-versatile",   # Groq 70B - quality
            "claude-3-haiku-20240307",   # Anthropic - backup
            "gpt-4.1-mini",               # OpenAI - backup
            "gemini-2.5-flash-lite",     # Gemini - backup
            "llama-3.1-70b-versatile",   # Groq - backup
        ],
        "default": "gpt-4.1-nano"
    },
    "fluency": {
        # Fluency: Start with Anthropic for variety
        "models": [
            "claude-3-haiku-20240307",   # Anthropic - different perspective
            "deepseek-chat",             # DeepSeek - cheap
            "llama-3.1-8b-instant",      # Groq - fast
            "gpt-4.1-nano",              # OpenAI - reliable
            "llama-3.3-70b-versatile",   # Groq 70B - quality
            "gpt-4o-mini",               # OpenAI - backup
            "gemini-2.0-flash",          # Gemini - backup
        ],
        "default": "claude-3-haiku-20240307"
    },
}


class BaseLLMScorer(BaseScorer):
    """
    An abstract base class for scorers that use batched LLM calls.
    """

    # Class-level model override for testing purposes
    _model_override = None

    def __init__(self, llm_handler: LLMAPIHandler, batch_size: int = 10, prompt_fn=None, **kwargs):
        if not isinstance(llm_handler, LLMAPIHandler):
            raise TypeError("BaseLLMScorer requires a valid LLMAPIHandler instance.")

        super().__init__(nlp=None)  # LLM scorers don't need spaCy
        self.batch_size = batch_size

        # Centralize all API handling into the LLMAPIHandler
        self.llm_handler = llm_handler

        # Optional: Inject custom prompt function (for subject-specific customization)
        self._prompt_fn = prompt_fn

        print(f"  ✅ LLM scoring enabled for {self.name} (batch size: {self.batch_size})")

    @classmethod
    def set_model_override(cls, model: str):
        """
        Set a model override for all scorers (useful for testing).
        This will force all scorers to use the specified model.
        """
        cls._model_override = model
        print(f"🔧 Model override set to: {model}")

    @classmethod
    def clear_model_override(cls):
        """Clear the model override."""
        cls._model_override = None
        print("🔧 Model override cleared")

    def get_and_reset_stats(self):
        """Get current API usage stats and reset counters."""
        if not hasattr(self, 'llm_handler'):
            return {} # Return empty stats if the handler was never initialized

        stats = self.llm_handler.get_stats()
        self.llm_handler.reset_stats()
        return stats

    def get_allowed_models(self) -> List[str]:
        """
        Get the list of allowed models for this scorer from the centralized config.
        """
        config = SCORER_MODEL_CONFIG.get(self.name, {})
        return config.get("models", ["gpt-3.5-turbo", "gpt-3.5-turbo-0125"])

    def get_default_model(self) -> str:
        """
        Get the default model for this scorer from the centralized config.
        """
        config = SCORER_MODEL_CONFIG.get(self.name, {})
        return config.get("default", "gpt-3.5-turbo")

    def get_prompt(self, exercises: List[Dict[str, Any]], request: Dict[str, Any]) -> str:
        """
        Get the prompt for a batch of exercises.

        If a prompt function was injected via prompt_fn, use that.
        Otherwise, child classes must implement this method.
        """
        if self._prompt_fn:
            return self._prompt_fn(exercises, request)
        raise NotImplementedError(f"{self.__class__.__name__} must implement get_prompt() or provide prompt_fn")

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the LLM.

        Override in subject-specific scorers to provide subject-specific instructions.
        Default is generic.
        """
        return "You are an expert evaluator. Respond accurately in the requested JSON format."

    async def score_batch(
        self, exercises: List[Dict[str, Any]], request: Dict[str, Any], semaphore: asyncio.Semaphore = None
    ) -> List[Tuple[float, List[str]]]:
        """
        Scores a batch of exercises using the centralized LLMAPIHandler.
        This is the main entry point for this scorer.
        """
        if not exercises:
            return []

        if semaphore:
            await semaphore.acquire()

        # Add jitter to prevent all concurrent requests from hitting API simultaneously
        # Use 1.5-4.0s jitter to stay within RPM (requests per minute) limits
        # Increased from 0.5-2.0s to handle high gradient_accumulation_steps
        jitter = random.uniform(1.5, 4.0)
        await asyncio.sleep(jitter)

        final_results = [(5.0, ["Scoring failed."])] * len(exercises)

        try:
            user_prompt = self.get_prompt(exercises, request)
            # Generic system prompt - subject-specific scorers can override get_system_prompt()
            system_prompt = self.get_system_prompt()

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

            # Get this scorer's allowed models for fallback chain
            allowed_models = self.get_allowed_models()
            preferred_model = allowed_models[0] if allowed_models else None
            preferred_provider = self.llm_handler.get_provider_for_model(preferred_model) if preferred_model else None

            # Retry the entire fallback chain until success
            # Strategy: Try all models with 90s timeout, if all fail, wait and retry the whole chain
            # This ensures we eventually get a score rather than returning a default
            max_retries = 5  # Try the entire chain up to 5 times
            retry_delay = 10.0  # Wait 10s between full chain retries

            for retry in range(max_retries):
                try:
                    # Delegate the call to the handler with scorer-specific model list
                    # Use 90s timeout per attempt through the fallback chain
                    result_text, model_used = await self.llm_handler.call_llm(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        json_schema=json_output_schema,
                        timeout=90.0,  # 90s to try all models in chain
                        preferred_provider=preferred_provider,
                        allowed_models=allowed_models
                    )
                    break  # Success! Exit retry loop
                except Exception as e:
                    if retry < max_retries - 1:
                        # Not the last retry - wait and try again
                        # print(f"    ⚠️  {self.name} failed (attempt {retry+1}/{max_retries}), retrying in {retry_delay}s...")
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        # Last retry failed - re-raise the exception
                        raise

            parsed_json = self._parse_llm_json(result_text)
            scores_data = parsed_json.get("scores", [])

            if len(scores_data) != len(exercises):
                raise ValueError(f"LLM returned {len(scores_data)} scores for {len(exercises)} exercises.")

            for i, data in enumerate(scores_data):
                score = float(data.get("score", 5.0))
                issue = data.get("issue", "")
                errors = [f"{self.name} issue ({model_used}): {issue}"] if score < 8 and issue else []
                final_results[i] = (score, errors)

        except asyncio.TimeoutError as e:
            # print(f"    ⚠️  {self.name} timed out: {str(e)[:100]}")  # Commented out to reduce log noise
            pass
        except Exception as e:
            # print(f"    ⚠️  {self.name} error: {str(e)[:100]}")  # Commented out to reduce log noise
            # final_results will contain the default error scores
            pass

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