"""
LLM API Handler - Unified interface for multiple LLM providers.

Supports:
- Google Gemini (multiple API keys)
- OpenAI (GPT-4o, GPT-4.1)
- Anthropic Claude (via AWS Bedrock or direct)
- Groq (Llama models, ultra-fast)
- DeepSeek (very cheap)
- Cerebras (ultra-fast, free tier)

Handles routing, retries, and fallbacks automatically.
"""

import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import threading
import time

import httpx


# Model configurations per provider
# Format: (provider, model_name, cost_per_1k_tokens, avg_latency_seconds)
# IMPORTANT: Keep these model names in sync with SCORER_MODEL_CONFIG in base_llm_scorer.py
# ORDER MATTERS: Fallback tries models in this dictionary order
MODEL_POOL = {
    "groq": [
        ("groq", "llama-3.1-8b-instant", 0.0, 1.8),  # FREE - fastest! Try first
        ("groq", "llama-3.3-70b-versatile", 0.0, 2.9),  # FREE - best quality
        ("groq", "llama-3.1-70b-versatile", 0.0, 2.5),  # FREE
    ],
    "openai": [
        ("openai", "gpt-4.1-nano", 0.001, 3),  # Fast and reliable - try before 4o-mini
        ("openai", "gpt-4.1-mini", 0.002, 3),
        ("openai", "gpt-4o-mini", 0.002, 3),
    ],
    "anthropic": [
        ("anthropic", "claude-3-haiku-20240307", 0.0005, 2),  # Faster model first
        ("anthropic", "claude-3-5-haiku-20241022", 0.001, 3),
    ],
    "deepseek": [
        ("deepseek", "deepseek-chat", 0.0001, 3),  # Slow but cheap - try after fast models
    ],
    "gemini": [
        ("gemini", "gemini-2.5-flash-lite", 0.0005, 2),  # LAST RESORT - has RPM issues
        ("gemini", "gemini-2.0-flash", 0.001, 5),
        ("gemini", "gemini-2.5-flash", 0.001, 20),
    ],
    # Cerebras temporarily disabled due to model name issues
    # "cerebras": [
    #     ("cerebras", "llama-3.1-70b", 0.0, 1.9),  # FREE - ultra-fast!
    # ],
}


class LLMAPIHandler:
    """
    Unified handler for all LLM API providers.

    Handles:
    - Provider selection and load balancing
    - Retries with exponential backoff
    - Automatic fallback to other providers
    - API key rotation (for providers with multiple keys)
    - Usage statistics tracking
    """

    def __init__(self):
        """Initialize all available LLM providers."""
        self.providers = {}
        self.api_keys = {}
        self.key_indices = {}
        self.stats = {}
        self._log_thread = None

        # Model-level cooldown tracking: "provider:model" -> timestamp of last failure
        # This allows us to skip slow/failing models while still using other models from the same provider
        self.model_failures = {}
        self.cooldown_duration = 120  # 2 minutes in seconds (longer cooldown to keep failing models out of rotation)

        # Track consecutive timeouts per model to trigger cooldown faster
        self.model_timeout_counts = {}  # "provider:model" -> consecutive timeout count
        self.timeout_threshold = 8  # Trigger cooldown after N consecutive timeouts (more lenient with request pacing)

        # Track models that have hit daily limits - these are permanently disabled for this session
        self.daily_limit_models = set()  # Set of "provider:model" keys that hit daily limits

        # Initialize each provider
        self._init_gemini()
        self._init_openai()
        self._init_anthropic()
        self._init_groq()
        self._init_deepseek()
        self._init_cerebras()

        # Build model routing list
        self.available_models = self._build_model_list()

        # DON'T start periodic logging - it creates daemon threads that can cause issues
        # Stats will be logged manually when needed
        # self.log_stats_periodically()

        print(f"  ✅ LLM API Handler initialized")
        print(f"     Providers: {', '.join(self.providers.keys())}")
        print(f"     Total models: {len(self.available_models)}")

    def _init_gemini(self):
        """Initialize Google Gemini with multiple API keys."""
        keys = []
        for key_name in ["GOOGLE_API_KEY", "GOOGLE_API_KEY_2", "GOOGLE_API_KEY_3", "GOOGLE_API_KEY_4"]:
            api_key = os.environ.get(key_name)
            if api_key:
                keys.append(api_key)

        if keys:
            try:
                import google.generativeai as genai
                genai.configure(api_key=keys[0])
                self.providers["gemini"] = genai
                self.api_keys["gemini"] = keys
                self.key_indices["gemini"] = 0
                self.stats["gemini"] = 0
                print(f"     ✅ Gemini: {len(keys)} API key(s)")
            except ImportError:
                print(f"     ⚠️ Gemini: google-generativeai not installed")

    def _init_openai(self):
        """Initialize OpenAI."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import AsyncOpenAI
                self.providers["openai"] = AsyncOpenAI(api_key=api_key, timeout=30.0)
                self.stats["openai"] = 0
                print(f"     ✅ OpenAI: configured")
            except ImportError:
                print(f"     ⚠️ OpenAI: openai library not installed")

    def _init_anthropic(self):
        """Initialize Anthropic Claude."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                import anthropic
                self.providers["anthropic"] = anthropic.AsyncAnthropic(api_key=api_key, timeout=30.0)
                self.stats["anthropic"] = 0
                print(f"     ✅ Anthropic: configured")
            except ImportError:
                print(f"     ⚠️ Anthropic: anthropic library not installed")

    def _init_groq(self):
        """Initialize Groq."""
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                from groq import AsyncGroq
                self.providers["groq"] = AsyncGroq(api_key=api_key, timeout=30.0)
                self.stats["groq"] = 0
                print(f"     ✅ Groq: configured")
            except ImportError:
                print(f"     ⚠️ Groq: groq library not installed")

    def _init_deepseek(self):
        """Initialize DeepSeek."""
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            try:
                from openai import AsyncOpenAI
                self.providers["deepseek"] = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1",
                    timeout=30.0
                )
                self.stats["deepseek"] = 0
                print(f"     ✅ DeepSeek: configured")
            except ImportError:
                print(f"     ⚠️ DeepSeek: openai library not installed")

    def _init_cerebras(self):
        """Initialize Cerebras."""
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if api_key:
            try:
                from openai import AsyncOpenAI
                self.providers["cerebras"] = AsyncOpenAI(
                    api_key=api_key,
                    base_url="https://api.cerebras.ai/v1",
                    timeout=30.0
                )
                self.stats["cerebras"] = 0
                print(f"     ✅ Cerebras: configured")
            except ImportError:
                print(f"     ⚠️ Cerebras: openai library not installed")

    def _build_model_list(self):
        """Build list of available models based on configured providers."""
        models = []
        for provider_name, provider_models in MODEL_POOL.items():
            if provider_name in self.providers:
                models.extend(provider_models)
        return models

    def get_provider_for_model(self, model_name: str) -> Optional[str]:
        """Get the provider for a given model name."""
        for provider, provider_models in MODEL_POOL.items():
            for model_info in provider_models:
                # model_info is a tuple like ("gemini", "gemini-2.0-flash", 0.001, 5)
                if len(model_info) > 1 and model_info[1] == model_name:
                    return provider
        return None

    def is_model_available(self, provider: str, model_name: str) -> bool:
        """
        Check if a specific model is available (not in cooldown or daily limit hit).

        Returns True if model should be tried, False if in cooldown period or daily limit reached.
        """
        model_key = f"{provider}:{model_name}"

        # Check if model hit daily limit - these are permanently disabled
        if model_key in self.daily_limit_models:
            return False

        if model_key not in self.model_failures:
            return True

        # Check if cooldown period has expired
        last_failure = self.model_failures[model_key]
        time_since_failure = time.time() - last_failure

        if time_since_failure >= self.cooldown_duration:
            # Cooldown expired, remove from failures and allow retry
            del self.model_failures[model_key]
            print(f"    🔄 {model_key} → back online")
            return True

        return False

    def mark_model_failed(self, provider: str, model_name: str):
        """Mark a specific model as failed, starting cooldown period."""
        model_key = f"{provider}:{model_name}"

        if model_key not in self.model_failures:
            self.model_failures[model_key] = time.time()
            cooldown_secs = self.cooldown_duration
            if cooldown_secs >= 60:
                print(f"    ⏸️  {model_key} → cooldown ({cooldown_secs/60:.0f}m)")
            else:
                print(f"    ⏸️  {model_key} → cooldown ({cooldown_secs:.0f}s)")

    def mark_model_daily_limit(self, provider: str, model_name: str):
        """Permanently disable a model that has hit its daily limit."""
        model_key = f"{provider}:{model_name}"

        if model_key not in self.daily_limit_models:
            self.daily_limit_models.add(model_key)
            print(f"    🚫 {model_key} → daily limit reached (disabled for session)")

    def track_timeout(self, provider: str, model_name: str):
        """
        Track a timeout for a specific model. If threshold is reached, trigger cooldown.
        """
        model_key = f"{provider}:{model_name}"
        self.model_timeout_counts[model_key] = self.model_timeout_counts.get(model_key, 0) + 1

        if self.model_timeout_counts[model_key] >= self.timeout_threshold:
            # Too many consecutive timeouts, trigger cooldown
            self.mark_model_failed(provider, model_name)
            self.model_timeout_counts[model_key] = 0  # Reset count

    def reset_timeout_count(self, provider: str, model_name: str):
        """Reset timeout count for a specific model after successful call."""
        model_key = f"{provider}:{model_name}"
        if model_key in self.model_timeout_counts:
            self.model_timeout_counts[model_key] = 0

    def is_rpm_limit_error(self, error_msg: str) -> bool:
        """
        Detect if an error message indicates an RPM (requests per minute) limit.

        Common patterns:
        - OpenAI: "Rate limit reached", "requests per minute"
        - Gemini: "429", "RESOURCE_EXHAUSTED"
        - Anthropic: "429", "rate_limit_error"
        """
        error_lower = error_msg.lower()
        rpm_indicators = [
            "rate limit reached",
            "requests per minute",
            "rpm:",
            "rate_limit_error",
            "too many requests",
        ]
        return any(indicator in error_lower for indicator in rpm_indicators)

    def is_daily_limit_error(self, error_msg: str) -> bool:
        """
        Detect if an error message indicates a daily limit has been reached.

        Common patterns:
        - OpenAI: "exceeded your current quota", "insufficient_quota", "requests per day (RPD)"
        - Gemini: "quota exceeded" (different from rate limit)
        - Groq: "daily limit", "per day"
        """
        error_lower = error_msg.lower()

        # First check if it's an RPM error (not daily)
        if self.is_rpm_limit_error(error_msg):
            return False

        daily_limit_indicators = [
            "daily limit",
            "requests per day",
            "rpd:",
            "quota exceeded",
            "insufficient_quota",
            "exceeded your current quota",
            "quota exhausted",
            "billing",
            "payment required",
        ]
        return any(indicator in error_lower for indicator in daily_limit_indicators)

    async def call_llm(
        self,
        prompt: str,
        system_prompt: str,
        json_schema: Dict[str, Any],
        timeout: float = 30.0,
        preferred_provider: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Call LLM with automatic provider selection and fallback.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            json_schema: Expected JSON output schema
            timeout: Request timeout in seconds
            preferred_provider: Preferred provider to try first (e.g., "gemini")

        Returns:
            Tuple of (result_text, model_used)

        Raises:
            Exception if all providers fail
        """
        # Organize models by preferred provider first, but exclude models in cooldown
        models_to_try = self.available_models.copy()

        # Filter out models that are in cooldown
        models_to_try = [m for m in models_to_try if self.is_model_available(m[0], m[1])]

        if not models_to_try:
            # Show how many models are in cooldown vs daily limit
            cooldown_count = len(self.model_failures)
            daily_limit_count = len(self.daily_limit_models)

            if daily_limit_count > 0:
                print(f"    ⚠️  All models unavailable: {cooldown_count} in cooldown, {daily_limit_count} at daily limit. Waiting 30s...")
            else:
                print(f"    ⚠️  All {cooldown_count} models in cooldown, waiting 30s...")

            await asyncio.sleep(30)
            # After wait, allow all models to retry (except daily limit ones)
            models_to_try = self.available_models.copy()

        # Prioritize preferred provider's models, but still respect cooldowns
        if preferred_provider:
            preferred_models = [m for m in models_to_try if m[0] == preferred_provider]
            other_models = [m for m in models_to_try if m[0] != preferred_provider]
            models_to_try = preferred_models + other_models

        deadline = time.time() + timeout
        last_error = None
        attempts = []

        for provider, model_name, cost, avg_latency in models_to_try:
            # Timeout strategy: Give each model 4x its expected latency, max 10s per attempt
            # With model-level cooldowns, we can be a bit more patient since we have many models to try
            attempt_timeout = min(avg_latency * 4.0, deadline - time.time(), 10.0)
            if attempt_timeout <= 0:
                break # Total timeout exceeded

            try:
                start_time = time.time()
                attempts.append(f"{provider}:{model_name}")

                # Call appropriate provider
                if provider == "gemini":
                    result = await self._call_gemini(model_name, prompt, system_prompt, json_schema, attempt_timeout)
                elif provider == "openai":
                    result = await self._call_openai(model_name, prompt, system_prompt, json_schema, attempt_timeout)
                elif provider == "anthropic":
                    result = await self._call_anthropic(model_name, prompt, system_prompt, json_schema, attempt_timeout)
                elif provider in ["groq", "deepseek", "cerebras"]:
                    result = await self._call_openai_compatible(provider, model_name, prompt, system_prompt, json_schema, attempt_timeout)
                else:
                    continue

                # Success! Reset timeout count for this model
                elapsed = time.time() - start_time
                self.stats[provider] = self.stats.get(provider, 0) + 1
                self.reset_timeout_count(provider, model_name)  # Clear timeout tracking on success
                # print(f"    ✅ {provider}:{model_name}")  # Debug: shows which model actually succeeded
                return result, f"{provider}:{model_name}"

            except asyncio.TimeoutError as e:
                last_error = f"{provider}:{model_name} timeout after {attempt_timeout:.1f}s"
                # Removed verbose logging - only log cooldown triggers

                # Track timeout for this specific model - will trigger cooldown after threshold
                self.track_timeout(provider, model_name)
                continue
            except Exception as e: # Catch all other errors
                error_msg = str(e)
                last_error = f"{provider}:{model_name} error: {error_msg[:150]}"

                # Check error type and handle appropriately
                if self.is_daily_limit_error(error_msg):
                    # Permanently disable this model for the session
                    self.mark_model_daily_limit(provider, model_name)
                elif self.is_rpm_limit_error(error_msg):
                    # RPM limit - just track timeout, will cooldown after threshold
                    # This is less aggressive than marking as failed immediately
                    self.track_timeout(provider, model_name)
                    # Don't log every RPM error, too noisy
                else:
                    # Other errors - put on cooldown
                    self.mark_model_failed(provider, model_name)
                continue

        # All models failed - log detailed error
        tried_models = " → ".join(attempts) if attempts else "none"
        print(f"    ❌ ALL MODELS FAILED! Tried: {tried_models}")
        if last_error:
            print(f"       Last error: {last_error[:200]}")
        raise Exception(f"All LLM providers failed (tried: {tried_models}). Last error: {last_error}")

    async def _call_gemini(self, model_name: str, prompt: str, system_prompt: str, json_schema: Dict, timeout: float) -> str:
        """Call Google Gemini API."""
        import google.generativeai as genai

        # Rotate API key if multiple available
        if len(self.api_keys["gemini"]) > 1:
            self.key_indices["gemini"] = (self.key_indices["gemini"] + 1) % len(self.api_keys["gemini"])
            genai.configure(api_key=self.api_keys["gemini"][self.key_indices["gemini"]])

        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=json_schema,
                temperature=0,
            )
        )

        combined_prompt = f"{system_prompt}\n\n{prompt}"

        # Run synchronous Gemini call in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, gemini_model.generate_content, combined_prompt),
            timeout=timeout,
        )
        return response.text

    async def _call_openai(self, model_name: str, prompt: str, system_prompt: str, json_schema: Dict, timeout: float) -> str:
        """Call OpenAI API."""
        client = self.providers["openai"]
        # Add instruction to system prompt to ensure JSON output
        system_prompt_with_json = f"{system_prompt}\n\nIMPORTANT: You must respond ONLY with a valid JSON object that conforms to the required schema. Do not include any other text or explanations."

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=[
                    # Use the system prompt to enforce JSON output
                    {"role": "system", "content": system_prompt_with_json},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            ),
            timeout=timeout
        )
        return response.choices[0].message.content

    async def _call_anthropic(self, model_name: str, prompt: str, system_prompt: str, json_schema: Dict, timeout: float) -> str:
        """Call Anthropic Claude API."""
        client = self.providers["anthropic"]

        response = await asyncio.wait_for(
            client.messages.create(
                model=model_name,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            ),
            timeout=timeout
        )
        # Anthropic returns content directly in text blocks
        return response.content[0].text

    async def _call_openai_compatible(self, provider: str, model_name: str, prompt: str, system_prompt: str, json_schema: Dict, timeout: float) -> str:
        """Call OpenAI-compatible API (Groq, DeepSeek, Cerebras)."""
        client = self.providers[provider]
        # CRITICAL FIX: Use system prompt instruction for JSON mode, not response_format or tools
        system_prompt_with_json = f"{system_prompt}\n\nIMPORTANT: You must respond ONLY with a valid JSON object that conforms to the required schema. Do not include any other text, explanations, or markdown."

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_name,
                messages=[
                    # Use the system prompt to enforce JSON output
                    {"role": "system", "content": system_prompt_with_json},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            ),
            timeout=timeout
        )
        
        # For OpenAI-compatible APIs, the JSON is expected directly in the content
        # as we are relying on the system prompt to enforce JSON output.
        return response.choices[0].message.content

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics per provider."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset usage statistics."""
        for provider in self.stats:
            self.stats[provider] = 0

    def log_stats(self):
        """Prints the model usage distribution for the current batch and resets stats."""
        total_api_calls = sum(self.stats.values())

        if total_api_calls == 0:
            # No successful API calls - this is a critical error!
            cooldown_count = len(self.model_failures)
            daily_limit_count = len(self.daily_limit_models)
            print(f"\n   ❌ WARNING: 0 successful LLM calls!")
            print(f"      {cooldown_count} models in cooldown, {daily_limit_count} at daily limit")
            if self.daily_limit_models:
                print(f"      Daily limits: {', '.join(sorted(self.daily_limit_models))}")
            print(f"      💡 TIP: Likely hitting RPM (requests/minute) limits")
            print(f"      💡 Solution: Reduce batch size or add request pacing")
            return

        if total_api_calls > 0:
            print(f"\n   📊 Model Usage Distribution ({total_api_calls} total requests):")
            # Sort by usage count (descending)
            sorted_providers = sorted(self.stats.items(), key=lambda x: x[1], reverse=True)
            for provider_name, count in sorted_providers:
                if count == 0:
                    continue
                percentage = (count / total_api_calls) * 100
                # Add provider emoji
                if provider_name == "groq":
                    emoji = "⚡"
                elif provider_name == "cerebras":
                    emoji = "🧠"
                elif provider_name == "openai":
                    emoji = "🟢"
                elif provider_name == "gemini":
                    emoji = "🔵"
                else:
                    emoji = "❓"
                print(f"      {emoji} {provider_name.capitalize()}: {count}/{total_api_calls} ({percentage:.1f}%)")

        # Show models that hit daily limits (important for user to know)
        if self.daily_limit_models:
            print(f"   🚫 Daily limits reached: {', '.join(sorted(self.daily_limit_models))}")

        self.reset_stats()

    def log_stats_periodically(self, interval: int = 10):
        """
        Starts a background thread to log stats periodically.
        This is useful in environments like notebooks where you can't easily
        modify the main training loop to call log_stats().
        """
        if self._log_thread is not None:
            return # Already running

        def log_loop():
            while True:
                time.sleep(interval)
                self.log_stats()

        self._log_thread = threading.Thread(target=log_loop, daemon=True)
        self._log_thread.start()
        print(f"     📈 Periodic stat logging enabled (every {interval} seconds)")
