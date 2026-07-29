"""
Providers: unified interface for LLM API providers.
All providers implement OpenAI-compatible chat completions.

Supported providers:
- Groq                 https://api.groq.com/openai/v1
- OpenRouter           https://openrouter.ai/api/v1
- Google AI Studio     https://generativelanguage.googleapis.com/v1beta/openai
- Cerebras             https://api.cerebras.ai/v1
- SambaNova            https://api.sambanova.ai/v1
- Mistral              https://api.mistral.ai/v1
- HuggingFace          https://router.huggingface.co/v1

Includes automatic retry on 429 with exponential backoff and Retry-After
header support.
"""
import os
import time
import json
import random
import requests
from abc import ABC, abstractmethod
from typing import Optional


class Provider(ABC):
    name: str
    base_url: str

    @abstractmethod
    def list_models(self) -> list[dict]:
        ...

    @abstractmethod
    def chat(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> dict:
        ...


def _parse_duration(s: str) -> float:
    s = s.strip().lower()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        pass
    total = 0.0
    num = ""
    for ch in s:
        if ch.isdigit() or ch == ".":
            num += ch
        elif ch == "h" and num:
            total += float(num) * 3600; num = ""
        elif ch == "m" and num:
            total += float(num) * 60; num = ""
        elif ch == "s" and num:
            total += float(num); num = ""
    if num:
        total += float(num)
    return total


class OpenAICompatibleProvider(Provider):
    """Generic OpenAI-compatible provider."""

    def __init__(self, name: str, base_url: str, api_key: str,
                 extra_headers: dict = None,
                 has_models_endpoint: bool = True):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.has_models_endpoint = has_models_endpoint

    def _headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "CAID-Bench/1.0 (research benchmark; https://github.com/revenue7-eng/caid-bench)",
        }
        h.update(self.extra_headers)
        return h

    def list_models(self) -> list[dict]:
        if not self.has_models_endpoint:
            return []
        try:
            r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=30)
            r.raise_for_status()
            data = r.json()
            return data.get("data", [])
        except Exception as e:
            print(f"[{self.name}] list_models failed: {e}")
            return []

    def _parse_retry_after(self, response) -> float:
        retry_after = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        for header in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
            val = response.headers.get(header)
            if val:
                return _parse_duration(val)
        for header in ("x-ratelimit-reset",):
            val = response.headers.get(header)
            if val:
                try:
                    ts = float(val)
                    now_ms = time.time() * 1000
                    if ts > now_ms:
                        return (ts - now_ms) / 1000.0
                except ValueError:
                    pass
        return 0.0

    def chat(
        self,
        model: str,
        system_prompt: Optional[str],
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> dict:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start = time.time()
        last_error = None
        last_retry_after = 0.0
        retries = 0

        for attempt in range(max_retries + 1):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=timeout,
                )

                if r.status_code == 429:
                    retry_after = self._parse_retry_after(r)
                    last_retry_after = retry_after
                    last_error = f"HTTP 429. Retry-After: {retry_after}s. Body: {r.text[:300]}"
                    if attempt < max_retries and 0 < retry_after <= 60:
                        wait = retry_after + random.uniform(0.5, 1.5)
                        time.sleep(wait)
                        retries += 1
                        continue
                    return {
                        "ok": False, "text": "", "raw": {"status": 429, "body": r.text[:1000]},
                        "error": last_error, "latency_ms": (time.time() - start) * 1000,
                        "retries": retries, "rate_limited": True, "retry_after": retry_after,
                    }

                if r.status_code in (500, 502, 503, 504):
                    last_error = f"HTTP {r.status_code}: {r.text[:300]}"
                    if attempt < max_retries:
                        time.sleep((2 ** attempt) + random.uniform(0, 1))
                        retries += 1
                        continue

                if r.status_code != 200:
                    return {
                        "ok": False, "text": "",
                        "raw": {"status": r.status_code, "body": r.text[:2000]},
                        "error": f"HTTP {r.status_code}: {r.text[:500]}",
                        "latency_ms": (time.time() - start) * 1000,
                        "retries": retries, "rate_limited": False, "retry_after": 0.0,
                    }

                data = r.json()
                text = ""
                try:
                    text = data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError, TypeError):
                    try:
                        msg = data["choices"][0]["message"]
                        text = msg.get("content") or msg.get("reasoning") or ""
                    except Exception:
                        pass

                return {
                    "ok": True, "text": text, "raw": data, "error": None,
                    "latency_ms": (time.time() - start) * 1000,
                    "retries": retries, "rate_limited": False, "retry_after": 0.0,
                }

            except requests.Timeout:
                last_error = f"Timeout after {timeout}s"
                if attempt < max_retries:
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                    retries += 1
                    continue
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_retries:
                    time.sleep((2 ** attempt) + random.uniform(0, 1))
                    retries += 1
                    continue

        return {
            "ok": False, "text": "", "raw": None, "error": last_error,
            "latency_ms": (time.time() - start) * 1000,
            "retries": retries, "rate_limited": False, "retry_after": last_retry_after,
        }


# ---------------------------------------------------------------------------
# Provider factory functions
# ---------------------------------------------------------------------------

def make_groq_provider(api_key: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )


def make_openrouter_provider(api_key: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        extra_headers={
            "HTTP-Referer": "https://github.com/revenue7-eng/caid-bench",
            "X-Title": "CAID Bench",
        },
    )


def make_google_provider(api_key: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="google",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key=api_key,
    )


def make_cerebras_provider(api_key: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key,
    )


def make_sambanova_provider(api_key: str) -> OpenAICompatibleProvider:
    # SambaNova does NOT expose /v1/models — must use whitelist
    return OpenAICompatibleProvider(
        name="sambanova",
        base_url="https://api.sambanova.ai/v1",
        api_key=api_key,
        has_models_endpoint=False,
    )


def make_mistral_provider(api_key: str) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        name="mistral",
        base_url="https://api.mistral.ai/v1",
        api_key=api_key,
    )


def make_huggingface_provider(api_key: str) -> OpenAICompatibleProvider:
    # HuggingFace Inference Router — OpenAI-compatible aggregator
    return OpenAICompatibleProvider(
        name="huggingface",
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )


def make_openai_provider(api_key: str) -> OpenAICompatibleProvider:
    # OpenAI native API — used for closed frontier models (Component B).
    # Reference impl of OpenAI-compatible; /v1/models works.
    return OpenAICompatibleProvider(
        name="openai",
        base_url="https://api.openai.com/v1",
        api_key=api_key,
    )
