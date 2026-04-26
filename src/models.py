"""
Model selection: which models to benchmark per provider.

For each provider we use either:
1. Whitelist (Groq, SambaNova, Cerebras, Mistral, HuggingFace) — known stable models
2. Dynamic discovery via /v1/models with filtering (OpenRouter free, Google Gemini)
"""
from typing import Optional


# -----------------------------------------------------------------------------
# Groq — text-gen models (excluding speech, moderation, vision-only)
# -----------------------------------------------------------------------------
GROQ_TEXT_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
    "groq/compound",
    "groq/compound-mini",
]


# -----------------------------------------------------------------------------
# Cerebras — fast-inference text models
# Note: Cerebras has GPT-OSS, Llama, Qwen, GLM. Ultra-fast inference (>2000 tok/s)
# -----------------------------------------------------------------------------
CEREBRAS_TEXT_MODELS = [
    "llama3.1-8b",
    "llama-3.3-70b",
    "qwen-3-32b",
    "qwen-3-235b-a22b-instruct-2507",
    "qwen-3-coder-480b",
    "gpt-oss-120b",
    "zai-glm-4.6",
    "zai-glm-4.7",
]


# -----------------------------------------------------------------------------
# SambaNova — does NOT expose /v1/models, must whitelist
# Free tier: 10-30 RPM depending on model
# -----------------------------------------------------------------------------
SAMBANOVA_TEXT_MODELS = [
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.3-70B-Instruct",
    "Llama-4-Maverick-17B-128E-Instruct",
    "DeepSeek-V3.1",
    "DeepSeek-R1-0528",
    "Qwen3-32B",
]


# -----------------------------------------------------------------------------
# Mistral La Plateforme — Mistral's own models
# Free tier limits apply
# -----------------------------------------------------------------------------
MISTRAL_TEXT_MODELS = [
    "mistral-large-latest",
    "mistral-small-latest",
    "mistral-medium-latest",
    "open-mistral-nemo",
    "ministral-8b-latest",
    "ministral-3b-latest",
    "codestral-latest",
    "magistral-medium-latest",
    "magistral-small-latest",
]


# -----------------------------------------------------------------------------
# HuggingFace Inference Providers (router) — aggregator across providers
# Free tier with credit limit
# Format: model_id:provider — e.g., "meta-llama/Llama-3.3-70B-Instruct:nebius"
# -----------------------------------------------------------------------------
HUGGINGFACE_TEXT_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
]


# -----------------------------------------------------------------------------
# Filter helpers (use API /v1/models response when available)
# -----------------------------------------------------------------------------

def filter_openrouter_free_text(models: list[dict]) -> list[str]:
    """Filter OpenRouter to free :free models."""
    out = []
    for m in models:
        mid = m.get("id", "")
        if not mid.endswith(":free"):
            continue

        pricing = m.get("pricing", {})
        prompt_price = str(pricing.get("prompt", "")).strip()
        completion_price = str(pricing.get("completion", "")).strip()
        if prompt_price not in ("0", "0.0", "") or completion_price not in ("0", "0.0", ""):
            continue

        arch = m.get("architecture", {})
        output_modalities = arch.get("output_modalities", [])
        if output_modalities and "text" not in output_modalities:
            continue

        lower = mid.lower()
        skip_keywords = ["image", "audio", "tts", "stt", "whisper", "dall-e", "lyria",
                         "seedream", "video", "vision-only", "flux"]
        if any(kw in lower for kw in skip_keywords):
            continue

        out.append(mid)

    return sorted(set(out))


def filter_groq_text(models: list[dict]) -> list[str]:
    available_ids = {m.get("id") for m in models}
    return [m for m in GROQ_TEXT_MODELS if m in available_ids]


def filter_cerebras_text(models: list[dict]) -> list[str]:
    """Cerebras /v1/models works; filter against whitelist for stable text models."""
    if not models:
        return CEREBRAS_TEXT_MODELS  # fallback if listing failed
    available_ids = {m.get("id") for m in models}
    found = [m for m in CEREBRAS_TEXT_MODELS if m in available_ids]
    # If our whitelist is stale, fall through to whatever Cerebras returned that looks text
    if not found:
        for m in models:
            mid = m.get("id", "")
            if any(kw in mid.lower() for kw in ["llama", "qwen", "gpt-oss", "glm", "deepseek"]):
                found.append(mid)
    return sorted(set(found))


def filter_mistral_text(models: list[dict]) -> list[str]:
    """Mistral API exposes /v1/models. Filter to non-embedding chat-capable models."""
    if not models:
        return MISTRAL_TEXT_MODELS
    out = []
    for m in models:
        mid = m.get("id", "")
        if any(kw in mid.lower() for kw in ["embed", "ocr", "moderation", "saba"]):
            continue
        # Only chat-capable
        capabilities = m.get("capabilities", {})
        if isinstance(capabilities, dict) and not capabilities.get("completion_chat", True):
            continue
        out.append(mid)
    return sorted(set(out)) if out else MISTRAL_TEXT_MODELS


def filter_google_gemini(models: list[dict]) -> list[str]:
    """Google AI Studio: only gemini text models (excluding embedding, vision-only)."""
    out = []
    for m in models:
        mid = m.get("id", "")
        lower = mid.lower()
        if "gemini" not in lower:
            continue
        if any(kw in lower for kw in ["embed", "aqa", "image", "vision-bison"]):
            continue
        out.append(mid)
    return sorted(set(out))


# -----------------------------------------------------------------------------
# Models requiring manual testing (no system prompt override via API)
# -----------------------------------------------------------------------------
MANUAL_MODELS = [
    {
        "id": "gitlab-duo-claude-sonnet-4.6",
        "vendor": "GitLab",
        "reason": "Native GitLab Duo behavior — system prompt is GitLab-controlled",
        "interface": "GitLab Duo Chat in GitLab.com UI",
    },
    {
        "id": "github-copilot-chat",
        "vendor": "GitHub/Microsoft",
        "reason": "Copilot system prompt not overridable via API",
        "interface": "VS Code / JetBrains Copilot Chat",
    },
    {
        "id": "cursor",
        "vendor": "Cursor",
        "reason": "Cursor wraps models with its own system prompt",
        "interface": "Cursor IDE",
    },
    {
        "id": "claude.ai-web",
        "vendor": "Anthropic",
        "reason": "Anthropic web UI; tests how claude.ai behaves vs raw API Claude",
        "interface": "claude.ai",
    },
    {
        "id": "chatgpt-web",
        "vendor": "OpenAI",
        "reason": "ChatGPT web UI with ChatGPT system prompt",
        "interface": "chat.openai.com",
    },
    {
        "id": "gemini-web",
        "vendor": "Google",
        "reason": "Gemini web UI",
        "interface": "gemini.google.com",
    },
    {
        "id": "windsurf",
        "vendor": "Codeium",
        "reason": "Codeium Windsurf IDE with wrapped prompts",
        "interface": "Windsurf IDE",
    },
]
