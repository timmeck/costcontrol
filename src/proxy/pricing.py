"""Token pricing tables for LLM providers.

All prices are per million tokens.
"""

from src.utils.logger import get_logger

log = get_logger("pricing")

# Pricing per million tokens (input / output)
PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o3": {"input": 2.0, "output": 8.0},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Ollama (free / local)
    "qwen3:14b": {"input": 0, "output": 0},
    "llama3.1:8b": {"input": 0, "output": 0},
    "mistral:7b": {"input": 0, "output": 0},
}

# Map model names to providers
PROVIDER_MAP: dict[str, str] = {
    "claude-opus-4-6": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-haiku-4-5-20251001": "anthropic",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "gpt-4.1": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-4.1-nano": "openai",
    "o3": "openai",
    "o4-mini": "openai",
    "qwen3:14b": "ollama",
    "llama3.1:8b": "ollama",
    "mistral:7b": "ollama",
}


def get_provider(model: str) -> str:
    """Determine the provider for a given model name."""
    if model in PROVIDER_MAP:
        return PROVIDER_MAP[model]
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o3", "o4")):
        return "openai"
    return "ollama"


def get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model. Returns zero pricing for unknown models (assumes local)."""
    if model in PRICING:
        return PRICING[model]
    # Unknown model — check by prefix
    if model.startswith("claude") or model.startswith(("gpt-", "o3", "o4")):
        log.warning(f"Unknown paid model '{model}' — defaulting to zero pricing")
    return {"input": 0, "output": 0}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost in USD for a given model and token counts.

    Prices are per million tokens, so divide by 1_000_000.
    """
    pricing = get_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)


def is_free_model(model: str) -> bool:
    """Check if a model is free (local Ollama)."""
    pricing = get_pricing(model)
    return pricing["input"] == 0 and pricing["output"] == 0


def list_models() -> list[dict]:
    """List all known models with their pricing info."""
    result = []
    for model, prices in PRICING.items():
        result.append({
            "model": model,
            "provider": get_provider(model),
            "input_per_million": prices["input"],
            "output_per_million": prices["output"],
            "free": prices["input"] == 0 and prices["output"] == 0,
        })
    return result
