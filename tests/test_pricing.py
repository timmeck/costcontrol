"""Tests for the pricing module."""

import pytest

from src.proxy.pricing import (
    PRICING,
    calculate_cost,
    get_pricing,
    get_provider,
    is_free_model,
    list_models,
)


class TestGetProvider:
    """Test provider identification."""

    def test_anthropic_models(self):
        assert get_provider("claude-opus-4-6") == "anthropic"
        assert get_provider("claude-sonnet-4-6") == "anthropic"
        assert get_provider("claude-haiku-4-5-20251001") == "anthropic"

    def test_openai_models(self):
        assert get_provider("gpt-4o") == "openai"
        assert get_provider("gpt-4o-mini") == "openai"
        assert get_provider("gpt-4.1") == "openai"
        assert get_provider("o3") == "openai"
        assert get_provider("o4-mini") == "openai"

    def test_ollama_models(self):
        assert get_provider("qwen3:14b") == "ollama"
        assert get_provider("llama3.1:8b") == "ollama"
        assert get_provider("mistral:7b") == "ollama"

    def test_unknown_model_defaults_ollama(self):
        assert get_provider("some-local-model") == "ollama"

    def test_unknown_claude_model(self):
        assert get_provider("claude-future-99") == "anthropic"


class TestGetPricing:
    """Test pricing lookup."""

    def test_known_model(self):
        p = get_pricing("claude-sonnet-4-6")
        assert p["input"] == 3.0
        assert p["output"] == 15.0

    def test_free_model(self):
        p = get_pricing("qwen3:14b")
        assert p["input"] == 0
        assert p["output"] == 0

    def test_unknown_model_zero_pricing(self):
        p = get_pricing("my-local-model:latest")
        assert p["input"] == 0
        assert p["output"] == 0


class TestCalculateCost:
    """Test cost calculation."""

    def test_claude_sonnet_cost(self):
        # 1000 input tokens, 500 output tokens
        # Input: (1000/1M) * 3.0 = 0.003
        # Output: (500/1M) * 15.0 = 0.0075
        # Total: 0.0105
        cost = calculate_cost("claude-sonnet-4-6", 1000, 500)
        assert abs(cost - 0.0105) < 0.0001

    def test_claude_opus_cost(self):
        # 1000 input, 500 output
        # Input: (1000/1M) * 15.0 = 0.015
        # Output: (500/1M) * 75.0 = 0.0375
        cost = calculate_cost("claude-opus-4-6", 1000, 500)
        assert abs(cost - 0.0525) < 0.0001

    def test_gpt4o_cost(self):
        # 1000 input, 500 output
        # Input: (1000/1M) * 2.50 = 0.0025
        # Output: (500/1M) * 10.0 = 0.005
        cost = calculate_cost("gpt-4o", 1000, 500)
        assert abs(cost - 0.0075) < 0.0001

    def test_free_model_zero_cost(self):
        cost = calculate_cost("qwen3:14b", 10000, 5000)
        assert cost == 0

    def test_zero_tokens(self):
        cost = calculate_cost("claude-sonnet-4-6", 0, 0)
        assert cost == 0

    def test_large_token_count(self):
        # 1M input tokens on Opus = $15.00
        cost = calculate_cost("claude-opus-4-6", 1_000_000, 0)
        assert abs(cost - 15.0) < 0.01

    def test_gpt4_1_nano_cost(self):
        # 10000 input, 5000 output
        # Input: (10000/1M) * 0.10 = 0.001
        # Output: (5000/1M) * 0.40 = 0.002
        cost = calculate_cost("gpt-4.1-nano", 10000, 5000)
        assert abs(cost - 0.003) < 0.0001


class TestIsFreeModel:
    """Test free model detection."""

    def test_ollama_models_are_free(self):
        assert is_free_model("qwen3:14b") is True
        assert is_free_model("llama3.1:8b") is True
        assert is_free_model("mistral:7b") is True

    def test_paid_models_not_free(self):
        assert is_free_model("claude-sonnet-4-6") is False
        assert is_free_model("gpt-4o") is False

    def test_unknown_model_is_free(self):
        assert is_free_model("custom-local:latest") is True


class TestListModels:
    """Test model listing."""

    def test_list_models_count(self):
        models = list_models()
        assert len(models) == len(PRICING)

    def test_list_models_structure(self):
        models = list_models()
        for m in models:
            assert "model" in m
            assert "provider" in m
            assert "input_per_million" in m
            assert "output_per_million" in m
            assert "free" in m
