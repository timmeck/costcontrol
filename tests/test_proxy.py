"""Tests for the proxy engine — token counting, cost calculation, budget enforcement."""

import pytest

from src.proxy.cache import ResponseCache
from src.proxy.engine import ProxyEngine, pick_cheaper_model, score_complexity
from src.proxy.pricing import calculate_cost, get_provider


class TestProxyEngineInit:
    """Test proxy engine initialization."""

    async def test_engine_creation(self, db):
        engine = ProxyEngine(db)
        assert engine.db is db
        assert engine.budget_mgr is not None


class TestProxyChatValidation:
    """Test proxy chat input validation."""

    async def test_invalid_api_key(self, db):
        engine = ProxyEngine(db)
        with pytest.raises(ValueError, match="Invalid API key"):
            await engine.proxy_chat(
                app_key="cc_nonexistent",
                model="qwen3:14b",
                messages=[{"role": "user", "content": "hello"}],
            )

    async def test_inactive_app(self, db, sample_app):
        engine = ProxyEngine(db)
        # Manually set app to inactive
        await db.db.execute("UPDATE apps SET status = 'inactive' WHERE id = ?", (sample_app["id"],))
        await db.db.commit()

        with pytest.raises(ValueError, match="not active"):
            await engine.proxy_chat(
                app_key="cc_test123456",
                model="qwen3:14b",
                messages=[{"role": "user", "content": "hello"}],
            )


class TestCostCalculationIntegration:
    """Test cost calculation through the proxy flow."""

    def test_cost_for_anthropic_request(self):
        # Simulate: 2000 input, 800 output on claude-sonnet-4-6
        cost = calculate_cost("claude-sonnet-4-6", 2000, 800)
        # Input: (2000/1M) * 3.0 = 0.006
        # Output: (800/1M) * 15.0 = 0.012
        expected = 0.018
        assert abs(cost - expected) < 0.001

    def test_cost_for_openai_request(self):
        cost = calculate_cost("gpt-4o-mini", 5000, 2000)
        # Input: (5000/1M) * 0.15 = 0.00075
        # Output: (2000/1M) * 0.60 = 0.0012
        expected = 0.00195
        assert abs(cost - expected) < 0.001

    def test_cost_for_ollama_is_zero(self):
        cost = calculate_cost("qwen3:14b", 50000, 20000)
        assert cost == 0

    def test_provider_detection(self):
        assert get_provider("claude-sonnet-4-6") == "anthropic"
        assert get_provider("gpt-4o") == "openai"
        assert get_provider("qwen3:14b") == "ollama"


class TestBudgetEnforcementInProxy:
    """Test that the proxy correctly enforces budgets."""

    async def test_auto_downgrade_check(self, db, sample_app):
        engine = ProxyEngine(db)
        # Exceed daily budget
        for _ in range(3):
            await db.record_request(
                sample_app["id"],
                "claude-opus-4-6",
                "anthropic",
                10000,
                5000,
                5.0,
                1000,
            )
        # Now check budget status
        status = await engine.budget_mgr.check_budget(sample_app["id"])
        assert status["should_downgrade"] is True
        assert status["fallback_model"] == "qwen3:14b"

    async def test_no_downgrade_when_within_budget(self, db, sample_app):
        engine = ProxyEngine(db)
        # Small spend
        await db.record_request(
            sample_app["id"],
            "gpt-4o-mini",
            "openai",
            1000,
            500,
            0.001,
            500,
        )
        status = await engine.budget_mgr.check_budget(sample_app["id"])
        assert status["should_downgrade"] is False

    async def test_no_downgrade_for_free_model(self, db, sample_app):
        ProxyEngine(db)
        # Even if over budget, free models don't need downgrade
        for _ in range(3):
            await db.record_request(
                sample_app["id"],
                "claude-opus-4-6",
                "anthropic",
                10000,
                5000,
                5.0,
                1000,
            )
        # The proxy would check is_free_model before downgrading
        from src.proxy.pricing import is_free_model

        assert is_free_model("qwen3:14b") is True


class TestCostEstimation:
    """Test cost estimation without executing requests."""

    async def test_estimate_paid_model(self, db, sample_app):
        engine = ProxyEngine(db)
        result = await engine.estimate_cost(
            "claude-sonnet-4-6",
            [{"role": "user", "content": "What is the meaning of life?"}],
            max_tokens=2000,
        )
        assert result["model"] == "claude-sonnet-4-6"
        assert result["provider"] == "anthropic"
        assert result["estimated_cost_usd"] > 0
        assert result["free"] is False

    async def test_estimate_free_model(self, db, sample_app):
        engine = ProxyEngine(db)
        result = await engine.estimate_cost(
            "qwen3:14b",
            [{"role": "user", "content": "Hello"}],
        )
        assert result["estimated_cost_usd"] == 0
        assert result["free"] is True


class TestComplexityScoring:
    """Test the complexity scorer for smart routing."""

    def test_simple_prompt_low_score(self):
        score = score_complexity("Hello, how are you?")
        assert score < 0.3

    def test_code_prompt_high_score(self):
        score = score_complexity("def fibonacci(n): return n if n <= 1 else fibonacci(n-1)")
        assert score >= 0.3

    def test_reasoning_prompt(self):
        score = score_complexity("Analyze and compare these two approaches step by step")
        assert score >= 0.2

    def test_math_prompt(self):
        score = score_complexity("Calculate the equation for proof of this algorithm")
        assert score >= 0.2

    def test_combined_reasoning_and_code(self):
        score = score_complexity("Analyze this ```python\ndef foo(): pass``` step by step, first plan then finally execute")
        assert score >= 0.6

    def test_long_prompt_gets_bonus(self):
        short = score_complexity("Hello")
        long = score_complexity("x " * 1500)  # > 2000 chars
        assert long > short

    def test_score_capped_at_one(self):
        # Trigger everything: code + reasoning + math + long + multi-step
        prompt = (
            "```python\ndef analyze(): pass```\n"
            "Analyze step by step, calculate the equation, "
            "first plan then finally " + "x " * 1500
        )
        score = score_complexity(prompt)
        assert score == 1.0

    def test_empty_prompt(self):
        score = score_complexity("")
        assert score == 0.0


class TestSmartRouting:
    """Test the pick_cheaper_model function."""

    def test_low_complexity_gets_cheaper_model(self):
        cheaper = pick_cheaper_model("claude-opus-4-6", 0.1)
        assert cheaper is not None
        assert cheaper != "claude-opus-4-6"

    def test_high_complexity_keeps_model(self):
        cheaper = pick_cheaper_model("claude-opus-4-6", 0.5)
        assert cheaper is None

    def test_already_cheapest_returns_none(self):
        cheaper = pick_cheaper_model("gpt-4.1-nano", 0.1)
        assert cheaper is None

    def test_free_model_returns_none(self):
        cheaper = pick_cheaper_model("qwen3:14b", 0.1)
        assert cheaper is None

    def test_very_low_complexity_skips_two_tiers(self):
        cheaper = pick_cheaper_model("claude-opus-4-6", 0.05)
        # Should skip 2 tiers from opus -> haiku
        assert cheaper == "claude-haiku-4-5-20251001"

    def test_medium_low_complexity_skips_one_tier(self):
        cheaper = pick_cheaper_model("claude-opus-4-6", 0.2)
        # Should skip 1 tier from opus -> sonnet
        assert cheaper == "claude-sonnet-4-6"


class TestResponseCache:
    """Test the response cache."""

    def test_cache_miss(self):
        cache = ResponseCache()
        result = cache.get("gpt-4o", [{"role": "user", "content": "hello"}])
        assert result is None

    def test_cache_hit(self):
        cache = ResponseCache()
        messages = [{"role": "user", "content": "hello"}]
        response = {"content": "Hi there!", "model": "gpt-4o"}
        cache.put("gpt-4o", messages, response)
        result = cache.get("gpt-4o", messages)
        assert result is not None
        assert result["content"] == "Hi there!"

    def test_cache_different_model_misses(self):
        cache = ResponseCache()
        messages = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4o", messages, {"content": "Hi"})
        result = cache.get("gpt-4o-mini", messages)
        assert result is None

    def test_cache_different_messages_misses(self):
        cache = ResponseCache()
        cache.put("gpt-4o", [{"role": "user", "content": "hello"}], {"content": "Hi"})
        result = cache.get("gpt-4o", [{"role": "user", "content": "goodbye"}])
        assert result is None

    def test_cache_expiry(self):
        cache = ResponseCache(default_ttl=0)  # Expires immediately
        messages = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4o", messages, {"content": "Hi"})
        import time
        time.sleep(0.01)
        result = cache.get("gpt-4o", messages)
        assert result is None

    def test_cache_stats(self):
        cache = ResponseCache()
        messages = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4o", messages, {"content": "Hi"})
        cache.get("gpt-4o", messages)  # hit
        cache.get("gpt-4o-mini", messages)  # miss

        stats = cache.stats()
        assert stats["cache_size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == 50.0

    def test_cache_clear(self):
        cache = ResponseCache()
        cache.put("gpt-4o", [{"role": "user", "content": "hello"}], {"content": "Hi"})
        assert cache.stats()["cache_size"] == 1
        cache.clear()
        assert cache.stats()["cache_size"] == 0

    def test_cleanup_expired(self):
        cache = ResponseCache(default_ttl=0)
        cache.put("gpt-4o", [{"role": "user", "content": "hello"}], {"content": "Hi"})
        import time
        time.sleep(0.01)
        cache.cleanup_expired()
        assert cache.stats()["cache_size"] == 0

    def test_engine_has_cache(self, db):
        """Verify ProxyEngine initializes with a cache."""
        engine = ProxyEngine(db)
        assert engine.cache is not None
        assert isinstance(engine.cache, ResponseCache)
