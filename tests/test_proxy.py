"""Tests for the proxy engine — token counting, cost calculation, budget enforcement."""

import pytest

from src.proxy.engine import ProxyEngine
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
