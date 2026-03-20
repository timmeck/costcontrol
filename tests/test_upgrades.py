"""Tests for the three upgrades: prompt optimization, billing export, webhook alerts."""

import time
from unittest.mock import AsyncMock, patch

import pytest

from src.proxy.engine import optimize_prompt, _estimate_tokens


# ── Upgrade 1: Prompt Compression / Token Optimization ──────


class TestEstimateTokens:
    """Test the rough token estimator."""

    def test_empty_string(self):
        assert _estimate_tokens("") == 1

    def test_single_word(self):
        assert _estimate_tokens("hello") >= 1

    def test_longer_text(self):
        tokens = _estimate_tokens("the quick brown fox jumps over the lazy dog")
        assert tokens > 5


class TestOptimizePrompt:
    """Test prompt optimization."""

    def test_empty_messages(self):
        result, meta = optimize_prompt([])
        assert result == []
        assert meta["tokens_saved"] == 0
        assert meta["messages_condensed"] == 0

    def test_whitespace_normalization(self):
        messages = [
            {"role": "user", "content": "hello    world   \t\t  foo"},
        ]
        result, meta = optimize_prompt(messages)
        assert "    " not in result[0]["content"]
        assert "\t" not in result[0]["content"]
        assert "hello world foo" == result[0]["content"]

    def test_newline_normalization(self):
        messages = [
            {"role": "user", "content": "line1\n\n\n\n\nline2"},
        ]
        result, meta = optimize_prompt(messages)
        # Multiple newlines collapsed to max 2
        assert "\n\n\n" not in result[0]["content"]
        assert "line1\n\nline2" == result[0]["content"]

    def test_short_messages_not_condensed(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(5)]
        result, meta = optimize_prompt(messages)
        assert len(result) == 5
        assert meta["messages_condensed"] == 0

    def test_long_messages_condensed(self):
        messages = [{"role": "user", "content": f"message number {i} with content"} for i in range(15)]
        result, meta = optimize_prompt(messages)
        # Should be fewer messages now
        assert len(result) < 15
        assert meta["messages_condensed"] > 0

    def test_system_message_preserved(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
        ] + [{"role": "user", "content": f"msg {i} with some text"} for i in range(14)]
        result, meta = optimize_prompt(messages)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_last_messages_preserved(self):
        messages = [{"role": "user", "content": f"unique_msg_{i}"} for i in range(15)]
        result, meta = optimize_prompt(messages)
        # Last 6 messages should be preserved
        last_contents = [m["content"] for m in result[-6:]]
        for i in range(9, 15):
            assert f"unique_msg_{i}" in last_contents

    def test_max_reduction_cap(self):
        # Create messages where condensing would remove too much
        messages = [
            {"role": "system", "content": "System."},
        ] + [{"role": "user", "content": "x"} for _ in range(20)]
        result, meta = optimize_prompt(messages, max_reduction=0.3)
        # Should not reduce by more than 30%
        if meta["tokens_before"] > 0:
            reduction = meta["tokens_saved"] / meta["tokens_before"]
            assert reduction <= 0.31  # small margin for rounding

    def test_returns_tokens_saved(self):
        messages = [
            {"role": "user", "content": "hello     world   with     spaces"},
        ]
        result, meta = optimize_prompt(messages)
        assert "tokens_before" in meta
        assert "tokens_after" in meta
        assert "tokens_saved" in meta
        assert meta["tokens_saved"] >= 0

    def test_preserves_role(self):
        messages = [
            {"role": "user", "content": "hi  there"},
            {"role": "assistant", "content": "hello  back"},
        ]
        result, meta = optimize_prompt(messages)
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"


# ── Upgrade 2: Usage Export / Billing Reports ────────────────


class TestBillingExport:
    """Test billing export database methods."""

    async def test_billing_export_empty(self, db):
        rows = await db.get_billing_export()
        assert rows == []

    async def test_billing_export_with_data(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 2000, 800, 0.01, 700)
        rows = await db.get_billing_export()
        assert len(rows) == 1  # Same date/app/model grouped
        assert rows[0]["requests"] == 2
        assert rows[0]["tokens_in"] == 3000
        assert rows[0]["tokens_out"] == 1300
        assert abs(rows[0]["cost_usd"] - 0.015) < 0.001

    async def test_billing_export_filter_by_key(self, db, sample_app, sample_app_no_budget):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        await db.record_request(sample_app_no_budget["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        rows = await db.get_billing_export(app_key="cc_test123456")
        assert len(rows) == 1
        assert rows[0]["app_name"] == "test-app"

    async def test_billing_export_filter_by_date(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        # Filter with future date range should return nothing
        rows = await db.get_billing_export(date_from="2099-01-01", date_to="2099-12-31")
        assert rows == []

    async def test_billing_summary_empty(self, db):
        summary = await db.get_billing_summary()
        assert summary["by_app"] == []
        assert summary["by_model"] == []
        assert summary["by_day"] == []

    async def test_billing_summary_with_data(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        await db.record_request(sample_app["id"], "claude-sonnet-4-6", "anthropic", 2000, 800, 0.02, 1200)
        summary = await db.get_billing_summary()

        assert len(summary["by_app"]) == 1
        assert summary["by_app"][0]["app_name"] == "test-app"
        assert summary["by_app"][0]["requests"] == 2

        assert len(summary["by_model"]) == 2
        model_names = {m["model"] for m in summary["by_model"]}
        assert "gpt-4o" in model_names
        assert "claude-sonnet-4-6" in model_names

        assert len(summary["by_day"]) == 1


class TestBillingAPI:
    """Test billing API endpoints."""

    @pytest.fixture
    async def client(self, tmp_db_path):
        from httpx import ASGITransport, AsyncClient

        from src.db.database import Database
        from src.proxy.analytics import Analytics
        from src.proxy.budgets import BudgetManager
        from src.proxy.engine import ProxyEngine
        from src.web import api as api_module
        from src.web.api import app

        test_db = Database(tmp_db_path)
        await test_db.connect()

        original_db = api_module.db
        original_engine = api_module.engine
        original_analytics = api_module.analytics
        original_budget_mgr = api_module.budget_mgr

        api_module.db = test_db
        api_module.engine = ProxyEngine(test_db)
        api_module.analytics = Analytics(test_db)
        api_module.budget_mgr = BudgetManager(test_db)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac, test_db

        api_module.db = original_db
        api_module.engine = original_engine
        api_module.analytics = original_analytics
        api_module.budget_mgr = original_budget_mgr
        await test_db.close()

    async def test_billing_export_json(self, client):
        ac, test_db = client
        app_data = await test_db.create_app(name="export-app", api_key="cc_export1")
        await test_db.record_request(app_data["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        resp = await ac.get("/api/billing/export")
        assert resp.status_code == 200
        data = resp.json()
        assert "export" in data
        assert len(data["export"]) == 1

    async def test_billing_export_csv(self, client):
        ac, test_db = client
        app_data = await test_db.create_app(name="csv-app", api_key="cc_csv1")
        await test_db.record_request(app_data["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        resp = await ac.get("/api/billing/export?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        text = resp.text
        assert "date" in text
        assert "app_name" in text
        assert "cost_usd" in text

    async def test_billing_export_with_app_key(self, client):
        ac, test_db = client
        app1 = await test_db.create_app(name="app-a", api_key="cc_a1")
        app2 = await test_db.create_app(name="app-b", api_key="cc_b1")
        await test_db.record_request(app1["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        await test_db.record_request(app2["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        resp = await ac.get("/api/billing/export?app_key=cc_a1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["export"]) == 1
        assert data["export"][0]["app_name"] == "app-a"

    async def test_billing_summary(self, client):
        ac, test_db = client
        app_data = await test_db.create_app(name="sum-app", api_key="cc_sum1")
        await test_db.record_request(app_data["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        resp = await ac.get("/api/billing/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "by_app" in data
        assert "by_model" in data
        assert "by_day" in data


# ── Upgrade 3: Budget Alerts via Webhook ─────────────────────


class TestWebhookDebounce:
    """Test webhook debounce logic."""

    def test_first_send_allowed(self):
        from src.proxy.budgets import _should_send_webhook, _webhook_debounce
        _webhook_debounce.clear()
        assert _should_send_webhook(999, 80) is True

    def test_second_send_blocked(self):
        from src.proxy.budgets import _should_send_webhook, _webhook_debounce
        _webhook_debounce.clear()
        _should_send_webhook(999, 80)
        assert _should_send_webhook(999, 80) is False

    def test_different_level_allowed(self):
        from src.proxy.budgets import _should_send_webhook, _webhook_debounce
        _webhook_debounce.clear()
        _should_send_webhook(999, 80)
        assert _should_send_webhook(999, 90) is True

    def test_different_app_allowed(self):
        from src.proxy.budgets import _should_send_webhook, _webhook_debounce
        _webhook_debounce.clear()
        _should_send_webhook(1, 80)
        assert _should_send_webhook(2, 80) is True

    def test_expired_debounce_allowed(self):
        from src.proxy.budgets import _should_send_webhook, _webhook_debounce, DEBOUNCE_SECONDS
        _webhook_debounce.clear()
        _should_send_webhook(999, 80)
        # Manually expire the debounce
        _webhook_debounce[(999, 80)] = time.time() - DEBOUNCE_SECONDS - 1
        assert _should_send_webhook(999, 80) is True


class TestWebhookAlertSending:
    """Test webhook alert sending."""

    @patch("src.proxy.budgets.ALERT_WEBHOOK_URL", "https://example.com/webhook")
    @patch("src.proxy.budgets.NTFY_TOPIC", "")
    async def test_webhook_called(self):
        from src.proxy.budgets import _send_webhook_alert
        payload = {
            "app": "test-app",
            "budget_used_pct": 85.0,
            "budget_limit": 100.0,
            "current_spend": 85.0,
            "alert_level": 80,
            "message": "Budget warning",
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = AsyncMock(status_code=200)
            await _send_webhook_alert(payload)
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://example.com/webhook"

    @patch("src.proxy.budgets.ALERT_WEBHOOK_URL", "")
    @patch("src.proxy.budgets.NTFY_TOPIC", "test-topic")
    async def test_ntfy_called(self):
        from src.proxy.budgets import _send_webhook_alert
        payload = {
            "app": "test-app",
            "budget_used_pct": 95.0,
            "budget_limit": 100.0,
            "current_spend": 95.0,
            "alert_level": 90,
            "message": "Budget warning 90%",
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = AsyncMock(status_code=200)
            await _send_webhook_alert(payload)
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert "ntfy.sh/test-topic" in call_args[0][0]

    @patch("src.proxy.budgets.ALERT_WEBHOOK_URL", "https://example.com/webhook")
    @patch("src.proxy.budgets.NTFY_TOPIC", "test-topic")
    async def test_both_called(self):
        from src.proxy.budgets import _send_webhook_alert
        payload = {
            "app": "test-app",
            "budget_used_pct": 100.0,
            "budget_limit": 100.0,
            "current_spend": 100.0,
            "alert_level": 100,
            "message": "Budget exceeded",
        }
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = AsyncMock(status_code=200)
            await _send_webhook_alert(payload)
            assert mock_post.call_count == 2

    @patch("src.proxy.budgets.ALERT_WEBHOOK_URL", "")
    @patch("src.proxy.budgets.NTFY_TOPIC", "")
    async def test_nothing_called_when_not_configured(self):
        from src.proxy.budgets import _send_webhook_alert
        payload = {"app": "test", "alert_level": 80, "message": "test"}
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            await _send_webhook_alert(payload)
            mock_post.assert_not_called()


class TestWebhookIntegration:
    """Test that webhook alerts fire during budget checks."""

    @patch("src.proxy.budgets.ALERT_WEBHOOK_URL", "https://example.com/hook")
    @patch("src.proxy.budgets.NTFY_TOPIC", "")
    async def test_webhook_fires_on_budget_alert(self, db, sample_app):
        from src.proxy.budgets import BudgetManager, _webhook_debounce
        _webhook_debounce.clear()
        mgr = BudgetManager(db)
        # Exceed daily budget
        await db.record_request(
            sample_app["id"], "claude-opus-4-6", "anthropic", 10000, 5000, 11.0, 1000,
        )
        with patch("src.proxy.budgets._send_webhook_alert", new_callable=AsyncMock) as mock_send:
            alerts = await mgr.check_and_alert(sample_app["id"])
            assert len(alerts) > 0
            # Webhook should have been called at least once
            assert mock_send.call_count >= 1
            # Verify payload structure
            call_payload = mock_send.call_args[0][0]
            assert "app" in call_payload
            assert "budget_used_pct" in call_payload
            assert "alert_level" in call_payload
            assert "current_spend" in call_payload
