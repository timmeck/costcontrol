"""Tests for the database layer."""

import pytest

from src.db.database import Database


class TestAppsCRUD:
    """Test app creation, retrieval, update, and deletion."""

    async def test_create_app(self, db):
        app = await db.create_app(name="my-app", api_key="cc_abc123", budget_monthly=50.0)
        assert app["name"] == "my-app"
        assert app["api_key"] == "cc_abc123"
        assert app["budget_monthly"] == 50.0
        assert app["status"] == "active"
        assert app["id"] is not None

    async def test_get_app_by_key(self, db, sample_app):
        found = await db.get_app_by_key("cc_test123456")
        assert found is not None
        assert found["name"] == "test-app"

    async def test_get_app_by_key_not_found(self, db):
        found = await db.get_app_by_key("cc_nonexistent")
        assert found is None

    async def test_get_app_by_name(self, db, sample_app):
        found = await db.get_app_by_name("test-app")
        assert found is not None
        assert found["api_key"] == "cc_test123456"

    async def test_get_app_by_id(self, db, sample_app):
        found = await db.get_app_by_id(sample_app["id"])
        assert found is not None
        assert found["name"] == "test-app"

    async def test_list_apps(self, db, sample_app, sample_app_no_budget):
        apps = await db.list_apps()
        assert len(apps) == 2
        names = {a["name"] for a in apps}
        assert "test-app" in names
        assert "unlimited-app" in names

    async def test_update_budget(self, db, sample_app):
        success = await db.update_app_budget(sample_app["id"], budget_monthly=200.0, budget_daily=20.0)
        assert success is True
        updated = await db.get_app_by_id(sample_app["id"])
        assert updated["budget_monthly"] == 200.0
        assert updated["budget_daily"] == 20.0

    async def test_delete_app(self, db, sample_app):
        success = await db.delete_app(sample_app["id"])
        assert success is True
        found = await db.get_app_by_id(sample_app["id"])
        assert found is None

    async def test_delete_app_not_found(self, db):
        success = await db.delete_app(9999)
        assert success is False

    async def test_duplicate_app_name_fails(self, db, sample_app):
        with pytest.raises(Exception):
            await db.create_app(name="test-app", api_key="cc_different")


class TestRequests:
    """Test request recording and retrieval."""

    async def test_record_request(self, db, sample_app):
        req = await db.record_request(
            app_id=sample_app["id"], model="claude-sonnet-4-6",
            provider="anthropic", input_tokens=1000, output_tokens=500,
            cost_usd=0.0105, latency_ms=1200,
        )
        assert req["total_tokens"] == 1500
        assert req["cost_usd"] == 0.0105
        assert req["status"] == "success"

    async def test_get_recent_requests(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 500, 200, 0.003, 800)
        await db.record_request(sample_app["id"], "claude-sonnet-4-6", "anthropic", 1000, 500, 0.01, 1200)
        reqs = await db.get_recent_requests(limit=10)
        assert len(reqs) == 2
        # Most recent first
        assert reqs[0]["model"] == "claude-sonnet-4-6"

    async def test_get_recent_requests_by_app(self, db, sample_app, sample_app_no_budget):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 500, 200, 0.003, 800)
        await db.record_request(sample_app_no_budget["id"], "qwen3:14b", "ollama", 500, 200, 0, 300)
        reqs = await db.get_recent_requests(limit=10, app_id=sample_app["id"])
        assert len(reqs) == 1
        assert reqs[0]["model"] == "gpt-4o"


class TestDailyCosts:
    """Test daily cost summaries."""

    async def test_daily_cost_creation(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        costs = await db.get_daily_costs(days=1, app_id=sample_app["id"])
        assert len(costs) == 1
        assert costs[0]["total_requests"] == 1
        assert costs[0]["total_cost"] == 0.005

    async def test_daily_cost_accumulation(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 2000, 800, 0.01, 700)
        costs = await db.get_daily_costs(days=1, app_id=sample_app["id"])
        assert len(costs) == 1
        assert costs[0]["total_requests"] == 2
        assert abs(costs[0]["total_cost"] - 0.015) < 0.0001


class TestAlerts:
    """Test alert creation and management."""

    async def test_create_alert(self, db, sample_app):
        alert = await db.create_alert(
            sample_app["id"], "monthly_budget", "Budget warning", 0.8, 80.0,
        )
        assert alert["alert_type"] == "monthly_budget"
        assert alert["acknowledged"] == 0

    async def test_get_unacknowledged_alerts(self, db, sample_app):
        await db.create_alert(sample_app["id"], "monthly_budget", "Warning 1", 0.8, 80.0)
        await db.create_alert(sample_app["id"], "daily_budget", "Warning 2", 0.9, 9.0)
        alerts = await db.get_unacknowledged_alerts()
        assert len(alerts) == 2

    async def test_acknowledge_alert(self, db, sample_app):
        alert = await db.create_alert(sample_app["id"], "test", "Test alert", 0.5, 50.0)
        success = await db.acknowledge_alert(alert["id"])
        assert success is True
        alerts = await db.get_unacknowledged_alerts()
        assert len(alerts) == 0


class TestStats:
    """Test statistics queries."""

    async def test_get_stats_empty(self, db):
        stats = await db.get_stats()
        assert stats["active_apps"] == 0
        assert stats["spend_today"] == 0
        assert stats["total_requests"] == 0

    async def test_get_stats_with_data(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        stats = await db.get_stats()
        assert stats["active_apps"] == 1
        assert stats["spend_today"] == 0.005
        assert stats["requests_today"] == 1

    async def test_get_app_spend_today(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        spend = await db.get_app_spend_today(sample_app["id"])
        assert spend == 0.005

    async def test_get_app_spend_month(self, db, sample_app):
        await db.record_request(sample_app["id"], "gpt-4o", "openai", 1000, 500, 0.005, 500)
        spend = await db.get_app_spend_month(sample_app["id"])
        assert spend == 0.005


class TestActivityLog:
    """Test activity logging."""

    async def test_log_activity(self, db, sample_app):
        await db.log_activity(sample_app["id"], "test_event", "Test message", {"key": "val"})
        log = await db.get_activity_log(limit=10, app_id=sample_app["id"])
        assert len(log) >= 1
        # Find our test event (app creation also logs)
        test_events = [e for e in log if e["event_type"] == "test_event"]
        assert len(test_events) == 1
        assert test_events[0]["message"] == "Test message"
