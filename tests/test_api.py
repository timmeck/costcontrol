"""Tests for FastAPI endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.web.api import app, db as app_db
from src.db.database import Database
import tempfile
from pathlib import Path


@pytest.fixture
async def client(tmp_db_path):
    """Create a test client with a temporary database."""
    from src.web import api as api_module

    # Override the global db
    test_db = Database(tmp_db_path)
    await test_db.connect()

    # Patch globals
    original_db = api_module.db
    original_engine = api_module.engine
    original_analytics = api_module.analytics
    original_budget_mgr = api_module.budget_mgr

    from src.proxy.engine import ProxyEngine
    from src.proxy.analytics import Analytics
    from src.proxy.budgets import BudgetManager

    api_module.db = test_db
    api_module.engine = ProxyEngine(test_db)
    api_module.analytics = Analytics(test_db)
    api_module.budget_mgr = BudgetManager(test_db)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    # Restore
    api_module.db = original_db
    api_module.engine = original_engine
    api_module.analytics = original_analytics
    api_module.budget_mgr = original_budget_mgr
    await test_db.close()


class TestStatusEndpoint:
    """Test the status endpoint."""

    async def test_get_status(self, client):
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["product"] == "CostControl"
        assert "active_apps" in data
        assert "spend_today" in data


class TestAppsEndpoints:
    """Test app registration and listing."""

    async def test_register_app(self, client):
        resp = await client.post("/api/apps/register", json={
            "name": "test-app",
            "budget_monthly": 100.0,
            "budget_daily": 10.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["app"]["name"] == "test-app"
        assert data["app"]["api_key"].startswith("cc_")
        assert data["app"]["budget_monthly"] == 100.0

    async def test_register_duplicate_app(self, client):
        await client.post("/api/apps/register", json={"name": "dup-app"})
        resp = await client.post("/api/apps/register", json={"name": "dup-app"})
        assert resp.status_code == 409

    async def test_register_app_no_name(self, client):
        resp = await client.post("/api/apps/register", json={})
        assert resp.status_code == 400

    async def test_list_apps(self, client):
        await client.post("/api/apps/register", json={"name": "app1"})
        await client.post("/api/apps/register", json={"name": "app2"})
        resp = await client.get("/api/apps")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["apps"]) == 2

    async def test_set_budget(self, client):
        reg = await client.post("/api/apps/register", json={"name": "budget-app"})
        app_id = reg.json()["app"]["id"]
        resp = await client.post(f"/api/apps/{app_id}/budget", json={
            "budget_monthly": 200.0,
            "budget_daily": 20.0,
        })
        assert resp.status_code == 200

    async def test_delete_app(self, client):
        reg = await client.post("/api/apps/register", json={"name": "del-app"})
        app_id = reg.json()["app"]["id"]
        resp = await client.delete(f"/api/apps/{app_id}")
        assert resp.status_code == 200


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""

    async def test_daily_report(self, client):
        resp = await client.get("/api/analytics/daily")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_type"] == "daily"
        assert "stats" in data

    async def test_weekly_report(self, client):
        resp = await client.get("/api/analytics/weekly")
        assert resp.status_code == 200
        assert resp.json()["report_type"] == "weekly"

    async def test_monthly_report(self, client):
        resp = await client.get("/api/analytics/monthly")
        assert resp.status_code == 200
        assert resp.json()["report_type"] == "monthly"

    async def test_spend_trend(self, client):
        resp = await client.get("/api/analytics/trend")
        assert resp.status_code == 200
        assert "trend" in resp.json()

    async def test_model_breakdown(self, client):
        resp = await client.get("/api/analytics/models")
        assert resp.status_code == 200
        assert "models" in resp.json()


class TestAlertsEndpoints:
    """Test alert endpoints."""

    async def test_get_alerts_empty(self, client):
        resp = await client.get("/api/alerts")
        assert resp.status_code == 200
        assert resp.json()["alerts"] == []


class TestBudgetsEndpoint:
    """Test budget status endpoint."""

    async def test_get_budgets(self, client):
        await client.post("/api/apps/register", json={"name": "b-app", "budget_monthly": 50.0})
        resp = await client.get("/api/budgets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["budgets"]) == 1
        assert data["budgets"][0]["app_name"] == "b-app"


class TestPricingEndpoint:
    """Test pricing endpoint."""

    async def test_get_pricing(self, client):
        resp = await client.get("/api/pricing")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) > 0
        # Check a known model
        sonnet = [m for m in data["models"] if m["model"] == "claude-sonnet-4-6"]
        assert len(sonnet) == 1
        assert sonnet[0]["input_per_million"] == 3.0


class TestDashboard:
    """Test dashboard serving."""

    async def test_dashboard_serves_html(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "CostControl" in resp.text
        assert "text/html" in resp.headers["content-type"]
