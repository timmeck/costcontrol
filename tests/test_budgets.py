"""Tests for budget management."""

from src.proxy.budgets import BudgetManager


class TestBudgetCheck:
    """Test budget checking logic."""

    async def test_within_budget(self, db, sample_app):
        mgr = BudgetManager(db)
        # No requests yet, should be within budget
        status = await mgr.check_budget(sample_app["id"])
        assert status["within_budget"] is True
        assert status["should_downgrade"] is False
        assert status["daily_spend"] == 0
        assert status["monthly_spend"] == 0

    async def test_daily_budget_exceeded(self, db, sample_app):
        mgr = BudgetManager(db)
        # Record enough requests to exceed daily budget of $10
        for _ in range(20):
            await db.record_request(
                sample_app["id"],
                "claude-opus-4-6",
                "anthropic",
                10000,
                5000,
                0.6,
                1000,
            )
        status = await mgr.check_budget(sample_app["id"])
        assert status["within_budget"] is False
        assert status["should_downgrade"] is True
        assert status["daily_spend"] >= 10.0

    async def test_monthly_budget_exceeded(self, db, sample_app):
        mgr = BudgetManager(db)
        # Record enough to exceed $100 monthly budget
        for _ in range(20):
            await db.record_request(
                sample_app["id"],
                "claude-opus-4-6",
                "anthropic",
                50000,
                20000,
                5.5,
                1000,
            )
        status = await mgr.check_budget(sample_app["id"])
        assert status["within_budget"] is False
        assert status["should_downgrade"] is True

    async def test_no_budget_always_within(self, db, sample_app_no_budget):
        mgr = BudgetManager(db)
        # Even with spend, no budget means always within
        await db.record_request(
            sample_app_no_budget["id"],
            "claude-opus-4-6",
            "anthropic",
            100000,
            50000,
            50.0,
            2000,
        )
        status = await mgr.check_budget(sample_app_no_budget["id"])
        assert status["within_budget"] is True
        assert status["should_downgrade"] is False

    async def test_nonexistent_app(self, db):
        mgr = BudgetManager(db)
        status = await mgr.check_budget(9999)
        assert status["within_budget"] is True
        assert status["should_downgrade"] is False

    async def test_budget_percentage_calculation(self, db, sample_app):
        mgr = BudgetManager(db)
        # Spend $5 against $10 daily budget = 50%
        await db.record_request(
            sample_app["id"],
            "claude-sonnet-4-6",
            "anthropic",
            10000,
            5000,
            5.0,
            1000,
        )
        status = await mgr.check_budget(sample_app["id"])
        assert status["daily_pct"] == 50.0

    async def test_fallback_model_in_status(self, db, sample_app):
        mgr = BudgetManager(db)
        status = await mgr.check_budget(sample_app["id"])
        assert status["fallback_model"] == "qwen3:14b"


class TestAlertGeneration:
    """Test budget alert creation."""

    async def test_no_alerts_when_within_budget(self, db, sample_app):
        mgr = BudgetManager(db)
        alerts = await mgr.check_and_alert(sample_app["id"])
        assert len(alerts) == 0

    async def test_alert_at_80_percent(self, db, sample_app):
        mgr = BudgetManager(db)
        # Spend $8.50 against $10 daily budget = 85%
        await db.record_request(
            sample_app["id"],
            "claude-opus-4-6",
            "anthropic",
            10000,
            5000,
            8.5,
            1000,
        )
        alerts = await mgr.check_and_alert(sample_app["id"])
        # Should get 80% alert for daily
        daily_alerts = [a for a in alerts if "daily" in a["alert_type"]]
        assert len(daily_alerts) >= 1

    async def test_alert_at_100_percent(self, db, sample_app):
        mgr = BudgetManager(db)
        # Exceed daily budget
        await db.record_request(
            sample_app["id"],
            "claude-opus-4-6",
            "anthropic",
            10000,
            5000,
            11.0,
            1000,
        )
        alerts = await mgr.check_and_alert(sample_app["id"])
        exceeded = [a for a in alerts if "exceeded" in a["alert_type"]]
        assert len(exceeded) >= 1

    async def test_no_duplicate_alerts(self, db, sample_app):
        mgr = BudgetManager(db)
        await db.record_request(
            sample_app["id"],
            "claude-opus-4-6",
            "anthropic",
            10000,
            5000,
            8.5,
            1000,
        )
        await mgr.check_and_alert(sample_app["id"])
        alerts2 = await mgr.check_and_alert(sample_app["id"])
        # Second call should not create duplicate alerts
        assert len(alerts2) == 0


class TestSetBudget:
    """Test budget setting."""

    async def test_set_monthly_budget(self, db, sample_app):
        mgr = BudgetManager(db)
        success = await mgr.set_budget(sample_app["id"], monthly=200.0)
        assert success is True
        app = await db.get_app_by_id(sample_app["id"])
        assert app["budget_monthly"] == 200.0

    async def test_set_daily_budget(self, db, sample_app):
        mgr = BudgetManager(db)
        success = await mgr.set_budget(sample_app["id"], daily=25.0)
        assert success is True
        app = await db.get_app_by_id(sample_app["id"])
        assert app["budget_daily"] == 25.0

    async def test_set_both_budgets(self, db, sample_app):
        mgr = BudgetManager(db)
        success = await mgr.set_budget(sample_app["id"], monthly=300.0, daily=30.0)
        assert success is True
        app = await db.get_app_by_id(sample_app["id"])
        assert app["budget_monthly"] == 300.0
        assert app["budget_daily"] == 30.0


class TestAllBudgetStatuses:
    """Test getting all budget statuses."""

    async def test_get_all_statuses(self, db, sample_app, sample_app_no_budget):
        mgr = BudgetManager(db)
        statuses = await mgr.get_all_budget_statuses()
        assert len(statuses) == 2
        names = {s["app_name"] for s in statuses}
        assert "test-app" in names
        assert "unlimited-app" in names
