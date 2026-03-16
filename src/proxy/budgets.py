"""Budget management — enforce limits, trigger alerts, auto-downgrade logic."""

from src.db.database import Database
from src.utils.logger import get_logger

log = get_logger("budgets")

# Alert thresholds as percentages
ALERT_THRESHOLDS = [0.80, 0.90, 1.00]


class BudgetManager:
    """Manages budget enforcement and alerting for registered apps."""

    def __init__(self, db: Database):
        self.db = db

    async def check_budget(self, app_id: int) -> dict:
        """Check budget status for an app.

        Returns:
            {
                "within_budget": bool,
                "should_downgrade": bool,
                "daily_spend": float,
                "monthly_spend": float,
                "daily_budget": float,
                "monthly_budget": float,
                "daily_pct": float,
                "monthly_pct": float,
            }
        """
        app = await self.db.get_app_by_id(app_id)
        if not app:
            return {"within_budget": True, "should_downgrade": False,
                    "daily_spend": 0, "monthly_spend": 0,
                    "daily_budget": 0, "monthly_budget": 0,
                    "daily_pct": 0, "monthly_pct": 0}

        daily_spend = await self.db.get_app_spend_today(app_id)
        monthly_spend = await self.db.get_app_spend_month(app_id)

        daily_budget = app["budget_daily"]
        monthly_budget = app["budget_monthly"]
        auto_downgrade = bool(app["auto_downgrade"])

        daily_pct = (daily_spend / daily_budget * 100) if daily_budget > 0 else 0
        monthly_pct = (monthly_spend / monthly_budget * 100) if monthly_budget > 0 else 0

        # Check if over budget
        daily_exceeded = daily_budget > 0 and daily_spend >= daily_budget
        monthly_exceeded = monthly_budget > 0 and monthly_spend >= monthly_budget
        over_budget = daily_exceeded or monthly_exceeded

        should_downgrade = over_budget and auto_downgrade

        return {
            "within_budget": not over_budget,
            "should_downgrade": should_downgrade,
            "daily_spend": round(daily_spend, 4),
            "monthly_spend": round(monthly_spend, 4),
            "daily_budget": daily_budget,
            "monthly_budget": monthly_budget,
            "daily_pct": round(daily_pct, 1),
            "monthly_pct": round(monthly_pct, 1),
            "daily_exceeded": daily_exceeded,
            "monthly_exceeded": monthly_exceeded,
            "fallback_model": app.get("fallback_model", "qwen3:14b"),
        }

    async def check_and_alert(self, app_id: int) -> list[dict]:
        """Check budget and generate alerts if thresholds are crossed.

        Returns list of newly created alerts.
        """
        app = await self.db.get_app_by_id(app_id)
        if not app:
            return []

        alerts_created = []

        # Check monthly budget thresholds
        if app["budget_monthly"] > 0:
            monthly_spend = await self.db.get_app_spend_month(app_id)
            monthly_pct = monthly_spend / app["budget_monthly"]

            for threshold in ALERT_THRESHOLDS:
                if monthly_pct >= threshold:
                    existing = await self._alert_exists(
                        app_id, "monthly_budget", threshold,
                    )
                    if not existing:
                        pct_label = int(threshold * 100)
                        if threshold >= 1.0:
                            msg = (f"App '{app['name']}' has EXCEEDED monthly budget! "
                                   f"${monthly_spend:.2f} / ${app['budget_monthly']:.2f}")
                            alert_type = "monthly_budget_exceeded"
                        else:
                            msg = (f"App '{app['name']}' has reached {pct_label}% of monthly budget: "
                                   f"${monthly_spend:.2f} / ${app['budget_monthly']:.2f}")
                            alert_type = "monthly_budget"
                        alert = await self.db.create_alert(
                            app_id, alert_type, msg, threshold, monthly_spend,
                        )
                        alerts_created.append(alert)
                        log.warning(msg)

        # Check daily budget thresholds
        if app["budget_daily"] > 0:
            daily_spend = await self.db.get_app_spend_today(app_id)
            daily_pct = daily_spend / app["budget_daily"]

            for threshold in ALERT_THRESHOLDS:
                if daily_pct >= threshold:
                    existing = await self._alert_exists(
                        app_id, "daily_budget", threshold,
                    )
                    if not existing:
                        pct_label = int(threshold * 100)
                        if threshold >= 1.0:
                            msg = (f"App '{app['name']}' has EXCEEDED daily budget! "
                                   f"${daily_spend:.2f} / ${app['budget_daily']:.2f}")
                            alert_type = "daily_budget_exceeded"
                        else:
                            msg = (f"App '{app['name']}' has reached {pct_label}% of daily budget: "
                                   f"${daily_spend:.2f} / ${app['budget_daily']:.2f}")
                            alert_type = "daily_budget"
                        alert = await self.db.create_alert(
                            app_id, alert_type, msg, threshold, daily_spend,
                        )
                        alerts_created.append(alert)
                        log.warning(msg)

        return alerts_created

    async def _alert_exists(self, app_id: int, alert_type_prefix: str,
                            threshold: float) -> bool:
        """Check if an alert already exists for this app/threshold combo today."""
        from src.db.database import _today
        today = _today()
        cursor = await self.db.db.execute(
            """SELECT COUNT(*) as cnt FROM alerts
               WHERE app_id = ? AND alert_type LIKE ? AND threshold = ?
               AND created_at LIKE ?""",
            (app_id, f"{alert_type_prefix}%", threshold, f"{today}%"),
        )
        row = await cursor.fetchone()
        return row["cnt"] > 0

    async def set_budget(self, app_id: int, monthly: float | None = None,
                         daily: float | None = None) -> bool:
        """Set budget limits for an app."""
        return await self.db.update_app_budget(app_id, monthly, daily)

    async def get_all_budget_statuses(self) -> list[dict]:
        """Get budget status for all apps."""
        apps = await self.db.list_apps()
        statuses = []
        for app in apps:
            status = await self.check_budget(app["id"])
            status["app_name"] = app["name"]
            status["app_id"] = app["id"]
            statuses.append(status)
        return statuses
