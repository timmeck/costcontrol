"""Cost analytics and reporting for CostControl."""

from src.db.database import Database
from src.utils.logger import get_logger

log = get_logger("analytics")


class Analytics:
    """Generate cost analytics and reports."""

    def __init__(self, db: Database):
        self.db = db

    async def daily_report(self) -> dict:
        """Generate a daily spending report."""
        stats = await self.db.get_stats()
        app_breakdown = await self.db.get_app_breakdown()
        model_breakdown = await self.db.get_model_breakdown(days=1)
        recent = await self.db.get_recent_requests(limit=20)
        alerts = await self.db.get_unacknowledged_alerts()

        return {
            "report_type": "daily",
            "stats": stats,
            "app_breakdown": app_breakdown,
            "model_breakdown": model_breakdown,
            "recent_requests": recent,
            "alerts": alerts,
        }

    async def weekly_report(self) -> dict:
        """Generate a weekly spending report."""
        stats = await self.db.get_stats()
        daily_costs = await self.db.get_daily_costs(days=7)
        app_breakdown = await self.db.get_app_breakdown()
        model_breakdown = await self.db.get_model_breakdown(days=7)

        total_week_spend = sum(d.get("total_cost", 0) for d in daily_costs)
        total_week_requests = sum(d.get("total_requests", 0) for d in daily_costs)
        total_week_tokens = sum(d.get("total_tokens", 0) for d in daily_costs)

        avg_cost_per_request = (
            total_week_spend / total_week_requests if total_week_requests > 0 else 0
        )

        return {
            "report_type": "weekly",
            "stats": stats,
            "daily_costs": daily_costs,
            "app_breakdown": app_breakdown,
            "model_breakdown": model_breakdown,
            "summary": {
                "total_spend": round(total_week_spend, 4),
                "total_requests": total_week_requests,
                "total_tokens": total_week_tokens,
                "avg_cost_per_request": round(avg_cost_per_request, 6),
            },
        }

    async def monthly_report(self) -> dict:
        """Generate a monthly spending report."""
        stats = await self.db.get_stats()
        daily_costs = await self.db.get_daily_costs(days=30)
        app_breakdown = await self.db.get_app_breakdown()
        model_breakdown = await self.db.get_model_breakdown(days=30)

        total_month_spend = sum(d.get("total_cost", 0) for d in daily_costs)
        total_month_requests = sum(d.get("total_requests", 0) for d in daily_costs)
        total_month_tokens = sum(d.get("total_tokens", 0) for d in daily_costs)

        avg_cost_per_request = (
            total_month_spend / total_month_requests if total_month_requests > 0 else 0
        )

        # Find most expensive app
        most_expensive_app = None
        if app_breakdown:
            most_expensive_app = max(app_breakdown, key=lambda a: a.get("spend_month", 0))

        # Find most used model
        most_used_model = None
        if model_breakdown:
            most_used_model = max(model_breakdown, key=lambda m: m.get("request_count", 0))

        return {
            "report_type": "monthly",
            "stats": stats,
            "daily_costs": daily_costs,
            "app_breakdown": app_breakdown,
            "model_breakdown": model_breakdown,
            "summary": {
                "total_spend": round(total_month_spend, 4),
                "total_requests": total_month_requests,
                "total_tokens": total_month_tokens,
                "avg_cost_per_request": round(avg_cost_per_request, 6),
                "most_expensive_app": most_expensive_app,
                "most_used_model": most_used_model,
            },
        }

    async def cost_per_query_avg(self, app_id: int | None = None, days: int = 30) -> float:
        """Calculate average cost per query over the last N days."""
        daily_costs = await self.db.get_daily_costs(days=days, app_id=app_id)
        total_cost = sum(d.get("total_cost", 0) for d in daily_costs)
        total_requests = sum(d.get("total_requests", 0) for d in daily_costs)
        if total_requests == 0:
            return 0.0
        return round(total_cost / total_requests, 6)

    async def spend_trend(self, days: int = 30) -> list[dict]:
        """Get spend trend data for charting (last N days)."""
        daily_costs = await self.db.get_daily_costs(days=days)
        # Reverse so oldest first for charts
        return list(reversed(daily_costs))

    async def format_text_report(self, report_type: str = "daily") -> str:
        """Format a human-readable text report."""
        if report_type == "weekly":
            data = await self.weekly_report()
        elif report_type == "monthly":
            data = await self.monthly_report()
        else:
            data = await self.daily_report()

        lines = []
        lines.append(f"=== CostControl {report_type.upper()} Report ===")
        lines.append("")

        stats = data["stats"]
        lines.append(f"  Active Apps:      {stats['active_apps']}")
        lines.append(f"  Spend Today:      ${stats['spend_today']:.4f}")
        lines.append(f"  Spend This Month: ${stats['spend_month']:.4f}")
        lines.append(f"  Requests Today:   {stats['requests_today']}")
        lines.append(f"  Total Requests:   {stats['total_requests']}")
        lines.append(f"  Total Spend:      ${stats['total_spend']:.4f}")
        lines.append(f"  Active Alerts:    {stats['active_alerts']}")
        lines.append("")

        if data.get("app_breakdown"):
            lines.append("--- App Breakdown ---")
            for app in data["app_breakdown"]:
                budget_str = ""
                if app.get("budget_monthly", 0) > 0:
                    pct = (app["spend_month"] / app["budget_monthly"]) * 100
                    budget_str = f" ({pct:.0f}% of ${app['budget_monthly']:.2f} monthly)"
                lines.append(f"  {app['name']}: ${app['spend_month']:.4f}{budget_str}")
            lines.append("")

        if data.get("model_breakdown"):
            lines.append("--- Model Breakdown ---")
            for m in data["model_breakdown"]:
                lines.append(
                    f"  {m['model']} ({m['provider']}): "
                    f"{m['request_count']} reqs, ${m['total_cost']:.4f}"
                )
            lines.append("")

        if data.get("alerts"):
            lines.append("--- Active Alerts ---")
            for a in data["alerts"]:
                lines.append(f"  [{a['alert_type']}] {a['message']}")
            lines.append("")

        return "\n".join(lines)
