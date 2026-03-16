"""SQLite database layer for CostControl — async with aiosqlite + WAL mode."""

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from src.utils.logger import get_logger

log = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS apps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    api_key TEXT NOT NULL,
    budget_monthly REAL DEFAULT 0,
    budget_daily REAL DEFAULT 0,
    auto_downgrade INTEGER DEFAULT 1,
    fallback_model TEXT DEFAULT 'qwen3:14b',
    status TEXT DEFAULT 'active',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    latency_ms INTEGER DEFAULT 0,
    status TEXT DEFAULT 'success',
    downgraded INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (app_id) REFERENCES apps(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id INTEGER,
    alert_type TEXT NOT NULL,
    message TEXT NOT NULL,
    threshold REAL,
    current_value REAL,
    acknowledged INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id INTEGER NOT NULL,
    date TEXT NOT NULL,
    total_requests INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0,
    models_used TEXT,
    UNIQUE(app_id, date)
);

CREATE TABLE IF NOT EXISTS activity_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id INTEGER,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    data TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_requests_app_id ON requests(app_id);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_daily_costs_date ON daily_costs(date);
CREATE INDEX IF NOT EXISTS idx_alerts_app_id ON alerts(app_id);
CREATE INDEX IF NOT EXISTS idx_activity_log_created ON activity_log(created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Database:
    """Async SQLite database for CostControl."""

    def __init__(self, db_path: Path | str):
        self.db_path = str(db_path)
        self._db: aiosqlite.Connection | None = None

    async def connect(self):
        """Open the database connection and initialize schema."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA)
        await self._db.commit()
        log.info(f"Database connected: {self.db_path}")

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database not connected"
        return self._db

    # ── Apps ─────────────────────────────────────────────────────

    async def create_app(self, name: str, api_key: str, budget_monthly: float = 0,
                         budget_daily: float = 0, auto_downgrade: bool = True,
                         fallback_model: str = "qwen3:14b") -> dict:
        """Register a new application."""
        now = _now()
        cursor = await self.db.execute(
            """INSERT INTO apps (name, api_key, budget_monthly, budget_daily,
               auto_downgrade, fallback_model, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (name, api_key, budget_monthly, budget_daily,
             1 if auto_downgrade else 0, fallback_model, now),
        )
        await self.db.commit()
        app_id = cursor.lastrowid
        await self.log_activity(app_id, "app_registered", f"App '{name}' registered")
        return {"id": app_id, "name": name, "api_key": api_key,
                "budget_monthly": budget_monthly, "budget_daily": budget_daily,
                "auto_downgrade": auto_downgrade, "fallback_model": fallback_model,
                "status": "active", "created_at": now}

    async def get_app_by_key(self, api_key: str) -> dict | None:
        """Look up an app by its API key."""
        cursor = await self.db.execute("SELECT * FROM apps WHERE api_key = ?", (api_key,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_app_by_name(self, name: str) -> dict | None:
        """Look up an app by name."""
        cursor = await self.db.execute("SELECT * FROM apps WHERE name = ?", (name,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_app_by_id(self, app_id: int) -> dict | None:
        """Look up an app by ID."""
        cursor = await self.db.execute("SELECT * FROM apps WHERE id = ?", (app_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_apps(self) -> list[dict]:
        """List all registered apps."""
        cursor = await self.db.execute("SELECT * FROM apps ORDER BY name")
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_app_budget(self, app_id: int, budget_monthly: float | None = None,
                                budget_daily: float | None = None) -> bool:
        """Update budget for an app."""
        fields = []
        vals = []
        if budget_monthly is not None:
            fields.append("budget_monthly = ?")
            vals.append(budget_monthly)
        if budget_daily is not None:
            fields.append("budget_daily = ?")
            vals.append(budget_daily)
        if not fields:
            return False
        vals.append(app_id)
        await self.db.execute(f"UPDATE apps SET {', '.join(fields)} WHERE id = ?", vals)
        await self.db.commit()
        await self.log_activity(app_id, "budget_updated",
                                f"Budget updated: monthly={budget_monthly}, daily={budget_daily}")
        return True

    async def delete_app(self, app_id: int) -> bool:
        """Delete an app and all related data."""
        cursor = await self.db.execute("DELETE FROM apps WHERE id = ?", (app_id,))
        await self.db.commit()
        return cursor.rowcount > 0

    # ── Requests ─────────────────────────────────────────────────

    async def record_request(self, app_id: int, model: str, provider: str,
                             input_tokens: int, output_tokens: int, cost_usd: float,
                             latency_ms: int, status: str = "success",
                             downgraded: bool = False) -> dict:
        """Record an LLM request."""
        now = _now()
        total_tokens = input_tokens + output_tokens
        cursor = await self.db.execute(
            """INSERT INTO requests (app_id, model, provider, input_tokens, output_tokens,
               total_tokens, cost_usd, latency_ms, status, downgraded, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (app_id, model, provider, input_tokens, output_tokens, total_tokens,
             cost_usd, latency_ms, status, 1 if downgraded else 0, now),
        )
        await self.db.commit()
        await self._update_daily_cost(app_id, model, input_tokens + output_tokens, cost_usd)
        return {"id": cursor.lastrowid, "app_id": app_id, "model": model,
                "provider": provider, "input_tokens": input_tokens,
                "output_tokens": output_tokens, "total_tokens": total_tokens,
                "cost_usd": cost_usd, "latency_ms": latency_ms, "status": status,
                "downgraded": downgraded, "created_at": now}

    async def get_recent_requests(self, limit: int = 50, app_id: int | None = None) -> list[dict]:
        """Get recent requests, optionally filtered by app."""
        if app_id:
            cursor = await self.db.execute(
                """SELECT r.*, a.name as app_name FROM requests r
                   JOIN apps a ON r.app_id = a.id
                   WHERE r.app_id = ? ORDER BY r.created_at DESC LIMIT ?""",
                (app_id, limit),
            )
        else:
            cursor = await self.db.execute(
                """SELECT r.*, a.name as app_name FROM requests r
                   JOIN apps a ON r.app_id = a.id
                   ORDER BY r.created_at DESC LIMIT ?""",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Daily Costs ──────────────────────────────────────────────

    async def _update_daily_cost(self, app_id: int, model: str, tokens: int, cost: float):
        """Update the daily cost summary for an app."""
        today = _today()
        cursor = await self.db.execute(
            "SELECT id, models_used FROM daily_costs WHERE app_id = ? AND date = ?",
            (app_id, today),
        )
        row = await cursor.fetchone()
        if row:
            existing_models = set(json.loads(row["models_used"])) if row["models_used"] else set()
            existing_models.add(model)
            await self.db.execute(
                """UPDATE daily_costs SET total_requests = total_requests + 1,
                   total_tokens = total_tokens + ?, total_cost = total_cost + ?,
                   models_used = ? WHERE id = ?""",
                (tokens, cost, json.dumps(list(existing_models)), row["id"]),
            )
        else:
            await self.db.execute(
                """INSERT INTO daily_costs (app_id, date, total_requests, total_tokens,
                   total_cost, models_used) VALUES (?, ?, 1, ?, ?, ?)""",
                (app_id, today, tokens, cost, json.dumps([model])),
            )
        await self.db.commit()

    async def get_daily_costs(self, days: int = 30, app_id: int | None = None) -> list[dict]:
        """Get daily cost summaries for the last N days."""
        if app_id:
            cursor = await self.db.execute(
                """SELECT * FROM daily_costs WHERE app_id = ?
                   ORDER BY date DESC LIMIT ?""",
                (app_id, days),
            )
        else:
            cursor = await self.db.execute(
                """SELECT date, SUM(total_requests) as total_requests,
                   SUM(total_tokens) as total_tokens, SUM(total_cost) as total_cost
                   FROM daily_costs GROUP BY date ORDER BY date DESC LIMIT ?""",
                (days,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_app_spend_today(self, app_id: int) -> float:
        """Get total spend for an app today."""
        today = _today()
        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as spend FROM requests WHERE app_id = ? AND created_at LIKE ?",
            (app_id, f"{today}%"),
        )
        row = await cursor.fetchone()
        return row["spend"] if row else 0.0

    async def get_app_spend_month(self, app_id: int) -> float:
        """Get total spend for an app this month."""
        month_prefix = _today()[:7]
        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as spend FROM requests WHERE app_id = ? AND created_at LIKE ?",
            (app_id, f"{month_prefix}%"),
        )
        row = await cursor.fetchone()
        return row["spend"] if row else 0.0

    # ── Alerts ───────────────────────────────────────────────────

    async def create_alert(self, app_id: int, alert_type: str, message: str,
                           threshold: float = 0, current_value: float = 0) -> dict:
        """Create a budget alert."""
        now = _now()
        cursor = await self.db.execute(
            """INSERT INTO alerts (app_id, alert_type, message, threshold, current_value, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (app_id, alert_type, message, threshold, current_value, now),
        )
        await self.db.commit()
        return {"id": cursor.lastrowid, "app_id": app_id, "alert_type": alert_type,
                "message": message, "threshold": threshold, "current_value": current_value,
                "acknowledged": 0, "created_at": now}

    async def get_unacknowledged_alerts(self, app_id: int | None = None) -> list[dict]:
        """Get unacknowledged alerts."""
        if app_id:
            cursor = await self.db.execute(
                """SELECT al.*, a.name as app_name FROM alerts al
                   LEFT JOIN apps a ON al.app_id = a.id
                   WHERE al.acknowledged = 0 AND al.app_id = ?
                   ORDER BY al.created_at DESC""",
                (app_id,),
            )
        else:
            cursor = await self.db.execute(
                """SELECT al.*, a.name as app_name FROM alerts al
                   LEFT JOIN apps a ON al.app_id = a.id
                   WHERE al.acknowledged = 0 ORDER BY al.created_at DESC""",
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def acknowledge_alert(self, alert_id: int) -> bool:
        """Acknowledge an alert."""
        cursor = await self.db.execute(
            "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    # ── Activity Log ─────────────────────────────────────────────

    async def log_activity(self, app_id: int | None, event_type: str,
                           message: str, data: dict | None = None):
        """Log an activity event."""
        now = _now()
        await self.db.execute(
            """INSERT INTO activity_log (app_id, event_type, message, data, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (app_id, event_type, message, json.dumps(data) if data else None, now),
        )
        await self.db.commit()

    async def get_activity_log(self, limit: int = 100, app_id: int | None = None) -> list[dict]:
        """Get recent activity log entries."""
        if app_id:
            cursor = await self.db.execute(
                "SELECT * FROM activity_log WHERE app_id = ? ORDER BY created_at DESC LIMIT ?",
                (app_id, limit),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM activity_log ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Stats ────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        """Get overall system statistics."""
        today = _today()
        month_prefix = today[:7]

        cursor = await self.db.execute(
            "SELECT COUNT(*) as count FROM apps WHERE status = 'active'",
        )
        active_apps = (await cursor.fetchone())["count"]

        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as spend FROM requests WHERE created_at LIKE ?",
            (f"{today}%",),
        )
        spend_today = (await cursor.fetchone())["spend"]

        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as spend FROM requests WHERE created_at LIKE ?",
            (f"{month_prefix}%",),
        )
        spend_month = (await cursor.fetchone())["spend"]

        cursor = await self.db.execute(
            "SELECT COUNT(*) as count FROM requests WHERE created_at LIKE ?",
            (f"{today}%",),
        )
        requests_today = (await cursor.fetchone())["count"]

        cursor = await self.db.execute("SELECT COUNT(*) as count FROM requests")
        total_requests = (await cursor.fetchone())["count"]

        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) as spend FROM requests",
        )
        total_spend = (await cursor.fetchone())["spend"]

        cursor = await self.db.execute(
            "SELECT COUNT(*) as count FROM alerts WHERE acknowledged = 0",
        )
        active_alerts = (await cursor.fetchone())["count"]

        return {
            "active_apps": active_apps,
            "spend_today": round(spend_today, 4),
            "spend_month": round(spend_month, 4),
            "requests_today": requests_today,
            "total_requests": total_requests,
            "total_spend": round(total_spend, 4),
            "active_alerts": active_alerts,
        }

    async def get_model_breakdown(self, days: int = 30) -> list[dict]:
        """Get spend breakdown by model over last N days."""
        cursor = await self.db.execute(
            """SELECT model, provider, COUNT(*) as request_count,
               SUM(input_tokens) as total_input, SUM(output_tokens) as total_output,
               SUM(total_tokens) as total_tokens, SUM(cost_usd) as total_cost
               FROM requests WHERE created_at >= date('now', ?)
               GROUP BY model, provider ORDER BY total_cost DESC""",
            (f"-{days} days",),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_app_breakdown(self) -> list[dict]:
        """Get spend breakdown by app with budget info."""
        month_prefix = _today()[:7]
        today = _today()
        cursor = await self.db.execute(
            """SELECT a.id, a.name, a.budget_monthly, a.budget_daily, a.auto_downgrade,
               a.fallback_model, a.status,
               COALESCE((SELECT SUM(cost_usd) FROM requests WHERE app_id = a.id
                AND created_at LIKE ?), 0) as spend_month,
               COALESCE((SELECT SUM(cost_usd) FROM requests WHERE app_id = a.id
                AND created_at LIKE ?), 0) as spend_today,
               COALESCE((SELECT COUNT(*) FROM requests WHERE app_id = a.id
                AND created_at LIKE ?), 0) as requests_today
               FROM apps a ORDER BY spend_month DESC""",
            (f"{month_prefix}%", f"{today}%", f"{today}%"),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
