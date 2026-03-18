"""FastAPI routes + SSE for CostControl dashboard."""

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from src.config import COSTCONTROL_PORT, DB_PATH, NEXUS_URL
from src.db.database import Database
from src.nexus_sdk import NexusAdapter
from src.proxy.analytics import Analytics
from src.proxy.budgets import BudgetManager
from src.proxy.engine import ProxyEngine
from src.proxy.pricing import list_models
from src.utils.logger import get_logger
from src.web.auth import AuthMiddleware

log = get_logger("api")

# Global instances
db: Database | None = None
engine: ProxyEngine | None = None
analytics: Analytics | None = None
budget_mgr: BudgetManager | None = None

# SSE event queue
sse_clients: list[asyncio.Queue] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global db, engine, analytics, budget_mgr
    db = Database(DB_PATH)
    await db.connect()
    engine = ProxyEngine(db)
    analytics = Analytics(db)
    budget_mgr = BudgetManager(db)
    log.info(f"CostControl started on port {COSTCONTROL_PORT}")
    yield
    if db:
        await db.close()
    log.info("CostControl shut down")


app = FastAPI(title="CostControl", version="1.0.0", lifespan=lifespan)
app.add_middleware(AuthMiddleware)

nexus = NexusAdapter(
    app=app, agent_name="costcontrol", nexus_url=NEXUS_URL,
    endpoint=f"http://localhost:{COSTCONTROL_PORT}",
    description="LLM Cost Controller — token tracking, budgets, cost analytics",
    capabilities=[
        {"name": "cost_tracking", "description": "Track and report LLM API costs", "languages": ["en"], "price_per_request": 0.001},
        {"name": "budget_management", "description": "Manage per-app LLM budgets", "languages": ["en"], "price_per_request": 0.001},
    ],
    tags=["cost", "budget", "analytics", "llm"],
)


@nexus.handle("cost_tracking")
async def handle_cost_tracking(query: str, params: dict) -> dict:
    if analytics:
        report = await analytics.daily_report()
        return {"result": json.dumps(report, default=str), "confidence": 0.95, "cost": 0.001}
    return {"result": "Analytics not initialized", "confidence": 0.0, "cost": 0.0}


@nexus.handle("budget_management")
async def handle_budget(query: str, params: dict) -> dict:
    if budget_mgr:
        app_id = params.get("app_id")
        if app_id:
            budgets = await budget_mgr.get_budget(app_id)
            answer = json.dumps(budgets, default=str) if budgets else "No budget found"
        else:
            answer = "Provide app_id in constraints"
        return {"result": answer, "confidence": 0.95, "cost": 0.001}
    return {"result": "Budget manager not initialized", "confidence": 0.0, "cost": 0.0}


def _broadcast_sse(event: str, data: dict):
    """Send an event to all SSE clients."""
    msg = json.dumps(data)
    for q in sse_clients:
        try:
            q.put_nowait({"event": event, "data": msg})
        except asyncio.QueueFull:
            pass


# ── Dashboard ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    template_path = Path(__file__).parent / "templates" / "dashboard.html"
    return HTMLResponse(template_path.read_text(encoding="utf-8"))


# ── SSE Stream ───────────────────────────────────────────────

@app.get("/api/events/stream")
async def event_stream(request: Request):
    """SSE endpoint for real-time updates."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    sse_clients.append(queue)

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield msg
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            sse_clients.remove(queue)

    return EventSourceResponse(generate())


# ── Status ───────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    """Get system status and stats."""
    stats = await db.get_stats()
    return {"status": "ok", "product": "CostControl", "version": "1.0.0", **stats}


# ── Apps ─────────────────────────────────────────────────────

@app.get("/api/apps")
async def list_apps():
    """List all registered apps with spend info."""
    breakdown = await db.get_app_breakdown()
    return {"apps": breakdown}


@app.post("/api/apps/register")
async def register_app(request: Request):
    """Register a new app."""
    body = await request.json()
    name = body.get("name")
    if not name:
        raise HTTPException(400, "App name is required")

    existing = await db.get_app_by_name(name)
    if existing:
        raise HTTPException(409, f"App '{name}' already exists")

    api_key = f"cc_{secrets.token_hex(24)}"
    budget_monthly = body.get("budget_monthly", 0)
    budget_daily = body.get("budget_daily", 0)
    auto_downgrade = body.get("auto_downgrade", True)
    fallback_model = body.get("fallback_model", "qwen3:14b")

    app_data = await db.create_app(
        name=name, api_key=api_key, budget_monthly=budget_monthly,
        budget_daily=budget_daily, auto_downgrade=auto_downgrade,
        fallback_model=fallback_model,
    )

    _broadcast_sse("app_registered", {"name": name})
    return {"app": app_data}


@app.post("/api/apps/{app_id}/budget")
async def set_app_budget(app_id: int, request: Request):
    """Set budget for an app."""
    body = await request.json()
    monthly = body.get("budget_monthly")
    daily = body.get("budget_daily")

    success = await budget_mgr.set_budget(app_id, monthly=monthly, daily=daily)
    if not success:
        raise HTTPException(404, "App not found or no changes made")

    _broadcast_sse("budget_updated", {"app_id": app_id})
    return {"status": "ok", "app_id": app_id}


@app.delete("/api/apps/{app_id}")
async def delete_app(app_id: int):
    """Delete an app."""
    success = await db.delete_app(app_id)
    if not success:
        raise HTTPException(404, "App not found")
    _broadcast_sse("app_deleted", {"app_id": app_id})
    return {"status": "ok"}


# ── Proxy ────────────────────────────────────────────────────

@app.post("/api/proxy/chat")
async def proxy_chat(request: Request):
    """Proxy a chat completion request through CostControl.

    Body: {
        "app_key": "cc_...",
        "model": "claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 2000,
        "temperature": 0.7
    }
    """
    body = await request.json()
    app_key = body.get("app_key")
    model = body.get("model")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 2000)
    temperature = body.get("temperature", 0.7)

    if not app_key:
        raise HTTPException(400, "app_key is required")
    if not model:
        raise HTTPException(400, "model is required")
    if not messages:
        raise HTTPException(400, "messages is required")

    try:
        result = await engine.proxy_chat(
            app_key=app_key, model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.error(f"Proxy error: {e}")
        raise HTTPException(502, f"LLM provider error: {e}")

    _broadcast_sse("request_completed", {
        "model": result["model"],
        "cost_usd": result["cost_usd"],
        "downgraded": result["downgraded"],
    })

    return result


@app.post("/api/proxy/estimate")
async def estimate_cost(request: Request):
    """Estimate cost for a request without executing it."""
    body = await request.json()
    model = body.get("model", "claude-sonnet-4-6")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 2000)

    result = await engine.estimate_cost(model, messages, max_tokens)
    return result


# ── Analytics ────────────────────────────────────────────────

@app.get("/api/analytics/daily")
async def daily_report():
    """Get daily spending report."""
    report = await analytics.daily_report()
    return report


@app.get("/api/analytics/weekly")
async def weekly_report():
    """Get weekly spending report."""
    report = await analytics.weekly_report()
    return report


@app.get("/api/analytics/monthly")
async def monthly_report():
    """Get monthly spending report."""
    report = await analytics.monthly_report()
    return report


@app.get("/api/analytics/trend")
async def spend_trend():
    """Get spend trend for charting (last 30 days)."""
    trend = await analytics.spend_trend(days=30)
    return {"trend": trend}


@app.get("/api/analytics/models")
async def model_breakdown():
    """Get spend breakdown by model."""
    breakdown = await db.get_model_breakdown(days=30)
    return {"models": breakdown}


# ── Requests ─────────────────────────────────────────────────

@app.get("/api/requests/recent")
async def recent_requests(limit: int = 50, app_id: int | None = None):
    """Get recent requests."""
    requests_data = await db.get_recent_requests(limit=limit, app_id=app_id)
    return {"requests": requests_data}


# ── Alerts ───────────────────────────────────────────────────

@app.get("/api/alerts")
async def get_alerts(app_id: int | None = None):
    """Get unacknowledged alerts."""
    alerts_data = await db.get_unacknowledged_alerts(app_id=app_id)
    return {"alerts": alerts_data}


@app.post("/api/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: int):
    """Acknowledge an alert."""
    success = await db.acknowledge_alert(alert_id)
    if not success:
        raise HTTPException(404, "Alert not found")
    _broadcast_sse("alert_acknowledged", {"alert_id": alert_id})
    return {"status": "ok"}


# ── Budget Status ────────────────────────────────────────────

@app.get("/api/budgets")
async def get_budgets():
    """Get budget status for all apps."""
    statuses = await budget_mgr.get_all_budget_statuses()
    return {"budgets": statuses}


# ── Pricing ──────────────────────────────────────────────────

@app.get("/api/pricing")
async def get_pricing():
    """Get all known model pricing."""
    return {"models": list_models()}
