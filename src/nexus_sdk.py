"""Nexus SDK (Standalone) — Drop-in integration for any FastAPI agent.

Copy this single file into your project. No nexus package dependency required.

Usage:
    from nexus_sdk import NexusAdapter

    adapter = NexusAdapter(
        app=app,
        agent_name="cortex",
        nexus_url="http://localhost:9500",
        endpoint="http://localhost:8100",
        capabilities=[
            {"name": "text_generation", "description": "Generates text"},
        ],
    )

    @adapter.handle("text_generation")
    async def handle_text_gen(query: str, params: dict) -> dict:
        result = await your_function(query)
        return {"result": result, "confidence": 0.9}
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac as hmac_mod
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

log = logging.getLogger("nexus.sdk")

# ── HMAC Auth (inline, no nexus dependency) ───────────────────────


_REPLAY_CACHE_MAX = 10000
_replay_cache: OrderedDict[str, float] = OrderedDict()


def _verify_signature(
    payload: str,
    api_key: str,
    timestamp: str,
    signature: str,
    max_age_seconds: int = 300,
) -> bool:
    """Verify HMAC-SHA256 signature with replay protection."""
    try:
        ts = int(timestamp)
    except (ValueError, TypeError):
        return False

    if abs(int(time.time()) - ts) > max_age_seconds:
        return False

    message = f"{ts}.{payload}".encode()
    expected = hmac_mod.new(api_key.encode(), message, hashlib.sha256).hexdigest()
    if not hmac_mod.compare_digest(signature, expected):
        return False

    if signature in _replay_cache:
        return False

    _replay_cache[signature] = time.time()

    # Evict old entries
    now = time.time()
    while _replay_cache:
        oldest_sig, oldest_time = next(iter(_replay_cache.items()))
        if now - oldest_time > max_age_seconds:
            _replay_cache.pop(oldest_sig)
        else:
            break
    while len(_replay_cache) > _REPLAY_CACHE_MAX:
        _replay_cache.popitem(last=False)

    return True


# ── Request/Response Models ───────────────────────────────────────


class NexusSDKRequest(BaseModel):
    request_id: str = ""
    from_agent: str = ""
    to_agent: str | None = None
    query: str = ""
    capability: str | None = None
    constraints: dict = Field(default_factory=dict)
    budget: float | None = None
    deadline_ms: int | None = None
    verification: str = "none"
    language: str = "en"
    context: dict = Field(default_factory=dict)
    created_at: str = ""


class NexusSDKResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    request_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    status: str = "completed"
    answer: str = ""
    confidence: float = 0.0
    sources: list[str] = Field(default_factory=list)
    cost: float = 0.0
    processing_ms: int = 0
    error: str | None = None
    meta: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# ── Handler Type ──────────────────────────────────────────────────

HandlerFunc = Callable[[str, dict], Coroutine[Any, Any, dict]]


# ── NexusAdapter ──────────────────────────────────────────────────


class NexusAdapter:
    """Drop-in Nexus integration for any FastAPI application.

    Adds /nexus/handle endpoint, auto-registers with Nexus, sends heartbeats,
    and verifies HMAC signatures.
    """

    def __init__(
        self,
        app: FastAPI,
        agent_name: str,
        nexus_url: str = "http://localhost:9500",
        endpoint: str | None = None,
        capabilities: list[dict] | None = None,
        tags: list[str] | None = None,
        description: str = "",
        api_key: str | None = None,
        heartbeat_interval: int = 30,
    ):
        self.app = app
        self.agent_name = agent_name
        self.nexus_url = nexus_url.rstrip("/")
        self.endpoint = endpoint or "http://localhost:8000"
        self.capabilities = capabilities or []
        self.tags = tags or []
        self.description = description
        self.api_key = api_key
        self.heartbeat_interval = heartbeat_interval
        self._handlers: dict[str, HandlerFunc] = {}
        self._agent_id: str | None = None
        self._heartbeat_task: asyncio.Task | None = None

        self._register_route()
        self._wrap_lifespan()

    def handle(self, capability: str):
        """Decorator to register a capability handler."""

        def decorator(func: HandlerFunc):
            self._handlers[capability] = func
            return func

        return decorator

    def _register_route(self):
        adapter = self

        @self.app.post("/nexus/handle")
        async def nexus_handle(request: Request):
            body = await request.body()
            body_str = body.decode()

            if adapter.api_key:
                ts = request.headers.get("X-Nexus-Timestamp", "")
                sig = request.headers.get("X-Nexus-Signature", "")
                if not _verify_signature(body_str, adapter.api_key, ts, sig):
                    return NexusSDKResponse(
                        from_agent=adapter.agent_name,
                        status="rejected",
                        error="Invalid HMAC signature",
                    ).model_dump()

            req_data = json.loads(body_str)
            req = NexusSDKRequest(**req_data)

            start = time.perf_counter_ns()
            handler = adapter._handlers.get(req.capability)

            if handler is None:
                return NexusSDKResponse(
                    request_id=req.request_id,
                    from_agent=adapter.agent_name,
                    to_agent=req.from_agent,
                    status="failed",
                    error=f"Unsupported capability: {req.capability}",
                ).model_dump()

            try:
                params = {**req.constraints, **req.context}
                result = await handler(req.query, params)
                elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000

                return NexusSDKResponse(
                    request_id=req.request_id,
                    from_agent=adapter.agent_name,
                    to_agent=req.from_agent,
                    status="completed",
                    answer=result.get("result", str(result)),
                    confidence=result.get("confidence", 0.8),
                    sources=result.get("sources", []),
                    cost=result.get("cost", 0.0),
                    processing_ms=elapsed_ms,
                    meta=result.get("meta", {}),
                ).model_dump()

            except Exception as e:
                elapsed_ms = (time.perf_counter_ns() - start) // 1_000_000
                log.exception("Handler error for capability %s", req.capability)
                return NexusSDKResponse(
                    request_id=req.request_id,
                    from_agent=adapter.agent_name,
                    to_agent=req.from_agent,
                    status="failed",
                    processing_ms=elapsed_ms,
                    error=str(e),
                ).model_dump()

    def _wrap_lifespan(self):
        original_lifespan = self.app.router.lifespan_context
        adapter = self

        @asynccontextmanager
        async def wrapped_lifespan(app):
            if original_lifespan:
                ctx = original_lifespan(app)
                await ctx.__aenter__()

            await adapter._register_with_nexus()
            adapter._heartbeat_task = asyncio.create_task(adapter._heartbeat_loop())

            yield

            if adapter._heartbeat_task:
                adapter._heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await adapter._heartbeat_task

            if original_lifespan:
                await ctx.__aexit__(None, None, None)

        self.app.router.lifespan_context = wrapped_lifespan

    async def _register_with_nexus(self):
        payload = {
            "name": self.agent_name,
            "description": self.description,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities,
            "tags": self.tags,
        }
        url = f"{self.nexus_url}/api/registry/agents"
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                self._agent_id = data.get("id")
                self.api_key = data.get("api_key", self.api_key)
                log.info("Registered %s with Nexus (id=%s)", self.agent_name, self._agent_id)
            except httpx.HTTPError as exc:
                log.warning("Nexus registration failed (will retry via heartbeat): %s", exc)

    async def _heartbeat_loop(self):
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            if not self._agent_id:
                await self._register_with_nexus()
                continue
            url = f"{self.nexus_url}/api/registry/agents/{self._agent_id}/heartbeat"
            async with httpx.AsyncClient(timeout=5) as client:
                try:
                    resp = await client.post(url)
                    resp.raise_for_status()
                except httpx.HTTPError:
                    pass
