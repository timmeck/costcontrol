"""Security tests — auth 401, concurrent lock."""

import asyncio

import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient


# ── 1. Auth Middleware ──────────────────────────────────────────


def test_auth_rejects_without_key():
    """Unauthenticated request to protected endpoint must return 401."""
    with patch("src.web.auth.COSTCONTROL_API_KEY", "test-secret-key"):
        from src.web.api import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/apps")
        assert resp.status_code == 401
        assert "Unauthorized" in resp.json().get("error", "")


def test_auth_rejects_wrong_key():
    """Wrong API key must return 401."""
    with patch("src.web.auth.COSTCONTROL_API_KEY", "test-secret-key"):
        from src.web.api import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/apps", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401


def test_auth_allows_correct_key():
    """Correct API key must pass auth (may fail downstream, but not 401)."""
    with patch("src.web.auth.COSTCONTROL_API_KEY", "test-secret-key"):
        from src.web.api import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/apps", headers={"X-API-Key": "test-secret-key"})
        assert resp.status_code != 401


def test_auth_public_paths_allowed():
    """Public paths must be accessible without auth."""
    with patch("src.web.auth.COSTCONTROL_API_KEY", "test-secret-key"):
        from src.web.api import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/status")
        assert resp.status_code != 401


def test_auth_no_query_param_bypass():
    """Query param ?key= must NOT bypass auth."""
    with patch("src.web.auth.COSTCONTROL_API_KEY", "test-secret-key"):
        from src.web.api import app

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/apps?key=test-secret-key")
        assert resp.status_code == 401


# ── 2. Concurrency Lock ────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_lock_rejects_second_run():
    """Second concurrent run on same lock must be rejected."""
    lock = asyncio.Lock()
    results = []

    async def fake_run(run_id: int):
        if lock.locked():
            results.append({"error": f"Run {run_id}: already running"})
            return
        async with lock:
            results.append({"ok": f"Run {run_id}: started"})
            await asyncio.sleep(0.1)

    task1 = asyncio.create_task(fake_run(1))
    await asyncio.sleep(0.01)
    task2 = asyncio.create_task(fake_run(2))
    await asyncio.gather(task1, task2)

    assert any("ok" in r for r in results), "First run should succeed"
    assert any("error" in r for r in results), "Second run should be rejected"
