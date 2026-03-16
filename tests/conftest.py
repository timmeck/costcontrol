"""Shared test fixtures for CostControl."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.db.database import Database


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def tmp_db_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d) / "test.db"


@pytest.fixture
async def db(tmp_db_path):
    """Provide a connected database instance for tests."""
    database = Database(tmp_db_path)
    await database.connect()
    yield database
    await database.close()


@pytest.fixture
async def sample_app(db):
    """Create a sample app for tests."""
    app = await db.create_app(
        name="test-app",
        api_key="cc_test123456",
        budget_monthly=100.0,
        budget_daily=10.0,
        auto_downgrade=True,
        fallback_model="qwen3:14b",
    )
    return app


@pytest.fixture
async def sample_app_no_budget(db):
    """Create a sample app with no budget limits."""
    app = await db.create_app(
        name="unlimited-app",
        api_key="cc_unlimited789",
        budget_monthly=0,
        budget_daily=0,
        auto_downgrade=False,
    )
    return app
