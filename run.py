"""CostControl CLI — manage apps, budgets, and run the dashboard."""

import asyncio
import secrets

import click

from src.config import COSTCONTROL_PORT, DB_PATH
from src.db.database import Database
from src.proxy.analytics import Analytics
from src.proxy.budgets import BudgetManager


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


@click.group()
def cli():
    """CostControl — AI Cost Controller. Track, budget, and optimize LLM spending."""
    pass


@cli.command()
def status():
    """Show system status."""

    async def _status():
        db = Database(DB_PATH)
        await db.connect()
        try:
            stats = await db.get_stats()
            click.echo("\n=== CostControl Status ===\n")
            click.echo(f"  Database:        {DB_PATH}")
            click.echo(f"  Active Apps:     {stats['active_apps']}")
            click.echo(f"  Spend Today:     ${stats['spend_today']:.4f}")
            click.echo(f"  Spend Month:     ${stats['spend_month']:.4f}")
            click.echo(f"  Requests Today:  {stats['requests_today']}")
            click.echo(f"  Total Requests:  {stats['total_requests']}")
            click.echo(f"  Total Spend:     ${stats['total_spend']:.4f}")
            click.echo(f"  Active Alerts:   {stats['active_alerts']}")
            click.echo()
        finally:
            await db.close()

    _run(_status())


@cli.command()
def apps():
    """List registered apps."""

    async def _apps():
        db = Database(DB_PATH)
        await db.connect()
        try:
            app_list = await db.get_app_breakdown()
            if not app_list:
                click.echo("\nNo apps registered. Use 'python run.py register <name>' to add one.\n")
                return
            click.echo(
                f"\n{'Name':<20} {'Monthly Budget':<16} {'Spend (Month)':<16} {'Spend (Today)':<16} {'Reqs Today':<12} {'Status'}"
            )
            click.echo("-" * 96)
            for a in app_list:
                budget = f"${a['budget_monthly']:.2f}" if a["budget_monthly"] > 0 else "none"
                click.echo(
                    f"{a['name']:<20} {budget:<16} "
                    f"${a['spend_month']:.4f}{'':<10} "
                    f"${a['spend_today']:.4f}{'':<10} "
                    f"{a['requests_today']:<12} {a['status']}"
                )
            click.echo()
        finally:
            await db.close()

    _run(_apps())


@cli.command()
@click.argument("name")
@click.option("--monthly", default=0.0, help="Monthly budget in USD")
@click.option("--daily", default=0.0, help="Daily budget in USD")
def register(name, monthly, daily):
    """Register a new app."""

    async def _register():
        db = Database(DB_PATH)
        await db.connect()
        try:
            existing = await db.get_app_by_name(name)
            if existing:
                click.echo(f"\nApp '{name}' already exists!\n")
                return
            api_key = f"cc_{secrets.token_hex(24)}"
            app_data = await db.create_app(
                name=name,
                api_key=api_key,
                budget_monthly=monthly,
                budget_daily=daily,
            )
            click.echo("\n=== App Registered ===")
            click.echo(f"  Name:     {app_data['name']}")
            click.echo(f"  API Key:  {app_data['api_key']}")
            click.echo(f"  Monthly:  ${monthly:.2f}")
            click.echo(f"  Daily:    ${daily:.2f}")
            click.echo("\nUse this API key in your app's requests to the proxy.\n")
        finally:
            await db.close()

    _run(_register())


@cli.command()
@click.argument("app_name")
@click.option("--monthly", default=None, type=float, help="Monthly budget in USD")
@click.option("--daily", default=None, type=float, help="Daily budget in USD")
def budget(app_name, monthly, daily):
    """Set budget for an app."""

    async def _budget():
        db = Database(DB_PATH)
        await db.connect()
        try:
            app = await db.get_app_by_name(app_name)
            if not app:
                click.echo(f"\nApp '{app_name}' not found.\n")
                return
            mgr = BudgetManager(db)
            success = await mgr.set_budget(app["id"], monthly=monthly, daily=daily)
            if success:
                click.echo(f"\nBudget updated for '{app_name}':")
                if monthly is not None:
                    click.echo(f"  Monthly: ${monthly:.2f}")
                if daily is not None:
                    click.echo(f"  Daily:   ${daily:.2f}")
                click.echo()
            else:
                click.echo("\nNo changes made.\n")
        finally:
            await db.close()

    _run(_budget())


@cli.command()
@click.option("--type", "report_type", default="daily", type=click.Choice(["daily", "weekly", "monthly"]))
def report(report_type):
    """Show spending report."""

    async def _report():
        db = Database(DB_PATH)
        await db.connect()
        try:
            a = Analytics(db)
            text = await a.format_text_report(report_type)
            click.echo(f"\n{text}")
        finally:
            await db.close()

    _run(_report())


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=COSTCONTROL_PORT, help="Port number")
def serve(host, port):
    """Start the CostControl dashboard server."""
    import uvicorn

    click.echo(f"\nStarting CostControl on http://{host}:{port}\n")
    uvicorn.run("src.web.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    cli()
