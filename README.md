# CostControl

Self-hosted AI Cost Controller. Tracks, budgets, and optimizes LLM API spending by acting as a proxy between your apps and LLM APIs.

## Features

- **LLM Proxy** — Sits between your app and LLM APIs (Anthropic, OpenAI, Ollama). Just change your base URL.
- **Token Counting** — Counts input/output tokens per request automatically.
- **Cost Tracking** — Per app, per model, per day cost tracking with full history.
- **Budget Management** — Set monthly/daily budgets per app with auto-downgrade to free local models.
- **Smart Alerts** — Notifications at 80%, 90%, and 100% budget thresholds.
- **Analytics Dashboard** — Gorgeous real-time dashboard with spend trends, model breakdowns, and budget bars.
- **Reports** — Daily, weekly, monthly spending reports via CLI or API.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Register an app
python run.py register my-app --monthly 50 --daily 5

# Start the dashboard
python run.py serve
```

Dashboard runs at **http://localhost:8600**

## Using the Proxy

Point your app to CostControl instead of the LLM API directly:

```python
import httpx

resp = httpx.post("http://localhost:8600/api/proxy/chat", json={
    "app_key": "cc_your_api_key_here",
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 2000
})
print(resp.json())
```

When your app hits its budget limit, CostControl automatically downgrades to a free local model (Ollama).

## CLI Commands

```bash
python run.py status              # System status
python run.py apps                # List registered apps
python run.py register <name>     # Register new app
python run.py budget <name> --monthly 50 --daily 5   # Set budget
python run.py report --type daily  # Spending report
python run.py serve               # Start dashboard (port 8600)
```

## Docker

```bash
docker-compose up -d
```

## Supported Models

| Provider | Models | Pricing |
|----------|--------|---------|
| Anthropic | claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5 | $0.80 - $75/M tokens |
| OpenAI | gpt-4o, gpt-4.1, o3, o4-mini | $0.10 - $10/M tokens |
| Ollama | qwen3:14b, llama3.1:8b, mistral:7b | Free (local) |

## Configuration

Copy `.env.example` to `.env` and configure:

```
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
OLLAMA_URL=http://localhost:11434
COSTCONTROL_PORT=8600
COSTCONTROL_API_KEY=optional_admin_key
```

## License

MIT
