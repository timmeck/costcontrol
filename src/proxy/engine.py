"""Core proxy engine — intercepts LLM calls, tracks costs, enforces budgets."""

import time

import httpx

from src.config import ANTHROPIC_API_KEY, OLLAMA_URL, OPENAI_API_KEY
from src.db.database import Database
from src.proxy.budgets import BudgetManager
from src.proxy.pricing import calculate_cost, get_provider, is_free_model
from src.utils.logger import get_logger

log = get_logger("engine")


class ProxyEngine:
    """LLM proxy engine that intercepts, tracks, and budgets all LLM calls."""

    def __init__(self, db: Database):
        self.db = db
        self.budget_mgr = BudgetManager(db)

    async def proxy_chat(
        self, app_key: str, model: str, messages: list[dict], max_tokens: int = 2000, temperature: float = 0.7
    ) -> dict:
        """Proxy a chat completion request.

        1. Look up the app by API key
        2. Check budget — downgrade if needed
        3. Forward to actual LLM provider
        4. Count tokens, calculate cost
        5. Store in DB
        6. Return response

        Returns:
            {"content": str, "model": str, "provider": str,
             "input_tokens": int, "output_tokens": int, "cost_usd": float,
             "downgraded": bool, "latency_ms": int}
        """
        # 1. Look up app
        app = await self.db.get_app_by_key(app_key)
        if not app:
            raise ValueError("Invalid API key — app not found")
        if app["status"] != "active":
            raise ValueError(f"App '{app['name']}' is not active")

        app_id = app["id"]
        downgraded = False
        original_model = model

        # 2. Check budget
        budget_status = await self.budget_mgr.check_budget(app_id)
        if budget_status["should_downgrade"] and not is_free_model(model):
            fallback = budget_status.get("fallback_model", "qwen3:14b")
            log.warning(f"App '{app['name']}' over budget — downgrading from {model} to {fallback}")
            model = fallback
            downgraded = True
            await self.db.log_activity(
                app_id,
                "auto_downgrade",
                f"Auto-downgraded from {original_model} to {model} (budget exceeded)",
            )

        # 3. Forward to provider
        provider = get_provider(model)
        start = time.monotonic()

        try:
            if provider == "anthropic":
                result = await self._call_anthropic(model, messages, max_tokens, temperature)
            elif provider == "openai":
                result = await self._call_openai(model, messages, max_tokens, temperature)
            else:
                result = await self._call_ollama(model, messages, max_tokens, temperature)
        except Exception as e:
            latency_ms = int((time.monotonic() - start) * 1000)
            await self.db.record_request(
                app_id,
                model,
                provider,
                0,
                0,
                0.0,
                latency_ms,
                status="error",
                downgraded=downgraded,
            )
            log.error(f"LLM call failed for app '{app['name']}': {e}")
            raise

        latency_ms = int((time.monotonic() - start) * 1000)

        # 4. Calculate cost
        input_tokens = result["input_tokens"]
        output_tokens = result["output_tokens"]
        cost_usd = calculate_cost(model, input_tokens, output_tokens)

        # 5. Store in DB
        await self.db.record_request(
            app_id,
            model,
            provider,
            input_tokens,
            output_tokens,
            cost_usd,
            latency_ms,
            status="success",
            downgraded=downgraded,
        )

        # 6. Check alerts after recording
        await self.budget_mgr.check_and_alert(app_id)

        return {
            "content": result["content"],
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "downgraded": downgraded,
            "original_model": original_model if downgraded else None,
            "latency_ms": latency_ms,
        }

    async def _call_anthropic(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        """Call Anthropic Claude API."""
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")

        import anthropic

        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

        # Extract system message if present
        system = "You are a helpful assistant."
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        if not chat_messages:
            chat_messages = [{"role": "user", "content": "Hello"}]

        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=chat_messages,
        )

        content = resp.content[0].text if resp.content else ""
        return {
            "content": content,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }

    async def _call_openai(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        """Call OpenAI API."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not configured")

        import openai

        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = resp.choices[0] if resp.choices else None
        content = choice.message.content if choice else ""
        input_tokens = resp.usage.prompt_tokens if resp.usage else 0
        output_tokens = resp.usage.completion_tokens if resp.usage else 0

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def _call_ollama(self, model: str, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        """Call Ollama local API."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data.get("message", {}).get("content", "")
        # Ollama provides token counts in eval_count / prompt_eval_count
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        # Estimate if not provided
        if input_tokens == 0:
            input_tokens = sum(len(m.get("content", "").split()) * 2 for m in messages)
        if output_tokens == 0:
            output_tokens = len(content.split()) * 2

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def estimate_cost(self, model: str, messages: list[dict], max_tokens: int = 2000) -> dict:
        """Estimate cost for a request without executing it."""
        # Rough token estimate: ~1.3 tokens per word
        input_text = " ".join(m.get("content", "") for m in messages)
        estimated_input = int(len(input_text.split()) * 1.3)
        estimated_output = min(max_tokens, estimated_input)  # rough guess

        cost = calculate_cost(model, estimated_input, estimated_output)
        return {
            "model": model,
            "provider": get_provider(model),
            "estimated_input_tokens": estimated_input,
            "estimated_output_tokens": estimated_output,
            "estimated_cost_usd": cost,
            "free": is_free_model(model),
        }
