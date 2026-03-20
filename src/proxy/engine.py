"""Core proxy engine — intercepts LLM calls, tracks costs, enforces budgets."""

import re
import time

import httpx

from src.config import ANTHROPIC_API_KEY, OLLAMA_URL, OPENAI_API_KEY
from src.db.database import Database
from src.proxy.budgets import BudgetManager
from src.proxy.cache import ResponseCache
from src.proxy.pricing import calculate_cost, get_provider, get_pricing, is_free_model
from src.utils.logger import get_logger

log = get_logger("engine")


# ── Prompt Compression / Token Optimization ─────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.3 tokens per word."""
    return max(1, int(len(text.split()) * 1.3))


def optimize_prompt(messages: list[dict], max_reduction: float = 0.3) -> tuple[list[dict], dict]:
    """Optimize messages to reduce token usage.

    - Strips excessive whitespace and normalizes newlines
    - If messages array is long (>10), summarizes older messages into condensed context

    Args:
        messages: List of chat messages.
        max_reduction: Maximum fraction of tokens to remove (0.0-1.0).

    Returns:
        (optimized_messages, optimization_meta) where meta contains tokens_before,
        tokens_after, tokens_saved, and messages_condensed count.
    """
    if not messages:
        return messages, {"tokens_before": 0, "tokens_after": 0, "tokens_saved": 0, "messages_condensed": 0}

    tokens_before = sum(_estimate_tokens(m.get("content", "")) for m in messages)

    # Step 1: Normalize whitespace in all messages
    cleaned: list[dict] = []
    for m in messages:
        content = m.get("content", "")
        # Collapse multiple spaces/tabs to single space
        content = re.sub(r"[ \t]+", " ", content)
        # Normalize multiple newlines to max two
        content = re.sub(r"\n{3,}", "\n\n", content)
        # Strip leading/trailing whitespace per line
        content = "\n".join(line.strip() for line in content.split("\n"))
        content = content.strip()
        cleaned.append({**m, "content": content})

    # Step 2: If >10 messages, condense older ones into a summary
    messages_condensed = 0
    if len(cleaned) > 10:
        # Keep the system message (if first), last 6 messages, condense the middle
        has_system = cleaned[0].get("role") == "system"
        prefix = cleaned[:1] if has_system else []
        middle = cleaned[1:-6] if has_system else cleaned[:-6]
        suffix = cleaned[-6:]

        if middle:
            # Condense middle messages into a brief summary
            summary_parts = []
            for m in middle:
                role = m.get("role", "user")
                content = m.get("content", "")
                # Truncate each message to first 80 chars for the summary
                snippet = content[:80].replace("\n", " ")
                if len(content) > 80:
                    snippet += "..."
                summary_parts.append(f"[{role}]: {snippet}")
            messages_condensed = len(middle)
            condensed_msg = {
                "role": "system",
                "content": f"[Condensed {len(middle)} earlier messages]\n" + "\n".join(summary_parts),
            }
            cleaned = prefix + [condensed_msg] + suffix

    tokens_after = sum(_estimate_tokens(m.get("content", "")) for m in cleaned)

    # Enforce max_reduction cap: if we removed too many tokens, back off
    if tokens_before > 0:
        reduction_pct = (tokens_before - tokens_after) / tokens_before
        if reduction_pct > max_reduction:
            # Too aggressive — return whitespace-cleaned only (no condensing)
            cleaned_only: list[dict] = []
            for m in messages:
                content = m.get("content", "")
                content = re.sub(r"[ \t]+", " ", content)
                content = re.sub(r"\n{3,}", "\n\n", content)
                content = "\n".join(line.strip() for line in content.split("\n"))
                content = content.strip()
                cleaned_only.append({**m, "content": content})
            tokens_after = sum(_estimate_tokens(m.get("content", "")) for m in cleaned_only)
            cleaned = cleaned_only
            messages_condensed = 0

    tokens_saved = max(0, tokens_before - tokens_after)
    return cleaned, {
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "tokens_saved": tokens_saved,
        "messages_condensed": messages_condensed,
    }


# ── Model tiers for smart routing ───────────────────────────

MODEL_TIERS: dict[str, list[str]] = {
    "anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    "openai": ["gpt-4o", "gpt-4.1", "o3", "gpt-4o-mini", "gpt-4.1-mini", "o4-mini", "gpt-4.1-nano"],
}


def score_complexity(prompt: str) -> float:
    """Score prompt complexity 0-1 for routing decisions."""
    score = 0.0
    # Length factor
    if len(prompt) > 2000:
        score += 0.2
    elif len(prompt) > 500:
        score += 0.1
    # Code detection
    if any(kw in prompt.lower() for kw in ["```", "def ", "function ", "class ", "import "]):
        score += 0.3
    # Reasoning keywords
    if any(kw in prompt.lower() for kw in ["analyze", "compare", "explain why", "step by step", "reasoning"]):
        score += 0.2
    # Multi-step
    if any(kw in prompt.lower() for kw in ["first", "then", "finally", "steps", "plan"]):
        score += 0.1
    # Math/logic
    if any(kw in prompt.lower() for kw in ["calculate", "equation", "proof", "algorithm"]):
        score += 0.2
    return min(score, 1.0)


def pick_cheaper_model(model: str, complexity: float) -> str | None:
    """Pick a cheaper model from the same provider based on complexity.

    Returns None if no downgrade is appropriate.
    Only suggests a cheaper model when complexity is low (<0.3).
    """
    if complexity >= 0.3:
        return None

    provider = get_provider(model)
    tier_list = MODEL_TIERS.get(provider)
    if not tier_list or model not in tier_list:
        return None

    current_idx = tier_list.index(model)
    # Already the cheapest in tier
    if current_idx >= len(tier_list) - 1:
        return None

    # Pick a model 1-2 tiers cheaper based on complexity
    steps = 2 if complexity < 0.1 else 1
    cheaper_idx = min(current_idx + steps, len(tier_list) - 1)
    cheaper_model = tier_list[cheaper_idx]

    # Only suggest if actually cheaper
    current_price = get_pricing(model)
    cheaper_price = get_pricing(cheaper_model)
    if cheaper_price["input"] < current_price["input"]:
        return cheaper_model
    return None


class ProxyEngine:
    """LLM proxy engine that intercepts, tracks, and budgets all LLM calls."""

    def __init__(self, db: Database):
        self.db = db
        self.budget_mgr = BudgetManager(db)
        self.cache = ResponseCache(default_ttl=3600)

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
        smart_routed = False
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
        elif not is_free_model(model):
            # 2b. Smart routing — use complexity to pick cheaper model when budget is getting tight
            budget_pct = max(budget_status.get("daily_pct", 0), budget_status.get("monthly_pct", 0))
            if budget_pct >= 50:  # Only smart-route when budget is >=50% consumed
                prompt_text = " ".join(m.get("content", "") for m in messages if m.get("role") == "user")
                complexity = score_complexity(prompt_text)
                cheaper = pick_cheaper_model(model, complexity)
                if cheaper:
                    log.info(
                        f"Smart routing: complexity={complexity:.2f}, budget={budget_pct:.0f}% — "
                        f"routing {model} -> {cheaper}"
                    )
                    model = cheaper
                    smart_routed = True

        # 2c. Token optimization (optional per app — enabled by default)
        optimization_meta = None
        app_optimize = app.get("optimize_prompts", 1)
        if app_optimize:
            messages, optimization_meta = optimize_prompt(messages, max_reduction=0.3)
            if optimization_meta["tokens_saved"] > 0:
                log.info(
                    f"Prompt optimization: saved ~{optimization_meta['tokens_saved']} tokens "
                    f"({optimization_meta['messages_condensed']} messages condensed)"
                )

        # 2d. Check cache before calling LLM
        cached = self.cache.get(model, messages)
        if cached is not None:
            log.info(f"Returning cached response for app '{app['name']}'")
            return {
                **cached,
                "cached": True,
                "downgraded": downgraded,
                "smart_routed": smart_routed,
                "original_model": original_model if (downgraded or smart_routed) else None,
            }

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

        response = {
            "content": result["content"],
            "model": model,
            "provider": provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "downgraded": downgraded,
            "smart_routed": smart_routed,
            "original_model": original_model if (downgraded or smart_routed) else None,
            "latency_ms": latency_ms,
            "cached": False,
            "optimization": optimization_meta,
        }

        # 7. Cache the response
        self.cache.put(model, messages, response)

        return response

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
