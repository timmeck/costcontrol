"""LLM abstraction -- Anthropic Claude + Ollama."""

import httpx

from src.config import ANTHROPIC_API_KEY, DEFAULT_MODEL, LLM_PROVIDER, OLLAMA_URL
from src.utils.logger import get_logger

log = get_logger("llm")


class LLM:
    """LLM client with retry logic and health tracking."""

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.model = DEFAULT_MODEL
        self._failures = 0
        self._max_failures = 5

    @property
    def is_healthy(self) -> bool:
        return self._failures < self._max_failures

    async def query(
        self, prompt: str, system: str = "You are a helpful assistant.", max_tokens: int = 2000
    ) -> str | None:
        """Query the LLM with retry logic."""
        for attempt in range(3):
            try:
                if self.provider == "anthropic":
                    return await self._anthropic(prompt, system, max_tokens)
                else:
                    return await self._ollama(prompt, system, max_tokens)
            except Exception as e:
                self._failures += 1
                log.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    import asyncio

                    await asyncio.sleep(1.0 * (2**attempt))
        return None

    async def _anthropic(self, prompt: str, system: str, max_tokens: int) -> str:
        """Call Anthropic Claude."""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        resp = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        self._failures = 0
        return resp.content[0].text if resp.content else None

    async def _ollama(self, prompt: str, system: str, max_tokens: int) -> str:
        """Call Ollama local model."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "options": {"num_predict": max_tokens},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            self._failures = 0
            return resp.json().get("message", {}).get("content")
