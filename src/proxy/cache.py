"""Response cache — exact-match caching for LLM responses to avoid duplicate calls."""

import hashlib
import json
import time

from src.utils.logger import get_logger

log = get_logger("cache")


class ResponseCache:
    """In-memory exact-match cache for LLM responses with TTL expiry."""

    def __init__(self, default_ttl: int = 3600):
        """Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds (default 1 hour).
        """
        self.default_ttl = default_ttl
        self._cache: dict[str, dict] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(model: str, messages: list[dict]) -> str:
        """Create a cache key from model + messages."""
        payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model: str, messages: list[dict]) -> dict | None:
        """Look up a cached response. Returns None on miss or expiry."""
        key = self._make_key(model, messages)
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None

        if time.time() > entry["expires_at"]:
            # Expired — remove and miss
            del self._cache[key]
            self._misses += 1
            return None

        self._hits += 1
        log.info(f"Cache hit for model={model} (key={key[:12]}...)")
        return entry["response"]

    def put(self, model: str, messages: list[dict], response: dict, ttl: int | None = None):
        """Store a response in the cache."""
        key = self._make_key(model, messages)
        self._cache[key] = {
            "response": response,
            "expires_at": time.time() + (ttl if ttl is not None else self.default_ttl),
            "created_at": time.time(),
        }
        log.info(f"Cached response for model={model} (key={key[:12]}...)")

    def invalidate(self, model: str, messages: list[dict]):
        """Remove a specific entry from cache."""
        key = self._make_key(model, messages)
        self._cache.pop(key, None)

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        log.info("Cache cleared")

    def cleanup_expired(self):
        """Remove all expired entries."""
        now = time.time()
        expired_keys = [k for k, v in self._cache.items() if now > v["expires_at"]]
        for k in expired_keys:
            del self._cache[k]
        if expired_keys:
            log.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def stats(self) -> dict:
        """Return cache statistics."""
        now = time.time()
        active = sum(1 for v in self._cache.values() if now <= v["expires_at"])
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        return {
            "cache_size": len(self._cache),
            "active_entries": active,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
        }
