"""Auth middleware -- API key protection for CostControl.

SECURITY: ALL endpoints require auth when COSTCONTROL_API_KEY is set.
No GET bypass. No query string keys.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import COSTCONTROL_API_KEY
from src.utils.logger import get_logger

log = get_logger("auth")

# Public paths (no auth required even when key is set)
PUBLIC_PATHS = {
    "/",
    "/health",
    "/api/status",
    "/nexus/handle",
}

# Public prefixes
PUBLIC_PREFIXES = ("/static",)


class AuthMiddleware(BaseHTTPMiddleware):
    """API key middleware. Active when COSTCONTROL_API_KEY is set.

    ALL methods (GET, POST, etc.) require auth except public paths.
    API key must be passed via X-API-Key header (not query string).
    """

    async def dispatch(self, request: Request, call_next):
        if not COSTCONTROL_API_KEY:
            return await call_next(request)

        path = request.url.path

        # Allow public paths
        if path in PUBLIC_PATHS:
            return await call_next(request)

        # Allow public prefixes
        if any(path.startswith(p) for p in PUBLIC_PREFIXES):
            return await call_next(request)

        # Require auth for ALL methods (including GET)
        key = request.headers.get("X-API-Key", "")
        if key != COSTCONTROL_API_KEY:
            log.warning(
                "Unauthorized %s %s from %s", request.method, path, request.client.host if request.client else "unknown"
            )
            return JSONResponse(
                {"error": "Unauthorized. Pass X-API-Key header."},
                status_code=401,
            )

        return await call_next(request)
