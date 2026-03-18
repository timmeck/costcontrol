"""Auth middleware for CostControl API."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.config import COSTCONTROL_API_KEY


class AuthMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication middleware."""

    async def dispatch(self, request, call_next):
        # Skip auth if no key configured
        if not COSTCONTROL_API_KEY:
            return await call_next(request)

        # Allow public GET endpoints
        if request.method == "GET" and request.url.path in (
            "/",
            "/api/status",
            "/api/events/stream",
        ):
            return await call_next(request)

        # Check API key for mutation endpoints
        key = request.headers.get("X-API-Key") or request.query_params.get("key")
        if key != COSTCONTROL_API_KEY:
            if request.method in ("POST", "PUT", "DELETE"):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)
