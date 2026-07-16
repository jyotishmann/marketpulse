# marketpulse/api/middleware.py
# HTTP middleware: CORS, per-request unique IDs, response timing.
# Middleware is applied LIFO — the last added runs first on requests.

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ── Middleware classes ─────────────────────────────────────────────────────────


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assign a unique short ID to every request.

    Adds X-Request-ID header to the response.
    Used to correlate log lines from the same request.

    Example header: X-Request-ID: a3f7bc12
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = str(uuid.uuid4())[:8]
        # Store on the request state so route handlers can access it if needed
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measure and log the wall-clock time for every request.

    Adds X-Process-Time header (seconds, 4 decimal places).
    Logs: METHOD /path — 0.0412s [200]

    Uses time.perf_counter() for high-resolution, monotonic measurement.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start

        response.headers["X-Process-Time"] = f"{duration:.4f}"

        # Log at DEBUG for health checks (noisy), INFO for all other routes
        log_fn = logger.debug if "/health" in request.url.path else logger.info
        log_fn(
            "%s %s — %.4fs [%d]",
            request.method,
            request.url.path,
            duration,
            response.status_code,
        )
        return response


# ── Setup function ─────────────────────────────────────────────────────────────


def setup_middleware(app: FastAPI) -> None:
    """
    Register all middleware on the FastAPI application.

    Middleware is applied in LIFO order (last added = runs first on requests).
    Order here: TimingMiddleware (innermost) → RequestID → CORS (outermost).

    CORS must be outermost so browser preflight OPTIONS requests are handled
    before RequestID or Timing middleware touch the request.

    Args:
        app: The FastAPI application instance (mutated in place).
    """
    # Innermost: timing (wraps only the route handler)
    app.add_middleware(TimingMiddleware)

    # Middle: request ID (available to all inner middleware and route)
    app.add_middleware(RequestIDMiddleware)

    # Outermost: CORS (intercepts preflight before anything else)
    # allow_origins=["*"]: permissive for development.
    # Production: replace with specific origins e.g. ["https://your-dashboard.com"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # allows any origin (Streamlit on port 8501)
        allow_credentials=True,
        allow_methods=["GET", "OPTIONS"],  # API is read-only
        allow_headers=["*"],
    )

    logger.debug(
        "Middleware registered: CORS (outermost) → RequestID → Timing (innermost)"
    )
