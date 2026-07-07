# marketpulse/db/session.py
# Database engine, connection pool, and session factory.
# get_db() is used as a FastAPI dependency to provide a DB session per request.

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from marketpulse.config import settings

# ── Engine: the connection pool to PostgreSQL ──────────────────────────────────
engine = create_engine(
    settings.database_url,
    # Test connections before use — silently replaces dropped connections
    pool_pre_ping=True,
    # Keep 5 connections open at all times
    pool_size=5,
    # Allow up to 10 extra connections under load (total max = 15)
    max_overflow=10,
    # Log all SQL statements when LOG_LEVEL=DEBUG (useful for debugging queries)
    echo=(settings.log_level == "DEBUG"),
)

# ── SessionLocal: factory that creates Session objects ────────────────────────
# A Session is a single "unit of work" — it tracks all pending changes
# and sends them to the database as one atomic transaction on commit().
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,   # we commit manually — explicit is safer
    autoflush=False,    # we flush manually — avoids surprise SQL mid-transaction
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session per HTTP request.

    Usage in a route:
        from sqlalchemy.orm import Session
        from fastapi import Depends
        from marketpulse.db.session import get_db

        @router.get("/prices")
        def get_prices(db: Session = Depends(get_db)):
            return db.query(StockPrice).all()

    The session is always closed after the response — even on exception.
    SQLAlchemy rolls back any uncommitted changes when the session closes.
    """
    db = SessionLocal()
    try:
        yield db          # FastAPI injects this session into the route function
    finally:
        db.close()        # always runs, even if the route raised an exception


def verify_connection() -> bool:
    """
    Lightweight database health check. Returns True if PostgreSQL is reachable.

    Used by GET /api/v1/health and Docker Compose healthcheck.
    """
    try:
        # engine.connect() borrows one connection from the pool
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
