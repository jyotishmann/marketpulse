# marketpulse/db/__init__.py
# Public API of the db package.
# Import from here rather than from the internal submodules.

from marketpulse.db.models import (
    Base,
    MLSignal,
    ModelRegistry,
    NewsArticle,
    StockPrice,
    TechnicalIndicator,
)
from marketpulse.db.session import SessionLocal, engine, get_db, verify_connection

__all__ = [
    # Models — SQLAlchemy ORM classes
    "Base",
    "StockPrice",
    "TechnicalIndicator",
    "NewsArticle",
    "MLSignal",
    "ModelRegistry",
    # Session utilities
    "SessionLocal",
    "engine",
    "get_db",
    "verify_connection",
]
