# tests/unit/test_db.py
from unittest.mock import MagicMock, patch


def test_verify_connection_returns_true_on_success():
    from marketpulse.db.session import verify_connection

    with patch("marketpulse.db.session.engine") as mock_engine:
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        result = verify_connection()
    assert result is True


def test_verify_connection_returns_false_on_error():
    from marketpulse.db.session import verify_connection

    with patch("marketpulse.db.session.engine") as mock_engine:
        mock_engine.connect.side_effect = Exception("connection refused")
        result = verify_connection()
    assert result is False


def test_get_db_yields_session():
    from marketpulse.db.session import get_db

    with patch("marketpulse.db.session.SessionLocal") as mock_session_local:
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        gen = get_db()
        session = next(gen)
        assert session == mock_session
        try:
            next(gen)
        except StopIteration:
            pass
        mock_session.close.assert_called_once()


def test_session_local_creates_session():
    from marketpulse.db.session import SessionLocal

    with patch("marketpulse.db.session.engine"):
        session = SessionLocal()
        assert session is not None
        session.close()
