# tests/unit/test_cache.py
# Unit tests for the Redis cache layer.
# All tests mock redis.Redis — no real Redis server required.

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest  # noqa: F401
import redis as redis_lib

# ══════════════════════════════════════════════════════════════════════════════
# RedisClient tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRedisClient:
    """Tests for the RedisClient wrapper class."""

    def _make_client(self, mock_redis_instance: MagicMock):
        """Create a RedisClient with a mocked underlying redis.Redis."""
        from marketpulse.cache.redis_client import RedisClient

        with patch("redis.Redis.from_url", return_value=mock_redis_instance):
            return RedisClient(redis_url="redis://localhost:6379/0")

    def test_get_returns_value_on_hit(self):
        mock_r = MagicMock()
        mock_r.get.return_value = '{"key": "value"}'
        client = self._make_client(mock_r)

        result = client.get("test:key")

        assert result == '{"key": "value"}'
        mock_r.get.assert_called_once_with("test:key")

    def test_get_returns_none_on_miss(self):
        mock_r = MagicMock()
        mock_r.get.return_value = None
        client = self._make_client(mock_r)

        result = client.get("test:missing")

        assert result is None

    def test_get_returns_none_on_redis_error(self):
        """Fail-open: RedisError is caught, None is returned."""
        mock_r = MagicMock()
        mock_r.get.side_effect = redis_lib.RedisError("connection refused")
        client = self._make_client(mock_r)

        result = client.get("test:key")  # should NOT raise

        assert result is None

    def test_set_calls_setex(self):
        mock_r = MagicMock()
        client = self._make_client(mock_r)

        result = client.set("test:key", "value", ttl=300)

        assert result is True
        mock_r.setex.assert_called_once_with(name="test:key", time=300, value="value")

    def test_set_returns_false_on_redis_error(self):
        mock_r = MagicMock()
        mock_r.setex.side_effect = redis_lib.RedisError("timeout")
        client = self._make_client(mock_r)

        result = client.set("test:key", "value", 300)

        assert result is False

    def test_ping_returns_true_on_success(self):
        mock_r = MagicMock()
        mock_r.ping.return_value = True
        client = self._make_client(mock_r)

        assert client.ping() is True

    def test_ping_returns_false_on_redis_error(self):
        mock_r = MagicMock()
        mock_r.ping.side_effect = redis_lib.RedisError("unavailable")
        client = self._make_client(mock_r)

        assert client.ping() is False

    def test_get_json_deserialises(self):
        mock_r = MagicMock()
        mock_r.get.return_value = json.dumps({"ticker": "AAPL", "close": 182.3})
        client = self._make_client(mock_r)

        result = client.get_json("test:key")

        assert result == {"ticker": "AAPL", "close": 182.3}

    def test_get_json_returns_none_on_miss(self):
        mock_r = MagicMock()
        mock_r.get.return_value = None
        client = self._make_client(mock_r)

        assert client.get_json("test:missing") is None

    def test_get_json_returns_none_on_corrupt_data(self):
        """Corrupt JSON is treated as a cache miss (not raised)."""
        mock_r = MagicMock()
        mock_r.get.return_value = "not valid json {{{"
        client = self._make_client(mock_r)

        result = client.get_json("test:corrupt")  # should NOT raise

        assert result is None

    def test_set_json_serialises(self):
        mock_r = MagicMock()
        client = self._make_client(mock_r)

        client.set_json("test:key", {"signal": "BUY"}, ttl=60)

        call_args = mock_r.setex.call_args
        stored_value = call_args[1]["value"] if "value" in call_args[1] else call_args[0][2]
        parsed = json.loads(stored_value)
        assert parsed == {"signal": "BUY"}

    def test_delete_pattern_uses_scan_iter(self):
        mock_r = MagicMock()
        mock_r.scan_iter.return_value = ["key:1", "key:2"]
        mock_r.delete.return_value = 2
        client = self._make_client(mock_r)

        count = client.delete_pattern("key:*")  # noqa: F841

        # Verify scan_iter was called (not keys)
        mock_r.scan_iter.assert_called_once_with(match="key:*")
        # Verify delete was called with the found keys
        mock_r.delete.assert_called_once_with("key:1", "key:2")
        # Verify keys() was never called (it would be blocking)
        assert not mock_r.keys.called


# ══════════════════════════════════════════════════════════════════════════════
# cached() decorator tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCachedDecorator:
    """Tests for the cache-aside @cached() decorator."""

    def test_calls_function_on_cache_miss(self):
        """First call → cache miss → function is called."""
        from marketpulse.cache.redis_client import RedisClient, cached

        mock_r = MagicMock()
        mock_r.get.return_value = None  # miss
        mock_r.setex.return_value = True

        call_count = 0

        with patch("marketpulse.cache.redis_client._create_client") as mock_factory:
            mock_factory.return_value = RedisClient.__new__(RedisClient)
            mock_factory.return_value._r = mock_r

            @cached(key_fn=lambda x: f"test:{x}", ttl=60)
            def my_func(x: int) -> list[int]:
                nonlocal call_count
                call_count += 1
                return [x, x * 2]

            result = my_func(5)

        assert call_count == 1
        assert result == [5, 10]

    def test_returns_cached_value_on_hit(self):
        """If Redis has the value, function should NOT be called."""
        from marketpulse.cache.redis_client import RedisClient, cached

        cached_data = json.dumps([42, 84])
        mock_r = MagicMock()
        mock_r.get.return_value = cached_data  # cache HIT

        call_count = 0

        with patch("marketpulse.cache.redis_client._create_client") as mock_factory:
            mock_factory.return_value = RedisClient.__new__(RedisClient)
            mock_factory.return_value._r = mock_r

            @cached(key_fn=lambda x: f"test:{x}", ttl=60)
            def expensive_func(x: int) -> list[int]:
                nonlocal call_count
                call_count += 1
                return [x]

            result = expensive_func(21)

        assert call_count == 0      # function was NOT called
        assert result == [42, 84]   # returned cached data


# ══════════════════════════════════════════════════════════════════════════════
# invalidate_ticker() tests
# ══════════════════════════════════════════════════════════════════════════════

class TestInvalidateTicker:
    """Tests for the Redis cache invalidation helper."""

    def test_deletes_ticker_pattern(self):
        from marketpulse.cache.redis_client import RedisClient, invalidate_ticker

        mock_r = MagicMock()
        mock_r.scan_iter.return_value = [
            "marketpulse:stocks:AAPL:prices:100",
            "marketpulse:stocks:AAPL:indicators:200",
        ]

        with patch("marketpulse.cache.redis_client._create_client") as mock_factory:
            mock_factory.return_value = RedisClient.__new__(RedisClient)
            mock_factory.return_value._r = mock_r

            count = invalidate_ticker("AAPL")

        # Verify scan was called with the correct pattern
        mock_r.scan_iter.assert_called_once_with(match="marketpulse:stocks:AAPL:*")
        assert count >= 0
