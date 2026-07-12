# Run with: python smoke_test_cache_scheduler.py

import asyncio  # noqa: F401
import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TICKERS", "AAPL,MSFT")
os.environ.setdefault("MODEL_DIR", "/tmp/models")

# ────────────────────────────────────────────────────────────────────────────
# SECTION A: Redis client (requires docker compose up -d)
# ────────────────────────────────────────────────────────────────────────────
print("=== SECTION A: RedisClient ===")

from marketpulse.cache import get_redis_client  # noqa: E402

client = get_redis_client()

if not client.ping():
    print("  ⚠ Redis not reachable — skipping Section A")
    print("    Run: docker compose up -d")
else:
    print("  ✓ Redis ping OK")

    # Test set + get
    client.set("test:key", '{"value": 42}', ttl=30)
    raw = client.get("test:key")
    assert raw == '{"value": 42}', f"Expected json string, got {raw!r}"
    print(f"  ✓ set + get: {raw}")

    # Test get_json + set_json
    client.set_json("test:json", {"ticker": "AAPL", "close": 182.3}, ttl=30)
    obj = client.get_json("test:json")
    assert obj["ticker"] == "AAPL"
    assert abs(obj["close"] - 182.3) < 0.001
    print(f"  ✓ set_json + get_json: {obj}")

    # Test miss (non-existent key)
    result = client.get("test:nonexistent:key:xyz")
    assert result is None, f"Expected None for missing key, got {result!r}"
    print("  ✓ get on missing key returns None")

    # Test delete
    client.set("test:del1", "v1", 30)
    client.set("test:del2", "v2", 30)
    deleted = client.delete("test:del1", "test:del2", "test:nonexistent")
    assert deleted == 2, f"Expected 2 deleted, got {deleted}"
    print(f"  ✓ delete: {deleted} keys removed")

    # Test delete_pattern (invalidate_ticker style)
    client.set_json("marketpulse:stocks:AAPL:prices:100", [{"close": 182.3}], 30)
    client.set_json("marketpulse:stocks:AAPL:indicators:200", [{"rsi_14": 62.4}], 30)
    client.set_json("marketpulse:stocks:GOOGL:prices:100", [{"close": 141.5}], 30)

    from marketpulse.cache import invalidate_ticker
    count = invalidate_ticker("AAPL")
    assert count == 2, f"Expected 2 keys deleted for AAPL, got {count}"
    # GOOGL should NOT be deleted
    googl_still = client.get("marketpulse:stocks:GOOGL:prices:100")
    assert googl_still is not None, "GOOGL cache should NOT have been deleted"
    print(f"  ✓ invalidate_ticker('AAPL'): deleted {count} keys, GOOGL unaffected")

    # Cleanup
    client.delete("marketpulse:stocks:GOOGL:prices:100", "test:key", "test:json")
    print("  ✓ All Section A tests passed")

# ────────────────────────────────────────────────────────────────────────────
# SECTION B: cached() decorator
# ────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION B: cached() decorator ===")

if client.ping():
    from marketpulse.cache import cached
    from marketpulse.config import settings  # noqa: F401

    call_count = [0]

    @cached(
        key_fn=lambda ticker, limit: f"test:decorated:{ticker}:{limit}",
        ttl=30,
    )
    def expensive_lookup(ticker: str, limit: int = 10) -> list[dict]:
        # nonlocal call_count
        call_count[0] += 1
        return [{"ticker": ticker, "call_count": call_count[0]}]

    # First call — cache miss
    result1 = expensive_lookup("AAPL", 10)
    assert call_count[0] == 1, "Should have called the function once"
    print(f"  First call (miss): {result1} — call_count={call_count}")

    # Second call — cache hit
    result2 = expensive_lookup("AAPL", 10)
    assert call_count[0] == 1, "Should NOT have called the function again"
    assert result2 == result1, "Cached result should match original"
    print(f"  Second call (hit): {result2} — call_count={call_count} (unchanged)")

    # Different args — new cache miss
    result3 = expensive_lookup("GOOGL", 10)
    assert call_count[0] == 2, "Different args should miss cache"
    print(f"  Third call (different args, miss): {result3} — call_count={call_count}")

    # Cleanup test keys
    client.delete_pattern("marketpulse:test:decorated:*")
    print("  ✓ cached() decorator works correctly")
else:
    print("  ⚠ Redis not available — skipping Section B")

# ────────────────────────────────────────────────────────────────────────────
# SECTION C: Scheduler creation (no Docker needed)
# ────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION C: create_scheduler() (no Docker needed) ===")

from marketpulse.scheduler import create_scheduler  # noqa: E402

scheduler = create_scheduler()

# Verify all four jobs are registered
job_ids = {job.id for job in scheduler.get_jobs()}
expected_ids = {"warmup_cache", "ingest_stock_data", "ingest_news", "run_ml_pipeline"}
assert expected_ids == job_ids, f"Expected {expected_ids}, got {job_ids}"
print(f"  ✓ All 4 jobs registered: {sorted(job_ids)}")

# Verify job triggers
for job in scheduler.get_jobs():
    print(f"  Job '{job.id}': trigger={job.trigger}")

# Verify get_prediction_service() returns the same instance twice
from marketpulse.scheduler import get_prediction_service  # noqa: E402

s1 = get_prediction_service()
s2 = get_prediction_service()
assert s1 is s2, "Should be the same singleton instance"
print("  ✓ get_prediction_service() returns singleton")

# Verify scheduler is not running yet
assert not scheduler.running, "Scheduler should not be running yet (not started)"
print("  ✓ Scheduler created but not yet started (call scheduler.start() in API)")

print("\n=== All smoke tests passed ✓ ===")
