"""
MarketPulse Dashboard
=====================
A live stock market intelligence dashboard powered by Streamlit.

Calls the MarketPulse FastAPI backend for:
  - OHLCV price bars and technical indicators (prices + indicators endpoints)
  - ML BUY/HOLD/SELL signals (signals endpoint)
  - News headlines with VADER sentiment scores (news endpoint)

Architecture: One Python file (Streamlit idiom for dashboards of this size).
Re-runs top-to-bottom on every user interaction; @st.cache_data prevents
redundant API calls within each 30-second window.

Run locally:  streamlit run dashboard/app.py
In Docker:    docker compose up -d  (starts on port 8501)
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timezone  # noqa: F401
from datetime import time as dt_time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── API configuration ──────────────────────────────────────────────────────────
# Set by docker-compose.yml for the dashboard service.
# Default allows running the dashboard locally without Docker.
API_URL: str = os.environ.get(
    "DASHBOARD_API_URL",
    "http://localhost:8000/api/v1",
)

# ── Colour palette (consistent across all charts) ─────────────────────────────
_C_BUY = "#26a69a"       # teal-green  — up candles, BUY signal
_C_SELL = "#ef5350"      # red         — down candles, SELL signal
_C_HOLD = "#FFA726"      # amber       — HOLD signal
_C_SMA20 = "#2196F3"     # blue        — SMA-20 line
_C_SMA50 = "#FF9800"     # orange      — SMA-50 line
_C_BB = "#9E9E9E"        # grey        — Bollinger Bands
_C_SENTIMENT = "#2196F3" # blue        — sentiment fill
_C_POS = "#66BB6A"       # green       — positive sentiment markers
_C_NEG = "#EF5350"       # red         — negative sentiment markers
_C_NEU = "#90A4AE"       # blue-grey   — neutral sentiment markers

# ── Market status utility ─────────────────────────────────────────────────────

def is_market_open() -> bool:
    """
    Estimate whether US equity markets (NYSE/NASDAQ) are currently open.

    Uses UTC times as a conservative approximation for Eastern Time:
    - Market open  ≈ 14:30 UTC  (9:30 AM EST / 10:30 AM EDT)
    - Market close ≈ 21:00 UTC  (4:00 PM EST / 5:00 PM EDT)

    Ignores US market holidays. Returns False on weekends.
    For production accuracy, use a proper market calendar library.
    """
    now = datetime.now(tz=UTC)
    if now.weekday() > 4:               # 5=Saturday, 6=Sunday
        return False
    open_utc = dt_time(14, 30)
    close_utc = dt_time(21, 0)
    return open_utc <= now.time() <= close_utc


def _f(v: object) -> float | None:
    """Convert a value to float; return None for None and NaN."""
    if v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
        return None if f != f else f  # NaN check: NaN != NaN is always True
    except (TypeError, ValueError):
        return None

# ── API fetch functions ────────────────────────────────────────────────────────
# All decorated with @st.cache_data to prevent redundant API calls on re-renders.
# Return empty DataFrames / None / [] on error (fail-silent for graceful UI).
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def _fetch_health() -> dict:
    """Fetch API health. Short TTL (10s) — used for the connection indicator."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "error", "db": "error", "redis": "error", "tickers": []}


@st.cache_data(ttl=30)
def _fetch_tickers() -> list[str]:
    """Fetch tracked ticker list. Fallback to ENV default if API unavailable."""
    try:
        r = httpx.get(f"{API_URL}/stocks", timeout=5.0)
        if r.status_code == 200:
            return r.json().get("tickers", [])
    except Exception:
        pass
    # Fallback: read from environment so the sidebar isn't empty
    tickers_env = os.environ.get("TICKERS", "AAPL,GOOGL,MSFT,TSLA")
    return [t.strip().upper() for t in tickers_env.split(",") if t.strip()]


@st.cache_data(ttl=30)
def _fetch_prices(ticker: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV bars. Streamlit TTL=30s; API Redis TTL=300s.

    Returns columns: ticker, timestamp, open, high, low, close, volume.
    Empty DataFrame on 404 (no data yet) or network error.
    """
    try:
        r = httpx.get(
            f"{API_URL}/stocks/{ticker}/prices",
            params={"limit": limit},
            timeout=10.0,
        )
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=30)
def _fetch_indicators(ticker: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch technical indicators. Same TTL as prices for chart timestamp alignment.

    Returns columns: ticker, timestamp, sma_20, sma_50, ..., bb_upper, bb_lower.
    Nullable columns are None for early bars with insufficient history.
    """
    try:
        r = httpx.get(
            f"{API_URL}/stocks/{ticker}/indicators",
            params={"limit": limit},
            timeout=10.0,
        )
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def _fetch_latest_signal(ticker: str) -> dict | None:
    """
    Fetch the single most recent ML signal.

    Returns None when the ML pipeline hasn't run yet (expected on first startup).
    Keys: ticker, timestamp, signal, confidence, is_anomaly, model_version.
    """
    try:
        r = httpx.get(
            f"{API_URL}/stocks/{ticker}/signals/latest",
            timeout=5.0,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=60)
def _fetch_signals(ticker: str, limit: int = 20) -> pd.DataFrame:
    """Fetch ML signal history for the anomaly panel."""
    try:
        r = httpx.get(
            f"{API_URL}/stocks/{ticker}/signals",
            params={"limit": limit},
            timeout=5.0,
        )
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=60)
def _fetch_news(limit: int = 30) -> pd.DataFrame:
    """
    Fetch news with VADER sentiment scores (newest-first).

    Columns: id, title, source_url, published_at,
             sentiment_positive, sentiment_negative,
             sentiment_neutral, sentiment_compound.
    """
    try:
        r = httpx.get(
            f"{API_URL}/news",
            params={"limit": limit},
            timeout=10.0,
        )
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
            return df
    except Exception:
        pass
    return pd.DataFrame()

# ── Chart builder functions ────────────────────────────────────────────────────

def _build_price_chart(
    ticker: str,
    prices_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
) -> go.Figure:
    """
    Build a Plotly candlestick chart with SMA-20, SMA-50, and Bollinger Band overlays.

    Traces (in order, so legend matches visual stacking):
    1. Candlestick — OHLCV bars
    2. SMA-20 — blue solid line
    3. SMA-50 — orange solid line
    4. BB Upper — grey dashed
    5. BB Lower — grey dashed + fills downward to BB Upper (shaded band)

    Args:
        ticker:        Stock symbol (used in title).
        prices_df:     OHLCV DataFrame (timestamp, open, high, low, close, volume).
        indicators_df: Indicator DataFrame (timestamp, sma_20, sma_50, bb_upper, bb_lower, …).

    Returns:
        Configured Plotly Figure ready for st.plotly_chart().
    """
    fig = go.Figure()

    # ── Trace 1: Candlestick ───────────────────────────────────────────────────
    if not prices_df.empty:
        fig.add_trace(go.Candlestick(
            x=prices_df["timestamp"],
            open=prices_df["open"],
            high=prices_df["high"],
            low=prices_df["low"],
            close=prices_df["close"],
            name=ticker,
            increasing_line_color=_C_BUY,   # green candles
            decreasing_line_color=_C_SELL,  # red candles
        ))

    # ── Traces 2–5: Indicator overlays ────────────────────────────────────────
    if not indicators_df.empty:
        ind = indicators_df

        if "sma_20" in ind.columns and ind["sma_20"].notna().any():
            fig.add_trace(go.Scatter(
                x=ind["timestamp"], y=ind["sma_20"],
                mode="lines", name="SMA-20",
                line={"color": _C_SMA20, "width": 1.2},
                opacity=0.9,
            ))

        if "sma_50" in ind.columns and ind["sma_50"].notna().any():
            fig.add_trace(go.Scatter(
                x=ind["timestamp"], y=ind["sma_50"],
                mode="lines", name="SMA-50",
                line={"color": _C_SMA50, "width": 1.2},
                opacity=0.9,
            ))

        # Bollinger Bands — add upper THEN lower with fill="tonexty"
        has_bb = (
            "bb_upper" in ind.columns and ind["bb_upper"].notna().any()
            and "bb_lower" in ind.columns and ind["bb_lower"].notna().any()
        )
        if has_bb:
            fig.add_trace(go.Scatter(
                x=ind["timestamp"], y=ind["bb_upper"],
                mode="lines", name="BB Upper",
                line={"color": _C_BB, "width": 0.8, "dash": "dash"},
                opacity=0.6,
            ))
            fig.add_trace(go.Scatter(
                x=ind["timestamp"], y=ind["bb_lower"],
                mode="lines", name="BB Lower",
                line={"color": _C_BB, "width": 0.8, "dash": "dash"},
                # fill="tonexty" fills from BB Lower UP to the previous trace (BB Upper)
                fill="tonexty",
                fillcolor="rgba(158,158,158,0.07)",  # nearly-invisible grey band
                opacity=0.6,
            ))

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        title={"text": f"{ticker} — Price & Indicators", "font": {"size": 13}},
        xaxis={
            "rangeslider": {"visible": False},  # disable mini-chart below (wastes space)
            "type": "date",
        },
        yaxis={"title": "Price (USD)", "side": "right"},
        height=460,
        margin={"l": 0, "r": 50, "t": 40, "b": 0},
        legend={
            "orientation": "h",
            "yanchor": "bottom", "y": 1.01,
            "xanchor": "right", "x": 1,
            "font": {"size": 11},
        },
        hovermode="x unified",  # show all trace values in one tooltip on hover
    )
    return fig


def _build_sentiment_chart(news_df: pd.DataFrame) -> go.Figure:
    """
    Build a Plotly scatter chart showing VADER compound sentiment over time.

    - Each data point is coloured green/red/grey based on compound threshold
    - Light blue fill from the line to y=0 (area chart style)
    - Dashed reference lines at ±0.05 (the positive/negative threshold)

    Args:
        news_df: DataFrame with published_at and sentiment_compound columns.

    Returns:
        Configured Plotly Figure.
    """
    fig = go.Figure()

    if not news_df.empty:
        df = news_df.sort_values("published_at")
        compound = df["sentiment_compound"].astype(float)

        # Colour each marker based on sentiment threshold
        marker_colours = compound.apply(
            lambda v: _C_POS if v > 0.05 else _C_NEG if v < -0.05 else _C_NEU
        ).tolist()

        fig.add_trace(go.Scatter(
            x=df["published_at"],
            y=compound,
            mode="lines+markers",
            name="Compound",
            line={"color": _C_SENTIMENT, "width": 1.5},
            marker={"size": 5, "color": marker_colours},
            fill="tozeroy",
            fillcolor="rgba(33, 150, 243, 0.07)",
            hovertemplate=(
                "<b>%{x|%b %d %H:%M}</b><br>"
                "Compound: %{y:.4f}<extra></extra>"
            ),
        ))

        # Threshold reference lines
        fig.add_hline(y=0.05, line_dash="dot", line_color=_C_POS, opacity=0.5)
        fig.add_hline(y=-0.05, line_dash="dot", line_color=_C_NEG, opacity=0.5)
        fig.add_hline(y=0, line_color="#78909C", line_width=0.5, opacity=0.4)

    fig.update_layout(
        title={"text": "News Sentiment (compound score)", "font": {"size": 13}},
        xaxis_title="",
        yaxis={"title": "Score", "range": [-1.15, 1.15]},
        height=280,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        hovermode="x unified",
        showlegend=False,
    )
    return fig

# ── Panel render functions ─────────────────────────────────────────────────────

def _render_sidebar() -> str:
    """
    Render the left sidebar and return the user-selected ticker symbol.

    Sidebar contents:
    - Logo + title
    - Ticker selector (dropdown from API ticker list)
    - Auto-refresh toggle
    - Market open/closed status
    - API / DB / Redis health indicators

    Returns:
        The selected ticker symbol (e.g., "AAPL").
    """
    with st.sidebar:
        st.markdown("## 📈 MarketPulse")
        st.caption("Stock Market Intelligence Pipeline")
        st.divider()

        # Ticker selector
        tickers = _fetch_tickers()
        ticker = st.selectbox(
            "Ticker",
            options=tickers,
            index=0,
            help="Select a tracked stock symbol",
        )

        st.divider()

        # Auto-refresh toggle
        auto = st.toggle(
            "Auto-refresh (30s)",
            value=st.session_state.get("auto_refresh", True),
        )
        st.session_state["auto_refresh"] = auto

        # Market status indicator
        if is_market_open():
            st.success("🟢 Market Open", icon=None)
        else:
            st.caption("🔴 Market Closed")

        st.caption(
            f"UTC: {datetime.now(tz=UTC).strftime('%H:%M:%S')}"
        )

        st.divider()

        # API health indicators in sidebar footer
        health = _fetch_health()
        api_ok = health.get("status") == "ok"
        db_ok = health.get("db") == "ok"
        cache_ok = health.get("redis") == "ok"

        if api_ok:
            st.success("API connected")
        else:
            st.error("API unavailable")

        col_db, col_cache = st.columns(2)
        col_db.caption(f"DB {'✓' if db_ok else '✗'}")
        col_cache.caption(f"Cache {'✓' if cache_ok else '✗'}")

        st.caption(f"Endpoint: `{API_URL}`")

    return str(ticker)


def _render_price_signal_panel(
    ticker: str,
    prices_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    latest_signal: dict | None,
) -> None:
    """
    Render the top section: candlestick chart (left, wide) + signal badge (right, narrow).

    Layout: 3:1 column split.
    - Left: Plotly candlestick with SMA-20, SMA-50, Bollinger Bands
    - Right: BUY/HOLD/SELL badge, confidence meter, anomaly flag
    """
    chart_col, signal_col = st.columns([3, 1])

    with chart_col:
        if not prices_df.empty:
            fig = _build_price_chart(ticker, prices_df, indicators_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                f"⏳ No price data for **{ticker}** yet.\n\n"
                "The ingestion job runs every 15 minutes. "
                "During startup, `warmup_cache()` triggers the first run. "
                "Check `docker compose logs api` for progress."
            )

    with signal_col:
        st.subheader("ML Signal")

        if latest_signal:
            sig = latest_signal["signal"]
            conf = float(latest_signal["confidence"])
            anomaly = latest_signal["is_anomaly"]
            version = latest_signal.get("model_version", "?")
            ts = latest_signal.get("timestamp", "")[:16].replace("T", " ")

            # Colour-coded signal status box
            if sig == "BUY":
                st.success(f"## 🟢 {sig}")
            elif sig == "SELL":
                st.error(f"## 🔴 {sig}")
            else:
                st.warning(f"## 🟡 {sig}")

            # Confidence: numeric metric + visual progress bar
            st.metric("Confidence", f"{conf:.0%}")
            st.progress(conf)

            st.caption(f"Model: `{version}`")
            st.caption(f"Generated: {ts} UTC")

            # Anomaly indicator
            if anomaly:
                st.error("⚠️ Anomaly detected in this bar")
            else:
                st.caption("✓ No anomaly")

        else:
            st.info(
                "No signal yet.\n\n"
                "The ML pipeline runs every hour and requires ≥ 50 rows "
                "of indicator data. Check the anomaly panel below for status."
            )


def _render_metrics_row(
    prices_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
) -> None:
    """
    Render four key metrics in a horizontal row.

    Metrics: Close price (+ % change), RSI-14 (+ overbought/oversold label),
             MACD (+ signal line delta), Volume (+ % vs 200-bar average).
    """
    st.subheader("Latest Bar")
    c1, c2, c3, c4 = st.columns(4)

    last_price = prices_df.iloc[-1] if not prices_df.empty else None
    prev_price = prices_df.iloc[-2] if len(prices_df) > 1 else None
    last_ind = indicators_df.iloc[-1] if not indicators_df.empty else None

    # ── Close + % change ──────────────────────────────────────────────────────
    with c1:
        close = _f(last_price["close"]) if last_price is not None else None
        prev = _f(prev_price["close"]) if prev_price is not None else None
        pct = ((close - prev) / prev * 100) if close and prev else None
        st.metric(
            "Close",
            f"${close:.2f}" if close else "—",
            delta=f"{pct:+.2f}%" if pct is not None else None,
        )

    # ── RSI-14 ────────────────────────────────────────────────────────────────
    with c2:
        rsi = _f(last_ind["rsi_14"]) if last_ind is not None else None
        rsi_label = None
        if rsi is not None:
            rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else None
        st.metric("RSI-14", f"{rsi:.1f}" if rsi else "—", delta=rsi_label)

    # ── MACD ─────────────────────────────────────────────────────────────────
    with c3:
        macd = _f(last_ind["macd"]) if last_ind is not None else None
        macd_sig = _f(last_ind["macd_signal"]) if last_ind is not None else None
        sig_delta = f"Sig: {macd_sig:+.4f}" if macd_sig is not None else None
        st.metric("MACD", f"{macd:+.4f}" if macd is not None else "—", delta=sig_delta)

    # ── Volume vs 200-bar average ─────────────────────────────────────────────
    with c4:
        vol = last_price["volume"] if last_price is not None else None
        avg_vol = prices_df["volume"].mean() if not prices_df.empty else None
        vol_delta = (
            f"{(vol - avg_vol) / avg_vol * 100:+.0f}% vs avg"
            if vol and avg_vol and avg_vol > 0
            else None
        )
        vol_str = f"{vol / 1_000_000:.2f}M" if vol else "—"
        st.metric("Volume", vol_str, delta=vol_delta)

def _render_sentiment_news_panel(news_df: pd.DataFrame) -> None:
    """
    Render the sentiment timeline chart (left) and news headlines feed (right).

    Layout: 3:2 column split.
    - Left: VADER compound score over time (area chart)
    - Right: Latest 8 headlines with colour-coded sentiment icons
             + expander for remaining headlines
    """
    st.divider()
    sent_col, news_col = st.columns([3, 2])

    with sent_col:
        st.subheader("Sentiment Timeline")
        if not news_df.empty:
            fig = _build_sentiment_chart(news_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "No news articles yet.\n\n"
                "Configure `RSS_FEED_URLS` in `.env` and restart the stack. "
                "The news job runs every 30 minutes."
            )

    with news_col:
        st.subheader("Headlines")
        if not news_df.empty:
            # Show latest 8 headlines directly
            for _, row in news_df.head(8).iterrows():
                compound = float(row["sentiment_compound"])
                icon = "🟢" if compound > 0.05 else "🔴" if compound < -0.05 else "⚪"
                title = str(row["title"])
                # Truncate long titles for readability
                display_title = title[:72] + "…" if len(title) > 72 else title
                pub = pd.to_datetime(row["published_at"])

                st.markdown(f"{icon} [{display_title}]({row['source_url']})")
                st.caption(
                    f"Sentiment: {compound:+.3f} | "
                    f"{pub.strftime('%b %d, %H:%M')} UTC"
                )

            # Remaining headlines inside a collapsible expander
            remaining = news_df.iloc[8:]
            if not remaining.empty:
                with st.expander(f"See {len(remaining)} more headlines"):
                    for _, row in remaining.iterrows():
                        compound = float(row["sentiment_compound"])
                        icon = "🟢" if compound > 0.05 else "🔴" if compound < -0.05 else "⚪"
                        title = str(row["title"])[:72]
                        st.markdown(f"{icon} {title}")
        else:
            st.info("No headlines yet.")


def _render_anomaly_panel(ticker: str, signals_df: pd.DataFrame) -> None:
    """
    Render the anomaly history panel showing IsolationForest detections.

    Shows:
    - Success message if no anomalies detected in last N signals
    - Warning with a list of anomaly timestamps + signal context if any exist
    - Info message if no signal data yet
    """
    st.divider()
    st.subheader("⚠️ Anomaly History")

    if signals_df.empty:
        st.info(
            f"No ML signal history for **{ticker}**.\n\n"
            "The ML pipeline (run every 60 minutes) requires ≥ 50 rows of "
            "feature data. Ensure the ingestion job has been running and the "
            "initial data is populated."
        )
        return

    anomalies = signals_df[signals_df["is_anomaly"] == True]  # noqa: E712

    if anomalies.empty:
        st.success(
            f"✓ No anomalies detected in the last **{len(signals_df)}** "
            "ML pipeline runs for this ticker."
        )
    else:
        st.warning(
            f"**{len(anomalies)} anomaly(ies)** detected "
            f"in the last {len(signals_df)} runs:"
        )
        for _, row in anomalies.iterrows():
            ts = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d %H:%M UTC")
            sig = row["signal"]
            conf = float(row["confidence"])
            sig_icon = "🟢" if sig == "BUY" else "🔴" if sig == "SELL" else "🟡"
            st.markdown(
                f"• **{ts}** — Signal: {sig_icon} {sig} "
                f"({conf:.0%} confidence) | Model: `{row['model_version']}`"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Main dashboard orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Main Streamlit application entry point.

    Execution flow on every render (user interaction or auto-refresh):
    1. Set page config (must be first Streamlit call)
    2. Render sidebar → get selected ticker
    3. Check API health → st.stop() if unavailable
    4. Fetch all data for the selected ticker (cached calls)
    5. Render panels top-to-bottom
    6. Sleep 30s + rerun (if auto-refresh enabled)
    """
    # ── Page configuration (must be the very first Streamlit call) ─────────────
    st.write("MAIN IN APP")
    st.set_page_config(
        page_title="MarketPulse",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": (
                "**MarketPulse** — stock market intelligence pipeline. "
                "Built with FastAPI, Streamlit, Redis, PostgreSQL, and scikit-learn."
            ),
        },
    )

    # ── Initialise session state ───────────────────────────────────────────────
    if "auto_refresh" not in st.session_state:
        st.session_state["auto_refresh"] = True

    # ── Sidebar (returns selected ticker) ─────────────────────────────────────
    ticker = _render_sidebar()

    # ── API availability check ─────────────────────────────────────────────────
    health = _fetch_health()
    if health.get("status") == "error":
        st.error(
            f"## ⚠️ Cannot reach the MarketPulse API\n\n"
            f"Endpoint: `{API_URL}`\n\n"
            "**Steps to fix:**\n"
            "1. Run `docker compose up -d`\n"
            "2. Wait 15–20 seconds for services to start\n"
            "3. Check `docker compose logs api` for errors\n"
            "4. Verify `docker compose exec db pg_isready -U marketpulse`"
        )
        st.stop()  # halt — do not render empty panels below

    # ── Page header ───────────────────────────────────────────────────────────
    head_left, head_right = st.columns([5, 1])
    with head_left:
        st.title(f"📊 {ticker}")
    with head_right:
        now_str = datetime.now(tz=UTC).strftime("%H:%M:%S UTC")
        st.caption(f"Updated: {now_str}")

    # ── Fetch all data for the selected ticker ────────────────────────────────
    # All calls are @st.cache_data decorated — fast on cache hit, one API call on miss
    prices_df = _fetch_prices(ticker, limit=200)
    indicators_df = _fetch_indicators(ticker, limit=200)
    latest_signal = _fetch_latest_signal(ticker)
    signals_df = _fetch_signals(ticker, limit=20)
    news_df = _fetch_news(limit=30)

    # ── No data banner (ingestion hasn't run yet) ──────────────────────────────
    if prices_df.empty:
        st.warning(
            f"**No data for {ticker} yet.** "
            "The first ingestion cycle runs on startup via `warmup_cache()`. "
            "Subsequent runs are every 15 minutes. Check `docker compose logs api`."
        )

    # ── Panel 1: Candlestick chart + Signal badge ─────────────────────────────
    _render_price_signal_panel(ticker, prices_df, indicators_df, latest_signal)

    # ── Panel 2: Metrics row ──────────────────────────────────────────────────
    if not prices_df.empty:
        _render_metrics_row(prices_df, indicators_df)

    # ── Panel 3: Sentiment timeline + News headlines ───────────────────────────
    _render_sentiment_news_panel(news_df)

    # ── Panel 4: Anomaly history ──────────────────────────────────────────────
    _render_anomaly_panel(ticker, signals_df)

    # ── Auto-refresh footer ───────────────────────────────────────────────────
    st.divider()
    if st.session_state["auto_refresh"]:
        st.caption("⏱ Auto-refresh active — page updates every 30 seconds.")
        # time.sleep(30) blocks the Streamlit thread.
        # After sleeping, st.rerun() triggers a fresh top-to-bottom execution.
        # The @st.cache_data TTL=30 on all fetch functions ensures fresh data.
        time.sleep(30)
        st.rerun()
    else:
        if st.button("🔄 Refresh now"):
            st.rerun()


# ── Entry point ────────────────────────────────────────────────────────────────
# Streamlit runs this file as a script: `streamlit run dashboard/app.py`
# The __name__ == "__main__" guard is conventional; Streamlit also works without it.
if __name__ == "__main__":
    main()

main()

