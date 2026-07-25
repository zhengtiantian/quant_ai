#!/usr/bin/env python3
"""MCP server exposing the AI-Driven Equity Signal Platform.

Runs over stdio, so each client spawns it as a subprocess rather than
connecting to a service. Two clients use it today: Claude Desktop (registered
in claude_desktop_config.json) and quant_ai's own research agent (mcp_client.py),
which discovers its entire tool surface here — registering a tool below is
enough for the agent to gain it.

Exposes LLM-labeled news sentiment over 845K+ articles, engineered features,
ranked daily signals, paper positions, and backtest performance. All tools are
read-only — this server analyzes, it never trades. Data is read through
quant_api (:18081), falling back to mongo only when that is unreachable.

Smoke test:
    RUN_MCP_E2E=1 .venv/bin/python -m unittest tests.test_mcp_server -v
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any

import requests
from mcp.server.fastmcp import FastMCP

QUANT_API = os.getenv("QUANT_API", "http://localhost:18081").rstrip("/")
TIMEOUT = float(os.getenv("QUANT_MCP_TIMEOUT", "15"))
MONGO_URI = os.getenv(
    "LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/?authSource=admin"
).strip()
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")

mcp = FastMCP("quant")

_mongo_client = None


def _db():
    """Lazy mongo handle for the read-only fallback path."""
    global _mongo_client
    if _mongo_client is None:
        from pymongo import MongoClient
        _mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
    return _mongo_client[DB_NAME]


def _get(path: str, params: dict[str, Any] | None = None) -> str | None:
    """GET from quant_api. Returns pretty JSON, or None if the call did not succeed
    so the caller can fall back to reading mongo directly."""
    try:
        resp = requests.get(f"{QUANT_API}{path}", params=params, timeout=TIMEOUT)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        return json.dumps(resp.json(), indent=2, ensure_ascii=False, default=str)
    except ValueError:
        return resp.text[:4000]


def _unavailable(path: str) -> str:
    return json.dumps({
        "error": f"quant_api at {QUANT_API} did not serve {path}",
        "hint": "Is the quant_api container running? `docker compose up -d quant_api`",
    }, indent=2)


@mcp.tool()
def get_news_sentiment(symbol: str, days: int = 90) -> str:
    """Aggregated LLM-labeled news sentiment for a stock symbol.

    Returns article count, average sentiment (-1 bearish .. +1 bullish), average
    disagreement between the two labeling models (low = high consensus), and the
    most recent headlines. Source: 840K+ articles labeled by a dual-LLM pipeline
    (Gemma + Qwen) and merged into a final sentiment score.

    Args:
        symbol: Ticker, e.g. AAPL.
        days: Lookback window in days (default 90). Use 0 for all history.
    """
    sym = symbol.upper().strip()
    via_api = _get(f"/api/agent-data/news/{sym}/sentiment", {"days": days})
    if via_api is not None:
        return via_api
    # Fallback: aggregate straight from mongo so a quant_api outage does not
    # take the tool down with it.
    try:
        col = _db()["news_articles_company_matched_v2"]
        match: dict[str, Any] = {"symbol": sym, "llm_sentiment_final": {"$exists": True}}
        if days > 0:
            cutoff = (dt.datetime.utcnow() - dt.timedelta(days=days)).strftime("%Y%m%d%H%M%S")
            match["date"] = {"$gte": cutoff}
        agg = list(col.aggregate([
            {"$match": match},
            {"$group": {"_id": None,
                        "articles": {"$sum": 1},
                        "avgSentiment": {"$avg": "$llm_sentiment_final"},
                        "avgModelDisagreement": {"$avg": "$llm_disagreement"}}},
        ]))
        if not agg:
            return json.dumps({"symbol": sym, "articles": 0,
                               "note": "no labeled articles for this symbol in window",
                               "source": "mongo-fallback"}, indent=2)
        g = agg[0]
        heads = list(col.find({"symbol": sym},
                              {"_id": 0, "title": 1, "date": 1, "llm_sentiment_final": 1})
                     .sort("date", -1).limit(3))
        return json.dumps({
            "symbol": sym,
            "articles": g["articles"],
            "avgSentiment": round(g["avgSentiment"], 4),
            "avgModelDisagreement": round(g.get("avgModelDisagreement") or 0.0, 4),
            "recentHeadlines": heads,
            "source": "mongo-fallback",
        }, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return _unavailable(f"/api/agent-data/news/{sym}/sentiment")


@mcp.tool()
def get_stock_features(symbol: str) -> str:
    """Latest engineered daily features for a stock symbol.

    Momentum, volatility, analyst consensus, institutional holdings, after-hours
    gap, retail/macro alt-data and aggregated news-sentiment features — the same
    feature row the ranking models score.

    Args:
        symbol: Ticker, e.g. AAPL.
    """
    sym = symbol.upper().strip()
    via_api = _get(f"/api/agent-data/features/{sym}/latest")
    if via_api is not None:
        return via_api
    try:
        doc = _db()["daily_symbol_features"].find_one(
            {"symbol": sym}, {"_id": 0}, sort=[("date", -1)]
        )
        if not doc:
            return json.dumps({"symbol": sym,
                               "note": "no features computed yet for this symbol",
                               "source": "mongo-fallback"}, indent=2)
        slim = {k: v for k, v in list(doc.items())[:25]}
        return json.dumps({"symbol": sym, "latestFeatures": slim,
                           "source": "mongo-fallback"},
                          indent=2, ensure_ascii=False, default=str)
    except Exception:
        return _unavailable(f"/api/agent-data/features/{sym}/latest")


@mcp.tool()
def get_latest_signals(limit: int = 10) -> str:
    """Latest ranked daily trading signals from the ensemble model.

    Each entry carries the trade date, symbol, composite score, rank, and signal
    type, produced by the Ridge + LightGBM ranker ensemble.

    Args:
        limit: How many top-ranked signals to return (default 10).
    """
    return _get("/api/signals/latest", {"limit": limit}) or _unavailable("/api/signals/latest")


@mcp.tool()
def get_positions() -> str:
    """Current paper-trading positions with entry price, current price, and P&L."""
    return _get("/api/positions") or _unavailable("/api/positions")


@mcp.tool()
def get_performance() -> str:
    """Portfolio backtest performance: Sharpe, returns, hit rate, drawdown, and
    the rebalance history behind them."""
    return _get("/api/performance") or _unavailable("/api/performance")


@mcp.tool()
def list_symbols() -> str:
    """List every stock symbol covered by the platform's universe."""
    return _get("/api/market/symbols") or _unavailable("/api/market/symbols")


if __name__ == "__main__":
    mcp.run(transport="stdio")
