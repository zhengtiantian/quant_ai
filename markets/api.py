"""HTTP routes for the portfolio, the global markets and the briefs (/api/markets/...).

Personal financial data: every route answers only requests from this Mac (127.0.0.1), whatever
address the server listens on. quant_ui calls these from the browser and the desktop widget
calls them with curl, both locally.
"""

import time
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from . import backup, global_markets, portfolio

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def local_only(request: Request) -> None:
    if not request.client or request.client.host not in LOOPBACK:
        raise HTTPException(status_code=403, detail="markets data is only served to this machine")


router = APIRouter(prefix="/api/markets", dependencies=[Depends(local_only)])


def _error(e: Exception, status: int) -> JSONResponse:
    msg = str(e) if isinstance(e, (ValueError, KeyError)) else f"{type(e).__name__}: {e}"
    return JSONResponse({"status": "error", "message": msg}, status_code=status)


@router.get("/portfolio/state")
def portfolio_state(force: str = ""):
    try:
        portfolio.init_db()
        # Never waits on Nasdaq: stored quotes now, a background thread refreshes them.
        state = portfolio.get_state(force_quotes=force == "1", background_refresh=True)
        from . import pipeline_health
        return {"status": "success", **state, "latest_briefs": _latest_briefs(), "pipeline": pipeline_health.latest()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


def _latest_briefs():
    """The newest written brief of each kind, for the desktop widget's headline lines."""
    from . import briefs
    out = {}
    for b in briefs.list_briefs(limit=30):
        if b["status"] == "done" and b["kind"] not in out:
            out[b["kind"]] = {k: b[k] for k in ("id", "kind", "session_date", "topic", "headline_zh", "headline_en", "finished_at")}
    # A flash only while it is fresh: today's or yesterday's.
    flash = out.get("flash")
    if flash and time.time() - (flash["finished_at"] or 0) > 36 * 3600:
        out.pop("flash")
    return [out[k] for k in ("flash", "outlook", "asia_europe", "us") if k in out]


@router.get("/portfolio/chart")
def portfolio_chart(symbol: str = "", range: str = "1d"):  # noqa: A002 - the page's parameter name
    try:
        portfolio.init_db()
        return {"status": "success", **portfolio.get_chart(symbol, range)}
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/portfolio/{action:path}")
def portfolio_write(action: str, payload: Dict[str, Any]):
    try:
        portfolio.init_db()
        result: Dict[str, Any] = {}
        if action == "tx/save":
            result["id"] = portfolio.save_transaction(payload.get("data") or {})
            try:   # a sale's profit goes to the budget's 股票 item; the trade is saved either way
                result["budget"] = portfolio.sync_budget().get(result["id"])
            except Exception as e:  # noqa: BLE001
                result["budget"] = {"error": f"{type(e).__name__}: {e}"}
        elif action == "tx/delete":
            portfolio.delete_transaction(int(payload["id"]))
            try:
                portfolio.sync_budget()     # the sale's line goes; later sales may change with the average
            except Exception:  # noqa: BLE001
                pass
        elif action == "watch/add":
            result["symbol"] = portfolio.add_watch(payload.get("symbol", ""), payload.get("note", ""))
        elif action == "watch/remove":
            portfolio.remove_watch(payload.get("symbol", ""))
        elif action == "cash/set":
            result["cash"] = portfolio.set_cash(payload.get("cash"))
        else:
            return JSONResponse({"status": "error", "message": f"unknown portfolio endpoint {action}"}, status_code=404)
        backup.schedule_backup()
        # Cached prices only: a save answers in milliseconds, the page's poll refreshes quotes.
        return {"status": "success", **result, "state": portfolio.get_state(cached_quotes=True)}
    except (ValueError, KeyError) as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/global")
def global_markets_state():
    try:
        return {"status": "success", **global_markets.global_state(background=True)}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/macro")
def macro():
    try:
        return {"status": "success", **global_markets.macro_state()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/briefs")
def briefs_list(kind: str = "", limit: int = 60):
    from . import briefs
    try:
        return {"status": "success", "briefs": briefs.list_briefs(kind, limit), **briefs.status()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/briefs/{brief_id}")
def briefs_get(brief_id: int):
    from . import briefs
    b = briefs.get_brief(brief_id)
    if not b:
        return JSONResponse({"status": "error", "message": "no such brief"}, status_code=404)
    return {"status": "success", "brief": b}


@router.post("/briefs/run")
def briefs_run(payload: Dict[str, Any]):
    """Write (or rewrite) the current session's brief now. Uses the subscription quota."""
    from . import briefs
    try:
        started = briefs.generate_in_background(str(payload.get("kind", "")), symbol=str(payload.get("symbol", "")))
        return {"status": "success", "started": started, **briefs.status()}
    except ValueError as e:
        return _error(e, 400)


@router.get("/advisor")
def advisor_state(limit: int = 30):
    """My Holdings' advisor panel: the reports (urgent ones flagged), the orders still in play, the
    orders' track record, the cash the advice is sized with."""
    from . import advisor, briefs
    try:
        st = briefs.status()["kinds"]["advisor"]
        return {"status": "success", "reports": briefs.list_advisor(limit), "orders": advisor.active_orders(),
                "scorecard": advisor.scorecard(), "cash": portfolio.get_cash(), "running": st["running"],
                "due_session": st["due_session"], "schedule": "weekdays 13:00 Dublin, plus focused updates on events"}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/predictions/scorecard")
def predictions_scorecard():
    from . import predictions
    try:
        return {"status": "success", **predictions.scorecard()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/pipeline/health")
def pipeline_health_route(refresh: str = ""):
    """Freshness of each quant_data stage; ?refresh=1 checks now instead of the last result."""
    from . import pipeline_health
    try:
        return {"status": "success", **(pipeline_health.check() if refresh == "1" else pipeline_health.latest())}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)
