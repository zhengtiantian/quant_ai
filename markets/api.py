"""HTTP routes for the portfolio, the global markets and the briefs (/api/markets/...).

Personal financial data: every route answers only requests from this Mac (127.0.0.1) or a signed-in
device on the home network (lan_auth), whatever address the server listens on. quant_ui calls them
from the browser.
"""

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from . import backup, global_markets, lan_auth, portfolio

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def local_only(request: Request) -> None:
    """This machine; or a device on the home network that is signed in (lan_auth)."""
    host = request.client.host if request.client else None
    if host in LOOPBACK:
        return
    if lan_auth.allowed(host, request.headers.get("authorization"), request.cookies.get("qt_token")):
        return
    raise HTTPException(status_code=403, detail="markets data is only served to this machine and signed-in home devices")


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
        from . import analysts
        syms = [h["symbol"] for h in state["holdings"]] + [w["symbol"] for w in state["watchlist"]]
        return {"status": "success", **state, "analysts": analysts.summaries(syms)}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/analysts/{symbol}")
def analyst_detail(symbol: str):
    """Every firm's newest rating and price target on one stock, with the headline that reported it."""
    try:
        from . import analysts
        analysts.ensure(symbol)
        return {"status": "success", **analysts.detail(symbol)}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/symbols")
def portfolio_symbols():
    """Just the tickers held (dust left out) and watched, for quant_data's news collectors: a stock
    bought today gets its own company news from the next 30-minute run."""
    try:
        from . import advisor
        st = portfolio.get_state(cached_quotes=True)
        return {"status": "success", "held": [h["symbol"] for h in advisor.followed(st)],
                "watchlist": [w["symbol"] for w in st.get("watchlist") or []]}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/daily-action")
def portfolio_daily_action(force: str = ""):
    """Today's action per holding through the risk gate (Q.12 / Q.13; a pilot on DAILY_ACTION_SYMBOLS)."""
    from . import daily_action
    try:
        portfolio.init_db()
        return {"status": "success", **daily_action.get(force=bool(force))}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/news-linkage")
def portfolio_news_linkage(force: str = ""):
    """The newest news-storyline judgment per followed stock (N.6): a hint under the Action pill only."""
    from . import news_linkage
    try:
        return {"status": "success", **news_linkage.get(force=bool(force))}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/devils-advocate")
def portfolio_devils_advocate():
    """The newest case against adding per stock (Q.17.1), from the last few days, with the page's words."""
    from . import devils_advocate
    try:
        return {"status": "success", **devils_advocate.latest()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/portfolio/devils-advocate")
def portfolio_devils_advocate_run(payload: Dict[str, Any]):
    """Write the case against adding for one stock now (Claude CLI, about a minute)."""
    from . import devils_advocate
    try:
        return {"status": "success", **devils_advocate.request(str(payload.get("symbol", "")), "manual")}
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/filing-changes")
def portfolio_filing_changes():
    """Held and watched stocks whose latest 10-K / 10-Q rewrote its risk factors (Q.17.5), with the words."""
    from . import filing_changes
    from . import portfolio as pf
    try:
        st = pf.get_state(cached_quotes=True)
        syms = [h["symbol"] for h in st.get("holdings") or []] + [w["symbol"] for w in st.get("watchlist") or []]
        return {"status": "success", **filing_changes.for_symbols(syms)}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/hint-scores")
def portfolio_hint_scores(held_only: bool = False):
    """Blocked buys and cases against adding, scored against having bought at that session's close (Q.17.0)."""
    from . import hint_scores
    try:
        return {"status": "success", **hint_scores.summary(held_only=held_only)}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/volume")
def portfolio_volume():
    """Volume of the held and watched stocks: last session, month and year (volume.py, from a file)."""
    from . import volume
    try:
        return {"status": "success", **volume.state()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/portfolio/lot-advice")
def portfolio_lot_advice(symbol: str = "", buy_price: str = ""):
    """Empirically calibrated dip-entry spacing check & per-lot ATR swing exit targets (Q.10)."""
    from . import lot_advisor
    try:
        portfolio.init_db()
        bp = float(buy_price) if buy_price and buy_price.strip() else None
        return {"status": "success", **lot_advisor.analyze_symbol(symbol, proposed_buy_price=bp)}
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


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
        elif action == "tx/delete":
            portfolio.delete_transaction(int(payload["id"]))
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


@router.get("/indices")
def indices():
    try:
        return {"status": "success", **global_markets.indices_state()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/indices")
def indices_set(body: Dict[str, Any]):
    """{"symbol": ".SOX", "shown": true|false}: show or hide one catalog index on the Holdings page."""
    try:
        return {"status": "success", **global_markets.set_index(body.get("symbol"), bool(body.get("shown", True)))}
    except (ValueError, KeyError) as e:
        return _error(e, 400)
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


@router.get("/pulse/scorecard")
def pulse_scorecard():
    """Earnings pulse: the live champion against its shadow, evening by evening (Q.5 / Q.6)."""
    from . import pulse
    try:
        return {"status": "success", **pulse.scorecard()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


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


@router.get("/keywords/review")
def keywords_review():
    """The weekly keyword proposals with the user's decisions (markets/keyword_review.py)."""
    from . import keyword_review
    try:
        return {"status": "success", **keyword_review.review()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/keywords/summary")
def keywords_summary():
    """Badge counts: suggestions waiting, coverage alerts (markets/keyword_review.py)."""
    from . import keyword_review
    try:
        return {"status": "success", **keyword_review.summary()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/keywords/coverage")
def keywords_coverage():
    """quant_data's daily coverage check (tools/coverage_watch.py)."""
    from . import keyword_review
    try:
        return {"status": "success", "coverage": keyword_review.coverage()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/keywords/auto")
def keywords_auto(payload: Dict[str, Any]):
    """{"enabled": true | false}: whether quant_data adds the suggestions that pass every check."""
    from . import keyword_review
    try:
        if not isinstance(payload.get("enabled"), bool):
            raise ValueError("enabled must be true or false")
        return {"status": "success", **keyword_review.set_auto(payload["enabled"])}
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/keywords/decide")
def keywords_decide(payload: Dict[str, Any]):
    """{"ids": [...], "decision": "add" | "weak" | "reject" | "pending" | "keep", "history"?: "full" | "first_seen"}."""
    from . import keyword_review
    try:
        ids = [str(i) for i in payload.get("ids") or []]
        if not ids:
            raise ValueError("no ids")
        return {"status": "success", **keyword_review.decide(ids, payload.get("decision"), payload.get("history"))}
    except (ValueError, KeyError) as e:
        return _error(e, 400)
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


@router.get("/tasks/monitor")
def tasks_monitor_route():
    """All scheduled and background task status: Airflow pipelines, job_agent (:5050), and daemons."""
    from . import task_monitor
    try:
        return task_monitor.get_task_monitor_status()
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/tasks/schedule")
def tasks_schedule_route():
    """Today's and tomorrow's runs (Dublin days) of every scheduled task, in time order, with their state."""
    from . import task_schedule
    try:
        return {"status": "success", **task_schedule.schedule()}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/tasks/log")
def tasks_log_route(
    source: str = "airflow",
    target_id: str = "",
    run_id: str = "",
    task_id: str = "",
    lines: int = 150,
):
    """Tail live log output for an Airflow DAG/task or Job Agent or daemon."""
    from . import task_monitor
    try:
        return task_monitor.get_task_log(
            source=source,
            target_id=target_id,
            run_id=run_id or None,
            task_id=task_id or None,
            lines=lines,
        )
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.get("/catalysts")
def catalysts_feed(scope: str = "all", days: int = 30, symbols: str = "", bellwethers: str = "1", force: str = ""):
    """3-layer upcoming catalysts + per-stock breakdown for My Holdings & Watchlist."""
    from . import catalysts
    try:
        sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
        feed = catalysts.upcoming_feed(
            scope=scope,
            days=days,
            symbols=sym_list,
            include_bellwethers=bellwethers != "0",
            force=force == "1",
        )
        return {"status": "success", **feed, "recent_queries": catalysts.list_queries(12)}
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/catalysts/collect")
def catalysts_collect(payload: Dict[str, Any]):
    """Trigger background Layer 2 (RSS/SEC/Mongo LLM extraction) and/or Layer 3 (Claude web sweep)."""
    from . import catalysts
    try:
        layer = str(payload.get("layer") or "l2").lower()
        days = int(payload.get("days") or 45)
        return {"status": "success", **catalysts.trigger_collection(layer=layer, days=days)}
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/catalysts/query")
def catalysts_query(payload: Dict[str, Any]):
    """Interactive AI query over the 3-layer catalyst store + live web search."""
    from . import catalysts
    try:
        raw_syms = payload.get("symbols")
        sym_list = [str(s).strip().upper() for s in raw_syms if str(s).strip()] if isinstance(raw_syms, list) else None
        res = catalysts.query_catalysts(
            question=str(payload.get("question") or ""),
            scope=str(payload.get("scope") or "all"),
            days=int(payload.get("days") or 30),
            symbols=sym_list,
            mode=str(payload.get("mode") or "fast"),
            lang=str(payload.get("lang") or "zh"),
        )
        return {"status": "success", "result": res, "recent_queries": catalysts.list_queries(12)}
    except ValueError as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


@router.post("/catalysts/history/delete")
def catalysts_history_delete(payload: Dict[str, Any]):
    from . import catalysts
    try:
        qid = int(payload["id"]) if payload.get("id") is not None else None
        catalysts.delete_query(qid=qid, clear_all=bool(payload.get("clear_all")))
        return {"status": "success", "recent_queries": catalysts.list_queries(12)}
    except (ValueError, TypeError) as e:
        return _error(e, 400)
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)
