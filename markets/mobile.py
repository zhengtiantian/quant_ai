"""The portfolio on the phone (/m/): the desktop widget's card as a page for an iPhone home screen.

Read-only. The phone reaches it through `tailscale serve --http=80`, which proxies
http://<mac>.<tailnet>.ts.net/ to http://127.0.0.1:18000/m/ for the user's own tailnet devices
(WireGuard-encrypted end to end), so nothing else of the API is exposed. serve passes the caller's
tailnet address in X-Forwarded-For, which uvicorn trusts only from 127.0.0.1: so these routes answer
this Mac and tailnet addresses (100.64.0.0/10, fd7a:115c:a1e0::/48), and the LAN still sees nothing.
/m/state is /api/markets/portfolio/state cut to what the card shows: the holdings, and the global
markets (every exchange's indices and stocks: name, price, change, currency).
"""

import ipaddress
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse

from . import portfolio
from .api import LOOPBACK, _error

HERE = Path(__file__).resolve().parent
HOLDING_KEYS = ("symbol", "quantity", "avg_cost", "cost", "market_value", "unrealized", "unrealized_pct", "day_change_pct", "day_pnl")
QUOTE_KEYS = ("price", "session", "fetched_at")
MARKET_KEYS = ("name", "price", "change_pct", "currency")
EXCHANGE_KEYS = ("code", "name", "open", "local_time", "hours")

TAILNET = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))


def mine_only(request: Request) -> None:
    host = request.client.host if request.client else ""
    try:
        ok = host in LOOPBACK or any(ipaddress.ip_address(host) in n for n in TAILNET)
    except ValueError:
        ok = False
    if not ok:
        raise HTTPException(status_code=403, detail="only served to this Mac and its tailnet")


router = APIRouter(prefix="/m", dependencies=[Depends(mine_only)])


@router.get("", include_in_schema=False)
def root():
    return RedirectResponse("/m/")                      # the page's relative links need the slash


@router.get("/")
def page():
    return FileResponse(HERE / "mobile.html", headers={"Cache-Control": "no-cache"})


@router.get("/icon.png")
def icon():
    return FileResponse(HERE / "mobile_icon.png", media_type="image/png")


@router.get("/manifest.json")
def manifest():
    return {"name": "Portfolio", "short_name": "Portfolio", "start_url": "./", "display": "standalone",
            "background_color": "#0b0b0d", "theme_color": "#0b0b0d", "icons": [{"src": "icon.png", "sizes": "180x180", "type": "image/png"}]}


@router.get("/state")
def state():
    try:
        portfolio.init_db()
        s = portfolio.get_state(force_quotes=False, background_refresh=True)
        holdings = [{**{k: h.get(k) for k in HOLDING_KEYS}, "quote": {k: (h.get("quote") or {}).get(k) for k in QUOTE_KEYS}}
                    for h in s.get("holdings") or []]
        return {"status": "success", "holdings": holdings, "totals": s.get("totals"), "session": s.get("session"),
                "us_time": s.get("us_time"), "server_time": s.get("server_time"), "refresh_seconds": s.get("refresh_seconds"),
                "global": _global(s.get("global") or {})}
    except Exception as e:  # noqa: BLE001
        return _error(e, 500)


def _global(g):
    pick = lambda q: {k: q.get(k) for k in MARKET_KEYS}
    return {"refresh_seconds": g.get("refresh_seconds"), "regions": [
        {"name": r.get("name"), "countries": [
            {"code": c.get("code"), "name": c.get("name"), "flag": c.get("flag"), "open": c.get("open"), "exchanges": [
                {**{k: e.get(k) for k in EXCHANGE_KEYS}, "indices": [pick(q) for q in e.get("indices") or []],
                 "stocks": [pick(q) for q in e.get("stocks") or []]}
                for e in c.get("exchanges") or []]}
            for c in r.get("countries") or []]}
        for r in g.get("regions") or []]}
