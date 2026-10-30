"""Three-layer upcoming catalyst collector, per-stock impact mapper, and AI query engine.

Covers both My Holdings and Watchlist across three complementary layers:
  - Layer 1 (`L1_calendar`): Deterministic structured schedules — Nasdaq & Finnhub earnings
    (including ADR underlying listings 000660.KS, 2330.TW, ASML.AS) + options ATM-straddle
    implied moves, Nasdaq cash dividends & stock splits, Nasdaq economic calendar, official
    FOMC / BLS CPI & Nonfarm Payrolls / US policy dates, 13F deadlines, OpEx, and TSMC's
    monthly revenue release on the 10th.
  - Layer 2 (`L2_extracted`): Automated forward-event extraction from three free feeds per
    symbol — Google News forward-catalyst RSS, SEC EDGAR recent 8-K / 6-K filings, and the
    platform's MongoDB `news_articles` corpus — structured into dated events by the local LLM
    (LM Studio) with verified source URLs and stored in SQLite (`collected_catalysts`).
  - Layer 3 (`L3_web`): Daily / on-demand Claude CLI `WebSearch` sweep discovering upcoming
    conferences (e.g. Ignite, SC, AWS re:Invent, CES, GTC), product launches, regulatory /
    tariff / antitrust deadlines, and Wall Street catalyst previews into `collected_catalysts`.

Every stock in Holdings and Watchlist receives its own aggregated view (`per_stock`) combining:
  1. Direct company events (L1 + L2 + L3)
  2. Supply-chain & peer spillover events (e.g. NVDA/AVGO/MU earnings -> TSM/SKHY/INTC/STX)
  3. High-impact macro & policy events.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from datetime import time as dtime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from . import advisor, events
from . import portfolio as pf
from .config import DB_PATH

NEW_YORK = ZoneInfo("America/New_York")
DUBLIN = ZoneInfo("Europe/Dublin")

IMPLIED_TTL = 2 * 3600
RSS_TTL = 2 * 3600
SEC_TICKERS_TTL = 7 * 86400
SEC_FILINGS_TTL = 4 * 3600
L2_INTERVAL = 6 * 3600          # refresh Layer 2 extraction every 6 hours
L3_INTERVAL = 20 * 3600         # refresh Layer 3 web sweep once per weekday (~20h cooldown)
L3_AT_DUBLIN = dtime(12, 30)

PURE_ETFS = {"SPY", "QQQ", "VOO", "IVV", "IWM", "DIA", "SMH", "SOXX", "XLK", "XLF", "XLE", "XLV"}

SEC_HEADERS = {
    "User-Agent": "QuantTradePortfolio/1.0 (personal-research@quant.local)",
    "Accept": "application/json",
}
RSS_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
}

SEC_ITEM_LABELS = {
    "1.01": "Material Definitive Agreement (重大协议)",
    "1.02": "Termination of Material Agreement",
    "2.01": "Acquisition or Disposition of Assets (资产并购/处置)",
    "2.02": "Results of Operations & Financial Condition",
    "2.05": "Restructuring / Cost Exit Activities (重组/裁员计划)",
    "3.01": "Listing / Delisting Notice",
    "5.02": "Executive / Board Change (高管或董事变动)",
    "7.01": "Regulation FD Disclosure / Investor Update (投资者披露/指引)",
    "8.01": "Other Material Events (重大事项公告)",
}

# Authoritative US macro, central bank & market-structure schedule (covers full 30-120d horizon).
OFFICIAL_MACRO_SCHEDULE: List[Dict[str, Any]] = [
    {"date": "2026-11-03", "country": "United States", "gmt": "All day",
     "event": "US Midterm Elections",
     "note_zh": "美国中期选举：国会席位决定关税政策、CHIPS半导体补贴、企业税与科技反垄断走向",
     "note_en": "US Midterm Elections: Congressional control shapes tariffs, CHIPS Act, tax & tech regulation"},
    {"date": "2026-11-06", "country": "United States", "gmt": "13:30",
     "event": "US Nonfarm Payrolls & Unemployment Rate (Oct)",
     "note_zh": "美国10月非农就业报告（08:30 ET），决定美联储12月降息路径与美债收益率",
     "note_en": "BLS Oct Employment Situation (08:30 ET), key input for Dec FOMC rate path & yields"},
    {"date": "2026-11-10", "country": "United States", "gmt": "13:30",
     "event": "US CPI Inflation Report (Oct)",
     "note_zh": "美国10月CPI通胀数据（08:30 ET），直接影响纳指与成长股估值",
     "note_en": "BLS Oct Consumer Price Index (08:30 ET), major driver for rates and tech multiples"},
    {"date": "2026-11-16", "country": "United States", "gmt": "21:00",
     "event": "SEC 13F Institutional Holdings Deadline (Q3)",
     "note_zh": "华尔街大型机构 Q3 13F 持仓披露截止日，关注巴菲特/顶级对冲基金对科技与芯片股调仓",
     "note_en": "Q3 13F institutional holdings filing deadline; reveals major fund positioning changes"},
    {"date": "2026-11-20", "country": "United States", "gmt": "21:00",
     "event": "US Monthly Options Expiration (OpEx)",
     "note_zh": "美股11月月度期权交割日（特别是 NVDA 财报当周 Gamma 敞口释放）",
     "note_en": "Monthly US equity & index options expiration (post-NVDA earnings week gamma reset)"},
    {"date": "2026-12-04", "country": "United States", "gmt": "13:30",
     "event": "US Nonfarm Payrolls & Unemployment Rate (Nov)",
     "note_zh": "美国11月非农就业数据（12月FOMC议息前最后一份就业报告）",
     "note_en": "BLS Nov Employment Situation (final jobs report ahead of Dec FOMC)"},
    {"date": "2026-12-09", "country": "United States", "gmt": "19:00",
     "event": "FOMC Rate Decision & SEP Dot Plot",
     "note_zh": "美联储12月利率决议 + 经济预测摘要（点阵图）+ 鲍威尔发布会",
     "note_en": "Fed rate decision, Summary of Economic Projections (dot plot) & Chair press conference"},
    {"date": "2026-12-10", "country": "United States", "gmt": "13:30",
     "event": "US CPI Inflation Report (Nov)",
     "note_zh": "美国11月CPI通胀报告（08:30 ET）",
     "note_en": "BLS Nov Consumer Price Index release (08:30 ET)"},
    {"date": "2026-12-18", "country": "United States", "gmt": "21:00",
     "event": "US Quadruple Witching (Quarterly OpEx)",
     "note_zh": "美股四巫日（股指期货/股指期权/个股期权季度集中交割，成交量与波动放大）",
     "note_en": "Quarterly index & equity options/futures expiration; elevated volume and rebalancing"},
    {"date": "2027-01-08", "country": "United States", "gmt": "13:30",
     "event": "US Nonfarm Payrolls & Unemployment Rate (Dec)",
     "note_zh": "美国12月非农就业报告（08:30 ET）",
     "note_en": "BLS Dec Employment Situation release (08:30 ET)"},
    {"date": "2027-01-13", "country": "United States", "gmt": "13:30",
     "event": "US CPI Inflation Report (Dec)",
     "note_zh": "美国12月CPI通胀报告（08:30 ET）",
     "note_en": "BLS Dec Consumer Price Index release (08:30 ET)"},
    {"date": "2027-01-27", "country": "United States", "gmt": "19:00",
     "event": "FOMC Interest Rate Decision",
     "note_zh": "美联储2027年首次利率决议与政策声明",
     "note_en": "First FOMC rate decision and policy statement of 2027"},
    {"date": "2027-02-16", "country": "United States", "gmt": "21:00",
     "event": "SEC 13F Institutional Holdings Deadline (Q4)",
     "note_zh": "机构 Q4 13F 持仓报告披露截止日",
     "note_en": "Q4 13F institutional holdings filing deadline"},
    {"date": "2027-03-17", "country": "United States", "gmt": "18:00",
     "event": "FOMC Rate Decision & SEP Dot Plot",
     "note_zh": "美联储3月利率决议 + 点阵图",
     "note_en": "March FOMC rate decision and SEP dot plot"},
]

# Thematic / supply-chain groups used to map spillover events across ALL Holdings and Watchlist stocks.
SECTOR_GROUPS: Dict[str, Tuple[str, str, Set[str]]] = {
    "ai_chips": (
        "AI 芯片与算力链", "AI chips & accelerators",
        {"NVDA", "AMD", "AVGO", "TSM", "SKHY", "MU", "INTC", "SMCI", "ARM", "MRVL"},
    ),
    "foundry_equip": (
        "晶圆代工与设备链", "Foundry & chip equipment",
        {"TSM", "INTC", "ASML", "LRCX", "AMAT", "KLAC"},
    ),
    "memory_storage": (
        "HBM / 存储与数据盘", "Memory (HBM/DRAM) & storage",
        {"SKHY", "MU", "STX", "WDC", "NVDA", "TSM"},
    ),
    "hyperscale_capex": (
        "云厂商 AI Capex 链", "Hyperscaler AI capex & cloud",
        {"GOOGL", "MSFT", "AMZN", "META", "ORCL", "NVDA", "AVGO", "TSM", "SKHY", "STX", "INTC"},
    ),
    "consumer_silicon": (
        "终端硬件与芯片链", "Consumer hardware & silicon",
        {"AAPL", "TSLA", "TSM", "INTC", "GOOGL", "QCOM"},
    ),
    "enterprise_ai": (
        "企业级 AI 与软件", "Enterprise AI & cloud software",
        {"PLTR", "ORCL", "MSFT", "GOOGL", "AMZN", "META"},
    ),
    "crypto_risk": (
        "加密资产与高Beta风偏", "Crypto & high-beta liquidity",
        {"COIN", "TSLA", "PLTR", "QQQ"},
    ),
    "broad_etf": (
        "纳指/大盘权重传导", "Index & Nasdaq heavyweight",
        {"QQQ", "SPY", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA"},
    ),
}

FORWARD_HINT_RE = re.compile(
    r"\b(upcoming|schedule[ds]?|expect(?:s|ed)?|set to|plan(?:s|ning)? to|will |ahead of|preview|"
    r"conference|keynote|summit|investor day|analyst day|gtc|ces|computex|ignite|re:invent|reinvent|"
    r"launch|unveil|rollout|mass production|shipment|wafer|hbm[0-9e]*|"
    r"tariff|export (?:ban|control|curb|rule)|sanction|chips act|subsid|"
    r"antitrust|doj|ftc|trial|hearing|ruling|deadline|approval|"
    r"buyback|repurchase|acqui|merger|spin-?off|guidance|outlook|"
    r"nov(?:ember)?|dec(?:ember)?|jan(?:uary)?|feb(?:ruary)?|q4|q1|2027|next week|next month)\b",
    re.I,
)

COUNTRY_FLAG = {
    "United States": "🇺🇸", "Euro Zone": "🇪🇺", "China": "🇨🇳", "Japan": "🇯🇵",
    "United Kingdom": "🇬🇧", "Germany": "🇩🇪", "South Korea": "🇰🇷", "Taiwan": "🇹🇼",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS collected_catalysts (
    id INTEGER PRIMARY KEY,
    dedupe_key TEXT NOT NULL UNIQUE,
    layer TEXT NOT NULL,                 -- 'L2_extracted' | 'L3_web'
    kind TEXT NOT NULL,                  -- 'product' | 'policy' | 'legal' | 'corporate' | 'analyst' | 'macro'
    symbol TEXT NOT NULL,                -- primary ticker or 'MACRO'
    affected_symbols TEXT NOT NULL,      -- JSON array of tickers affected
    event_date TEXT NOT NULL,            -- YYYY-MM-DD (exact or window start)
    date_precision TEXT NOT NULL DEFAULT 'exact',  -- 'exact' | 'approx'
    date_label TEXT,                     -- e.g. 'Mid-Nov 2026' when approx
    impact_level TEXT NOT NULL DEFAULT 'medium',
    title_zh TEXT NOT NULL,
    title_en TEXT NOT NULL,
    detail_zh TEXT NOT NULL,
    detail_en TEXT NOT NULL,
    source_title TEXT,
    source_url TEXT,
    collected_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_col_cat_date ON collected_catalysts(event_date);

CREATE TABLE IF NOT EXISTS catalyst_runs (
    layer TEXT PRIMARY KEY,
    last_started_at REAL,
    last_finished_at REAL,
    status TEXT,
    items_saved INTEGER DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS catalyst_queries (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    question TEXT NOT NULL,
    scope TEXT NOT NULL,
    days INTEGER NOT NULL,
    symbols TEXT,
    mode TEXT NOT NULL,
    lang TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT NOT NULL,
    model TEXT,
    elapsed_s REAL
);
CREATE INDEX IF NOT EXISTS idx_cat_q_created ON catalyst_queries(created_at DESC);
"""

_collect_lock = threading.Lock()
_collecting: Dict[str, bool] = {"l2": False, "l3": False}


def _conn(path: Optional[Path] = None) -> sqlite3.Connection:
    pf.init_db(path)
    conn = sqlite3.connect(str(path or DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _us_today() -> date:
    return datetime.now(NEW_YORK).date()


def _related_symbols(symbol: str, pool: Set[str]) -> Tuple[List[str], List[str], List[str]]:
    """Return (affected_pool_symbols, chain_labels_zh, chain_labels_en) for `symbol`."""
    rel: Set[str] = set()
    zh_labels: List[str] = []
    en_labels: List[str] = []
    for _, (l_zh, l_en, members) in SECTOR_GROUPS.items():
        if symbol in members:
            overlap = (members & pool) - {symbol}
            if overlap:
                rel |= overlap
                if l_zh not in zh_labels:
                    zh_labels.append(l_zh)
                    en_labels.append(l_en)
    return sorted(rel), zh_labels, en_labels


def _fmt_rev(val: Optional[float], currency: str = "USD") -> Optional[str]:
    if not val or val <= 0:
        return None
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "KRW": "₩", "TWD": "NT$", "HKD": "HK$"}.get(currency, f"{currency} ")
    if currency == "KRW" and val >= 1e12:
        return f"{sym}{val / 1e12:.2f}T"
    if currency == "TWD" and val >= 1e9:
        return f"{sym}{val / 1e9:.1f}B"
    if val >= 1e9:
        return f"{sym}{val / 1e9:.2f}B"
    if val >= 1e6:
        return f"{sym}{val / 1e6:.0f}M"
    return f"{sym}{val:,.0f}"


def _fmt_eps(val: Optional[float], currency: str = "USD") -> Optional[str]:
    if val is None:
        return None
    sym = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "KRW": "₩", "TWD": "NT$", "HKD": "HK$"}.get(currency, f"{currency} ")
    if currency == "KRW" and abs(val) >= 100:
        return f"{sym}{val:,.0f}"
    return f"{sym}{val:.2f}"


def _when_zh(when: Optional[str]) -> str:
    w = (when or "").lower()
    if "seoul" in w:
        return "美股盘后 / 首尔早盘"
    if "before" in w:
        return "盘前 (BMO)"
    if "after" in w:
        return "盘后 (AMC)"
    return "时间待定"


def _when_en(when: Optional[str]) -> str:
    w = (when or "").strip()
    if not w or w == "time not given":
        return "Time TBD"
    return w[0].upper() + w[1:]


def _cached_implied_move(symbol: str, on_date: date, price: Optional[float], path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    if not price or price <= 0 or symbol in PURE_ETFS:
        return None

    def fetch():
        return events.implied_move(symbol, on_date, price)

    try:
        return events._cache(path, f"implied:{symbol}:{on_date.isoformat()}", fetch, ttl=IMPLIED_TTL)
    except Exception:  # noqa: BLE001
        return None


def _norm_econ_name(name: str) -> str:
    s = re.sub(r"\s*\((?:MoM|YoY|QoQ|Prelim|Final|SA|NSA|Oct|Nov|Dec|Jan|Feb|Q[1-4]).*?\)", "", name or "", flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def _tsm_monthly_revenue_dates(start: date, end: date) -> List[date]:
    """TSMC reports monthly consolidated revenue on the 10th of each month (or next weekday if weekend)."""
    out: List[date] = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        d = date(y, m, 10)
        while d.weekday() >= 5:
            d += timedelta(days=1)
        if start <= d <= end:
            out.append(d)
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


# ---------------------------------------------------------------- Layer 2 Feeds: Google News RSS + SEC EDGAR + Mongo

def fetch_google_news_rss(query: str, max_items: int = 8, path: Optional[Path] = None, cache_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch forward-looking articles from Google News RSS (free, no key)."""
    def fetch():
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers=RSS_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        rows = []
        for it in root.findall("./channel/item")[:max_items]:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            src_el = it.find("source")
            src_name = (src_el.text.strip() if src_el is not None and src_el.text else "") or "Google News"
            pub_iso = ""
            if pub:
                try:
                    pub_iso = parsedate_to_datetime(pub).strftime("%Y-%m-%d")
                except Exception:  # noqa: BLE001
                    pub_iso = pub[:16]
            if title and link:
                rows.append({"title": title, "url": link, "date": pub_iso, "source": src_name})
        return rows

    if cache_key:
        try:
            return events._cache(path, cache_key, fetch, ttl=RSS_TTL)
        except Exception:  # noqa: BLE001
            return []
    try:
        return fetch()
    except Exception:  # noqa: BLE001
        return []


def _sec_cik_map(path: Optional[Path] = None) -> Dict[str, str]:
    def fetch():
        req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            str(v.get("ticker") or "").upper(): f"{int(v['cik_str']):010d}"
            for v in (data or {}).values()
            if v.get("ticker") and v.get("cik_str") is not None
        }

    try:
        return events._cache(path, "sec:company_tickers", fetch, ttl=SEC_TICKERS_TTL)
    except Exception:  # noqa: BLE001
        return {}


def fetch_sec_recent_filings(symbol: str, cik_map: Dict[str, str], since_days: int = 14, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Recent SEC 8-K / 6-K material event filings for `symbol`."""
    sym = symbol.upper()
    cik = cik_map.get(sym)
    if not cik or sym in PURE_ETFS:
        return []

    def fetch():
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        recent = (data.get("filings") or {}).get("recent") or {}
        forms = recent.get("form") or []
        dates = recent.get("filingDate") or []
        accs = recent.get("accessionNumber") or []
        docs = recent.get("primaryDocument") or []
        items_list = recent.get("items") or []
        out = []
        for i in range(min(30, len(forms))):
            f = forms[i]
            if f not in ("8-K", "6-K"):
                continue
            fdate = dates[i] if i < len(dates) else ""
            raw_items = str(items_list[i] if i < len(items_list) else "")
            item_codes = [c.strip() for c in raw_items.split(",") if c.strip()]
            # Skip pure earnings release 2.02-only 8-Ks when no forward corporate item is attached
            if item_codes and set(item_codes) <= {"2.02", "9.01"}:
                continue
            acc_clean = (accs[i] if i < len(accs) else "").replace("-", "")
            doc_name = docs[i] if i < len(docs) else ""
            cik_int = str(int(cik))
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{doc_name}"
                if acc_clean and doc_name
                else f"https://www.sec.gov/edgar/browse/?CIK={cik}"
            )
            labels = [SEC_ITEM_LABELS.get(c, f"Item {c}") for c in item_codes if c != "9.01"]
            desc = "; ".join(labels) if labels else ("Foreign Issuer Report (6-K)" if f == "6-K" else "Current Report (8-K)")
            out.append({
                "symbol": sym,
                "form": f,
                "date": fdate,
                "items": item_codes,
                "title": f"{sym} SEC {f} Filing: {desc}",
                "url": filing_url,
                "source": "SEC EDGAR",
            })
            if len(out) >= 3:
                break
        return out

    try:
        rows = events._cache(path, f"sec:filings:{sym}", fetch, ttl=SEC_FILINGS_TTL)
    except Exception:  # noqa: BLE001
        return []
    cutoff = (_us_today() - timedelta(days=since_days)).isoformat()
    return [r for r in rows if (r.get("date") or "") >= cutoff]


def _get_llm(temperature: float = 0.15):
    for mod_name in ("main", "__main__"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "get_chat_llm"):
            return mod.get_chat_llm(temperature=temperature)
    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.insert(0, parent)
    import main as main_mod
    return main_mod.get_chat_llm(temperature=temperature)


def _record_run(layer: str, status: str, started: float, saved: int = 0, error: Optional[str] = None, path: Optional[Path] = None) -> None:
    conn = _conn(path)
    try:
        conn.execute(
            "INSERT OR REPLACE INTO catalyst_runs(layer, last_started_at, last_finished_at, status, items_saved, error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (layer, started, time.time() if status != "running" else None, status, saved, error),
        )
        conn.commit()
    finally:
        conn.close()


def _upsert_collected(rows: List[Dict[str, Any]], layer: str, today: date, path: Optional[Path] = None) -> int:
    conn = _conn(path)
    saved = 0
    now = time.time()
    try:
        # Prune stale events already past today - 1 day
        conn.execute("DELETE FROM collected_catalysts WHERE event_date < ?", ((today - timedelta(days=1)).isoformat(),))
        for r in rows:
            conn.execute(
                "INSERT OR REPLACE INTO collected_catalysts("
                "dedupe_key, layer, kind, symbol, affected_symbols, event_date, date_precision, date_label, "
                "impact_level, title_zh, title_en, detail_zh, detail_en, source_title, source_url, collected_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["dedupe_key"], layer, r["kind"], r["symbol"],
                    json.dumps(r.get("affected_symbols") or []),
                    r["event_date"], r.get("date_precision") or "exact", r.get("date_label") or "",
                    r.get("impact_level") or "medium",
                    r["title_zh"], r["title_en"], r["detail_zh"], r["detail_en"],
                    r.get("source_title") or "", r.get("source_url") or "", now,
                ),
            )
            saved += 1
        conn.commit()
    finally:
        conn.close()
    return saved


def collect_l2_extracted(days: int = 45, path: Optional[Path] = None, today: Optional[date] = None) -> int:
    """Layer 2: Pull Google News forward RSS + SEC 8-K/6-K + MongoDB news for every Holding & Watchlist
    symbol, and use the local LLM to extract structured upcoming catalysts into `collected_catalysts`."""
    started = time.time()
    _record_run("L2_extracted", "running", started, path=path)
    start_d = today or _us_today()
    end_d = start_d + timedelta(days=days)

    try:
        st = pf.get_state(path=path, cached_quotes=True)
        held_syms = [h["symbol"] for h in advisor.followed(st)]
        watch_syms = [w["symbol"] for w in st.get("watchlist") or [] if w["symbol"] not in held_syms]
        all_syms = held_syms + watch_syms
        if not all_syms:
            _record_run("L2_extracted", "done", started, 0, path=path)
            return 0

        names: Dict[str, str] = {}
        for row in (st.get("holdings") or []) + (st.get("watchlist") or []):
            q = row.get("quote") or {}
            if q.get("name"):
                names[row["symbol"]] = re.sub(r"\s+(Common Stock|Class [A-C].*|Ordinary Shares.*)$", "", q["name"], flags=re.I)

        stock_syms = [s for s in all_syms if s not in PURE_ETFS]
        cik_map = _sec_cik_map(path)

        # 1. Gather Google News forward RSS + SEC filings in parallel
        def _pull_rss(sym: str) -> Tuple[str, List[Dict[str, Any]]]:
            co = names.get(sym, "")
            term = f'"{sym}" OR "{co}"' if co and len(co) >= 3 else f'"{sym}"'
            q = f'({term}) (upcoming OR conference OR keynote OR launch OR "investor day" OR tariff OR "export control" OR antitrust OR ruling OR guidance) when:7d'
            return sym, fetch_google_news_rss(q, max_items=6, path=path, cache_key=f"gnrss:fwd:{sym}")

        def _pull_sec(sym: str) -> Tuple[str, List[Dict[str, Any]]]:
            return sym, fetch_sec_recent_filings(sym, cik_map, since_days=14, path=path)

        with ThreadPoolExecutor(max_workers=6) as pool:
            rss_map = dict(pool.map(_pull_rss, stock_syms))
        with ThreadPoolExecutor(max_workers=4) as pool:
            sec_map = dict(pool.map(_pull_sec, stock_syms))

        # 2. Gather recent MongoDB news_articles with forward hints
        mongo_map: Dict[str, List[Dict[str, Any]]] = {}
        try:
            mongo_map = advisor.news(stock_syms, days=7, per_symbol=6)
        except Exception:  # noqa: BLE001
            mongo_map = {}

        parent = str(Path(__file__).resolve().parents[1])
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            from injection_guard import screen
        except Exception:  # noqa: BLE001
            screen = lambda s: type("S", (), {"text": s})()

        candidates: List[Dict[str, Any]] = []
        seen_titles: Set[str] = set()
        for sym in stock_syms:
            per_sym = 0
            for item in (rss_map.get(sym) or []) + (mongo_map.get(sym) or []) + (sec_map.get(sym) or []):
                title = screen((item.get("title") or "").strip()).text
                t_key = f"{sym}:{title[:60].lower()}"
                if not title or t_key in seen_titles:
                    continue
                if item.get("source") != "SEC EDGAR" and not FORWARD_HINT_RE.search(title):
                    continue
                seen_titles.add(t_key)
                candidates.append({
                    "symbol": sym,
                    "published": (item.get("date") or item.get("time_utc") or "")[:10],
                    "title": title,
                    "source": item.get("source") or "News",
                    "url": item.get("url") or "",
                })
                per_sym += 1
                if per_sym >= 5:
                    break

        if not candidates:
            _record_run("L2_extracted", "done", started, 0, path=path)
            return 0

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        llm = _get_llm(temperature=0.1)
        model_name = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "")

        valid_universe = set(all_syms) | {"MACRO"}
        extracted_rows: List[Dict[str, Any]] = []

        # Process in compact batches of 18 headlines per call so the local 9B model stays crisp and fast.
        batch_size = 18
        for b_start in range(0, min(len(candidates), 90), batch_size):
            batch = candidates[b_start: b_start + batch_size]
            lines = [
                f"[{i}] symbol={c['symbol']} | pub={c['published']} | src={c['source']} | {c['title']}"
                for i, c in enumerate(batch, 1)
            ]
            sys_prompt = f"""You extract UPCOMING stock catalysts & material corporate/policy events from financial headlines & SEC filings.
Today is {start_d.isoformat()}. Target window: {start_d.isoformat()} to {end_d.isoformat()}.
Valid portfolio symbols: {', '.join(all_syms)}.

Rules:
1. Extract ONLY events that (a) happen in the future within [{start_d.isoformat()}, {end_d.isoformat()}] (conferences, product launches, keynotes, investor days, court/antitrust hearings, tariff/export-control deadlines, scheduled announcements), OR (b) are newly filed SEC 8-K/6-K corporate actions or major forward guidance/policy shifts published in the last 7 days with ongoing forward impact.
2. SKIP routine past price commentary ("Why X rose 3% today", "Top stocks to buy") AND skip plain "X sets Q3 earnings call date" announcements (scheduled earnings dates are already in Layer 1).
3. If an event has an exact date in [{start_d.isoformat()}, {end_d.isoformat()}], set `event_date` = "YYYY-MM-DD" and `date_precision` = "exact". If only a month/window is known (e.g. "mid-November", "December"), set `event_date` to an estimated YYYY-MM-DD inside [{start_d.isoformat()}, {end_d.isoformat()}], `date_precision` = "approx", and `date_label` (e.g. "2026-11 中旬 / Mid-Nov"). For an active recent SEC 8-K / policy catalyst with immediate forward impact, use `{start_d.isoformat()}` and `date_precision` = "approx", `date_label` = "近期催化 / Active catalyst".
4. Deduplicate: emit at most ONE object per distinct real-world event.
5. Output ONLY a valid JSON array (or `[]` if none qualify):
[
  {{
    "source_id": <int 1..{len(batch)}>,
    "symbol": "<primary ticker from valid symbols>",
    "affected_symbols": ["<other affected tickers from valid symbols>"],
    "kind": "product|policy|legal|corporate|analyst",
    "event_date": "YYYY-MM-DD",
    "date_precision": "exact|approx",
    "date_label": "<optional label>",
    "impact_level": "high|medium",
    "title_zh": "<concise Chinese event title, <= 32 chars>",
    "title_en": "<concise English event title, <= 12 words>",
    "detail_zh": "<1 sentence in Chinese on how/why it impacts the stock>",
    "detail_en": "<1 sentence in English on how/why it impacts the stock>"
  }}
]"""
            msgs = [SystemMessage(content=sys_prompt), HumanMessage(content="CANDIDATES:\n" + "\n".join(lines))]
            if "qwen" in model_name.lower():
                msgs.append(AIMessage(content="<think>\n\n</think>\n\n"))
            raw_out = (llm.invoke(msgs).content or "").strip()
            raw_out = re.sub(r"^<think>.*?</think>\s*", "", raw_out, flags=re.S)
            s_idx, e_idx = raw_out.find("["), raw_out.rfind("]")
            if s_idx < 0 or e_idx <= s_idx:
                continue
            try:
                parsed = json.loads(raw_out[s_idx: e_idx + 1])
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list):
                continue

            for obj in parsed:
                if not isinstance(obj, dict):
                    continue
                try:
                    sid = int(obj.get("source_id") or 0)
                except (TypeError, ValueError):
                    sid = 0
                if not (1 <= sid <= len(batch)):
                    continue
                src_item = batch[sid - 1]
                sym = str(obj.get("symbol") or src_item["symbol"]).upper().strip()
                if sym not in valid_universe:
                    sym = src_item["symbol"]
                ev_date_str = str(obj.get("event_date") or "")[:10]
                try:
                    ev_d = date.fromisoformat(ev_date_str)
                except ValueError:
                    ev_d = start_d
                    ev_date_str = start_d.isoformat()
                if ev_d < start_d:
                    ev_date_str = start_d.isoformat()
                elif ev_d > end_d:
                    continue
                kind = str(obj.get("kind") or "corporate").lower()
                if kind not in ("product", "policy", "legal", "corporate", "analyst", "macro"):
                    kind = "corporate"
                t_zh = str(obj.get("title_zh") or "").strip()
                t_en = str(obj.get("title_en") or "").strip()
                if not t_zh and not t_en:
                    continue
                t_zh = t_zh or t_en
                t_en = t_en or t_zh
                aff = [s.upper() for s in (obj.get("affected_symbols") or []) if isinstance(s, str) and s.upper() in valid_universe]
                if sym != "MACRO" and sym not in aff:
                    aff = [sym] + aff
                topic_sig = hashlib.sha1(f"L2|{sym}|{kind}|{ev_date_str[:7]}|{t_en[:36].lower()}".encode()).hexdigest()[:16]
                extracted_rows.append({
                    "dedupe_key": f"L2:{sym}:{topic_sig}",
                    "kind": kind,
                    "symbol": sym,
                    "affected_symbols": aff,
                    "event_date": ev_date_str,
                    "date_precision": "approx" if obj.get("date_precision") == "approx" else "exact",
                    "date_label": str(obj.get("date_label") or "")[:40],
                    "impact_level": "high" if obj.get("impact_level") == "high" else "medium",
                    "title_zh": t_zh[:80],
                    "title_en": t_en[:100],
                    "detail_zh": str(obj.get("detail_zh") or src_item["title"])[:220],
                    "detail_en": str(obj.get("detail_en") or src_item["title"])[:240],
                    "source_title": f"{src_item['source']}: {src_item['title'][:80]}",
                    "source_url": src_item["url"],
                })

        saved = _upsert_collected(extracted_rows, "L2_extracted", start_d, path=path)
        _record_run("L2_extracted", "done", started, saved, path=path)
        return saved
    except Exception as e:  # noqa: BLE001
        _record_run("L2_extracted", "failed", started, 0, f"{type(e).__name__}: {e}"[:500], path=path)
        raise



def _collect_l3_thematic_web_fallback(
    all_syms: List[str],
    start_d: date,
    end_d: date,
    path: Optional[Path] = None,
) -> int:
    """When Claude CLI hits weekly quota or is absent, run 8 cross-market thematic web discovery
    queries (conferences, semiconductor nodes/HBM, tariffs/export controls, antitrust trials,
    product roadmaps) and extract structured L3_web catalysts with local LLM."""
    thematic_queries = [
        '("Microsoft Ignite" OR "AWS re:Invent" OR "SC26" OR "CES 2027" OR "NeurIPS") (AI OR chip OR keynote OR November OR December)',
        '("TSMC" OR "SK Hynix" OR "Micron" OR "ASML") ("HBM4" OR "2nm" OR "CoWoS" OR "Arizona" OR "investor day" OR conference) when:14d',
        '("export control" OR "tariff" OR "CHIPS Act" OR "BIS") ("TSMC" OR "NVIDIA" OR "Intel" OR "SK Hynix" OR "AMD" OR "ASML") when:14d',
        '("antitrust" OR "DOJ" OR "remedies" OR "trial" OR "ruling" OR "EU") ("Google" OR "Apple" OR "Meta" OR "Microsoft" OR "Amazon" OR "NVIDIA") when:14d',
        '("Seagate" OR "HAMR" OR "data center storage" OR "HBM3E") (shipment OR qualification OR launch OR capacity OR conference) when:14d',
        '("Intel 18A" OR "Panther Lake" OR "Blackwell" OR "Rubin" OR "MI350" OR "TPU") (launch OR shipment OR keynote OR event) when:14d',
        '("Tesla" OR "Robotaxi" OR "FSD" OR "Palantir" OR "Coinbase" OR "Oracle") (event OR regulatory OR deadline OR conference OR launch) when:14d',
        '("US election" OR "Federal Reserve" OR "FOMC" OR "tariff") ("semiconductors" OR "Big Tech" OR "Nasdaq") when:7d',
    ]

    def _pull_theme(idx_q: Tuple[int, str]) -> List[Dict[str, Any]]:
        idx, q = idx_q
        return fetch_google_news_rss(q, max_items=7, path=path, cache_key=f"gnrss:l3theme:{idx}")

    with ThreadPoolExecutor(max_workers=6) as pool:
        nested = list(pool.map(_pull_theme, enumerate(thematic_queries)))

    parent = str(Path(__file__).resolve().parents[1])
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        from injection_guard import screen
    except Exception:  # noqa: BLE001
        screen = lambda s: type("S", (), {"text": s})()

    items: List[Dict[str, Any]] = []
    seen_u: Set[str] = set()
    for sub_l in nested:
        for it in sub_l:
            u = it.get("url") or ""
            t = screen((it.get("title") or "").strip()).text
            if not t or not u or u in seen_u:
                continue
            seen_u.add(u)
            items.append({"published": (it.get("date") or "")[:10], "title": t, "source": it.get("source") or "Web", "url": u})

    if not items:
        return 0

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    llm = _get_llm(temperature=0.1)
    model_name = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "")
    valid_universe = set(all_syms) | {"MACRO"}
    rows: List[Dict[str, Any]] = []

    for b_start in range(0, min(len(items), 48), 16):
        batch = items[b_start: b_start + 16]
        lines = [f"[{i}] pub={x['published']} | src={x['source']} | {x['title']}" for i, x in enumerate(batch, 1)]
        sys_p = f"""You are a cross-market equity catalyst scout building Layer 3 (Web Thematic Discovery).
Today is {start_d.isoformat()}. Window: {start_d.isoformat()} to {end_d.isoformat()}.
Portfolio & Watchlist tickers: {', '.join(all_syms)} (or use "MACRO" for market-wide policy/macro catalysts).

Extract ONLY genuine upcoming or active forward-looking catalysts (industry conferences like Ignite/SC26/re:Invent/CES, product/node/HBM launches, export-control/tariff rules, antitrust/court rulings, major capex/supply-chain catalysts).
SKIP plain quarterly earnings date announcements and routine stock price recaps.
Output ONLY a JSON array:
[
  {{
    "source_id": <int 1..{len(batch)}>,
    "symbol": "<primary ticker from list or MACRO>",
    "affected_symbols": ["<all impacted tickers from list>"],
    "kind": "product|policy|legal|corporate|macro",
    "event_date": "YYYY-MM-DD",
    "date_precision": "exact|approx",
    "date_label": "<optional window label>",
    "impact_level": "high|medium",
    "title_zh": "<concise Chinese title <= 34 chars>",
    "title_en": "<concise English title <= 14 words>",
    "detail_zh": "<1 sentence Chinese explanation of stock/supply-chain impact>",
    "detail_en": "<1 sentence English explanation of stock/supply-chain impact>"
  }}
]"""
        msgs = [SystemMessage(content=sys_p), HumanMessage(content="WEB_RESULTS:\n" + "\n".join(lines))]
        if "qwen" in model_name.lower():
            msgs.append(AIMessage(content="<think>\n\n</think>\n\n"))
        raw_out = re.sub(r"^<think>.*?</think>\s*", "", (llm.invoke(msgs).content or "").strip(), flags=re.S)
        s_i, e_i = raw_out.find("["), raw_out.rfind("]")
        if s_i < 0 or e_i <= s_i:
            continue
        try:
            parsed = json.loads(raw_out[s_i: e_i + 1])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            try:
                sid = int(obj.get("source_id") or 0)
            except (TypeError, ValueError):
                sid = 0
            if not (1 <= sid <= len(batch)):
                continue
            src = batch[sid - 1]
            sym = str(obj.get("symbol") or "").upper().strip()
            if sym not in valid_universe:
                continue
            ev_date_str = str(obj.get("event_date") or "")[:10]
            try:
                ev_d = date.fromisoformat(ev_date_str)
            except ValueError:
                ev_d = start_d
                ev_date_str = start_d.isoformat()
            if ev_d < start_d:
                ev_date_str = start_d.isoformat()
            elif ev_d > end_d + timedelta(days=15):
                continue
            kind = str(obj.get("kind") or "corporate").lower()
            if kind not in ("product", "policy", "legal", "corporate", "analyst", "macro"):
                kind = "corporate"
            t_zh = str(obj.get("title_zh") or "").strip()
            t_en = str(obj.get("title_en") or "").strip()
            if not (t_zh or t_en):
                continue
            aff = [s.upper() for s in (obj.get("affected_symbols") or []) if isinstance(s, str) and s.upper() in valid_universe]
            if sym != "MACRO" and sym not in aff:
                aff = [sym] + aff
            sig = hashlib.sha1(f"L3|{sym}|{kind}|{ev_date_str[:7]}|{(t_en or t_zh)[:36].lower()}".encode()).hexdigest()[:16]
            rows.append({
                "dedupe_key": f"L3:{sym}:{sig}",
                "kind": kind,
                "symbol": sym,
                "affected_symbols": aff,
                "event_date": ev_date_str,
                "date_precision": "approx" if obj.get("date_precision") == "approx" else "exact",
                "date_label": str(obj.get("date_label") or "")[:40],
                "impact_level": "high" if obj.get("impact_level") == "high" else "medium",
                "title_zh": (t_zh or t_en)[:80],
                "title_en": (t_en or t_zh)[:100],
                "detail_zh": str(obj.get("detail_zh") or src["title"])[:240],
                "detail_en": str(obj.get("detail_en") or src["title"])[:260],
                "source_title": f"{src['source']}: {src['title'][:75]}",
                "source_url": src["url"],
            })

    return _upsert_collected(rows, "L3_web", start_d, path=path)


def collect_l3_web(days: int = 45, path: Optional[Path] = None, today: Optional[date] = None) -> int:
    """Layer 3: Use Claude CLI with live WebSearch to sweep upcoming 45-day non-calendar catalysts
    (conferences, product launches, regulatory/antitrust/tariff dates, Investor Days) for Holdings & Watchlist."""
    from . import briefs
    started = time.time()
    _record_run("L3_web", "running", started, path=path)
    start_d = today or _us_today()
    end_d = start_d + timedelta(days=days)

    try:
        st = pf.get_state(path=path, cached_quotes=True)
        held_syms = [h["symbol"] for h in advisor.followed(st)]
        watch_syms = [w["symbol"] for w in st.get("watchlist") or [] if w["symbol"] not in held_syms]
        all_syms = held_syms + watch_syms
        if not all_syms:
            _record_run("L3_web", "done", started, 0, path=path)
            return 0
        if not briefs.find_claude():
            saved_fb = _collect_l3_thematic_web_fallback(all_syms, start_d, end_d, path=path)
            _record_run("L3_web", "done", started, saved_fb, path=path)
            return saved_fb

        prompt = f"""You are building a forward-looking Catalyst Calendar for a US tech/equity portfolio from {start_d.isoformat()} to {end_d.isoformat()}.
Holdings: {', '.join(held_syms)}
Watchlist: {', '.join(watch_syms)}

Note: Standard quarterly earnings dates, regular dividends, FOMC meetings, and BLS CPI/NFP dates are ALREADY collected in Layer 1.
Use WebSearch (6 to 10 focused searches) to find NON-EARNINGS upcoming catalysts in [{start_d.isoformat()}, {end_d.isoformat()}] affecting these stocks:
1. Industry conferences, developer summits & keynotes (e.g. Microsoft Ignite, SC26 supercomputing, AWS re:Invent, CES 2027, NeurIPS, Investor/Analyst Days, product launch events)
2. Regulatory, antitrust, court & trade/tariff milestones (e.g. DOJ/EU antitrust hearings or remedies on GOOGL/AAPL/META/NVDA, US BIS semiconductor export control rules, CHIPS Act awards, China/US tariff deadlines)
3. Major supply-chain / product roadmap milestones (e.g. Blackwell/Rubin shipments, HBM3E/HBM4 qualification or pricing, foundry node milestones).

Reply with ONLY JSON lines between =====EVENTS===== and =====END===== (8 to 18 verified events; every item MUST include a real https:// source_url you found):
=====EVENTS=====
{{"symbol": "<primary ticker from Holdings/Watchlist or MACRO>", "affected_symbols": ["<tickers>"], "kind": "product|policy|legal|corporate", "event_date": "YYYY-MM-DD", "date_precision": "exact|approx", "date_label": "<e.g. Nov 18-22 or Mid-Nov>", "impact_level": "high|medium", "title_zh": "<Chinese title>", "title_en": "<English title>", "detail_zh": "<Chinese impact explanation>", "detail_en": "<English impact explanation>", "source_title": "<Publisher / Page title>", "source_url": "https://..."}}
=====END=====
"""
        try:
            out = briefs.run_claude(prompt)
        except Exception as claude_exc:  # noqa: BLE001
            print(f"[catalysts] Claude CLI unavailable/limited ({claude_exc}); running thematic web RSS + LLM sweep for L3")
            saved_fb = _collect_l3_thematic_web_fallback(all_syms, start_d, end_d, path=path)
            _record_run("L3_web", "done", started, saved_fb, path=path)
            return saved_fb
        raw = out.get("result") or ""
        m = re.search(r"=====EVENTS=====(.*?)=====END=====", raw, flags=re.S)
        body = m.group(1) if m else raw
        valid_universe = set(all_syms) | {"MACRO"}
        rows: List[Dict[str, Any]] = []
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sym = str(obj.get("symbol") or "").upper().strip()
            if sym not in valid_universe:
                continue
            ev_date_str = str(obj.get("event_date") or "")[:10]
            try:
                ev_d = date.fromisoformat(ev_date_str)
            except ValueError:
                continue
            if not (start_d <= ev_d <= end_d + timedelta(days=15)):
                continue
            url = str(obj.get("source_url") or "").strip()
            if not url.startswith("http"):
                continue
            kind = str(obj.get("kind") or "corporate").lower()
            if kind not in ("product", "policy", "legal", "corporate", "analyst", "macro"):
                kind = "corporate"
            t_zh = str(obj.get("title_zh") or "").strip()
            t_en = str(obj.get("title_en") or "").strip()
            if not (t_zh or t_en):
                continue
            aff = [s.upper() for s in (obj.get("affected_symbols") or []) if isinstance(s, str) and s.upper() in valid_universe]
            if sym != "MACRO" and sym not in aff:
                aff = [sym] + aff
            sig = hashlib.sha1(f"L3|{sym}|{kind}|{ev_date_str[:7]}|{(t_en or t_zh)[:36].lower()}".encode()).hexdigest()[:16]
            rows.append({
                "dedupe_key": f"L3:{sym}:{sig}",
                "kind": kind,
                "symbol": sym,
                "affected_symbols": aff,
                "event_date": ev_date_str,
                "date_precision": "approx" if obj.get("date_precision") == "approx" else "exact",
                "date_label": str(obj.get("date_label") or "")[:40],
                "impact_level": "high" if obj.get("impact_level") == "high" else "medium",
                "title_zh": (t_zh or t_en)[:80],
                "title_en": (t_en or t_zh)[:100],
                "detail_zh": str(obj.get("detail_zh") or t_zh or t_en)[:240],
                "detail_en": str(obj.get("detail_en") or t_en or t_zh)[:260],
                "source_title": str(obj.get("source_title") or "Web source")[:90],
                "source_url": url,
            })

        saved = _upsert_collected(rows, "L3_web", start_d, path=path)
        _record_run("L3_web", "done", started, saved, path=path)
        return saved
    except Exception as e:  # noqa: BLE001
        _record_run("L3_web", "failed", started, 0, f"{type(e).__name__}: {e}"[:500], path=path)
        raise


def trigger_collection(layer: str = "l2", days: int = 45, path: Optional[Path] = None) -> Dict[str, Any]:
    """Start Layer 2, Layer 3, or both (`all`) in background daemon threads without blocking."""
    targets = ["l2", "l3"] if layer == "all" else (["l3"] if layer == "l3" else ["l2"])
    started_layers: List[str] = []
    with _collect_lock:
        for t in targets:
            if _collecting.get(t):
                continue
            _collecting[t] = True
            started_layers.append(t)

            def _runner(which: str = t):
                try:
                    if which == "l2":
                        collect_l2_extracted(days=days, path=path)
                    else:
                        collect_l3_web(days=days, path=path)
                except Exception as exc:  # noqa: BLE001
                    print(f"[catalysts] {which} collection failed: {exc}")
                finally:
                    with _collect_lock:
                        _collecting[which] = False

            threading.Thread(target=_runner, name=f"catalysts-{t}", daemon=True).start()

    return {"started": started_layers, "layers": collection_status(path)}


def collection_status(path: Optional[Path] = None) -> Dict[str, Any]:
    conn = _conn(path)
    try:
        runs = {r["layer"]: dict(r) for r in conn.execute("SELECT * FROM catalyst_runs").fetchall()}
        counts = {
            r[0]: r[1]
            for r in conn.execute("SELECT layer, COUNT(*) FROM collected_catalysts GROUP BY layer").fetchall()
        }
    finally:
        conn.close()
    from . import briefs
    return {
        "L2_extracted": {
            **(runs.get("L2_extracted") or {}),
            "running": bool(_collecting.get("l2")),
            "stored_count": counts.get("L2_extracted", 0),
        },
        "L3_web": {
            **(runs.get("L3_web") or {}),
            "running": bool(_collecting.get("l3")),
            "stored_count": counts.get("L3_web", 0),
            "claude_available": bool(briefs.find_claude()),
        },
    }


def run_due(path: Optional[Path] = None) -> None:
    """Called once a minute by `markets/background.py` to keep Layer 2 and Layer 3 pre-collected."""
    st = collection_status(path)
    now = time.time()

    l2 = st["L2_extracted"]
    l2_last = l2.get("last_finished_at") or l2.get("last_started_at") or 0
    if not l2["running"] and now - l2_last >= L2_INTERVAL:
        trigger_collection("l2", days=45, path=path)

    l3 = st["L3_web"]
    l3_last = l3.get("last_finished_at") or l3.get("last_started_at") or 0
    dub_now = datetime.fromtimestamp(now, DUBLIN)
    if (
        l3["claude_available"]
        and not l3["running"]
        and dub_now.weekday() < 5
        and dub_now.time() >= L3_AT_DUBLIN
        and now - l3_last >= L3_INTERVAL
    ):
        trigger_collection("l3", days=45, path=path)


def _stored_catalysts(start: date, end: date, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = _conn(path)
    try:
        rows = conn.execute(
            "SELECT * FROM collected_catalysts WHERE event_date >= ? AND event_date <= ? ORDER BY event_date ASC, id ASC",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["affected_symbols"] = json.loads(d["affected_symbols"]) if d.get("affected_symbols") else []
        out.append(d)
    return out


# ---------------------------------------------------------------- Unified 3-Layer Feed + Per-Stock View

def upcoming_feed(
    scope: str = "all",
    days: int = 30,
    symbols: Optional[List[str]] = None,
    include_bellwethers: bool = True,
    force: bool = False,
    path: Optional[Path] = None,
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Return all 3 layers of upcoming events + per-stock breakdown for every Holding & Watchlist stock."""
    days = max(3, min(int(days or 30), 180))
    scope = scope if scope in ("all", "holdings", "watchlist") else "all"
    start = today or _us_today()
    end = start + timedelta(days=days)

    if force:
        conn = pf._conn(path)
        try:
            conn.execute("DELETE FROM calendar_cache WHERE fetched_at < ?", (time.time() - 60,))
            conn.commit()
        except Exception:  # noqa: BLE001
            pass
        finally:
            conn.close()

    st = pf.get_state(path=path, cached_quotes=True)
    held_rows = advisor.followed(st)
    watch_rows = st.get("watchlist") or []

    held_map: Dict[str, Dict[str, Any]] = {h["symbol"]: h for h in held_rows}
    watch_map: Dict[str, Dict[str, Any]] = {w["symbol"]: w for w in watch_rows if w["symbol"] not in held_map}
    held_set = set(held_map)
    watch_set = set(watch_map)
    pool_set = held_set | watch_set

    symbols_meta: List[Dict[str, Any]] = []
    for h in held_rows:
        q = h.get("quote") or {}
        symbols_meta.append({
            "symbol": h["symbol"],
            "name": q.get("name") or h["symbol"],
            "bucket": "held",
            "quantity": round(h.get("quantity") or 0, 6),
            "weight_pct": round(h["weight"], 1) if h.get("weight") is not None else None,
            "unrealized_pct": round(h["unrealized_pct"], 2) if h.get("unrealized_pct") is not None else None,
            "price": q.get("price"),
            "day_change_pct": round(h["day_change_pct"], 2) if h.get("day_change_pct") is not None else None,
        })
    for w in watch_rows:
        if w["symbol"] in held_set:
            continue
        q = w.get("quote") or {}
        symbols_meta.append({
            "symbol": w["symbol"],
            "name": q.get("name") or w["symbol"],
            "bucket": "watchlist",
            "quantity": 0,
            "weight_pct": None,
            "unrealized_pct": None,
            "price": q.get("price"),
            "day_change_pct": round(w["day_change_pct"], 2) if w.get("day_change_pct") is not None else None,
        })

    if scope == "holdings":
        active_syms = list(held_map)
    elif scope == "watchlist":
        active_syms = list(watch_map)
    else:
        active_syms = list(held_map) + list(watch_map)

    explicit_filter: Optional[Set[str]] = None
    if symbols:
        cleaned = {s.strip().upper() for s in symbols if s and s.strip()}
        if cleaned:
            explicit_filter = cleaned
            active_syms = [s for s in active_syms if s in cleaned] or sorted(cleaned)

    active_set = set(active_syms)
    bellwether_only: Set[str] = set()
    if include_bellwethers and not explicit_filter:
        bellwether_only = set(events.BELLWETHERS) - pool_set

    prices: Dict[str, Optional[float]] = {m["symbol"]: m["price"] for m in symbols_meta}
    names: Dict[str, str] = {m["symbol"]: m["name"] for m in symbols_meta}
    for b_sym, (b_name, _) in events.BELLWETHERS.items():
        names.setdefault(b_sym, b_name)

    gaps: List[str] = []
    all_events: List[Dict[str, Any]] = []

    weekdays: List[date] = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:
            weekdays.append(cur)
        cur += timedelta(days=1)

    near_weekdays = weekdays[:15]
    all_stock_syms = [s for s in (list(held_map) + list(watch_map)) if s not in PURE_ETFS]

    def _safe_earnings_on(d: date):
        try:
            return d, events.earnings_on(d, path)
        except Exception as e:  # noqa: BLE001
            gaps.append(f"nasdaq earnings {d}: {e}")
            return d, []

    def _safe_fh(sym: str):
        try:
            return sym, events.finnhub_earnings(sym, start, end, path)
        except Exception as e:  # noqa: BLE001
            gaps.append(f"finnhub {sym}: {e}")
            return sym, []

    def _safe_div(sym: str):
        try:
            return sym, events.dividends_for(sym, path=path)
        except Exception:  # noqa: BLE001
            return sym, []

    def _safe_splits(d: date):
        try:
            return events.splits_on(d, path)
        except Exception:  # noqa: BLE001
            return []

    def _safe_econ(d: date):
        try:
            return d, events.econ_on(d, path)
        except Exception as e:  # noqa: BLE001
            gaps.append(f"econ {d}: {e}")
            return d, []

    with ThreadPoolExecutor(max_workers=10) as pool:
        nasdaq_earnings = list(pool.map(_safe_earnings_on, near_weekdays))
        fh_by_sym = dict(pool.map(_safe_fh, all_stock_syms))
        divs_by_sym = dict(pool.map(_safe_div, list(held_map) + list(watch_map)))
        splits_nested = list(pool.map(_safe_splits, near_weekdays[:10]))
        econ_by_day = list(pool.map(_safe_econ, near_weekdays))

    splits_rows = [r for sub_l in splits_nested for r in sub_l]

    # ---------------- LAYER 1A: Earnings (Nasdaq + Finnhub + Implied Move)
    wanted_earn = pool_set | bellwether_only
    merged_earn: List[Dict[str, Any]] = []
    for d, rows in nasdaq_earnings:
        for r in rows:
            sym = r.get("symbol")
            if sym in wanted_earn:
                merged_earn.append({
                    "symbol": sym,
                    "name": r.get("name") or names.get(sym) or sym,
                    "date": d.isoformat(),
                    "when": r.get("when") or "time not given",
                    "eps_forecast": r.get("eps_forecast"),
                    "revenue_forecast": None,
                    "forecast_currency": "USD",
                    "revenue_currency": "USD",
                    "fiscal_quarter": r.get("fiscal_quarter"),
                    "listed_as": sym,
                })

    for sym, rows in fh_by_sym.items():
        for r in rows:
            d_str = str(r.get("date") or "")[:10]
            if not d_str:
                continue
            try:
                ed = date.fromisoformat(d_str)
            except ValueError:
                continue
            if not (start <= ed <= end):
                continue
            existing = next(
                (m for m in merged_earn if m["symbol"] == sym and abs((date.fromisoformat(m["date"]) - ed).days) <= 6),
                None,
            )
            if existing:
                if r.get("revenue_forecast") and not existing.get("revenue_forecast"):
                    existing["revenue_forecast"] = r["revenue_forecast"]
                    existing["revenue_currency"] = r.get("forecast_currency") or "USD"
                if existing.get("when") == "time not given" and r.get("when") and r.get("when") != "time not given":
                    existing["when"] = r["when"]
                if not existing.get("fiscal_quarter") and r.get("fiscal_quarter"):
                    existing["fiscal_quarter"] = r["fiscal_quarter"]
            else:
                merged_earn.append({
                    "symbol": sym,
                    "name": names.get(sym) or sym,
                    "date": d_str,
                    "when": r.get("when") or "time not given",
                    "eps_forecast": r.get("eps_forecast"),
                    "revenue_forecast": r.get("revenue_forecast"),
                    "forecast_currency": r.get("forecast_currency") or "USD",
                    "revenue_currency": r.get("forecast_currency") or "USD",
                    "fiscal_quarter": r.get("fiscal_quarter"),
                    "listed_as": r.get("listed_as") or sym,
                })

    merged_earn.sort(key=lambda x: x["date"])
    imp_targets = [
        m for m in merged_earn
        if (date.fromisoformat(m["date"]) - start).days <= 45 and prices.get(m["symbol"])
    ][:10]
    if imp_targets:
        with ThreadPoolExecutor(max_workers=4) as pool:
            imp_results = list(pool.map(
                lambda m: _cached_implied_move(m["symbol"], date.fromisoformat(m["date"]), prices.get(m["symbol"]), path),
                imp_targets,
            ))
        for m, imp in zip(imp_targets, imp_results):
            m["implied"] = imp

    l1_earn_keys: Set[Tuple[str, str]] = set()
    for m in merged_earn:
        sym = m["symbol"]
        ed = date.fromisoformat(m["date"])
        l1_earn_keys.add((sym, m["date"][:7]))
        days_away = (ed - start).days
        bucket = "held" if sym in held_set else ("watchlist" if sym in watch_set else "bellwether")
        rel_all, chains_zh, chains_en = _related_symbols(sym, pool_set)
        rel_held = [s for s in rel_all if s in held_set]
        cur_code = m.get("forecast_currency") or "USD"
        rev_cur = m.get("revenue_currency") or cur_code
        eps_s = _fmt_eps(m.get("eps_forecast"), cur_code)
        rev_s = _fmt_rev(m.get("revenue_forecast"), rev_cur)
        imp_pct = ((m.get("implied") or {}).get("implied_move_pct"))
        bw_group = events.BELLWETHERS.get(sym, (None, None))[1]

        parts_zh = [_when_zh(m.get("when"))]
        parts_en = [_when_en(m.get("when"))]
        if m.get("fiscal_quarter"):
            parts_zh.append(str(m["fiscal_quarter"]))
            parts_en.append(str(m["fiscal_quarter"]))
        if eps_s:
            parts_zh.append(f"预期 EPS {eps_s}" + (f" ({m['listed_as']})" if cur_code != "USD" else ""))
            parts_en.append(f"Est. EPS {eps_s}" + (f" ({m['listed_as']})" if cur_code != "USD" else ""))
        if rev_s:
            parts_zh.append(f"预期营收 {rev_s}")
            parts_en.append(f"Est. Rev {rev_s}")
        if imp_pct:
            parts_zh.append(f"期权隐含 ±{imp_pct}%")
            parts_en.append(f"Implied ±{imp_pct}%")
        if rel_held:
            parts_zh.append(f"传导持仓: {', '.join(rel_held)}")
            parts_en.append(f"Spillover to held: {', '.join(rel_held)}")

        affected = ([sym] if sym in pool_set else []) + ([s for s in rel_all if s != sym] if (bw_group or sym in held_set or sym in {"NVDA", "AVGO", "MU", "AAPL", "AMD", "MSFT", "ASML", "GOOGL", "AMZN", "META", "TSM", "SKHY"}) else [])
        all_events.append({
            "id": f"earn:{sym}:{m['date']}",
            "layer": "L1_calendar",
            "kind": "earnings",
            "bucket": bucket,
            "symbol": sym,
            "name": m["name"],
            "date": m["date"],
            "date_precision": "exact",
            "date_label": "",
            "days_away": days_away,
            "when": m.get("when"),
            "fiscal_quarter": m.get("fiscal_quarter"),
            "eps_forecast": m.get("eps_forecast"),
            "revenue_forecast": m.get("revenue_forecast"),
            "forecast_currency": cur_code,
            "implied_move_pct": imp_pct,
            "bellwether": bw_group,
            "weight_pct": round(held_map[sym]["weight"], 1) if sym in held_map and held_map[sym].get("weight") is not None else None,
            "affected_symbols": affected,
            "related_holdings": rel_held,
            "impact_level": "high" if (bucket == "held" or bw_group or (imp_pct and imp_pct >= 6)) else "medium",
            "title_zh": f"{sym} 财报发布",
            "title_en": f"{sym} Earnings Release",
            "detail_zh": " · ".join(parts_zh),
            "detail_en": " · ".join(parts_en),
            "source_title": "Nasdaq / Finnhub Calendar",
            "source_url": f"https://www.nasdaq.com/market-activity/stocks/{sym.lower()}/earnings",
        })

    # ---------------- LAYER 1B: TSMC Monthly Revenue Schedule (10th of month)
    if "TSM" in pool_set:
        for td in _tsm_monthly_revenue_dates(start, end):
            rel_all, _, _ = _related_symbols("TSM", pool_set)
            rel_held = [s for s in rel_all if s in held_set]
            prev_m = (td.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")
            all_events.append({
                "id": f"corp:TSM:rev:{td.isoformat()}",
                "layer": "L1_calendar",
                "kind": "corporate",
                "bucket": "held" if "TSM" in held_set else "watchlist",
                "symbol": "TSM",
                "name": names.get("TSM", "TSMC"),
                "date": td.isoformat(),
                "date_precision": "exact",
                "date_label": "",
                "days_away": (td - start).days,
                "weight_pct": round(held_map["TSM"]["weight"], 1) if "TSM" in held_map and held_map["TSM"].get("weight") is not None else None,
                "affected_symbols": ["TSM"] + [s for s in rel_all if s in {"NVDA", "AMD", "AVGO", "AAPL", "SKHY", "MU", "INTC"}],
                "related_holdings": rel_held,
                "impact_level": "high",
                "title_zh": f"TSM 公布 {prev_m} 月度合并营收",
                "title_en": f"TSM Monthly Consolidated Revenue ({prev_m})",
                "detail_zh": "每月10日披露：全球 AI 芯片出货与先进制程稼动率最高频先行验证指标",
                "detail_en": "10th-of-month release: highest-frequency barometer for AI wafer demand & CoWoS utilization",
                "source_title": "TSMC Investor Relations",
                "source_url": "https://investor.tsmc.com/english/monthly-revenue-report",
            })

    # ---------------- LAYER 1C: Dividends & Splits
    for sym, rows in divs_by_sym.items():
        if sym not in pool_set:
            continue
        bucket = "held" if sym in held_set else "watchlist"
        qty_held = (held_map.get(sym) or {}).get("quantity") or 0.0
        for r in rows:
            ex_d = date.fromisoformat(r["ex_date"]) if r.get("ex_date") else None
            pay_d = date.fromisoformat(r["pay_date"]) if r.get("pay_date") else None
            amt = r.get("amount") or 0.0
            cur_code = r.get("currency") or "USD"
            payout = round(qty_held * amt, 2) if bucket == "held" and qty_held > 0 and cur_code == "USD" else None
            if ex_d and start <= ex_d <= end:
                pay_zh = f" · 派息日 {r['pay_date']}" if r.get("pay_date") else ""
                pay_en = f" · Pay date {r['pay_date']}" if r.get("pay_date") else ""
                est_zh = f" · 持仓预计分红 ${payout:,.2f}" if payout else ""
                est_en = f" · Est. payout ${payout:,.2f}" if payout else ""
                all_events.append({
                    "id": f"div:ex:{sym}:{r['ex_date']}",
                    "layer": "L1_calendar",
                    "kind": "dividend",
                    "bucket": bucket,
                    "symbol": sym,
                    "name": names.get(sym) or sym,
                    "date": r["ex_date"],
                    "date_precision": "exact",
                    "date_label": "",
                    "days_away": (ex_d - start).days,
                    "amount": amt,
                    "currency": cur_code,
                    "pay_date": r.get("pay_date"),
                    "estimated_payout_usd": payout,
                    "weight_pct": round(held_map[sym]["weight"], 1) if sym in held_map and held_map[sym].get("weight") is not None else None,
                    "affected_symbols": [sym],
                    "related_holdings": [sym] if bucket == "held" else [],
                    "impact_level": "medium",
                    "title_zh": f"{sym} 除权除息日 (Ex-Div)",
                    "title_en": f"{sym} Ex-Dividend Date",
                    "detail_zh": f"每股现金派息 ${amt:.4g}{pay_zh}{est_zh}",
                    "detail_en": f"Cash dividend ${amt:.4g}/sh{pay_en}{est_en}",
                    "source_title": "Nasdaq Dividend Calendar",
                    "source_url": f"https://www.nasdaq.com/market-activity/stocks/{sym.lower()}/dividend-history",
                })
            elif pay_d and start <= pay_d <= end and (not ex_d or ex_d < start):
                est_zh = f" · 预计到账 ${payout:,.2f}" if payout else ""
                est_en = f" · Est. payout ${payout:,.2f}" if payout else ""
                all_events.append({
                    "id": f"div:pay:{sym}:{r['pay_date']}",
                    "layer": "L1_calendar",
                    "kind": "dividend",
                    "bucket": bucket,
                    "symbol": sym,
                    "name": names.get(sym) or sym,
                    "date": r["pay_date"],
                    "date_precision": "exact",
                    "date_label": "",
                    "days_away": (pay_d - start).days,
                    "amount": amt,
                    "currency": cur_code,
                    "pay_date": r["pay_date"],
                    "estimated_payout_usd": payout,
                    "weight_pct": round(held_map[sym]["weight"], 1) if sym in held_map and held_map[sym].get("weight") is not None else None,
                    "affected_symbols": [sym],
                    "related_holdings": [sym] if bucket == "held" else [],
                    "impact_level": "medium",
                    "title_zh": f"{sym} 股息发放日",
                    "title_en": f"{sym} Dividend Payment",
                    "detail_zh": f"每股 ${amt:.4g} (除息日 {r.get('ex_date') or '—'}){est_zh}",
                    "detail_en": f"${amt:.4g}/sh (ex-div {r.get('ex_date') or '—'}){est_en}",
                    "source_title": "Nasdaq Dividend Calendar",
                    "source_url": f"https://www.nasdaq.com/market-activity/stocks/{sym.lower()}/dividend-history",
                })

    seen_splits: Set[Tuple[str, str]] = set()
    for sp in splits_rows:
        sym = sp["symbol"]
        ex_date_str = sp.get("execution_date") or ""
        if sym not in pool_set or not ex_date_str or (sym, ex_date_str) in seen_splits:
            continue
        try:
            sd = date.fromisoformat(ex_date_str)
        except ValueError:
            continue
        if not (start <= sd <= end):
            continue
        seen_splits.add((sym, ex_date_str))
        bucket = "held" if sym in held_set else "watchlist"
        all_events.append({
            "id": f"split:{sym}:{ex_date_str}",
            "layer": "L1_calendar",
            "kind": "split",
            "bucket": bucket,
            "symbol": sym,
            "name": sp.get("name") or names.get(sym) or sym,
            "date": ex_date_str,
            "date_precision": "exact",
            "date_label": "",
            "days_away": (sd - start).days,
            "ratio": sp.get("ratio"),
            "affected_symbols": [sym],
            "related_holdings": [sym] if bucket == "held" else [],
            "impact_level": "high" if bucket == "held" else "medium",
            "title_zh": f"{sym} 股票拆分 ({sp.get('ratio') or ''})",
            "title_en": f"{sym} Stock Split ({sp.get('ratio') or ''})",
            "detail_zh": f"比例 {sp.get('ratio') or '—'} · 生效日 {ex_date_str}",
            "detail_en": f"Ratio {sp.get('ratio') or '—'} · Effective {ex_date_str}",
            "source_title": "Nasdaq Splits Calendar",
            "source_url": "https://www.nasdaq.com/market-activity/stock-splits",
        })

    # ---------------- LAYER 1D: Official & Nasdaq Macro Calendar
    seen_econ: Set[Tuple[str, str, str]] = set()
    for m_item in OFFICIAL_MACRO_SCHEDULE:
        md = date.fromisoformat(m_item["date"])
        if start <= md <= end:
            norm = _norm_econ_name(m_item["event"]).lower()
            seen_econ.add((m_item["date"], m_item["country"], norm))
            flag = COUNTRY_FLAG.get(m_item["country"], "🌐")
            all_events.append({
                "id": f"macro:{m_item['date']}:{norm[:24]}",
                "layer": "L1_calendar",
                "kind": "macro",
                "bucket": "macro",
                "symbol": "MACRO",
                "name": m_item["country"],
                "country": m_item["country"],
                "flag": flag,
                "date": m_item["date"],
                "date_precision": "exact",
                "date_label": "",
                "days_away": (md - start).days,
                "gmt": m_item.get("gmt"),
                "affected_symbols": sorted(pool_set),
                "related_holdings": sorted(held_set),
                "impact_level": "high",
                "title_zh": f"{flag} {m_item['event']}",
                "title_en": f"{flag} {m_item['event']}",
                "detail_zh": m_item["note_zh"],
                "detail_en": m_item["note_en"],
                "source_title": "Official Fed / BLS / SEC Schedule",
                "source_url": "",
            })

    for d, rows in econ_by_day:
        d_str = d.isoformat()
        for r in rows:
            country = r.get("country") or ""
            ev_name = (r.get("event") or "").strip()
            if country not in events.ECON_COUNTRIES or not ev_name or not events.ECON_KEYWORDS.search(ev_name):
                continue
            # Keep timeline crisp: filter out noisy secondary sub-components & minor non-US releases
            if re.search(r"(Continuing Jobless|4-Week Avg|GDP Sales|GDP Price Index|Consumer Credit|Inflation Expectation|Redbook|Bill Auction|Bond Auction)", ev_name, re.I):
                continue
            if country != "United States" and not re.search(r"\b(Rate Decision|Interest Rate|ECB|BoE|BoJ|GDP|Exports|CPI)\b", ev_name, re.I):
                continue
            norm = _norm_econ_name(ev_name).lower()
            if "pce" in norm:
                norm = "core pce"
            elif norm.startswith("gdp"):
                norm = "gdp"
            # Avoid duplicating official US CPI / Nonfarm / FOMC rows on the exact same date
            if country == "United States" and any(
                k[0] == d_str and (
                    ("cpi" in norm and "cpi" in k[2])
                    or (("nonfarm" in norm or "payroll" in norm) and "nonfarm" in k[2])
                    or (("fomc" in norm or "rate decision" in norm) and "fomc" in k[2])
                )
                for k in seen_econ
            ):
                continue
            if (d_str, country, norm) in seen_econ:
                continue
            seen_econ.add((d_str, country, norm))
            flag = COUNTRY_FLAG.get(country, "🌐")
            major_macro = bool(re.search(r"\b(CPI|PCE|Nonfarm|Payrolls?|FOMC|Rate Decision|GDP|ISM)\b", ev_name, re.I))
            parts_zh, parts_en = [], []
            if r.get("gmt"):
                parts_zh.append(f"{r['gmt']} GMT")
                parts_en.append(f"{r['gmt']} GMT")
            if r.get("consensus"):
                parts_zh.append(f"预期 {r['consensus']}")
                parts_en.append(f"Consensus {r['consensus']}")
            if r.get("previous"):
                parts_zh.append(f"前值 {r['previous']}")
                parts_en.append(f"Prev {r['previous']}")
            aff_syms = sorted(pool_set) if (country == "United States" and major_macro) else (
                [s for s in ("TSM", "SKHY", "NVDA", "AMD", "AVGO", "MU", "INTC", "ASML") if s in pool_set]
                if country in ("South Korea", "Taiwan")
                else [s for s in ("QQQ", "SPY") if s in pool_set]
            )
            all_events.append({
                "id": f"econ:{d_str}:{country}:{norm[:24]}",
                "layer": "L1_calendar",
                "kind": "macro",
                "bucket": "macro",
                "symbol": "MACRO",
                "name": country,
                "country": country,
                "flag": flag,
                "date": d_str,
                "date_precision": "exact",
                "date_label": "",
                "days_away": (d - start).days,
                "gmt": r.get("gmt"),
                "consensus": r.get("consensus"),
                "previous": r.get("previous"),
                "affected_symbols": aff_syms,
                "related_holdings": [s for s in aff_syms if s in held_set],
                "impact_level": "high" if (major_macro and country == "United States") else "medium",
                "title_zh": f"{flag} {country} · {ev_name}",
                "title_en": f"{flag} {country} · {ev_name}",
                "detail_zh": " · ".join(parts_zh) or country,
                "detail_en": " · ".join(parts_en) or country,
                "source_title": "Nasdaq Economic Calendar",
                "source_url": "https://www.nasdaq.com/market-activity/economic-calendar",
            })

    # ---------------- LAYERS 2 & 3: Pre-collected Extracted & Web-Swept Catalysts
    for row in _stored_catalysts(start, end, path=path):
        sym = row["symbol"]
        if sym != "MACRO" and sym not in pool_set and sym not in events.BELLWETHERS:
            continue
        # Skip if L2/L3 mistakenly extracted a plain earnings row already covered by L1
        if (sym, row["event_date"][:7]) in l1_earn_keys and re.search(r"\b(earnings|q[1-4]\s+results)\b|财报", row["title_en"] + row["title_zh"], re.I):
            continue
        ed = date.fromisoformat(row["event_date"])
        bucket = "macro" if sym == "MACRO" else ("held" if sym in held_set else ("watchlist" if sym in watch_set else "bellwether"))
        aff = [s for s in (row.get("affected_symbols") or []) if s in pool_set]
        if sym in pool_set and sym not in aff:
            aff = [sym] + aff
        if sym == "MACRO" and not aff:
            aff = sorted(pool_set)
        if not aff and sym in events.BELLWETHERS:
            rel_all, _, _ = _related_symbols(sym, pool_set)
            aff = rel_all
        rel_held = [s for s in aff if s in held_set]
        all_events.append({
            "id": f"col:{row['id']}",
            "layer": row["layer"],
            "kind": row["kind"],
            "bucket": bucket,
            "symbol": sym,
            "name": names.get(sym) or sym,
            "date": row["event_date"],
            "date_precision": row.get("date_precision") or "exact",
            "date_label": row.get("date_label") or "",
            "days_away": (ed - start).days,
            "weight_pct": round(held_map[sym]["weight"], 1) if sym in held_map and held_map[sym].get("weight") is not None else None,
            "affected_symbols": aff,
            "related_holdings": rel_held,
            "impact_level": row.get("impact_level") or "medium",
            "title_zh": row["title_zh"],
            "title_en": row["title_en"],
            "detail_zh": row["detail_zh"],
            "detail_en": row["detail_en"],
            "source_title": row.get("source_title") or "",
            "source_url": row.get("source_url") or "",
        })

    bucket_rank = {"held": 0, "watchlist": 1, "macro": 2, "bellwether": 3}
    layer_rank = {"L1_calendar": 0, "L3_web": 1, "L2_extracted": 2}
    all_events.sort(key=lambda x: (x["date"], bucket_rank.get(x["bucket"], 9), layer_rank.get(x["layer"], 9), x.get("symbol") or ""))

    # ---------------- Build Per-Stock Catalyst Map for EVERY Holding & Watchlist Stock
    per_stock: Dict[str, Dict[str, Any]] = {}
    for m in symbols_meta:
        sym = m["symbol"]
        direct: List[Dict[str, Any]] = []
        spillover: List[Dict[str, Any]] = []
        macro_list: List[Dict[str, Any]] = []
        for ev in all_events:
            if ev["symbol"] == sym:
                direct.append(ev)
            elif ev["bucket"] == "macro" and ev["impact_level"] == "high" and sym in (ev.get("affected_symbols") or []):
                macro_list.append(ev)
            elif sym in (ev.get("affected_symbols") or []):
                spillover.append(ev)
        per_stock[sym] = {
            **m,
            "direct_count": len(direct),
            "spillover_count": len(spillover),
            "macro_count": len(macro_list),
            "total_count": len(direct) + len(spillover) + len(macro_list[:4]),
            "next_direct": direct[0] if direct else None,
            "direct": direct,
            "spillover": spillover[:6],
            "macro": macro_list[:4],
        }

    # Filter timeline to requested `scope` / `symbols`
    def _matches_scope(ev: Dict[str, Any]) -> bool:
        if explicit_filter:
            return ev["symbol"] in explicit_filter or bool(set(ev.get("affected_symbols") or []) & explicit_filter)
        if scope == "holdings":
            return ev["bucket"] == "held" or bool(ev.get("related_holdings")) or (ev["bucket"] == "macro" and ev["impact_level"] == "high")
        if scope == "watchlist":
            return ev["bucket"] == "watchlist" or any(s in watch_set for s in (ev.get("affected_symbols") or []))
        return True

    timeline = [ev for ev in all_events if _matches_scope(ev)]

    held_direct = [e for e in timeline if e["bucket"] == "held"]
    watch_direct = [e for e in timeline if e["bucket"] == "watchlist"]

    return {
        "from": start.isoformat(),
        "to": end.isoformat(),
        "days": days,
        "scope": scope,
        "symbols_meta": symbols_meta,
        "per_stock": per_stock,
        "collection": collection_status(path),
        "summary": {
            "total": len(timeline),
            "l1": sum(1 for e in timeline if e["layer"] == "L1_calendar"),
            "l2": sum(1 for e in timeline if e["layer"] == "L2_extracted"),
            "l3": sum(1 for e in timeline if e["layer"] == "L3_web"),
            "earnings": sum(1 for e in timeline if e["kind"] == "earnings"),
            "macro": sum(1 for e in timeline if e["kind"] == "macro"),
            "dividends": sum(1 for e in timeline if e["kind"] in ("dividend", "split")),
            "catalysts": sum(1 for e in timeline if e["kind"] in ("product", "policy", "legal", "corporate", "analyst")),
            "held_events": len(held_direct),
            "watch_events": len(watch_direct),
            "next_held": held_direct[0] if held_direct else None,
            "next_watch": watch_direct[0] if watch_direct else None,
        },
        "timeline": timeline,
        "data_gaps": gaps,
    }


# ---------------------------------------------------------------- Search & Interactive AI Query

def list_queries(limit: int = 12, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = _conn(path)
    try:
        rows = conn.execute(
            "SELECT * FROM catalyst_queries ORDER BY created_at DESC, id DESC LIMIT ?",
            (max(1, min(int(limit), 50)),),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["symbols"] = json.loads(d["symbols"]) if d.get("symbols") else []
        d["sources"] = json.loads(d["sources"]) if d.get("sources") else []
        out.append(d)
    return out


def delete_query(qid: Optional[int] = None, clear_all: bool = False, path: Optional[Path] = None) -> None:
    conn = _conn(path)
    try:
        if clear_all:
            conn.execute("DELETE FROM catalyst_queries")
        elif qid is not None:
            conn.execute("DELETE FROM catalyst_queries WHERE id = ?", (int(qid),))
        conn.commit()
    finally:
        conn.close()


def _save_query(record: Dict[str, Any], path: Optional[Path] = None) -> int:
    conn = _conn(path)
    try:
        cur = conn.execute(
            "INSERT INTO catalyst_queries(created_at, question, scope, days, symbols, mode, lang, answer, sources, model, elapsed_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["created_at"], record["question"], record["scope"], record["days"],
                json.dumps(record.get("symbols") or []), record["mode"], record["lang"],
                record["answer"], json.dumps(record.get("sources") or [], ensure_ascii=False),
                record.get("model"), record["elapsed_s"],
            ),
        )
        qid = cur.lastrowid
        conn.execute(
            "DELETE FROM catalyst_queries WHERE id NOT IN (SELECT id FROM catalyst_queries ORDER BY created_at DESC, id DESC LIMIT 40)"
        )
        conn.commit()
        return int(qid)
    finally:
        conn.close()


def _detect_mentioned_symbols(question: str, known: Set[str]) -> List[str]:
    tokens = {t.upper() for t in re.findall(r"\b[A-Za-z]{2,5}\b", question or "")}
    return sorted(tokens & known)


def query_catalysts(
    question: str,
    scope: str = "all",
    days: int = 30,
    symbols: Optional[List[str]] = None,
    mode: str = "fast",
    lang: str = "zh",
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Answer a question over the 3-layer pre-collected catalyst store + live web RSS / Claude search."""
    q = (question or "").strip()
    if not q:
        raise ValueError("Please enter a question to query upcoming events")
    if len(q) > 800:
        q = q[:800]

    days = max(3, min(int(days or 30), 180))
    mode = "web" if mode == "web" else "fast"
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", q))
    out_lang = "zh" if (has_cjk or lang != "en") else "en"

    t0 = time.time()
    feed = upcoming_feed(scope=scope, days=days, symbols=symbols, include_bellwethers=True, path=path)
    all_known = {m["symbol"] for m in feed["symbols_meta"]} | set(events.BELLWETHERS)
    mentioned = _detect_mentioned_symbols(q, all_known)
    if mentioned and not set(mentioned).issubset({e.get("symbol") for e in feed["timeline"]}):
        feed = upcoming_feed(scope="all", days=days, include_bellwethers=True, path=path)

    held_summary = [m for m in feed["symbols_meta"] if m["bucket"] == "held"]
    watch_summary = [m for m in feed["symbols_meta"] if m["bucket"] == "watchlist"]

    prioritized_events = sorted(
        feed["timeline"],
        key=lambda e: (
            0 if e["bucket"] in ("held", "watchlist") or e["layer"] in ("L2_extracted", "L3_web")
            else (1 if e.get("impact_level") == "high" else 2),
            e["date"],
        ),
    )[:65]
    prioritized_events.sort(key=lambda e: e["date"])
    compact_events = [
        {k: e[k] for k in ("date", "days_away", "layer", "kind", "bucket", "symbol", "title_en", "detail_en", "implied_move_pct", "affected_symbols")
         if e.get(k) not in (None, [], "")}
        for e in prioritized_events
    ]
    snapshot_payload = {
        "window": {"from": feed["from"], "to": feed["to"], "days": days, "scope": scope},
        "holdings": held_summary,
        "watchlist": watch_summary,
        "collected_events_3layers": compact_events,
    }

    lang_rule = (
        "请使用清晰、专业的简体中文回答（股票代码保留英文，如 TSM、NVDA）。"
        if out_lang == "zh"
        else "Write clearly and concisely in English."
    )

    answer_text = ""
    sources_out: List[Dict[str, Any]] = []
    model_used = mode

    fallback_note = ""
    if mode == "web":
        from . import briefs
        if briefs.find_claude():
            web_prompt = f"""You are a quantitative catalyst analyst for the user's portfolio and watchlist.
{lang_rule}

Below is the 3-layer pre-collected snapshot of the user's holdings, watchlist, and upcoming events ({feed['from']} to {feed['to']}).
Use WebSearch (4 to 8 targeted searches) to supplement any fresh developments, unscheduled conferences, regulatory rulings, or Wall Street consensus expectations relevant to the user's question.

User question: {q}

Structure concisely in Markdown:
1. 核心结论 / Bottom line
2. 关键事件时间线 / Catalyst timeline (exact dates, layer/type, consensus & implied moves)
3. 对持仓与 Watchlist 的影响传导 / Impact & supply-chain spillover (distinguish Held vs Watchlist)
4. 风险与观察点 / Key risks

Reply in this exact format:
=====ANSWER=====
(Markdown answer)
=====SOURCES=====
- [Source title](https://...)

SNAPSHOT (JSON):
{json.dumps(snapshot_payload, ensure_ascii=False, separators=(",", ":"))}
"""
            try:
                out = briefs.run_claude(web_prompt)
                raw = out.get("result") or ""
                parts = re.split(r"^=====([A-Z_]+)=====\s*$", raw, flags=re.M)
                blocks = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
                answer_text = blocks.get("ANSWER") or raw.strip()
                sources_out = [
                    {"id": idx, "title": m.group(1).strip(), "url": m.group(2).strip()}
                    for idx, m in enumerate(re.finditer(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", blocks.get("SOURCES", "")), 1)
                ]
                models = list((out.get("modelUsage") or {}).keys())
                model_used = ",".join(models) if models else "claude-web"
            except Exception as web_exc:  # noqa: BLE001
                print(f"[catalysts] web query fallback to live RSS + local LLM: {web_exc}")
                mode = "fast"
                fallback_note = (
                    "> ℹ️ *Claude CLI 当前额度暂满，已自动切换为 **实时全网新闻检索 + 三层预抓库 + 本地大模型** 完成分析。*\n\n"
                    if out_lang == "zh"
                    else "> ℹ️ *Claude CLI quota is temporarily reached; automatically answered using **Live Web News RSS + 3-Layer Catalyst DB + Local LLM**.*\n\n"
                )
        else:
            mode = "fast"

    if mode == "fast":
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        # Combine sources from pre-collected L2/L3 events + a fast live Google News RSS query on the question!
        rss_syms = mentioned[:4] or [h["symbol"] for h in held_summary[:4]]
        rss_q = f"({' OR '.join(rss_syms)}) (upcoming OR conference OR earnings OR tariff OR launch OR outlook) when:7d" if rss_syms else f"{q} stock upcoming catalyst"
        live_rss = fetch_google_news_rss(rss_q, max_items=8, path=path)

        source_pool: List[Dict[str, Any]] = []
        seen_u: Set[str] = set()
        for ev in feed["timeline"]:
            if ev["layer"] in ("L2_extracted", "L3_web") and ev.get("source_url") and ev["source_url"] not in seen_u:
                seen_u.add(ev["source_url"])
                source_pool.append({
                    "symbol": ev["symbol"],
                    "date": ev["date"],
                    "title": f"[{ev['symbol']}] {ev['title_en']} — {ev['detail_en']}",
                    "url": ev["source_url"],
                })
        for r in live_rss:
            if r["url"] not in seen_u:
                seen_u.add(r["url"])
                source_pool.append({
                    "symbol": "",
                    "date": r["date"],
                    "title": f"{r['source']}: {r['title']}",
                    "url": r["url"],
                })

        numbered_sources = [{**s, "id": i} for i, s in enumerate(source_pool[:12], 1)]
        news_block = "\n".join(f"[{s['id']}] ({s['date']}) {s['title']}" for s in numbered_sources)

        sys_msg = f"""You are a quantitative equity catalyst analyst for the user's portfolio (`holdings`) and `watchlist`.
{lang_rule}

Rules:
1. Ground all dates, consensus EPS/revenue, dividends, and options-implied moves in `SNAPSHOT_3LAYERS` and `LIVE_SOURCES`. Never invent dates.
2. Cite external news / catalyst claims inline as [1], [2], etc. matching `LIVE_SOURCES`.
3. Explicitly distinguish **Held stocks (持仓)** from **Watchlist stocks (观察池)**, and highlight supply-chain spillover from Watchlist/sector leaders to Held positions.
4. Format cleanly in Markdown with short headings and bullet points."""

        user_msg = (
            f"SNAPSHOT_3LAYERS (JSON):\n{json.dumps(snapshot_payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
            f"LIVE_SOURCES:\n{news_block or '(none)'}\n\n"
            f"USER_QUESTION: {q}"
        )
        llm = _get_llm(temperature=0.2)
        model_name = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "local-llm")
        messages = [SystemMessage(content=sys_msg), HumanMessage(content=user_msg)]
        if "qwen" in model_name.lower():
            messages.append(AIMessage(content="<think>\n\n</think>\n\n"))
        raw_ans = llm.invoke(messages).content or ""
        answer_text = fallback_note + re.sub(r"^<think>.*?</think>\s*", "", raw_ans.strip(), flags=re.S)
        cited_nums = {int(x) for x in re.findall(r"\[(\d+)\]", answer_text)}
        sources_out = [s for s in numbered_sources if s["id"] in cited_nums and s.get("url")] or [s for s in numbered_sources[:5] if s.get("url")]
        model_used = model_name

    elapsed = round(time.time() - t0, 2)
    record = {
        "created_at": time.time(),
        "question": q,
        "scope": scope,
        "days": days,
        "symbols": symbols or mentioned,
        "mode": mode,
        "lang": out_lang,
        "answer": answer_text,
        "sources": sources_out,
        "model": model_used,
        "elapsed_s": elapsed,
    }
    record["id"] = _save_query(record, path=path)
    return record
