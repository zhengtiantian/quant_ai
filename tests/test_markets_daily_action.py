"""Risk gate (Q.12) and today's action (Q.13). No network: the state and the Q.10 analysis are given."""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from markets import daily_action, risk_budget, risk_rules  # noqa: E402


def state(intc_qty=142.18, intc_px=64.89, others=(("SKHY", 6755.82), ("TSM", 2760.33), ("STX", 2154.21), ("GOOGL", 281.45)), cash=0.0):
    hs = [{"symbol": "INTC", "quantity": intc_qty, "avg_cost": 87.69, "market_value": intc_qty * intc_px, "quote": {"price": intc_px}}]
    hs += [{"symbol": s, "quantity": 1.0, "avg_cost": v, "market_value": v, "quote": {"price": v}} for s, v in others]
    return {"holdings": hs, "cash": cash}


def advice(stabilized=False, lowest=64.44, events=True, report=False, big_drop=None):
    cats = [{"title": "Apple earnings", "impact": "high", "bucket": "spillover", "category": "earnings", "days_away": 2}] if events else []
    if report:
        cats.append({"title": "INTC earnings", "impact": "high", "bucket": "direct", "category": "earnings", "days_away": 0})
    return {"stats": {"atr14": 6.86, "stabilized": stabilized, "big_drop": big_drop}, "catalysts_7d": cats,
            "dip_guardrail": {"anchor_cost": lowest}}


class GateTest(unittest.TestCase):
    def test_exposure_counts_cash_in_the_total(self):
        ex = risk_rules.exposure(state(cash=1000.0))
        self.assertAlmostEqual(ex["total"], 142.18 * 64.89 + 6755.82 + 2760.33 + 2154.21 + 281.45 + 1000.0)
        self.assertGreater(ex["cluster_pct"], 90)                  # GOOGL is the only name outside the cluster

    def test_trim_back_to_the_cap(self):
        ex = risk_rules.exposure(state())
        n = risk_rules.trim_shares(142.18, ex["mv"]["INTC"], ex["total"], 30)
        self.assertEqual(n, 45)                                    # 43.6% -> 30%: 1 - 30/43.6 of 142.18, rounded up
        self.assertEqual(risk_rules.trim_shares(10, 100, 1000, 30), 0)

    def test_a_sale_is_capped_at_the_whole_shares_held(self):
        g = risk_rules.gate({"symbol": "STX", "side": "sell", "shares": 5, "limit": 700}, risk_rules.exposure(state()), held=3.32)
        self.assertEqual((g["shares"], g["allowed"]), (3, True))

    def test_a_buy_meets_cash_caps_spacing_and_the_falling_rule(self):
        ex = risk_rules.exposure(state(cash=5000.0))
        g = risk_rules.gate({"symbol": "INTC", "side": "buy", "shares": 10, "limit": 64.0}, ex, held=142.18, atr=6.86,
                            lowest_lot=64.44, stabilized=False, event_7d=True)
        self.assertFalse(g["allowed"])
        self.assertEqual(set(g["blocked_by"]), {"falling", "spacing", "size"})   # over the symbol cap: no room at all
        self.assertTrue(any("$50.72" in n for n in g["notes"]))                  # 64.44 - 2 x 6.86

    def test_a_buy_that_fits(self):
        ex = risk_rules.exposure({"holdings": [{"symbol": "GOOGL", "market_value": 1000.0}], "cash": 9000.0})
        g = risk_rules.gate({"symbol": "GOOGL", "side": "buy", "shares": 5, "limit": 300.0}, ex, held=3, atr=10, lowest_lot=340,
                            stabilized=True)
        self.assertEqual((g["allowed"], g["shares"]), (True, 5))
        g = risk_rules.gate({"symbol": "GOOGL", "side": "buy", "shares": 20, "limit": 300.0}, ex, held=3)
        self.assertEqual(g["shares"], 6)                            # 30% of 10,000 = 3,000; 2,000 of room left

    def test_a_buy_keeps_the_cash_target(self):
        ex = risk_rules.exposure({"holdings": [{"symbol": "GOOGL", "market_value": 9000.0}], "cash": 1500.0})
        g = risk_rules.gate({"symbol": "AAPL", "side": "buy", "shares": 3, "limit": 200.0}, ex)
        self.assertEqual(g["shares"], 2)                            # 10% of 10,500 stays: 450 to spend
        self.assertTrue(any(b.startswith("Cash is 0.0%") for b in risk_rules.summary(risk_rules.exposure(state()))))


class DailyActionTest(unittest.TestCase):
    def test_overweight_intc_is_trimmed_into_a_rebound_and_buys_are_blocked(self):
        st = state()
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice())
        self.assertEqual(a["action"], "trim")
        self.assertEqual([(o["shares"], o["limit"]) for o in a["orders"]], [(23, 75.18), (22, 78.61)])   # +1.5 / +2 x ATR
        self.assertFalse(a["buy_allowed"])
        self.assertFalse(any("CGT" in r or "tax" in r for r in a["reasons"]))  # the user asked for no tax notes
        self.assertEqual(a["after"]["cash_pct"], 13.8)                          # 45 x 64.89 of 21,180
        self.assertGreater(a["after"]["cluster_excess"], 0)                     # chips still above 70% after INTC

    def test_reach_rates_follow_the_drop_cause(self):
        st = state()
        intc = {"date": "2026-10-30", "prev_date": "2026-10-29", "sessions_ago": 0, "drop_atr": 5.1, "earnings": True}
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(big_drop=intc))
        self.assertEqual([o["hit_pct"] for o in a["orders"]], [39, 28])
        self.assertIn("an earnings drop of >= 4 x ATR", a["reasons"][1])
        other = {**intc, "drop_atr": 2.4, "earnings": False}
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(big_drop=other))
        self.assertEqual([o["hit_pct"] for o in a["orders"]], [65, 54])
        for drop in (None, {**intc, "earnings": None}):          # no recent drop, or its cause unknown: no rate quoted
            a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(big_drop=drop))
            self.assertEqual([o["hit_pct"] for o in a["orders"]], [None, None])
            self.assertEqual(a["evidence"], "rule (symbol cap)")

    def test_report_into_the_drop_from_the_calendar(self):
        from unittest import mock
        drop = {"date": "2026-10-30", "prev_date": "2026-10-29"}
        cal = lambda rows: (lambda d, path=None: rows.get(d.isoformat(), []))
        for rows, expected in (({"2026-10-29": [{"symbol": "INTC", "when": "time not given"}]}, True),     # after the close, time unknown
                               ({"2026-10-30": [{"symbol": "INTC", "when": "before the open"}]}, True),
                               ({"2026-10-29": [{"symbol": "INTC", "when": "before the open"}]}, False),   # a day earlier: not this drop
                               ({}, False)):
            with mock.patch("markets.events.earnings_on", side_effect=cal(rows)):
                self.assertEqual(daily_action.reported_into("INTC", drop), expected, rows)
        with mock.patch("markets.events.earnings_on", side_effect=RuntimeError("down")):
            self.assertIsNone(daily_action.reported_into("INTC", drop))

    def test_risk_plan_can_size_the_trim_above_the_symbol_cap(self):
        st = state()
        plan = {"sell": {"INTC": {"usd": 4000.0, "shares": 62}, "SKHY": {"usd": 600.0, "shares": 4}}, "total_usd": 4600.0,
                "vol_before_pct": 60.1, "vol_after_pct": 52.3}
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(), plan)
        self.assertEqual(sum(o["shares"] for o in a["orders"]), 62)              # the cap alone: 45
        self.assertIn("45 shares over", a["reasons"][0])
        self.assertTrue(any(r.startswith("Risk plan: sell $4,600") and "INTC $4,000, SKHY $600" in r for r in a["reasons"]))
        self.assertTrue(a["evidence"].endswith("+ risk plan"))

    def test_report_night_comes_first_and_is_sized_to_the_loss_budget(self):
        st = state()
        ex = risk_rules.exposure(st)
        straddle = lambda sym, d, px: {"implied_move_pct": 9.0, "expiry": "2026-10-31"}       # 9% straddle -> 7.6% typical
        a = daily_action.decide("INTC", st["holdings"][0], ex, advice(report=True), implied_fn=straddle)
        self.assertEqual(a["action"], "earnings")
        self.assertAlmostEqual(a["event"]["loss_pct"], 20.0, delta=0.1)                      # 2.635 x 7.6%
        keep = daily_action.risk_budget.BUDGET_PCT / 100 * ex["total"] / 0.20
        self.assertEqual(a["orders"][0]["shares"], math.ceil((ex["mv"]["INTC"] - keep) / 64.89))
        self.assertTrue(a["orders"][0]["before_report"])
        self.assertIn("over the", a["headline"])
        small = state(intc_qty=5)                                                              # within the budget: hold through
        a = daily_action.decide("INTC", small["holdings"][0], risk_rules.exposure(small), advice(report=True), implied_fn=straddle)
        self.assertEqual((a["action"], a["orders"]), ("earnings", []))

    def test_every_line_in_english_and_chinese(self):
        st = state()
        plan = {"sell": {"INTC": {"usd": 4000.0, "shares": 62}}, "total_usd": 4000.0, "vol_before_pct": 60.1, "vol_after_pct": 52.3,
                "bad_day_loss_before": 1500.0, "bad_day_loss_after": 600.0}
        drop = {"date": "2026-10-30", "prev_date": "2026-10-29", "sessions_ago": 0, "drop_atr": 5.1, "earnings": True}
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(big_drop=drop), plan, None,
                                implied_fn=lambda *x: None)
        self.assertTrue(a["headline_zh"].startswith("等反弹减仓 62 股"))
        self.assertTrue(all(n["en"] and n["zh"] and n["zh"] != n["en"] for n in a["notes"]))
        self.assertGreaterEqual(sum(n["key"] for n in a["notes"]), 3)
        self.assertLess(sum(n["key"] for n in a["notes"]), len(a["notes"]))     # some lines wait under "more"
        self.assertEqual(a["orders"][0]["text"]["zh"], "卖 31 股 @ $75.18 · 历史成交率 39%")
        self.assertEqual(a["evidence_zh"], "规则（单只上限） + 按下跌原因的成交率 + 风险计划")
        self.assertIn("大跌后尚未企稳，暂不买入", a["blocks_zh"])

    def test_every_action_ends_with_budget_and_market(self):
        st = state()
        plan = {"sell": {}, "total_usd": 0.0, "bad_day_loss_before": 1500.0, "bad_day_loss_after": 600.0}
        market = {"state": "tense", "label": "VIX in its top 20% of the year", "vix": 22.0, "vix_pctile": 0.85, "spy_5d_pct": -1.2,
                  "n": 152, "up5": 61, "mean5": 0.37, "p10_5": -3.9}
        adv = advice()
        adv["catalysts_7d"].append({"title": "US Midterm Elections", "impact": "high", "bucket": "macro", "category": "macro", "days_away": 1})
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), adv, plan, market, implied_fn=lambda *x: None)
        text = " | ".join(a["reasons"])
        self.assertIn("a 1-in-20 bad day would lose $1,500", text)
        self.assertIn("after the plan's sales $600", text)
        self.assertIn("Market: VIX in its top 20% of the year", text)
        self.assertIn("5 US election days", text)

    def test_hold_when_within_the_caps(self):
        st = {"holdings": [{"symbol": "INTC", "quantity": 10, "avg_cost": 80, "market_value": 650.0, "quote": {"price": 65.0}},
                           {"symbol": "GOOGL", "quantity": 10, "avg_cost": 280, "market_value": 2800.0, "quote": {"price": 280.0}}],
              "cash": 0.0}
        st["holdings"].append({"symbol": "AAPL", "quantity": 10, "avg_cost": 200, "market_value": 2000.0, "quote": {"price": 200.0}})
        st["holdings"].append({"symbol": "MSFT", "quantity": 10, "avg_cost": 300, "market_value": 3000.0, "quote": {"price": 300.0}})
        a = daily_action.decide("INTC", st["holdings"][0], risk_rules.exposure(st), advice(stabilized=True, events=False))
        self.assertEqual(a["action"], "hold")
        self.assertIn("Cash $0.00 buys 0 shares at $65.00 (keeping 10% in cash)", a["blocks"])

    def test_compute_keeps_the_answer_for_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            out = daily_action.compute(path=db, state=state(), analyze=lambda s, path=None, state=None: advice(),
                                       plan_fn=lambda st, ex, path=None: None, market_fn=lambda path=None: None,
                                       model_fn=lambda *a, **k: None)
            self.assertEqual(out["actions"][0]["action"], "trim")
            import sqlite3
            rows = sqlite3.connect(db).execute("SELECT symbol, action FROM daily_actions").fetchall()
            self.assertEqual(rows, [("INTC", "trim")])


if __name__ == "__main__":
    unittest.main()


class WeekReportsTest(unittest.TestCase):
    """Q.17 round 4: holdings reporting within a week count as one report-night exposure."""

    def nights(self, *rows):
        return {s: {"days_away": d, "loss_usd": l, "source": "atr"} for s, d, l in rows}

    def test_two_reports_over_the_budget_together_get_a_key_note(self):
        acts = [{"symbol": "AAA", "notes": [], "reasons": []}, {"symbol": "ZZZ", "notes": [], "reasons": []}]
        w = daily_action.week_reports(self.nights(("BBB", 3, 700.0), ("AAA", 0, 600.0), ("CCC", 20, 900.0)), acts, 1000.0, 20000.0)
        self.assertEqual((w["symbols"], w["loss_usd"], w["over"]), (["AAA", "BBB"], 1300.0, True))   # by days away
        self.assertIn("2 只持仓一周内发财报", w["text"]["zh"])
        self.assertTrue(acts[0]["notes"][0]["key"])
        self.assertEqual(acts[1]["notes"], [])                          # not reporting this week

    def test_under_the_budget_or_alone_no_note(self):
        acts = [{"symbol": "AAA", "notes": [], "reasons": []}]
        w = daily_action.week_reports(self.nights(("AAA", 0, 300.0), ("BBB", 2, 300.0)), acts, 1000.0, 20000.0)
        self.assertFalse(w["over"])
        self.assertNotIn("text", w)
        self.assertEqual(acts[0]["notes"], [])
        self.assertIsNone(daily_action.week_reports(self.nights(("AAA", 0, 5000.0)), acts, 1000.0, 20000.0))


class NightsFromAdviceTest(unittest.TestCase):
    def test_reports_within_five_sessions_from_latest_json(self):
        from unittest import mock
        adv = {"symbols": {"AAA": {"barrier": {"atr": 2.0, "close": 100.0, "calendar": {"to_report_days": 2}}},
                           "BBB": {"barrier": {"atr": 2.0, "close": 100.0, "calendar": {"to_report_days": 40}}}}}
        st = {"holdings": [{"symbol": "AAA"}, {"symbol": "BBB"}, {"symbol": "CCC"}]}
        ex = {"mv": {"AAA": 10000.0, "BBB": 10000.0, "CCC": 5000.0}}
        with mock.patch.object(daily_action.model_advice, "load", return_value=adv):
            n = daily_action._nights_from_advice(st, ex)
        self.assertEqual(list(n), ["AAA"])
        # 1.6 x 2% ATR = 3.2% typical, x EVENT_K 2.635 = 8.4% (rounded) of $10,000
        self.assertAlmostEqual(n["AAA"]["loss_usd"], 10000.0 * n["AAA"]["loss_pct"] / 100, places=2)
        self.assertEqual(n["AAA"]["sessions_away"], 2)


class MarketWeatherTest(unittest.TestCase):
    def _candles(self, vals):
        return {"candles": [{"time": f"2026-01-{i+1:02d}", "close": v} for i, v in enumerate(vals)]}

    def test_live_panic_chip_selloff_and_macro_headwind(self):
        live = {
            ".VIX": {"price": 34.0, "change_pct": 19.5},
            ".SPX": {"price": 6968.0, "change_pct": -2.9},
            ".SOX": {"price": 8523.9, "change_pct": -9.51},
            "US10Y": {"price": 4.92},
            "US2Y": {"price": 4.62},
            ".DXY": {"price": 103.8},
            "@CL.1": {"price": 84.1},
            "@LCO.1": {"price": 92.6},
        }
        charts = {
            "SPY": self._candles([100, 100, 100, 100, 100, 96.0]),      # -4.0% 5d
            "SOXX": self._candles([100, 100, 100, 100, 100, 91.0]),     # -9.0% 5d
            "HYG": self._candles([100.0] * 60 + [98.5]),
            "IEF": self._candles([100.0] * 61),
        }
        hist = {
            "as_of": "2026-11-19", "y10_60d_ago": 4.49, "y2_60d_ago": 3.99,
            "dxy_60d_ago": 99.2, "credit_chg20": -0.2, "credit_chg60": -1.1,
            "oil_20d_ago": 80.0, "oil_60d_ago": 83.5,
        }
        fake_events = [{"date": "2026-11-27", "days_away": 7, "kind": "policy", "impact": "medium",
                        "title_en": "Export curbs pause ends", "title_zh": "出口限制暂停期届满"}]
        m = risk_budget.market_state(
            features_fn=lambda: {"macro_vix": 24.6, "macro_vix_pctile_252d": 0.91},
            chart_fn=lambda sym, rng, path=None: charts.get(sym, {"candles": []}),
            live_fn=lambda path=None: live,
            macro_hist_fn=lambda: hist,
            events_fn=lambda path=None: fake_events,
        )
        self.assertEqual(m["state"], "fearful")
        self.assertEqual(m["vix"], 34.0)
        self.assertEqual(m["vix_source"], "cnbc_live")
        self.assertTrue(m["sector_stress"])
        self.assertEqual(m["macro"]["level"], "heavy")
        self.assertEqual(set(m["macro"]["flags"]), {"rates", "dollar", "credit", "oil"})
        self.assertEqual(m["macro"]["oil"], 84.1)
        self.assertEqual(m["macro"]["brent"], 92.6)
        self.assertEqual(m["upcoming"], fake_events)
        self.assertEqual(m["severity"], "critical")
        self.assertTrue(m["tighten_buys"])
        self.assertIn("市场恐慌", m["title"]["zh"])
        self.assertIn("半导体踩踏", m["title"]["zh"])
        self.assertIn("宏观紧缩重压", m["title"]["zh"])
        by_id = {p["id"]: p for p in m["pills"]}
        for pid in ("vix", "spx1d", "spy5d", "sox1d", "soxx5d", "us10y", "dxy", "credit", "oil"):
            self.assertTrue(by_id[pid]["stress"], pid)

    def test_calm_market_stays_info_without_tightening(self):
        live = {
            ".VIX": {"price": 14.2, "change_pct": -2.1},
            ".SPX": {"price": 7100.0, "change_pct": 0.4},
            ".SOX": {"price": 9200.0, "change_pct": 0.8},
            "US10Y": {"price": 4.10},
            "US2Y": {"price": 3.80},
            ".DXY": {"price": 99.5},
            "@CL.1": {"price": 68.0},
            "@LCO.1": {"price": 72.0},
        }
        charts = {
            "SPY": self._candles([100, 100.2, 100.4, 100.6, 100.8, 101.2]),
            "SOXX": self._candles([100, 100.5, 101.0, 101.2, 101.5, 102.0]),
            "HYG": self._candles([100.0] * 61),
            "IEF": self._candles([100.0] * 61),
        }
        hist = {"as_of": "2026-11-19", "y10_60d_ago": 4.05, "y2_60d_ago": 3.82, "dxy_60d_ago": 99.0,
                "oil_20d_ago": 67.5, "oil_60d_ago": 69.0}
        m = risk_budget.market_state(
            features_fn=lambda: {"macro_vix": 14.5, "macro_vix_pctile_252d": 0.30},
            chart_fn=lambda sym, rng, path=None: charts.get(sym, {"candles": []}),
            live_fn=lambda path=None: live,
            macro_hist_fn=lambda: hist,
            events_fn=lambda path=None: [],
        )
        self.assertEqual(m["state"], "normal")
        self.assertFalse(m["sector_stress"])
        self.assertEqual(m["macro"]["level"], "calm")
        self.assertEqual(m["severity"], "info")
        self.assertFalse(m["tighten_buys"])

    def test_upcoming_macro_policy_filters_noise_and_deduplicates(self):
        import sqlite3
        from datetime import date
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            conn = sqlite3.connect(db)
            conn.execute(
                "CREATE TABLE collected_catalysts (id INTEGER PRIMARY KEY, event_date TEXT, kind TEXT, "
                "symbol TEXT, impact_level TEXT, title_zh TEXT, title_en TEXT)"
            )
            conn.executemany(
                "INSERT INTO collected_catalysts(event_date, kind, symbol, impact_level, title_zh, title_en) VALUES (?,?,?,?,?,?)",
                [
                    ("2026-11-27", "policy", "MACRO", "medium", "中国对美镓锗出口限制暂停期届满", "China gallium/germanium export curbs pause ends"),
                    ("2026-11-30", "policy", "AMD", "high", "AMD 市场份额超博通", "AMD expected to beat Broadcom share"),
                    ("2026-12-04", "policy", "MACRO", "medium", "美国232条款多晶硅关税生效", "Section 232 polysilicon tariff takes effect"),
                    ("2026-12-04", "policy", "MACRO", "medium", "美国第232条多晶硅关税生效", "US Section 232 polysilicon tariff takes effect"),
                ],
            )
            conn.commit()
            conn.close()
            out = risk_budget._upcoming_macro_policy(path=db, today=date(2026, 11, 20), days=14)
            titles = [x["title_zh"] for x in out]
            self.assertTrue(any("期权交割" in t for t in titles))
            self.assertTrue(any("非农" in t for t in titles))
            self.assertTrue(any("镓锗" in t for t in titles))
            self.assertFalse(any("AMD" in t for t in titles))
            self.assertEqual(sum(1 for t in titles if "多晶硅" in t), 1)

    def test_compute_passes_market_and_exposes_chinese_breaches_and_budget(self):
        seen = {}

        def fake_model(*args, **kwargs):
            seen["market"] = kwargs.get("market")
            return None

        fake_market = {"state": "fearful", "severity": "critical", "tighten_buys": True,
                       "label": "fearful", "label_zh": "恐慌", "n": 66, "up5": 67, "mean5": 0.86, "p10_5": -4.6}
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            out = daily_action.compute(
                path=db, state=state(), analyze=lambda s, path=None, state=None: advice(),
                plan_fn=lambda st, ex, path=None: {"sell": {}, "total_usd": 0.0, "bad_day_loss_before": 1400.0},
                market_fn=lambda path=None: fake_market,
                model_fn=fake_model,
            )
            self.assertEqual(seen["market"], fake_market)
            p = out["portfolio"]
            self.assertEqual(p["market"], fake_market)
            self.assertEqual(p["bad_day_loss"], 1400.0)
            self.assertGreater(p["loss_budget_usd"], 0)
            self.assertTrue(any("芯片 / 存储类" in line for line in p["breaches_zh"]))
            self.assertTrue(any("现金仅占组合" in line for line in p["breaches_zh"]))
