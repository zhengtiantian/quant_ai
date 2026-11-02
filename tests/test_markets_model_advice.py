"""The daily action from the models' forecast (markets/model_advice.py). No files: the forecast is given."""
import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from markets import model_advice  # noqa: E402


def forecast(call="hold", p20=52.0, generated=None, reliable=True):
    hz = lambda p, q10, q50, q90: {"p_up": p, "q10": q10, "q50": q50, "q90": q90}
    return {"generated_at": generated or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_quality": {"20": {"ic": 0.05, "ic_t": 2.6, "q10_breached_pct": 10.4, "base_up_pct": 57.0}},
            "decision_test": {"2023-2026 (held out for thresholds)": {"reduce": {"mean20": -0.8}, "hold": {"mean20": 1.6},
                                                                      "add": {"mean20": 2.9}, "reduce_minus_hold_t": -2.1}},
            "symbols": {"INTC": {"as_of": "2026-11-02", "close": 60.19, "call": call, "direction_reliable": reliable,
                                 "events": [{"date": "2026-10-30", "ret_pct": -28.6, "cause_en": "Weak Q4 outlook", "cause_zh": "四季度指引疲软",
                                             "sources": [{"title": "Intel plunges as weak Q4 forecast overshadows beat"}]}],
                                 "horizons": {"5": hz(48.0, -9.0, -0.4, 8.0), "20": hz(p20, -18.0, 1.0, 20.0), "60": hz(55.0, -28.0, 3.0, 35.0)},
                                 "drivers": [{"feature": "dd_252", "value": -53.3, "effect_pct": 1.2, "en": "below the 1-year high", "zh": "距一年高点"}]}}}


HOLD = {"symbol": "INTC", "quantity": 142.18, "quote": {"price": 60.19}}
STATS = {"atr14": 6.86, "stabilized": True}


class ModelAdviceTest(unittest.TestCase):
    def test_hold_shows_the_odds_in_both_languages(self):
        a = model_advice.render("INTC", HOLD, STATS, 64.44, 0.0, forecast())
        self.assertEqual((a["action"], a["orders"]), ("hold", []))
        self.assertTrue(a["headline_zh"].startswith("模型建议：持有"))
        odds = [n for n in a["notes"] if n["section"] == "odds"]
        self.assertEqual(len(odds), 3)
        self.assertIn("类似情况上涨的概率 52%", odds[1]["zh"])
        self.assertTrue(any(n["section"] == "why" and "四季度指引疲软" in n["zh"] for n in a["notes"]))
        self.assertEqual(a["notes"][0]["section"], "why")                       # why it fell comes first
        self.assertTrue(all(n["en"] and n["zh"] for n in a["notes"]))

    def test_reduce_sells_a_third(self):
        a = model_advice.render("INTC", HOLD, STATS, 64.44, 0.0, forecast("reduce", 35.0))
        self.assertEqual((a["action"], a["orders"][0]["side"], a["orders"][0]["shares"]), ("reduce", "sell", 47))

    def test_add_is_never_shown_too_close_to_the_cheapest_lot(self):
        a = model_advice.render("INTC", HOLD, STATS, 64.44, 0.0, forecast("add", 70.0))
        self.assertEqual((a["action"], a["orders"]), ("hold", []))           # 60.19 is within 2 x 6.86 of 64.44
        self.assertTrue(any("$50.72 或更低" in n["zh"] for n in a["notes"] if n["section"] == "add"))
        far = {**HOLD, "quote": {"price": 50.0}}
        a = model_advice.render("INTC", far, STATS, 64.44, 10000.0, forecast("add", 70.0))
        self.assertEqual((a["action"], a["orders"][0]["shares"]), ("add", 14))
        a = model_advice.render("INTC", far, {**STATS, "stabilized": False}, 64.44, 10000.0, forecast("add", 70.0))
        self.assertEqual(a["action"], "hold")                                  # still falling

    def test_an_unreliable_direction_is_not_acted_on(self):
        a = model_advice.render("INTC", HOLD, STATS, 64.44, 0.0, forecast("reduce", 35.0, reliable=False))
        self.assertEqual((a["action"], a["orders"]), ("hold", []))
        self.assertIn("没有可靠的方向信号", a["headline_zh"])

    def test_getting_out_shows_the_models_view_without_sizing(self):
        f = forecast(reliable=False)
        st = lambda u, l, n, hm, hw, bm, bw: {"sell_at": u, "stop_at": l, "n": n, "hold_mean": hm, "hold_worst5": hw, "barrier_mean": bm,
                                              "barrier_worst5": bw, "upper_first": 52.2, "lower_first": 32.3}
        f["symbols"]["INTC"]["barrier"] = {"situation": "deep_badnews", "model_5": {"p_upper": 47.6, "p_lower": 1.9, "p_none": 50.5},
                                           "news_20": {"guide_cut_20": 28.0},
                                           "sets": {"5": st(66.95, 49.5, 4793, 0.55, -10.8, 0.21, -9.0), "20": st(70.44, 46.01, 4758, 2.13, -18.0, 0.62, -14.0)}}
        f["news_study"] = {"guidance cut": {"days": 916, "60d_vs_all": {"mean": -2.35, "t": -3.62}}}
        held = {**HOLD, "avg_cost": 87.69}
        a = model_advice.render("INTC", held, STATS, 64.44, 0.0, f, total=20200.0, budget_pct=5.0)
        self.assertEqual((a["action"], a["orders"]), ("hold", []))
        rec = [n["zh"] for n in a["notes"] if n["section"] == "recover"]
        self.assertTrue(any("先反弹到 $66.95 的概率 48%" in r for r in rec))
        self.assertTrue(any("不要补仓" in r for r in rec))
        self.assertFalse(any("风险预算" in n["zh"] for n in a["notes"]))

    def test_events_ahead_with_their_history(self):
        f = forecast(reliable=False)
        f["calendar_study"] = {"fomc_ahead_5d": {"mean": 0.046}, "no_fomc_5d": {"mean": 0.319}, "opex_week": {"mean": -0.092, "t": -2.21},
                               "day_before_report": {"mean": 0.155, "t": 3.16}}
        f["symbols"]["INTC"]["barrier"] = {"calendar": {"to_fomc": 3, "to_opex": 2, "to_nfp": 9, "pre_holiday": False, "to_report_days": 1}}
        a = model_advice.render("INTC", HOLD, STATS, 64.44, 0.0, f)
        ev = [n["zh"] for n in a["notes"] if n["section"] == "events"]
        self.assertTrue(any("3 个交易日后美联储议息" in e and "+0.05%" in e for e in ev))
        self.assertTrue(any("期权到期周" in e for e in ev))
        self.assertTrue(any("1 天后发财报" in e and "+0.15%" in e for e in ev) or any("1 天后发财报" in e and "+0.16%" in e for e in ev))

    def test_no_forecast_or_stale(self):
        self.assertIsNone(model_advice.render("TSM", HOLD, STATS, None, 0.0, forecast()))
        a = model_advice.render("INTC", HOLD, STATS, 64.44, 0.0, forecast(generated="2026-10-01T00:00:00+00:00"))
        self.assertTrue(a["stale"])


if __name__ == "__main__":
    unittest.main()
