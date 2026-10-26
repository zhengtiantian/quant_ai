"""Earnings pulse watch (Q.5 v4): release minute, placements, whole-share ladders, saved orders. No network."""
import sys
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import advisor, pulse  # noqa: E402

NY = ZoneInfo("America/New_York")
DAY = date(2026, 10, 29)


def at(h, m, s=0):
    return datetime(2026, 10, 29, h, m, s, tzinfo=NY)


class FakeModel:
    """Acts at the first placement after the release whose move is at least act_above."""

    def __init__(self, act_above=None):
        self.act_above, self.calls = act_above, []

    def __call__(self, req):
        self.calls.append(req)
        act = self.act_above is not None and req["place"] != "pre" and req["move_at"] >= self.act_above
        return {"act": act, "best": {"plan": f"{req['place']} | 1/2/3 | lim2", "ladder": [1, 2, 3], "weights": [1, 1, 1],
                                      "buyback": "lim2", "pred_pct": 0.8},
                "stock_record": {"reports": 30, "acted": 13, "mean_when_acting": 1.65, "win": 0.77}, "model_record": {}}


def run(watch, prices):
    """prices: [(datetime, price)]; returns the orders, if any."""
    out = None
    for t, p in prices:
        out = watch.step(t, p) or out
    return out


class WatchTest(unittest.TestCase):
    def evening(self, path):
        """Close 100, flat to 16:00, jumps to +8% at 16:01, +11% by 16:05, fades to +5% by 17:10."""
        pts = [(at(15, 50) + timedelta(seconds=30 * i), 100.0) for i in range(20)]
        pts += [(at(16, 1), 108.0), (at(16, 2), 109.0), (at(16, 4), 110.0), (at(16, 6), 111.0), (at(16, 11), 109.0),
                (at(16, 21), 107.0), (at(16, 31), 106.0), (at(16, 56), 105.5), (at(17, 10), 105.0)]
        return pts

    def test_release_minute_and_placements_in_order(self):
        m = FakeModel()
        w = pulse.Watch("INTC", 98.9, 106.74, 100.0, DAY, {"run5": -3.0}, predict=m)
        self.assertIsNone(run(w, self.evening(None)))
        self.assertEqual(w.release, at(16, 1))
        self.assertEqual([c["place"] for c in m.calls], ["r+1", "r+3", "r+5", "r+10", "r+20", "r+30", "c-5"])
        r5 = next(c for c in m.calls if c["place"] == "r+5")
        self.assertAlmostEqual(r5["move_at"], 11.0)                 # the price at 16:06, when r+5 is asked
        self.assertEqual(r5["release_min"], 16 * 60 + 1)

    def test_first_placement_that_clears_becomes_whole_share_orders(self):
        w = pulse.Watch("INTC", 98.9, 106.74, 100.0, DAY, {}, predict=FakeModel(act_above=9.5))
        o = run(w, self.evening(None))
        self.assertEqual(o["plan"], "r+3 | 1/2/3 | lim2")
        self.assertEqual([s["shares"] for s in o["sells"]], [10, 10, 9])          # 30% of 98 whole shares, equal weights
        self.assertEqual([s["limit"] for s in o["sells"]], [111.1, 112.2, 113.3])  # +1/2/3% on 110
        self.assertEqual([r["limit"] for r in o["rebuys"]], [108.88, 109.96, 111.03])
        self.assertEqual(len(w.decisions), 2)                                      # asks nothing after it acted

    def test_close_is_the_last_price_before_the_bell(self):
        w = pulse.Watch("INTC", 10, None, 90.0, DAY, {}, predict=FakeModel())
        w.step(at(15, 58), 100.0)
        w.step(at(16, 0, 30), 100.2)                                # +0.2%: not the release
        self.assertEqual(w.close, 100.0)
        self.assertIsNone(w.release)

    def test_shares_split_by_the_policy_weights(self):
        self.assertEqual(pulse.shares_for(98.9, [1, 1, 1]), [10, 10, 9])           # 29 shares (30% of 98)
        self.assertEqual(pulse.shares_for(98.9, [0.6, 0.3, 0.1]), [17, 9, 3])
        self.assertEqual(pulse.shares_for(6.3, [0.5, 0.3, 0.2]), [1, 0, 0])         # STX: 30% of 6 shares is one share
        self.assertEqual(pulse.shares_for(10, [0.9, 0.05, 0.05]), [1, 1, 1])        # 3 shares: one per step first
        self.assertEqual(pulse.shares_for(0.4, [1]), [])

    def test_saved_with_buy_backs_waiting_and_kept_out_of_daily_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "a.db"
            w = pulse.Watch("INTC", 98.9, 106.74, 100.0, DAY, {}, predict=FakeModel(act_above=9.5))
            o = run(w, self.evening(None))
            pulse.save(o, DAY, db)
            rows = advisor.active_orders(db)
            self.assertEqual(sorted((r["side"], r["status"]) for r in rows), [("buy", "waiting")] * 3 + [("sell", "open")] * 3)
            self.assertTrue(all(r["valid_until"] == "2026-10-30" and r["plan"].startswith("pulse:") for r in rows))
            self.assertEqual(advisor.score_due(db, now=at(23, 0).timestamp() + 2 * 86400, bars_fn=lambda *a: []), 0)

    def test_next_morning_scoring_on_minute_bars(self):
        def bar(h, m, o, hi, lo, c, day=29):
            return {"t": datetime(2026, 10, day, h, m, tzinfo=NY), "o": o, "h": hi, "l": lo, "c": c}
        # sold the +1% step (111.1) at 16:08 and the +2% step (112.2) at 16:20; both bought back at 16:40
        # (the bar opened at 109.5, below 112.2 x 0.98); the +3% step never traded
        bars = [bar(16, 7, 110, 110.5, 109.8, 110.2), bar(16, 8, 110.5, 111.3, 110.4, 111.0), bar(16, 20, 111.5, 112.4, 111.4, 112.0),
                bar(16, 40, 109.5, 109.6, 108.5, 108.9), bar(19, 59, 107, 107, 107, 107), bar(9, 30, 106, 106.5, 105.5, 106, day=30)]
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "a.db"
            w = pulse.Watch("INTC", 98.9, 106.74, 100.0, DAY, {}, predict=FakeModel(act_above=9.5))
            pulse.save(run(w, WatchTest.evening(self, None)), DAY, db)          # orders at 16:06, p0 110
            self.assertEqual(pulse.score_due(db, now=datetime(2026, 10, 30, 9, 0, tzinfo=NY), bars_fn=lambda *a: bars), 0)
            self.assertEqual(pulse.score_due(db, now=datetime(2026, 10, 30, 11, 0, tzinfo=NY), bars_fn=lambda *a: bars), 3)
            card = {(r["side"], r["limit_price"]): r for r in advisor.scorecard(db)["recent"]}
            self.assertEqual(card[("sell", 111.1)]["status"], "scored")
            self.assertAlmostEqual(card[("sell", 111.1)]["edge_pct"], round((111.1 - 111.1 * 0.98) / 110 * 100, 3))   # back 2% under the sale, as in training
            self.assertAlmostEqual(card[("sell", 112.2)]["edge_pct"], round((112.2 - 109.5) / 110 * 100, 3))
            self.assertEqual(card[("sell", 113.3)]["status"], "expired")
            self.assertEqual(card[("buy", 111.03)]["status"], "void")

    def test_premarket_watch_uses_the_previous_close_and_this_mornings_open(self):
        m = FakeModel(act_above=6.0)
        w = pulse.Watch("TSM", 6.75, 444.27, 400.0, DAY, {}, predict=m, session="bmo")   # yesterday's close 400
        pts = [(at(4, 0), 400.5), (at(6, 30), 401.0), (at(7, 1), 420.0), (at(7, 2), 424.0), (at(7, 4), 426.0), (at(7, 6), 425.0)]
        o = run(w, pts)
        self.assertEqual((w.close, w.release), (400.0, at(7, 1)))              # 04:00-06:30 drift is not the release
        self.assertEqual(m.calls[0]["bmo"], 1.0)
        self.assertEqual(o["plan"], "r+1 | 1/2/3 | lim2")                     # +6% at 07:02
        self.assertEqual(o["session"], "bmo")
        bars = [{"t": datetime(2026, 10, 29, 7, 3, tzinfo=NY), "o": 424, "h": 428.6, "l": 423, "c": 428},
                {"t": datetime(2026, 10, 29, 8, 0, tzinfo=NY), "o": 427, "h": 427.5, "l": 418, "c": 419},
                {"t": datetime(2026, 10, 29, 9, 30, tzinfo=NY), "o": 417, "h": 418, "l": 415, "c": 416}]
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "a.db"
            pulse.save(o, DAY, db)
            self.assertTrue(all(r["valid_until"] == "2026-10-29" for r in advisor.active_orders(db)))  # this morning only
            self.assertEqual(pulse.score_due(db, now=datetime(2026, 10, 29, 11, 0, tzinfo=NY), bars_fn=lambda *a: bars), 1)
            card = {(r["side"], r["limit_price"]): r for r in advisor.scorecard(db)["recent"]}
            sold = card[("sell", round(424 * 1.01, 2))]
            self.assertEqual(sold["status"], "scored")                          # sold 428.24 at 07:03, back at 419.68 08:00
            self.assertEqual(sold["eval_date"][:16], "2026-10-29T08:00")

    def test_premarket_only_before_the_open(self):
        held = [{"symbol": "TSM"}, {"symbol": "INTC"}]
        cal = [{"symbol": "TSM", "date": "2026-10-29", "when": "before the open"},
               {"symbol": "INTC", "date": "2026-10-29", "when": "after the close"}]
        self.assertEqual([e["symbol"] for e in pulse.reporting_premarket(held, DAY, cal_fn=lambda s: cal)], ["TSM"])

    def test_shadow_is_kept_scored_and_never_announced(self):
        class TwoModels(FakeModel):
            def __call__(self, req):
                ans = super().__call__(req)
                # the shadow acts one placement later than the champion, with +2/4% steps
                sh_act = req["place"] == "r+5"
                ans["model"] = "catboost"
                ans["shadow"] = {"model": "lgbm_seeds", "role": "shadow", "act": sh_act, "stock_record": {},
                                 "best": {"plan": "r+5 | 2/4 | lim2", "ladder": [2, 4], "weights": [0.5, 0.5], "buyback": "lim2", "pred_pct": 0.9}}
                return ans
        w = pulse.Watch("INTC", 98.9, 106.74, 100.0, DAY, {}, predict=TwoModels(act_above=9.5))
        o = run(w, WatchTest.evening(self, None))
        self.assertEqual((o["plan"], o["model"]), ("r+3 | 1/2/3 | lim2", "catboost"))       # the champion at r+3
        self.assertEqual(w.shadow_orders["plan"], "r+5 | 2/4 | lim2")                       # asked on after it acted
        self.assertEqual([c["place"] for c in w.predict.calls], ["r+1", "r+3", "r+5"])        # and then stopped
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "a.db"
            pulse.save(o, DAY, db)
            pulse.save(w.shadow_orders, DAY, db, shadow=True)
            live = advisor.active_orders(db)
            self.assertTrue(all(r["plan"].startswith("pulse:") for r in live))                # the shadow is not an order to place
            bars = [{"t": datetime(2026, 10, 29, 16, 7, tzinfo=NY), "o": 110.5, "h": 112.5, "l": 110.4, "c": 112.0},
                    {"t": datetime(2026, 10, 29, 16, 40, tzinfo=NY), "o": 109.0, "h": 109.2, "l": 108.0, "c": 108.5},
                    {"t": datetime(2026, 10, 30, 9, 30, tzinfo=NY), "o": 107.0, "h": 107.5, "l": 106.5, "c": 107.0}]
            pulse.score_due(db, now=datetime(2026, 10, 30, 11, 0, tzinfo=NY), bars_fn=lambda *a: bars)
            card = pulse.scorecard(db)
            self.assertEqual((card["champion"]["evenings"], card["shadow"]["evenings"]), (1, 1))
            self.assertFalse(any(r["plan"].startswith("shadow:") for r in advisor.scorecard(db)["recent"]))

    def test_reporting_today_only_after_the_close(self):
        held = [{"symbol": "INTC"}, {"symbol": "TSM"}, {"symbol": "SKHY"}]
        cal = [{"symbol": "INTC", "date": "2026-10-29", "when": "after the close"},
               {"symbol": "TSM", "date": "2026-10-29", "when": "before the open"},
               {"symbol": "SKHY", "date": "2026-10-29", "when": "after the close (Seoul morning)"}]
        self.assertEqual([e["symbol"] for e in pulse.reporting_today(held, DAY, cal_fn=lambda s: cal)], ["INTC"])


if __name__ == "__main__":
    unittest.main()
