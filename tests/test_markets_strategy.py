"""Strategy engine (Q.5): whole-share candidates, the replay on real bars, eligibility. No network."""
import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import strategy as stg  # noqa: E402


def bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


T1 = [{"side": "sell", "shares": 10, "pct": 5.0, "after": None}, {"side": "buy", "shares": 10, "pct": -5.0, "after": 0}]


class CandidatesTest(unittest.TestCase):
    def test_whole_shares_and_the_plans_offered(self):
        c = {p["id"]: p for p in stg.candidates(98.9, 87.0, 106.74, 9.9, 0)}
        self.assertIn("hold", c)
        self.assertNotIn("add", c)                                  # no cash
        self.assertNotIn("breakeven", c)                            # cost 22.7% away, beyond 1.5 x the move
        self.assertEqual(c["t1_50_50"]["legs"][0]["shares"], 24)    # floor(98.9 x 25%)
        self.assertEqual(c["t2"]["legs"][0]["shares"], 14)
        self.assertTrue(all(isinstance(l["shares"], int) for p in c.values() for l in p["legs"]))
        self.assertIn("breakeven", {p["id"] for p in stg.candidates(6.3, 630.0, 680.0, 9.8, 0)})
        self.assertEqual([p["id"] for p in stg.candidates(0.4, 100.0, None, 5.0, 0)], ["hold"])

    def test_add_uses_half_the_cash(self):
        add = next(p for p in stg.candidates(10, 100.0, 90.0, 5.0, 1000.0) if p["id"] == "add")
        self.assertEqual(add["legs"][0]["shares"], 5)                # 500 // 95


class ReplayTest(unittest.TestCase):
    def test_buy_back_only_after_the_sale_and_gain_against_holding(self):
        # Day 1 dips to 94 before anything is sold; day 2 sells at 105; day 3 buys back at 95; ends 100.
        bars = [bar(100, 101, 94, 99), bar(100, 106, 99, 104), bar(98, 99, 94, 96), bar(96, 101, 95, 100)]
        r = stg.replay(T1, 100.0, bars)
        self.assertEqual(r["fills"], [0, 1])
        self.assertAlmostEqual(r["gain"], 10 * (105 - 100) + 10 * (100 - 95))
        self.assertFalse(r["regret"])

    def test_gap_through_fills_at_the_open_and_regret(self):
        bars = [bar(112, 115, 110, 114), bar(114, 118, 113, 117)]   # gaps up through 105, never comes back
        r = stg.replay(T1, 100.0, bars)
        self.assertEqual(r["fills"], [0])
        self.assertAlmostEqual(r["gain"], 10 * (112 - 117))
        self.assertTrue(r["regret"])
        self.assertAlmostEqual(r["hold_pct"], 17.0)

    def test_history_scale_moves_the_limits(self):
        windows = [(100.0, [bar(100, 103, 99, 101), bar(101, 102, 97, 100)])]
        plan = {"legs": T1}
        self.assertEqual(stg.history_test(plan, 100, windows)["leg_fill_rate"], [0.0, 0.0])
        self.assertEqual(stg.history_test(plan, 100, windows, scale=0.5)["leg_fill_rate"], [1.0, 1.0])   # +-2.5%


class WindowsTest(unittest.TestCase):
    def test_entry_lead_matches_today(self):
        bars = [bar(i, i + 1, i - 1, i) for i in range(100, 116)]      # 16 bars, reaction at index 5
        ev = {"bars": bars, "reaction_offset": 5, "move_pct": 3.0}
        entry, after = stg.earnings_windows([ev], lead=2)[0]
        self.assertEqual(entry, 103)                                  # two sessions before the reaction
        self.assertEqual(after[1]["close"], 105)                      # the reaction day at index lead - 1
        self.assertEqual(len(after), 2 + stg.HOLD_AFTER)

    def test_jumps_are_centred_and_scaled_to_the_implied_move(self):
        j = stg.scaled_jumps([{"move_pct": m} for m in (10, 6, -2, 2)], 8.0)
        self.assertAlmostEqual(sum(j), 0.0)
        self.assertAlmostEqual(sorted(abs(x) for x in j)[1], 4.0)   # centred moves 6, 2, -6, -2: median |.| 4 -> 8


class EligibleTest(unittest.TestCase):
    def plan(self, legs, **h):
        return {"legs": legs, "history": {"n": 16, "n_eff": 16, "t": 2.5, "mean_pct": 0.3, "win_rate": 0.6, "loss_rate": 0.3, **h}}

    def test_profit_first(self):
        below = [{**T1[0], "below_cost": True}, {**T1[1], "below_cost": False}]
        self.assertTrue(stg.eligible(self.plan(below))[0])                        # below cost, but bought back
        self.assertFalse(stg.eligible(self.plan(below[:1]))[0])                    # below cost, no buy-back
        self.assertFalse(stg.eligible(self.plan(below, mean_pct=-0.6))[0])         # trailed holding
        self.assertFalse(stg.eligible(self.plan(below, n=5))[0])                   # too few windows
        self.assertFalse(stg.eligible(self.plan(below, win_rate=0.2, loss_rate=0.7))[0])
        self.assertTrue(stg.eligible({"legs": [], "history": {}})[0])               # hold is always allowed
        self.assertFalse(stg.eligible(self.plan(below, t=1.0, n=47, n_eff=15.7))[0])  # TSM's ladder: +0.36%, t 1.0


class EvaluateTest(unittest.TestCase):
    def test_earnings_horizon_and_validity(self):
        candles = [{"time": i, **bar(100, 102, 98, 100 + (i % 3 - 1))} for i in range(250)]
        reactions = [{"move_pct": m, "reaction_offset": 5, "history_source": "X",
                      "bars": [bar(100, 101, 99, 100)] * 5 + [bar(100 + m, 101 + m, 99 + m, 100 + m)] * 11}
                     for m in (8, -6, 5, -9, 7, 4, -3, 6)]
        e = {"date": "2026-10-29", "when": "after the close", "implied": {"implied_move_pct": 9.9}}
        r = stg.evaluate_holding("INTC", 98.9, 87.0, 106.74, 0, candles, 3.0, e, reactions, date(2026, 10, 25))
        self.assertTrue(r["earnings_in_horizon"])
        self.assertEqual(r["sessions_to_reaction"], 4)                 # Mon-Thu, reacts Friday (index 4)
        self.assertEqual(r["valid_until"], "2026-11-13")               # 5 + 10 sessions after Sunday
        self.assertEqual(r["replay_windows"], 8)
        self.assertAlmostEqual(r["history_scale"], 6.0 / 9.9, places=3)     # median of the past abs moves / the implied move
        self.assertTrue(all("eligible" in c for c in r["candidates"]))
        quiet = stg.evaluate_holding("TSM", 6.75, 418.0, 444.0, 0, candles, 3.0, None, [], date(2026, 10, 25))
        self.assertFalse(quiet["earnings_in_horizon"])
        self.assertEqual(quiet["move_scale_basis"], "2 x ATR(14)")


if __name__ == "__main__":
    unittest.main()
