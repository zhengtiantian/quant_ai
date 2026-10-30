"""Portfolio: average cost, sale checks, dividends / fees, watchlist dedupe. No network (quotes are stubbed)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from markets import budget_link  # noqa: E402
from markets import portfolio as pf  # noqa: E402

REAL_FETCH_QUOTE = pf.fetch_quote


def fake_quote(symbol, asset_class=None, known_prev_close=None):
    prices = {"NVDA": (213.0, 217.0), "VOO": (700.0, 704.0), "AAPL": (250.0, 248.0)}
    if symbol not in prices:
        raise ValueError("not found")
    price, prev = prices[symbol]
    return {"symbol": symbol, "asset_class": "stocks", "name": symbol, "price": price, "session": "post",
            "prev_close": prev, "regular_price": prev, "session_change": 0.0, "session_change_pct": 0.0}


class PortfolioTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "p.db"
        pf.init_db(self.db)
        self._q = mock.patch.object(pf, "fetch_quote", side_effect=fake_quote)
        self._q.start()
        # No network in tests: the regular-session batch path answers nothing unless a test opts in.
        self.real_batch = pf.fetch_quotes_batch
        self._b = mock.patch.object(pf, "fetch_quotes_batch", return_value={})
        self._b.start()
        pf._BATCH_MISSES.clear()
        self._fx = mock.patch.object(budget_link, "usd_cny", return_value=None)   # no call to job_agent
        self._fx.start()
        self.addCleanup(self._fx.stop)

    def tearDown(self):
        self._q.stop(); self._b.stop(); self.tmp.cleanup()

    def tx(self, **kw):
        return pf.save_transaction({"trade_date": "2026-09-01", "symbol": "NVDA", "side": "buy", **kw}, self.db)

    def test_average_cost_includes_fees(self):
        self.tx(quantity=10, price=180, fee=2)
        self.tx(quantity=5, price=210, trade_date="2026-09-15")
        h = pf.get_state(path=self.db)["holdings"][0]
        self.assertAlmostEqual(h["avg_cost"], (1800 + 2 + 1050) / 15)
        self.assertAlmostEqual(h["market_value"], 15 * 213)
        self.assertAlmostEqual(h["day_pnl"], 15 * (213 - 217))

    def test_sale_realises_against_average_cost_and_keeps_it(self):
        self.tx(quantity=10, price=100)
        self.tx(quantity=10, price=200, trade_date="2026-09-02")
        self.tx(side="sell", quantity=5, price=180, fee=1, trade_date="2026-09-03")
        st = pf.get_state(path=self.db)
        h = st["holdings"][0]
        self.assertAlmostEqual(h["avg_cost"], 150)
        self.assertAlmostEqual(h["quantity"], 15)
        self.assertAlmostEqual(st["totals"]["realized"], 5 * (180 - 150) - 1)

    def test_overselling_is_rejected_and_not_saved(self):
        self.tx(quantity=1, price=100)
        with self.assertRaises(ValueError):
            self.tx(side="sell", quantity=2, price=100, trade_date="2026-09-02")
        self.assertEqual(len(pf.get_state(path=self.db)["transactions"]), 1)

    def test_deleting_a_buy_that_a_sale_depends_on_is_refused(self):
        buy = self.tx(quantity=1, price=100)
        self.tx(side="sell", quantity=1, price=120, trade_date="2026-09-02")
        with self.assertRaises(ValueError):
            pf.delete_transaction(buy, self.db)

    def test_fractional_full_exit_closes_the_position(self):
        self.tx(symbol="VOO", quantity=0.3, price=600)
        self.tx(symbol="VOO", quantity=0.2, price=650, trade_date="2026-09-02")
        self.tx(symbol="VOO", side="sell", quantity=0.5, price=700, trade_date="2026-09-03")
        st = pf.get_state(path=self.db)
        self.assertEqual(st["holdings"], [])
        self.assertAlmostEqual(st["closed"][0]["realized"], 0.5 * 700 - (180 + 130))

    def test_dividend_and_fee(self):
        self.tx(quantity=1, price=100)
        self.tx(side="dividend", price=0.25, trade_date="2026-10-01")
        self.tx(side="fee", price=0.10, trade_date="2026-10-02")
        t = pf.get_state(path=self.db)["totals"]
        self.assertAlmostEqual(t["dividends"], 0.25)
        self.assertAlmostEqual(t["realized"], -0.10)

    def test_validation(self):
        for bad in ({"symbol": "12AB"}, {"trade_date": "yesterday"}, {"quantity": 0, "price": 10}, {"side": "short"}):
            with self.assertRaises(ValueError, msg=bad):
                self.tx(**{"quantity": 1, "price": 10, **bad})

    def test_watchlist_excludes_held_symbols_and_rejects_unknown(self):
        pf.add_watch("aapl", path=self.db)
        pf.add_watch("NVDA", path=self.db)
        self.tx(quantity=1, price=100)
        st = pf.get_state(path=self.db)
        self.assertEqual([w["symbol"] for w in st["watchlist"]], ["AAPL"])
        self.assertAlmostEqual(st["watchlist"][0]["day_change"], 2.0)
        pf.add_watch("ZZZZQ", path=self.db)          # accepted at once; the lookup says it does not exist
        bad = [w for w in pf.get_state(path=self.db)["watchlist"] if w["symbol"] == "ZZZZQ"][0]
        self.assertIsNone(bad["quote"]["price"])
        pf.remove_watch("ZZZZQ", path=self.db)
        pf.remove_watch("AAPL", path=self.db)
        self.assertEqual(pf.get_state(path=self.db)["watchlist"], [])

    def test_users_example_average_cost_vs_fifo(self):
        # 100 at $10 (10 sh) + 100 at $8 (12.5 sh) = 22.5 sh for $200; then sell 5 at $9.
        self.tx(quantity=10, price=10)
        self.tx(quantity=12.5, price=8, trade_date="2026-09-02")
        h = pf.get_state(path=self.db)["holdings"][0]
        self.assertAlmostEqual(h["avg_cost"], 200 / 22.5)          # 8.889, not 9
        self.assertAlmostEqual(h["fifo_avg_cost"], 200 / 22.5)     # same until something is sold
        self.tx(side="sell", quantity=5, price=9, trade_date="2026-09-03")
        st = pf.get_state(path=self.db)
        h = st["holdings"][0]
        self.assertAlmostEqual(h["quantity"], 17.5)
        self.assertAlmostEqual(h["avg_cost"], 200 / 22.5)          # the user's rule: unchanged by a sale
        self.assertAlmostEqual(h["fifo_avg_cost"], 150 / 17.5)     # FIFO: the five $10 shares went first -> 8.571
        self.assertAlmostEqual(st["totals"]["realized"], 5 * (9 - 200 / 22.5))
        self.assertAlmostEqual(st["totals"]["fifo_realized"], 5 * (9 - 10))
        # a new buy re-averages both
        self.tx(quantity=2.5, price=12, trade_date="2026-09-04")
        h = pf.get_state(path=self.db)["holdings"][0]
        self.assertAlmostEqual(h["avg_cost"], (17.5 * 200 / 22.5 + 30) / 20)
        self.assertAlmostEqual(h["fifo_avg_cost"], (150 + 30) / 20)

    def test_fifo_sale_spanning_lots(self):
        self.tx(quantity=2, price=10)
        self.tx(quantity=3, price=20, trade_date="2026-09-02")
        self.tx(side="sell", quantity=4, price=25, trade_date="2026-09-03")
        st = pf.get_state(path=self.db)
        self.assertAlmostEqual(st["totals"]["fifo_realized"], 4 * 25 - (2 * 10 + 2 * 20))
        self.assertAlmostEqual(st["holdings"][0]["fifo_avg_cost"], 20)
        self.assertEqual(len(st["holdings"][0]["lots"]), 1)

    def test_entry_by_amount(self):
        self.tx(entry_mode="amount", amount=100, price=10, quantity="")
        self.tx(entry_mode="amount", amount=100, price=8, quantity="", trade_date="2026-09-02")
        h = pf.get_state(path=self.db)["holdings"][0]
        self.assertAlmostEqual(h["quantity"], 22.5)
        self.assertAlmostEqual(h["avg_cost"], 200 / 22.5)
        with self.assertRaises(ValueError):
            self.tx(entry_mode="amount", amount=0, price=8)

    def test_same_request_id_is_saved_once(self):
        ids = [self.tx(quantity=1, price=100, request_id="form-1") for _ in range(3)]
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(pf.get_state(path=self.db, cached_quotes=True)["transactions"]), 1)

    def test_concurrent_sales_cannot_both_pass_the_holding_check(self):
        from concurrent.futures import ThreadPoolExecutor
        self.tx(quantity=1, price=100)

        def sell(i):
            try:
                return self.tx(side="sell", quantity=1, price=110, trade_date="2026-09-02", request_id=f"s{i}")
            except ValueError:
                return None
        with ThreadPoolExecutor(8) as ex:
            ok = [r for r in ex.map(sell, range(8)) if r]
        self.assertEqual(len(ok), 1)

    def test_editing_a_deleted_transaction_says_so(self):
        tid = self.tx(quantity=1, price=100)
        pf.delete_transaction(tid, self.db)
        pf.delete_transaction(tid, self.db)          # a repeated delete is harmless
        with self.assertRaises(ValueError):
            self.tx(id=tid, quantity=2, price=100)

    def test_chart_daily_appends_todays_partial_candle_and_caches(self):
        from datetime import datetime, timezone
        today = datetime.now(pf.US_EAST).date()
        # Nasdaq encodes ET wall-clock as UTC: 08:00 is pre-market, 10:00 and 10:05 regular.
        base = int(datetime(today.year, today.month, today.day, tzinfo=timezone.utc).timestamp())
        minutes = {"chart": [{"x": (base + h * 3600 + m * 60) * 1000, "y": y, "w": 100}
                             for h, m, y in ((8, 0, 99.0), (10, 0, 100.0), (10, 1, 103.0), (10, 6, 101.0))],
                   "chartPrevClose": "98.50"}
        hist = {"tradesTable": {"rows": [{"date": "01/02/2026", "open": "$90", "high": "$95", "low": "$89",
                                          "close": "$94", "volume": "1,000"}]}}

        def fake_nasdaq(symbol, endpoint, cls):
            return minutes if endpoint == "chart" else hist if endpoint == "historical" else None
        with mock.patch.object(pf, "_nasdaq", side_effect=fake_nasdaq) as m:
            day = pf.get_chart("NVDA", "1d", self.db)
            self.assertEqual([c["session"] for c in day["candles"]], ["pre", "regular", "regular"])
            self.assertEqual(day["prev_close"], 98.5)
            first = day["candles"][1]
            self.assertEqual((first["open"], first["high"], first["close"]), (100.0, 103.0, 103.0))
            daily = pf.get_chart("NVDA", "1y", self.db)
            self.assertEqual(daily["candles"][0]["time"], "2026-01-02")
            today_bar = daily["candles"][-1]
            self.assertTrue(today_bar["partial"])
            self.assertEqual((today_bar["open"], today_bar["high"], today_bar["close"]), (100.0, 103.0, 101.0))
            calls = m.call_count
            pf.get_chart("NVDA", "1y", self.db)
            self.assertEqual(m.call_count, calls)          # served from the cache
        with self.assertRaises(ValueError):
            pf.get_chart("NVDA", "10y", self.db)

    def test_regular_session_uses_one_batch_request_and_falls_back_per_symbol(self):
        reply = {"status": {"rCode": 200}, "data": [
            {"symbol": "NVDA", "companyName": "NVIDIA", "lastSalePrice": "$213.50", "netChange": "-3.59",
             "percentageChange": "-1.65%", "assetClass": "STOCKS", "previousClosePrice": 217.09, "marketStatus": "Market Open"},
            {"symbol": "QQQ", "companyName": "Invesco QQQ", "lastSalePrice": "$720.00", "netChange": "1.64",
             "percentageChange": "0.23%", "assetClass": "ETF", "previousClosePrice": 718.36, "marketStatus": "Market Open"}]}
        with mock.patch.object(pf, "session_now", return_value="regular"), \
             mock.patch.object(pf, "fetch_quotes_batch", side_effect=self.real_batch), \
             mock.patch.object(pf, "_get_json", return_value=reply) as batch_call:
            q = pf.get_quotes(["NVDA", "QQQ", "AAPL"], force=True, path=self.db)
        self.assertEqual(batch_call.call_count, 1)
        self.assertEqual((q["NVDA"]["price"], q["NVDA"]["prev_close"], q["NVDA"]["source"]), (213.5, 217.09, "nasdaq-batch"))
        self.assertEqual(q["QQQ"]["asset_class"], "etf")
        self.assertEqual(q["AAPL"]["price"], 250.0)            # not in the batch reply -> per-symbol fetch (fake)

    def test_extended_hours_stay_per_symbol(self):
        with mock.patch.object(pf, "session_now", return_value="post"), \
             mock.patch.object(pf, "fetch_quotes_batch") as batch:
            pf.get_quotes(["NVDA"], force=True, path=self.db)
        batch.assert_not_called()

    def test_holdings_refresh_faster_than_watchlist(self):
        import time as _t
        self.tx(quantity=1, price=100)                  # NVDA held
        pf.add_watch("AAPL", path=self.db)
        pf.get_quotes(["NVDA", "AAPL"], force=True, path=self.db)
        conn = pf._conn(self.db)
        conn.execute("UPDATE quotes SET fetched_at = ?", (_t.time() - 10,))   # both 10 s old
        conn.commit(); conn.close()
        fetched = []
        with mock.patch.object(pf, "session_now", return_value="regular"), \
             mock.patch.object(pf, "fetch_quote", side_effect=lambda s, *a: fetched.append(s) or fake_quote(s)):
            pf.get_quotes(["NVDA", "AAPL"], path=self.db, ttl_by_symbol={"NVDA": 3, "AAPL": 60})
        self.assertEqual(fetched, ["NVDA"])             # per-symbol TTLs: 10 s is past 3 s, not past 60 s

    def test_a_sale_shows_its_average_cost_and_profit(self):
        self.tx(quantity=10, price=10)
        self.tx(quantity=12.5, price=8, trade_date="2026-09-02")          # average 200 / 22.5
        sale = self.tx(side="sell", quantity=5, price=11, fee=1, trade_date="2026-09-03")
        with mock.patch.object(pf, "fetch_quote", side_effect=lambda s, *a: None):
            txs = {t["id"]: t for t in pf.get_state(path=self.db, cached_quotes=True)["transactions"]}
        self.assertAlmostEqual(txs[sale]["sale_avg_cost"], 200 / 22.5)
        self.assertAlmostEqual(txs[sale]["sale_profit"], 5 * (11 - 200 / 22.5) - 1)
        self.assertNotIn("sale_profit", next(t for i, t in txs.items() if i != sale))

    def test_saving_a_sale_does_not_push_to_the_budget(self):
        with mock.patch("urllib.request.urlopen") as url:
            self.tx(quantity=10, price=10)
            self.tx(side="sell", quantity=5, price=11, trade_date="2026-09-03", budget_sync="1")
        url.assert_not_called()

    def test_extended_hours_print_outside_the_quote_uses_the_midpoint(self):
        self.assertEqual(pf._sane_extended_price(399.55, "post", 430.00, 430.50), (430.25, "mid"))
        self.assertEqual(pf._sane_extended_price(430.40, "post", 430.00, 430.50), (430.40, "last"))
        self.assertEqual(pf._sane_extended_price(399.55, "regular", 430.00, 430.50), (399.55, "last"))
        self.assertEqual(pf._sane_extended_price(399.55, "post", 380.0, 440.0), (399.55, "last"))   # too wide to judge

    def test_unconfirmed_jump_waits_for_a_second_sighting(self):
        old = {"price": 430.4, "session": "post"}
        first = pf._hold_back_jump({"price": 399.5, "session": "post", "price_basis": "last"}, old)
        self.assertEqual((first["price"], first["pending_jump"]), (430.4, 399.5))
        second = pf._hold_back_jump({"price": 399.9, "session": "post", "price_basis": "last"}, first)
        self.assertEqual(second["price"], 399.9)                    # it repeated: a real move
        back = pf._hold_back_jump({"price": 430.2, "session": "post", "price_basis": "last"}, first)
        self.assertEqual(back["price"], 430.2)                      # the spike never came back

    def test_minute_spikes_are_dropped_but_real_moves_kept(self):
        pts = [(i, p, 1) for i, p in enumerate([429.15, 429.17, 411.51, 429.68, 430.46, 429.9])]
        self.assertNotIn(411.51, [p[1] for p in pf._drop_spikes(pts)])
        step = [(i, p, 1) for i, p in enumerate([100, 100.1, 100, 106, 106.2, 106.1, 106.3])]
        self.assertEqual(len(pf._drop_spikes(step)), len(step))

    def test_frozen_nasdaq_replica_is_detected(self):
        live = {"marketStatus": "Open", "primaryData": {"lastSalePrice": "$91.32", "bidPrice": "$91.32", "askPrice": "$91.33"}}
        frozen = {"marketStatus": "Open", "primaryData": {"lastSalePrice": "$91.215", "bidPrice": "N/A", "askPrice": "N/A"}}
        after = {"marketStatus": "After-Hours", "primaryData": {"lastSalePrice": "$91.2", "bidPrice": "N/A", "askPrice": "N/A"}}
        self.assertFalse(pf._frozen_reply(live))
        self.assertTrue(pf._frozen_reply(frozen))
        self.assertFalse(pf._frozen_reply(after))             # thin extended hours may lack a quote

    def _nasdaq_replies(self, status, secondary=None, primary_change="+0.61"):
        info = {"marketStatus": status, "companyName": "Intel",
                "primaryData": {"lastSalePrice": "$86.10", "netChange": primary_change,
                                "bidPrice": "$86.05", "askPrice": "$86.15"},
                "secondaryData": secondary}
        summary = {"summaryData": {"PreviousClose": {"value": "$92.96"}}}      # the 10-21 close, stale
        return lambda symbol, endpoint, cls: info if endpoint == "info" else summary

    def test_baseline_comes_from_the_info_reply_not_the_stale_summary(self):
        # 2026-10-23: some summary replies still said 92.96 (10-21) while 10-22 closed at 85.49.
        cases = [("Pre-Market", {"lastSalePrice": "$85.49", "netChange": "-7.47"}, "+0.61", 85.49),
                 ("Open", None, "+0.61", 85.49),
                 ("After-Hours", {"lastSalePrice": "$86.96", "netChange": "+1.47"}, "-0.86", 85.49)]
        for status, secondary, change, want in cases:
            with self.subTest(status), \
                 mock.patch.object(pf, "_nasdaq", side_effect=self._nasdaq_replies(status, secondary, change)):
                self.assertEqual(REAL_FETCH_QUOTE("INTC", "stocks")["prev_close"], want)
                self.assertEqual(REAL_FETCH_QUOTE("INTC", "stocks", known_prev_close=92.96)["prev_close"], want)

    def test_premarket_baseline_ignores_a_stale_secondary_close(self):
        # GOOGL 2026-10-26 pre-market: the same primary pair, secondaryData alternating between replicas
        for secondary in ({"lastSalePrice": "$320.21", "netChange": "-4.01"}, {"lastSalePrice": "$322.35", "netChange": "+2.14"}):
            with self.subTest(secondary["lastSalePrice"]):
                self.assertEqual(pf._prev_close_from_info("pre", {"lastSalePrice": "$321.9999", "netChange": "+1.7899"}, secondary), 320.21)
        self.assertEqual(pf._prev_close_from_info("pre", {"lastSalePrice": "$321.99", "netChange": "N/A"},
                                                  {"lastSalePrice": "$320.21"}), 320.21)   # no change in the reply: the fallback

    def test_previous_close_read_while_closed_is_not_reused(self):
        with mock.patch.object(pf, "_nasdaq", side_effect=self._nasdaq_replies("Closed")):
            q = REAL_FETCH_QUOTE("INTC", "stocks")
        self.assertEqual(q["prev_close"], 92.96)                   # only the summary to go by
        self.assertIsNone(q["prev_close_date"])

    def _closed_replies(self, ext_date="Oct 28, 2026", trades=("$292.5", "$292.5", "$292.4985")):
        # GOOGL 2026-10-28 21:12 ET: info back on the regular close, the after-hours last on the extended page
        info = {"marketStatus": "Closed", "companyName": "Alphabet",
                "primaryData": {"lastSalePrice": "$324.89", "netChange": "+4.31", "percentageChange": "+1.34%",
                                "lastTradeTimestamp": "Oct 28, 2026"}, "secondaryData": None}
        ext = {"lastUpdateInfo": [f"Data last updated {ext_date} 08:00 PM ET.", "This page will resume updating on Oct 29, 2026 04:00 PM ET."],
               "infoTable": {"rows": [{"consolidated": "$292.5 -28.08 (-8.76%)"}]},
               "tradeDetailTable": {"rows": [{"time": "19:59:59", "price": p, "shareVolume": "10"} for p in trades]}}
        summary = {"summaryData": {"PreviousClose": {"value": "$320.58"}}}
        return lambda symbol, endpoint, cls: {"info": info, "extended-trading": ext}.get(endpoint, summary)

    def test_closed_market_shows_the_last_after_hours_price(self):
        with mock.patch.object(pf, "session_now", return_value="closed"), \
             mock.patch.object(pf, "_nasdaq", side_effect=self._closed_replies()):
            q = REAL_FETCH_QUOTE("GOOGL", "stocks")
        self.assertEqual(q["session"], "closed")                   # no polling overnight
        self.assertEqual((q["price"], q["price_basis"], q["regular_price"]), (292.5, "post_close", 324.89))
        self.assertEqual(q["prev_close"], 320.58)                  # the day still runs from the close before
        self.assertAlmostEqual(q["session_change"], -32.39)

    def test_after_hours_odd_lot_print_does_not_set_the_closed_price(self):
        with mock.patch.object(pf, "session_now", return_value="closed"), \
             mock.patch.object(pf, "_nasdaq", side_effect=self._closed_replies(trades=("$270.00", "$292.5", "$292.4"))):
            self.assertEqual(REAL_FETCH_QUOTE("GOOGL", "stocks")["price"], 292.4)

    def test_after_hours_page_from_another_day_is_ignored(self):
        with mock.patch.object(pf, "session_now", return_value="closed"), \
             mock.patch.object(pf, "_nasdaq", side_effect=self._closed_replies(ext_date="Oct 27, 2026")):
            q = REAL_FETCH_QUOTE("GOOGL", "stocks")
        self.assertEqual((q["price"], q["price_basis"]), (324.89, "last"))

    def test_batch_row_with_volume_going_backwards_is_refetched(self):
        pf.get_quotes(["NVDA"], force=True, path=self.db)
        conn = pf._conn(self.db)
        q = json.loads(conn.execute("SELECT data FROM quotes WHERE symbol='NVDA'").fetchone()[0])
        conn.execute("UPDATE quotes SET data = ?", (json.dumps({**q, "volume": 13_700_000}),))
        conn.commit(); conn.close()
        stale_row = {"NVDA": {"symbol": "NVDA", "price": 91.215, "volume": 10_644_128, "session": "regular", "asset_class": "stocks"}}
        with mock.patch.object(pf, "session_now", return_value="regular"), \
             mock.patch.object(pf, "fetch_quotes_batch", return_value=stale_row):
            out = pf.get_quotes(["NVDA"], force=True, path=self.db)
        self.assertEqual(out["NVDA"]["price"], 213.0)          # the per-symbol fetch (fake) won, not 91.215


    def test_explicit_lot_allocation_swing_profit_statuses_and_invariants(self):
        # User's exact scenario: 10 sh @ $106 + 20 sh @ $73.14, then sell 20 @ $85 closing the $73.14 lot.
        b_high = self.tx(symbol="NVDA", quantity=10, price=106.0, trade_date="2026-10-20")
        b_low = self.tx(symbol="NVDA", quantity=20, price=73.14, trade_date="2026-10-29")
        sale = self.tx(
            symbol="NVDA", side="sell", entry_mode="lots", price=85.0, trade_date="2026-10-30",
            lot_allocations=[{"buy_id": b_low, "quantity": 20}],
        )
        st = pf.get_state(path=self.db)
        h = st["holdings"][0]
        by_id = {bo["id"]: bo for bo in h["buy_orders"]}

        # Selected low lot is closed with +$237.20 swing profit; high lot stays open
        self.assertEqual(by_id[b_low]["status"], "closed")
        self.assertAlmostEqual(by_id[b_low]["open_qty"], 0.0)
        self.assertAlmostEqual(by_id[b_low]["closed_qty"], 20.0)
        self.assertAlmostEqual(by_id[b_low]["realized_pnl"], 20 * (85.0 - 73.14))
        self.assertEqual(by_id[b_high]["status"], "open")
        self.assertAlmostEqual(by_id[b_high]["open_qty"], 10.0)

        # Strict FIFO view on the same buy orders shows the opposite (oldest 10 closed + 10 of second lot)
        self.assertEqual(by_id[b_high]["fifo_status"], "closed")
        self.assertEqual(by_id[b_low]["fifo_status"], "partial")
        self.assertAlmostEqual(by_id[b_low]["fifo_open_qty"], 10.0)

        # Headline Mine avg_cost and Irish CGT FIFO avg_cost are completely unaffected by custom lot choice
        expected_avg = (10 * 106.0 + 20 * 73.14) / 30.0
        self.assertAlmostEqual(h["avg_cost"], expected_avg)
        self.assertAlmostEqual(h["fifo_avg_cost"], 73.14)
        self.assertAlmostEqual(h["open_lots_avg"], 106.0)
        self.assertAlmostEqual(h["swing_realized"], 237.20)
        # Net cash breakeven drops from $106.00 to $82.28 thanks to the +$237.20 swing gain
        self.assertAlmostEqual(h["breakeven_price"], (1060.0 + 1462.80 - 1700.0) / 10.0)

        # Fundamental conservation law: unrealized + realized is identical across all 3 accounting views
        self.assertAlmostEqual(h["open_lots_unrealized"] + h["swing_realized"], h["unrealized"] + h["realized"])
        self.assertAlmostEqual(h["fifo_unrealized"] + h["fifo_realized"], h["unrealized"] + h["realized"])

        # Editing the sale's note without passing lot_allocations preserves custom lot selection
        pf.save_transaction({
            "id": sale, "symbol": "NVDA", "side": "sell", "trade_date": "2026-10-30",
            "quantity": 20, "price": 85.0, "fee": 0, "note": "swing profit",
        }, self.db)
        h2 = pf.get_state(path=self.db)["holdings"][0]
        self.assertAlmostEqual(h2["swing_realized"], 237.20)

        # Explicitly clearing lot_allocations=[] reverts that sale back to FIFO
        pf.save_transaction({
            "id": sale, "symbol": "NVDA", "side": "sell", "trade_date": "2026-10-30",
            "quantity": 20, "price": 85.0, "fee": 0, "note": "revert fifo", "lot_allocations": [],
        }, self.db)
        h3 = pf.get_state(path=self.db)["holdings"][0]
        self.assertAlmostEqual(h3["swing_realized"], h3["fifo_realized"])

    def test_partial_multi_lot_close_fractional_dust_and_validation(self):
        b1 = self.tx(symbol="NVDA", entry_mode="amount", amount=1000, price=137.54376278, quantity="", trade_date="2026-10-20")
        b2 = self.tx(symbol="NVDA", quantity=20, price=73.14, trade_date="2026-10-29")
        st0 = pf.get_state(path=self.db)
        q1 = next(bo["open_qty"] for bo in st0["holdings"][0]["buy_orders"] if bo["id"] == b1)

        # Over-allocating beyond b2's open shares is rejected and rolled back
        with self.assertRaises(ValueError):
            self.tx(
                symbol="NVDA", side="sell", entry_mode="lots", price=85.0, trade_date="2026-10-30",
                lot_allocations=[{"buy_id": b2, "quantity": 20.5}],
            )

        # Closing 100% of 8-decimal fractional lot b1 + 8 shares of b2 in a single sale
        self.tx(
            symbol="NVDA", side="sell", entry_mode="lots", price=150.0, fee=2.0, trade_date="2026-10-30",
            lot_allocations=[{"buy_id": b1, "quantity": q1}, {"buy_id": b2, "quantity": 8}],
        )
        h = pf.get_state(path=self.db)["holdings"][0]
        by_id = {bo["id"]: bo for bo in h["buy_orders"]}
        self.assertEqual(by_id[b1]["status"], "closed")
        self.assertEqual(by_id[b1]["open_qty"], 0.0)
        self.assertEqual(by_id[b2]["status"], "partial")
        self.assertAlmostEqual(by_id[b2]["open_qty"], 12.0)
        self.assertAlmostEqual(by_id[b2]["closed_qty"], 8.0)

        # Deleting a buy order explicitly referenced by a sale is refused and rolled back
        with self.assertRaises(ValueError):
            pf.delete_transaction(b2, self.db)


if __name__ == "__main__":
    unittest.main()


class FastWindowTest(unittest.TestCase):
    """10:00-22:00 Dublin: every 5 s in any trading session (asked 2026-10-22)."""

    def at(self, hh, mm=0):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime(2026, 10, 22, hh, mm, tzinfo=ZoneInfo("Europe/Dublin"))

    def test_pre_regular_post_inside_the_window_are_5s(self):
        for session in ("pre", "regular", "post"):
            self.assertEqual(pf.refresh_interval(session, pf.REFRESH_SECONDS, self.at(10)), 5)
            self.assertEqual(pf.refresh_interval(session, pf.WATCH_TTL, self.at(21, 59)), 5)

    def test_outside_the_window_the_session_table_applies(self):
        self.assertEqual(pf.refresh_interval("pre", pf.REFRESH_SECONDS, self.at(9, 30)), 30)
        self.assertEqual(pf.refresh_interval("post", pf.REFRESH_SECONDS, self.at(22, 0)), 30)
        self.assertIsNone(pf.refresh_interval("closed", pf.REFRESH_SECONDS, self.at(12)))   # weekend / holiday
