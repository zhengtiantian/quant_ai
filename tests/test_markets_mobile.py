"""Phone page (/m/): served only to this Mac and its tailnet (tailscale serve), the state cut down. No network."""
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from markets import mobile, portfolio  # noqa: E402

STATE = {"holdings": [{"symbol": "INTC", "quantity": 98.9, "avg_cost": 106.7, "cost": 10556.4, "market_value": 11000.0, "unrealized": 443.6,
                       "unrealized_pct": 4.2, "day_change_pct": 1.1, "lots": [{"quantity": 8.3}], "trades": 9,
                       "quote": {"price": 111.2, "session": "regular", "fetched_at": 1.0, "asset_class": "stocks", "raw": {"x": 1}}}],
         "transactions": [{"id": 1}], "watchlist": [{"symbol": "MU"}], "totals": {"market_value": 11000.0, "cost": 10556.4},
         "session": "regular", "us_time": "10:00", "server_time": 2.0, "refresh_seconds": 5,
         "global": {"refresh_seconds": 60, "any_open": True, "regions": [{"name": "Asia", "countries": [
             {"code": "CN", "name": "China", "flag": "🇨🇳", "open": False, "exchanges": [
                 {"code": "SSE", "name": "Shanghai Stock Exchange", "open": False, "local_time": "19:38", "hours": "09:30–15:00", "tz": "Asia/Shanghai",
                  "indices": [{"symbol": ".SSEC", "name": "Shanghai Composite", "price": 3648.3, "change_pct": -1.33, "currency": "CNY", "prev_close": 3697.5}],
                  "stocks": [{"symbol": "600519-CN", "name": "Kweichow Moutai", "price": 1355.9, "change_pct": 0.44, "currency": "CNY",
                              "full_name": "Kweichow Moutai Co., Ltd."}]}]}]}]}}


def client(host):
    app = FastAPI()
    app.include_router(mobile.router)
    return TestClient(app, client=(host, 50000))


class MobileTest(unittest.TestCase):
    def setUp(self):
        p = [mock.patch.object(portfolio, "init_db"), mock.patch.object(portfolio, "get_state", return_value=STATE)]
        for x in p:
            x.start()
            self.addCleanup(x.stop)

    def test_state_keeps_only_what_the_card_shows(self):
        d = client("127.0.0.1").get("/m/state").json()
        self.assertEqual(d["status"], "success")
        self.assertNotIn("transactions", d)
        self.assertNotIn("watchlist", d)
        h = d["holdings"][0]
        self.assertNotIn("lots", h)
        self.assertEqual(h["quote"], {"price": 111.2, "session": "regular", "fetched_at": 1.0})
        self.assertEqual(d["refresh_seconds"], 5)

    def test_global_markets_keep_every_exchange_and_stock_cut_to_the_shown_fields(self):
        g = client("127.0.0.1").get("/m/state").json()["global"]
        self.assertEqual(g["refresh_seconds"], 60)
        e = g["regions"][0]["countries"][0]["exchanges"][0]
        self.assertEqual((e["name"], e["local_time"], e["hours"], e["open"]), ("Shanghai Stock Exchange", "19:38", "09:30–15:00", False))
        self.assertEqual(e["indices"], [{"name": "Shanghai Composite", "price": 3648.3, "change_pct": -1.33, "currency": "CNY"}])
        self.assertEqual(e["stocks"], [{"name": "Kweichow Moutai", "price": 1355.9, "change_pct": 0.44, "currency": "CNY"}])

    def test_page_icon_and_manifest(self):
        c = client("127.0.0.1")
        self.assertIn("Portfolio", c.get("/m/").text)
        self.assertEqual(c.get("/m/icon.png").headers["content-type"], "image/png")
        self.assertEqual(c.get("/m/manifest.json").json()["display"], "standalone")
        r = c.get("/m", follow_redirects=False)
        self.assertEqual(r.headers["location"], "/m/")

    def test_tailnet_devices_are_served(self):
        self.assertEqual(client("100.127.212.87").get("/m/state").status_code, 200)

    def test_other_machines_are_refused(self):
        c = client("192.168.1.50")
        self.assertEqual(c.get("/m/state").status_code, 403)
        self.assertEqual(c.get("/m/").status_code, 403)


if __name__ == "__main__":
    unittest.main()
