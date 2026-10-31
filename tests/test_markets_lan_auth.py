import os
import sys
import time
import unittest

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from markets import lan_auth  # noqa: E402


class _Key:
    def __init__(self, key):
        self.key = key


class _Jwks:
    def __init__(self, key):
        self.k = key

    def get_signing_key_from_jwt(self, token):
        return _Key(self.k)


class LanAuthTest(unittest.TestCase):
    def setUp(self):
        self.priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        lan_auth._jwks = _Jwks(self.priv.public_key())
        lan_auth.ENABLED = True

    def tearDown(self):
        lan_auth._jwks = None
        lan_auth.ENABLED = False

    def token(self, exp_in=300, key=None):
        return jwt.encode({"sub": "u", "exp": int(time.time()) + exp_in}, key or self.priv, algorithm="RS256")

    def test_home_network_with_a_valid_token(self):
        t = self.token()
        self.assertTrue(lan_auth.allowed("192.168.1.42", f"Bearer {t}", None))
        self.assertTrue(lan_auth.allowed("10.0.0.7", None, t))                 # the cookie, for the iframes

    def test_refused(self):
        other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.assertFalse(lan_auth.allowed("192.168.1.42", None, None))                          # no token
        self.assertFalse(lan_auth.allowed("192.168.1.42", None, self.token(exp_in=-10)))        # expired
        self.assertFalse(lan_auth.allowed("192.168.1.42", None, self.token(key=other)))         # not the realm's key
        self.assertFalse(lan_auth.allowed("192.168.1.42", None, "not-a-jwt"))
        self.assertFalse(lan_auth.allowed("100.101.2.3", None, self.token()))                   # tailnet stays out
        self.assertFalse(lan_auth.allowed("8.8.8.8", None, self.token()))                       # public


if __name__ == "__main__":
    unittest.main()
