"""Home-network access to the markets data (2026-10-31).

The data is personal, so the API was served to this machine only. The user now opens the dashboard
from an iPad or phone on the home Wi-Fi: a request from a private (RFC 1918) address is let in when it
carries a valid sign-in token of the quant realm, the same Keycloak token quant_ui logs in with
(an Authorization: Bearer header, or the qt_token cookie quant_ui sets). The signature is checked
against the realm's public keys and the expiry is enforced; nothing else about the token is needed,
since every account in the realm is the user's own. Tailnet (100.64/10) and public addresses stay
out. Off unless MARKETS_LAN_ACCESS=1 (run_host.sh).
"""

import ipaddress
import os
from typing import Optional

import jwt

JWKS_URL = os.getenv("KEYCLOAK_JWKS_URL", "http://127.0.0.1:18082/realms/quant/protocol/openid-connect/certs")
ENABLED = os.getenv("MARKETS_LAN_ACCESS", "0") == "1"
_jwks: Optional[jwt.PyJWKClient] = None


def is_home_network(host: Optional[str]) -> bool:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return ip.is_private and not ip.is_loopback


def token_from(authorization: Optional[str], cookie: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return cookie or None


def valid(token: Optional[str]) -> bool:
    global _jwks
    if not token:
        return False
    try:
        if _jwks is None:
            _jwks = jwt.PyJWKClient(JWKS_URL, cache_keys=True, lifespan=3600)
        key = _jwks.get_signing_key_from_jwt(token).key
        jwt.decode(token, key, algorithms=["RS256"], options={"verify_aud": False, "require": ["exp"]})
        return True
    except Exception:  # noqa: BLE001 - any failure (bad signature, expired, Keycloak down) means no
        return False


def allowed(host: Optional[str], authorization: Optional[str], cookie: Optional[str]) -> bool:
    return ENABLED and is_home_network(host) and valid(token_from(authorization, cookie))
