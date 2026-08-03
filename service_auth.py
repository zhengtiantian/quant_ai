"""R.5 phase 1b — obtain and cache a service token for calls into quant_api.

**What this is, stated plainly rather than discovered later.** A client-credentials grant
authenticates *this service*, not the person who asked. Every request quant_api sees will
carry the identity `service-account-quant-ai`, whatever user prompted it. That is the
confused-deputy arrangement: an agent holding a standing credential broader than any one
caller can return data the caller was never entitled to, and nothing in the logs would
show it, because every component behaved correctly.

It is the right *first* step and the wrong *final* one. It gives the API something to
authenticate so phase 2 can require it at all, and it makes the caller identifiable in a
way "no header at all" never could. The correct end state is on-behalf-of: quant_ai
exchanges the caller's token for one downscoped to that caller's rights, so a retrieval
performed for user A can only reach what A may see. That is tracked as R.5c and is the
part worth the interview conversation — this file is the scaffolding it needs.

Fails open on purpose while phase 1 is in flight: quant_api still ends its filter chain
with `.anyRequest().permitAll()`, so a request without a token succeeds. If token
acquisition breaks, calls proceed unauthenticated rather than the platform going dark. The
moment phase 2 lands, the same code path fails closed by itself — the API starts answering
401 — which is the intended direction and needs no change here.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import requests

# Load the service's own .env before reading config.
#
# mcp_server.py is spawned as a subprocess by whoever is using it — Claude Desktop, Codex,
# quant_ai's own client — and none of them pass these variables. A server that can only
# find its credentials when launched one particular way is a server that is unauthenticated
# for two of its three clients, silently. Reading the file next to the code makes the
# credential a property of the deployment rather than of the launcher.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"), override=False)
except ImportError:
    pass

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:18082").rstrip("/")
REALM = os.getenv("KEYCLOAK_REALM", "quant")
CLIENT_ID = os.getenv("QUANT_AI_CLIENT_ID", "quant-ai")
CLIENT_SECRET = os.getenv("QUANT_AI_CLIENT_SECRET", "").strip()
TOKEN_TIMEOUT = float(os.getenv("QUANT_AI_TOKEN_TIMEOUT", "10"))

# Tokens live 300s. Refreshing 30s early avoids the race where a token passes the check
# here and expires in flight, which would surface as a sporadic 401 that reproduces only
# under load.
_SKEW_SECONDS = 30

_lock = threading.Lock()
_token: str | None = None
_expires_at: float = 0.0
_warned = False


def _fetch() -> tuple[str, float] | None:
    url = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/token"
    r = requests.post(
        url,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=TOKEN_TIMEOUT,
    )
    r.raise_for_status()
    body = r.json()
    tok = body.get("access_token")
    if not tok:
        return None
    return tok, time.time() + float(body.get("expires_in", 300))


def get_token() -> str | None:
    """Cached service token, or None when one cannot be obtained.

    Returning None rather than raising is what keeps the failure open. The caller adds
    the header when there is one and proceeds when there is not.
    """
    global _token, _expires_at, _warned

    if not CLIENT_SECRET:
        if not _warned:
            print("[auth] QUANT_AI_CLIENT_SECRET unset — calling quant_api without a "
                  "token. Fine while the API permits all; a 401 after phase 2 means this.")
            _warned = True
        return None

    with _lock:
        if _token and time.time() < _expires_at - _SKEW_SECONDS:
            return _token
        try:
            got = _fetch()
        except requests.RequestException as e:
            print(f"[auth] could not reach Keycloak ({e}); proceeding without a token")
            return None
        if got is None:
            print("[auth] Keycloak returned no access_token; proceeding without one")
            return None
        _token, _expires_at = got
        return _token


def auth_headers() -> dict[str, str]:
    tok = get_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}
