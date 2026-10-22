"""Markets: the user's own portfolio, global market quotes and the daily market briefs.

Host-only (enabled by MARKETS_ENABLED=1 in run_host.sh): it keeps personal data in SQLite under
the user's Library folder, calls the `claude` CLI, and must never run inside the Docker image.
"""
