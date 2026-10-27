"""Where the markets data lives and what it talks to. All overridable by environment."""

import os
from pathlib import Path

DATA_DIR = Path(os.getenv("MARKETS_DATA_DIR", "~/Library/Application Support/quant_ai")).expanduser()
DB_PATH = DATA_DIR / "markets.db"
# A copy after every change, in Google Drive (written by Python: launchd's bash cannot read
# ~/Library/CloudStorage). Skipped quietly when the folder does not exist.
BACKUP_PATH = Path(os.getenv(
    "MARKETS_BACKUP_PATH",
    "~/Library/CloudStorage/GoogleDrive-biaszh00001@gmail.com/My Drive/quant_ai_backup/markets.db")).expanduser()
# job_agent's dashboard owns the budget: the USD rate is read from it.
JOB_AGENT_URL = os.getenv("JOB_AGENT_URL", "http://127.0.0.1:5050").rstrip("/")

DATA_DIR.mkdir(parents=True, exist_ok=True)
