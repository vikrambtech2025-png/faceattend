"""Backup SQLite database to a timestamped file."""

from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path

from faceattend.config import get_settings


def main() -> None:
    settings = get_settings()
    src = str(settings.db_path)
    out_dir = Path("backups")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    dst = out_dir / f"attendance_{ts}.db"
    con = sqlite3.connect(src)
    try:
        backup = sqlite3.connect(str(dst))
        con.backup(backup)
        backup.close()
    finally:
        con.close()
    print(f"backup written to {dst}")


if __name__ == "__main__":
    main()