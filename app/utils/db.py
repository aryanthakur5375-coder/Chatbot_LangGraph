"""SQLite checkpointing for persistent multi-turn conversations."""
from __future__ import annotations

import sqlite3
from typing import List

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import SQLITE_DB_PATH

_conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(_conn)


def retrieve_all_threads() -> List[str]:
    thread_ids = set()
    try:
        for checkpoint in checkpointer.list(None):
            thread_ids.add(checkpoint.config["configurable"]["thread_id"])
    except Exception:
        pass
    return list(thread_ids)
