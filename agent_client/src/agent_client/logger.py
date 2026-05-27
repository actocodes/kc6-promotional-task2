from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FILE = Path("agent_system.log")
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

class _PrefixFormatter(logging.Formatter):
    def __init__(self, prefix: str) -> None:
        super().__init__(
            fmt=f"[%(asctime)s] [{prefix}] [%(levelname)s] %(message)s",
            datefmt=_DATE_FMT,
        )

def _build_logger(name: str, prefix: str, level: int = logging.DEBUG) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(_PrefixFormatter(prefix))
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_PrefixFormatter(prefix))
    stream_handler.setLevel(level)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.propagate = False
    return log

client_log: logging.Logger = _build_logger("agent_client.client", "CLIENT")
server_log: logging.Logger = _build_logger("agent_client.server", "SERVER")

def ingest_server_log(level: str, message: str) -> None:
    level_map = {
        "debug":    logging.DEBUG,
        "info":     logging.INFO,
        "notice":   logging.INFO,
        "warning":  logging.WARNING,
        "error":    logging.ERROR,
        "critical": logging.CRITICAL,
        "alert":    logging.CRITICAL,
        "emergency":logging.CRITICAL,
    }
    server_log.log(level_map.get(level.lower(), logging.INFO), message)


def log_separator(label: str = "") -> None:
    sep = "=" * 70
    line = f"{sep}  {label}  {sep}" if label else sep
    client_log.info(line)
