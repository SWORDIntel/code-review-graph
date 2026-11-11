"""Best-effort bounded progress events for an explicitly configured parent pipe."""
from __future__ import annotations

import json
import os
import stat


def report(phase: str, unit: str, completed: int, total: int | None = None) -> None:
    """Never block work or fall back to logs; only write to a nonblocking pipe."""
    raw = os.environ.get("QIHSE_CODEGRAPH_PROGRESS_FD", "")
    if not raw.isascii() or not raw.isdecimal():
        return
    try:
        fd = int(raw)
        if fd < 3 or os.get_blocking(fd) or not stat.S_ISFIFO(os.fstat(fd).st_mode):
            return
        payload = (json.dumps({"version": 1, "phase": phase, "unit": unit,
                               "completed": completed, "total": total},
                              separators=(",", ":")) + "\n").encode("ascii")
        if len(payload) <= 512:
            os.write(fd, payload)
    except (OSError, ValueError, OverflowError):
        pass
