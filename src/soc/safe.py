"""Untrusted text shown as Markdown (owner 2026-10-16, hostile-input stress test).

Log fields (user names, hosts, URLs, event details) are written by whoever produced
the log - an attacker included. Placed into Markdown as-is, a value such as
``![x](http://attacker/pixel)`` makes the analyst's browser fetch a remote image
(leaking that someone looked, and from where), and ``[Microsoft](http://evil)``
shows a misleading link. ``md()`` escapes Markdown syntax so such values display
literally. Raw HTML is never rendered by the dashboard (no unsafe_allow_html).
Use it only where text is rendered as Markdown; stored data stays unchanged.
"""

from __future__ import annotations

import re
from typing import Any

_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~])")


def md(value: Any, limit: int | None = None) -> str:
    """``value`` as literal text inside Markdown (optionally cut to ``limit`` chars
    before escaping, so an escape is never split)."""
    s = "" if value is None else str(value)
    if limit is not None:
        s = s[:limit]
    s = s.replace("\r", " ").replace("\n", " ")  # no new Markdown blocks from a field
    return _SPECIAL.sub(r"\\\1", s)
