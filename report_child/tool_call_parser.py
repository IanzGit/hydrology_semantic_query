from __future__ import annotations

import re

_INTERNAL_PROTOCOL_MARKERS = re.compile(
    r"<\s*/?\s*tool_call\b|<\s*/?\s*function\s*=|"
    r"<\s*/?\s*parameter\s*=|(?im:^\s*Action\s*:)|"
    r"(?im:^\s*Action Input\s*:)|[\"']tool_calls[\"']\s*:|"
    r"(?im:^\s*Observation\s*:)|"
    r"[\"']semantic_query[\"']\s*:|[\"']query_mode[\"']\s*:|"
    r"[\"']rows_truncated[\"']\s*:|"
    r"[\"']kind[\"']\s*:\s*[\"'](?:semantic_query_result|error)[\"']",
    re.IGNORECASE,
)
_SQL_BLOCK = re.compile(
    r"(?is)(?:^|\n)\s*(?:```sql\s*)?(?:SELECT|WITH)\s+.+(?:```\s*)?$"
)


def contains_internal_protocol(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    return bool(_INTERNAL_PROTOCOL_MARKERS.search(text) or _SQL_BLOCK.search(text))


__all__ = ["contains_internal_protocol"]
