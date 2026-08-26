from __future__ import annotations

import json

from .models import SemanticContext


def context_for_prompt(context: SemanticContext) -> str:
    payload = {
        "strategy": context.strategy,
        "retrieval_round": context.retrieval_round,
        "items": [
            item.model_dump(mode="json", exclude_none=True)
            for item in context.items
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
