from __future__ import annotations

import json

from .models import SemanticContext


def context_for_prompt(context: SemanticContext) -> str:
    payload = {
        "retrieval_intent": context.retrieval_intent.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "candidate_models": [
            context.model_details[name] for name in context.candidate_models
        ],
        "members": [
            context.member_details[name] for name in context.allowed_members
        ],
        "filter_members": context.filter_members,
        "binding_candidates": {
            key: [candidate.model_dump(mode="json") for candidate in candidates]
            for key, candidates in context.binding_candidates.items()
        },
        "suggested_members": context.suggested_members,
        "projection_mode": context.projection_mode,
        "projection_policy": context.projection_policy,
        "fixed_business_context": context.fixed_business_context,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
