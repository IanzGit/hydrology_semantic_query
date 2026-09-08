from __future__ import annotations

import logging
import time
from typing import Any

from .models import StepRecord, StepStatus

logger = logging.getLogger("uvicorn.error")


def build_step(
    stage: str,
    started: float,
    *,
    attempt: int,
    status: StepStatus,
    summary: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> StepRecord:
    step = StepRecord(
        stage=stage,
        status=status,
        duration_ms=round((time.perf_counter() - started) * 1000, 3),
        attempt=attempt,
        summary=summary,
        metadata=metadata or {},
    )
    logger.info(
        "hydrology_semantic_query node timing: stage=%s status=%s attempt=%s duration_ms=%.3f",
        step.stage,
        step.status.value,
        step.attempt,
        step.duration_ms,
    )
    return step


__all__ = ["build_step"]
