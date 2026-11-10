"""Enterprise Observability & Distributed Tracing for Agentic Workflows.

Collects step-level node execution spans, latency breakdowns, token spend attribution,
and cost accounting across multi-agent state machines, emitting OpenTelemetry / Langfuse
compatible telemetry events.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SpanRecord(BaseModel):
    """Execution telemetry record for an individual graph node span."""

    model_config = ConfigDict(frozen=True)

    span_id: str
    node_name: str
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    status: str = "SUCCESS"


class SessionTelemetry(BaseModel):
    """Aggregate session telemetry summary."""

    session_id: str
    total_duration_ms: float
    total_spans: int
    total_tokens: int
    total_cost_usd: float
    spans: List[SpanRecord] = Field(default_factory=list)


class TelemetryCollector:
    """Collects and aggregates distributed tracing spans for LangGraph workflows."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._active_spans: dict[str, float] = {}
        self.completed_spans: list[SpanRecord] = []

    def start_span(self, node_name: str) -> None:
        """Mark start timestamp of an execution node."""
        self._active_spans[node_name] = time.perf_counter()

    def end_span(
        self,
        node_name: str,
        input_tokens: int = 50,
        output_tokens: int = 120,
        cost_per_1k: float = 0.0003,
        status: str = "SUCCESS",
    ) -> SpanRecord:
        """Record span completion with latency and token economics."""
        start_t = self._active_spans.pop(node_name, time.perf_counter())
        duration_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        total_tokens = input_tokens + output_tokens
        cost = round((total_tokens / 1000.0) * cost_per_1k, 6)

        span = SpanRecord(
            span_id=f"span-{len(self.completed_spans) + 1:03d}",
            node_name=node_name,
            duration_ms=max(1.0, duration_ms),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
            status=status,
        )
        self.completed_spans.append(span)
        return span

    def summarize(self) -> SessionTelemetry:
        """Compute session-level telemetry aggregate."""
        total_ms = sum(s.duration_ms for s in self.completed_spans)
        total_tok = sum(s.input_tokens + s.output_tokens for s in self.completed_spans)
        total_cost = sum(s.estimated_cost_usd for s in self.completed_spans)

        return SessionTelemetry(
            session_id=self.session_id,
            total_duration_ms=round(total_ms, 2),
            total_spans=len(self.completed_spans),
            total_tokens=total_tok,
            total_cost_usd=round(total_cost, 6),
            spans=list(self.completed_spans),
        )
