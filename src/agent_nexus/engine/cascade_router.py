"""Intelligent Model Cascade & Token Cost Optimization Router.

Classifies incoming enterprise requests by complexity, routing simple extraction
and filtering to lightweight Small Language Models (SLMs) and escalating multi-step
reasoning to frontier models, enforcing latency budgets and token economics.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class ModelTier(StrEnum):
    """Model execution tiers with varying latency and cost characteristics."""

    SLM_TIER_1 = "SLM_TIER_1"          # Lightweight (e.g. Llama-3.2-3B, Haiku, Gemini-1.5-Flash)
    FRONTIER_TIER_2 = "FRONTIER_TIER_2"  # Frontier Reasoning (e.g. Claude-3.5-Sonnet, Gemini-1.5-Pro)


class QueryComplexity(StrEnum):
    """Estimated cognitive and multi-step reasoning complexity."""

    LOW = "LOW"          # Simple lookup, formatting, single entity extraction
    MEDIUM = "MEDIUM"    # Single-turn tool calling, summarization, moderate reasoning
    HIGH = "HIGH"        # Multi-agent debate, code repair, constraint optimization


class RoutingDecision(BaseModel):
    """Output decision of the cascade model router."""

    model_config = ConfigDict(frozen=True)

    tier: ModelTier
    selected_model: str
    complexity: QueryComplexity
    estimated_latency_ms: float
    cost_per_1k_input_tokens_usd: float
    cost_per_1k_output_tokens_usd: float
    savings_vs_frontier_percent: float
    routing_rationale: str


class ModelCascadeRouter:
    """Dynamic cost and latency optimizer routing tasks across SLMs and Frontier models."""

    # Frontier pricing baseline: Claude 3.5 Sonnet / Gemini Pro ($3.00 / $15.00 per M tokens)
    FRONTIER_MODEL = "claude-3-5-sonnet-20241022"
    FRONTIER_INPUT_COST_1K = 0.003
    FRONTIER_OUTPUT_COST_1K = 0.015
    FRONTIER_P95_LATENCY_MS = 2400.0

    # SLM pricing baseline: Llama 3.2 3B / Gemini Flash ($0.075 / $0.30 per M tokens)
    SLM_MODEL = "llama-3.2-3b-instruct"
    SLM_INPUT_COST_1K = 0.00015
    SLM_OUTPUT_COST_1K = 0.0006
    SLM_P95_LATENCY_MS = 380.0

    # Complex reasoning indicator tokens triggering frontier escalation
    COMPLEX_INDICATORS: list[re.Pattern] = [
        re.compile(r"\b(debate|consensus|critique|red-team|arbitrate)\b", re.I),
        re.compile(r"\b(self-heal|patch|stack\s*trace|remediation|code\s*repair)\b", re.I),
        re.compile(r"\b(synthesize\s*diff|ast\s*analysis|call\s*graph)\b", re.I),
        re.compile(r"\b(multi-modal|satellite\s*image|radar|bol\s*audit)\b", re.I),
        re.compile(r"\b(reconcil|anomaly\s*detect|variance\s*analysis)\b", re.I),
        re.compile(r"\b(negotiat|game-theoretic|root\s*cause|optimiz|re-plan)\b", re.I),
    ]

    @classmethod
    def classify_complexity(
        cls,
        prompt: str,
        attachment_count: int = 0,
        has_tool_dependencies: bool = False,
    ) -> QueryComplexity:
        """Classify prompt cognitive complexity using structural analysis and token heuristics."""
        text = prompt.strip()

        # Check for complex multi-agent or code repair triggers
        for pattern in cls.COMPLEX_INDICATORS:
            if pattern.search(text):
                return QueryComplexity.HIGH

        if attachment_count > 1 or len(text.split()) > 400:
            return QueryComplexity.HIGH

        if has_tool_dependencies or len(text.split()) > 100 or "?" in text:
            return QueryComplexity.MEDIUM

        return QueryComplexity.LOW

    @classmethod
    def route(
        cls,
        prompt: str,
        attachment_count: int = 0,
        has_tool_dependencies: bool = False,
        force_frontier: bool = False,
    ) -> RoutingDecision:
        """Compute optimal model routing decision with cost savings metrics."""
        complexity = cls.classify_complexity(prompt, attachment_count, has_tool_dependencies)

        if force_frontier or complexity == QueryComplexity.HIGH:
            return RoutingDecision(
                tier=ModelTier.FRONTIER_TIER_2,
                selected_model=cls.FRONTIER_MODEL,
                complexity=complexity,
                estimated_latency_ms=cls.FRONTIER_P95_LATENCY_MS,
                cost_per_1k_input_tokens_usd=cls.FRONTIER_INPUT_COST_1K,
                cost_per_1k_output_tokens_usd=cls.FRONTIER_OUTPUT_COST_1K,
                savings_vs_frontier_percent=0.0,
                routing_rationale=(
                    f"Escalated to Frontier model {cls.FRONTIER_MODEL} due to {complexity.value} "
                    f"reasoning complexity requiring deep semantic deduction."
                ),
            )

        # Route to Tier 1 SLM
        cost_diff = (cls.FRONTIER_INPUT_COST_1K - cls.SLM_INPUT_COST_1K) / cls.FRONTIER_INPUT_COST_1K
        savings_pct = round(cost_diff * 100.0, 1)

        return RoutingDecision(
            tier=ModelTier.SLM_TIER_1,
            selected_model=cls.SLM_MODEL,
            complexity=complexity,
            estimated_latency_ms=cls.SLM_P95_LATENCY_MS,
            cost_per_1k_input_tokens_usd=cls.SLM_INPUT_COST_1K,
            cost_per_1k_output_tokens_usd=cls.SLM_OUTPUT_COST_1K,
            savings_vs_frontier_percent=savings_pct,
            routing_rationale=(
                f"Routed to Tier 1 SLM {cls.SLM_MODEL} for {complexity.value} complexity task, "
                f"achieving {savings_pct}% token cost reduction and sub-400ms latency."
            ),
        )
