"""Unit tests for Agent Nexus heavy AI systems engineering features:
- Model Cascade Cost Router (SLM vs Frontier)
- Multi-Agent Consensus & Debate Graph
- OpenTelemetry / Langfuse distributed telemetry collector
"""

from __future__ import annotations

import pytest

from agent_nexus.engine.cascade_router import (
    ModelCascadeRouter,
    ModelTier,
    QueryComplexity,
)
from agent_nexus.engine.consensus_graph import (
    create_consensus_graph,
)
from agent_nexus.observability.telemetry import (
    TelemetryCollector,
)


class TestModelCascadeRouter:
    """Test dynamic query complexity classification and SLM vs Frontier cost routing."""

    def test_routes_simple_lookup_to_slm(self) -> None:
        """Simple entity extraction or short query routes to Tier 1 SLM with ~95% savings."""
        prompt = "Extract tracking ID and origin port code from this bill of lading."
        decision = ModelCascadeRouter.route(prompt)

        assert decision.tier == ModelTier.SLM_TIER_1
        assert decision.complexity == QueryComplexity.LOW
        assert decision.savings_vs_frontier_percent >= 90.0
        assert decision.estimated_latency_ms < 500.0

    def test_routes_complex_reasoning_to_frontier(self) -> None:
        """Complex prompt with consensus debate or code repair escalates to Frontier."""
        prompt = "Run multi-agent debate and consensus verification to red-team this execution plan."
        decision = ModelCascadeRouter.route(prompt)

        assert decision.tier == ModelTier.FRONTIER_TIER_2
        assert decision.complexity == QueryComplexity.HIGH
        assert decision.selected_model == ModelCascadeRouter.FRONTIER_MODEL

    def test_force_frontier_override(self) -> None:
        """Explicit force_frontier flag routes to Tier 2 regardless of simple prompt."""
        decision = ModelCascadeRouter.route("Hello world", force_frontier=True)
        assert decision.tier == ModelTier.FRONTIER_TIER_2


class TestConsensusDebateGraph:
    """Test cyclic multi-agent Planner-Executor-Critic-Arbiter state machine."""

    def test_consensus_graph_converges_within_2_rounds(self) -> None:
        """Verifies state machine executes planner, critic reflection, and commits consensus."""
        graph = create_consensus_graph()
        initial_state = {
            "goal": "Reconcile payment discrepancy across multi-currency ledger",
            "debate_round": 1,
            "audit_trail": [],
        }

        final_state = graph.invoke(initial_state)

        assert final_state.get("consensus_achieved") is True
        assert "Enterprise Consensus Resolution" in final_state.get("final_output", "")
        assert final_state.get("debate_round") >= 2

        # Verify audit trail contains records from all agent personas
        logs = " ".join(final_state.get("audit_trail", []))
        assert "PLANNER:" in logs
        assert "EXECUTOR:" in logs
        assert "CRITIC:" in logs
        assert "ARBITER:" in logs


class TestDistributedTelemetry:
    """Test span tracking and session cost attribution."""

    def test_collector_records_spans_and_aggregates_cost(self) -> None:
        """Records node lifecycles and outputs session summary with token economics."""
        collector = TelemetryCollector(session_id="test-session-001")

        collector.start_span("node_ingest")
        collector.end_span("node_ingest", input_tokens=100, output_tokens=50)

        collector.start_span("node_solve")
        collector.end_span("node_solve", input_tokens=400, output_tokens=300)

        summary = collector.summarize()

        assert summary.session_id == "test-session-001"
        assert summary.total_spans == 2
        assert summary.total_tokens == 850
        assert summary.total_cost_usd > 0.0
        assert summary.total_duration_ms > 0.0
