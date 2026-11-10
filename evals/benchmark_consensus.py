"""Benchmark evaluation runner for Agent Nexus model cascade & multi-agent consensus.

Evaluates 25 enterprise workflow prompts measuring:
1. Cascade Routing Accuracy (% correctly classified as SLM vs Frontier)
2. Consensus Debate Convergence Rate (% reaching >=0.85 confidence)
3. Token Cost Reduction (% saved over uniform frontier model usage)
4. Latency Improvement (sub-second execution on Tier 1 tasks)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_nexus.engine.cascade_router import (
    ModelCascadeRouter,
    ModelTier,
    QueryComplexity,
)
from agent_nexus.engine.consensus_graph import (
    create_consensus_graph,
)

BENCHMARK_SCENARIOS = [
    {"id": "SLM-01", "prompt": "Extract the carrier name and BOL tracking number from this manifest.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-02", "prompt": "Format this JSON payload into a markdown table with 3 columns.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-03", "prompt": "Check if port USLAX is currently listed in the active port registry.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-04", "prompt": "Summarize these 3 status bullet points in 1 sentence.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-05", "prompt": "Identify any email address mentions in this carrier note.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-06", "prompt": "Filter out duplicate line items from this shipment manifest.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-07", "prompt": "Translate this French delivery receipt text to English.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-08", "prompt": "Validate whether postal code 60601 belongs to Illinois.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-09", "prompt": "Parse weight and volume numbers from this raw dispatch snippet.", "expected_tier": "SLM_TIER_1"},
    {"id": "SLM-10", "prompt": "Convert UTC timestamps into human-readable Chicago CST format.", "expected_tier": "SLM_TIER_1"},
    {"id": "FRT-01", "prompt": "Execute multi-agent debate and consensus verification to red-team this supply chain re-planning strategy.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-02", "prompt": "Analyze this AST call graph and synthesize a self-heal patch to resolve test stack trace regression.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-03", "prompt": "Perform multi-modal satellite image and radar analysis to detect hurricane eye perimeter in Gulf corridor.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-04", "prompt": "Arbitrate cross-currency ledger variance analysis with multi-tier reconciliation rules.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-05", "prompt": "Run autonomous red-team critique on enterprise role-based access control policies.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-06", "prompt": "Synthesize diff patch resolving breaking changes between PyYAML 5.3 and 6.0 in sandbox.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-07", "prompt": "Evaluate complex game-theoretic multi-agent negotiation between shipper and carrier brokers.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-08", "prompt": "Perform root cause analysis on distributed microservice trace failures during high-concurrency surge.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-09", "prompt": "Conduct multi-jurisdictional compliance critique across UKGC, MGA, and New Jersey DGE gaming rules.", "expected_tier": "FRONTIER_TIER_2"},
    {"id": "FRT-10", "prompt": "Re-plan multimodal intermodal detours optimizing across 5 conflicting SLA demurrage constraints.", "expected_tier": "FRONTIER_TIER_2"},
]


def run_benchmark() -> dict[str, Any]:
    """Execute evaluation benchmark and return quantitative scorecard."""
    results = []
    correct_routes = 0
    total_slm_cost = 0.0
    total_frontier_only_cost = 0.0

    print("=== Agent Nexus Cascade Router & Consensus Benchmark ===")

    for sc in BENCHMARK_SCENARIOS:
        start_t = time.perf_counter()
        decision = ModelCascadeRouter.route(sc["prompt"])
        duration_ms = (time.perf_counter() - start_t) * 1000.0

        is_match = decision.tier.value == sc["expected_tier"]
        if is_match:
            correct_routes += 1

        # Token cost estimation (assuming 500 input + 200 output tokens per query)
        input_tokens = 500
        output_tokens = 200

        frontier_baseline_cost = (
            (input_tokens / 1000.0) * ModelCascadeRouter.FRONTIER_INPUT_COST_1K
            + (output_tokens / 1000.0) * ModelCascadeRouter.FRONTIER_OUTPUT_COST_1K
        )
        total_frontier_only_cost += frontier_baseline_cost

        actual_cost = (
            (input_tokens / 1000.0) * decision.cost_per_1k_input_tokens_usd
            + (output_tokens / 1000.0) * decision.cost_per_1k_output_tokens_usd
        )
        total_slm_cost += actual_cost

        status = "PASS" if is_match else "FAIL"
        print(f"[{status}] {sc['id']:<8} tier={decision.tier.value:<16} complexity={decision.complexity.value:<8} ({duration_ms:.2f}ms)")

        results.append({
            "id": sc["id"],
            "prompt": sc["prompt"],
            "expected_tier": sc["expected_tier"],
            "actual_tier": decision.tier.value,
            "status": status,
            "latency_ms": round(duration_ms, 2),
            "actual_cost_usd": round(actual_cost, 6),
            "baseline_frontier_cost_usd": round(frontier_baseline_cost, 6),
        })

    # Test Consensus Convergence on 5 complex goals
    graph = create_consensus_graph()
    consensus_passed = 0
    for i in range(5):
        goal = f"Enterprise Task {i+1}: Multi-stage policy consensus verification"
        out = graph.invoke({"goal": goal, "debate_round": 1, "audit_trail": []})
        if out.get("consensus_achieved"):
            consensus_passed += 1

    routing_accuracy = (correct_routes / len(BENCHMARK_SCENARIOS)) * 100.0
    cost_savings_pct = ((total_frontier_only_cost - total_slm_cost) / total_frontier_only_cost) * 100.0
    consensus_convergence_rate = (consensus_passed / 5.0) * 100.0

    summary = {
        "total_scenarios": len(BENCHMARK_SCENARIOS),
        "routing_accuracy_percent": round(routing_accuracy, 2),
        "cost_savings_percent": round(cost_savings_pct, 2),
        "consensus_convergence_rate_percent": round(consensus_convergence_rate, 2),
        "total_cascade_cost_usd": round(total_slm_cost, 6),
        "baseline_frontier_only_cost_usd": round(total_frontier_only_cost, 6),
    }

    print("\n--- Summary Performance ---")
    print(f"Routing Accuracy:             {summary['routing_accuracy_percent']}%")
    print(f"Token Cost Reduction:         {summary['cost_savings_percent']}% (Saved ${total_frontier_only_cost - total_slm_cost:.4f} on 20 calls)")
    print(f"Consensus Convergence Rate:   {summary['consensus_convergence_rate_percent']}%")

    out_file = Path(__file__).parent / "benchmark_results.json"
    out_file.write_text(json.dumps({"summary": summary, "results": results}, indent=2), encoding="utf-8")
    print(f"Exported benchmark results to {out_file}")

    return summary


if __name__ == "__main__":
    run_benchmark()
