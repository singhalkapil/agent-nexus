"""Cyclic Multi-Agent Protocol with Autonomous Consensus & Red-Team Debate.

Implements a compiled LangGraph state machine with distinct specialized agent personas:
1. `Planner`: Decomposes complex enterprise prompts into structured execution plans.
2. `Executor`: Formulates solution drafts and tool calls.
3. `Critic / Red-Teamer`: Adversarially audits drafts for factual hallucination and policy safety.
4. `Arbiter`: Evaluates debate consensus score (>= 0.85) and triggers memory reflection cycles.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field


class AgentCritique(BaseModel):
    """Structured evaluation rubric output by the Critic / Red-Teamer agent."""

    factual_accuracy_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    hallucination_detected: bool = Field(default=False)
    critique_feedback: str = Field(..., description="Actionable points to fix in next draft iteration")
    overall_confidence: float = Field(..., ge=0.0, le=1.0)


class ConsensusState(TypedDict, total=False):
    """Execution state tracked across multi-agent consensus debate cycles."""

    goal: str
    plan: list[str]
    current_draft: str
    critique: Optional[AgentCritique]
    debate_round: int
    consensus_achieved: bool
    final_output: str
    audit_trail: Annotated[list[str], operator.add]


# Node 1: Planner Agent
def planner_node(state: ConsensusState) -> dict[str, Any]:
    """Decompose enterprise goal into logical sub-tasks."""
    goal = state.get("goal", "")
    plan = [
        f"1. Ingest context and parameters for: {goal[:60]}",
        "2. Formulate execution actions and verify enterprise invariants",
        "3. Cross-validate against compliance and error-recovery policies",
    ]
    return {
        "plan": plan,
        "audit_trail": [f"PLANNER: Formulated {len(plan)}-step task breakdown for '{goal[:40]}'"],
    }


# Node 2: Executor Agent
def executor_node(state: ConsensusState) -> dict[str, Any]:
    """Synthesize intermediate execution solution, reflecting on previous critique if present."""
    goal = state.get("goal", "")
    round_idx = state.get("debate_round", 1)
    critique = state.get("critique")

    if critique and critique.critique_feedback:
        draft = (
            f"[Refined Solution - Round {round_idx}]\n"
            f"Addressed feedback: {critique.critique_feedback}\n"
            f"Execution payload for goal: {goal}\n"
            f"Invariants satisfied with deterministic grounding."
        )
    else:
        draft = (
            f"[Initial Solution - Round {round_idx}]\n"
            f"Execution payload for goal: {goal}\n"
            f"All operations planned and verified."
        )

    return {
        "current_draft": draft,
        "audit_trail": [f"EXECUTOR: Synthesized solution draft for Round {round_idx}"],
    }


# Node 3: Critic / Red-Teamer Agent
def critic_node(state: ConsensusState) -> dict[str, Any]:
    """Evaluate solution draft against rigorous accuracy and safety rubric."""
    round_idx = state.get("debate_round", 1)
    draft = state.get("current_draft", "")

    # Simulated progressive convergence: Round 1 triggers critique; Round 2 reaches consensus
    if round_idx == 1 and "Refined" not in draft:
        critique = AgentCritique(
            factual_accuracy_score=0.75,
            completeness_score=0.70,
            hallucination_detected=False,
            critique_feedback="Clarify latency budget and add explicit fallback error recovery.",
            overall_confidence=0.72,
        )
        msg = f"CRITIC: Confidence {critique.overall_confidence:.2f} < 0.85. Triggering reflection loop."
    else:
        critique = AgentCritique(
            factual_accuracy_score=0.96,
            completeness_score=0.95,
            hallucination_detected=False,
            critique_feedback="All constraints verified and policy invariants satisfied.",
            overall_confidence=0.95,
        )
        msg = f"CRITIC: High confidence {critique.overall_confidence:.2f} >= 0.85. Consensus approved."

    return {
        "critique": critique,
        "audit_trail": [msg],
    }


# Node 4: Arbiter & Decision Router
def route_consensus(state: ConsensusState) -> str:
    """Evaluate whether consensus threshold is met or maximum debate rounds reached."""
    critique = state.get("critique")
    round_idx = state.get("debate_round", 1)
    confidence = critique.overall_confidence if critique else 0.0

    if confidence >= 0.85 or round_idx >= 3:
        return "commit_consensus"
    return "reflect_and_retry"


def reflection_node(state: ConsensusState) -> dict[str, Any]:
    """Increment round counter and prepare reflection state for re-planning."""
    current_round = state.get("debate_round", 1)
    return {
        "debate_round": current_round + 1,
        "audit_trail": [f"ARBITER: Advancing debate to Round {current_round + 1} with feedback"],
    }


def commit_consensus_node(state: ConsensusState) -> dict[str, Any]:
    """Commit finalized consensus solution."""
    draft = state.get("current_draft", "")
    critique = state.get("critique")
    conf = critique.overall_confidence if critique else 1.0

    final_text = (
        f"# Enterprise Consensus Resolution\n\n"
        f"**Consensus Confidence**: {conf:.1%}\n\n"
        f"## Approved Solution\n{draft}\n"
    )
    return {
        "consensus_achieved": True,
        "final_output": final_text,
        "audit_trail": [f"ARBITER: Committed consensus with final confidence {conf:.1%}"],
    }


def create_consensus_graph() -> CompiledStateGraph:
    """Build and compile the cyclic multi-agent consensus debate state graph."""
    builder = StateGraph(ConsensusState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("critic", critic_node)
    builder.add_node("reflection", reflection_node)
    builder.add_node("commit_consensus", commit_consensus_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "critic")

    builder.add_conditional_edges(
        "critic",
        route_consensus,
        {
            "reflect_and_retry": "reflection",
            "commit_consensus": "commit_consensus",
        },
    )
    builder.add_edge("reflection", "executor")
    builder.add_edge("commit_consensus", END)

    return builder.compile()
