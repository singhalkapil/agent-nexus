"""Unified protocol schemas for Agent Nexus.

Defines strictly typed, immutable Pydantic v2 event protocols used across the
gateway, workflow execution engine, and client communication interfaces.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "NODE_START",
    "NODE_END",
    "TOKEN_STREAM",
    "HUMAN_APPROVAL_REQUIRED",
    "EXECUTION_COMPLETE",
    "ERROR",
]


class AgentMetadata(BaseModel):
    """Metadata describing an agent's identity and execution capabilities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(description="Unique identifier for the registered agent")
    name: str = Field(description="Human-readable name of the agent")
    description: str = Field(description="Detailed description of what the agent accomplishes")
    input_types: list[str] = Field(
        default_factory=list,
        description="Supported input MIME or semantic types",
    )
    supported_files: list[str] = Field(
        default_factory=list,
        description="List of file extension patterns or format identifiers supported",
    )


class ExecutionRequest(BaseModel):
    """Payload initiated by clients to start or resume agent execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(description="Target agent identifier to invoke")
    session_id: str = Field(description="Session identifier tracking caller conversation context")
    prompt: str = Field(description="Primary user prompt or instruction")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary execution parameters or hyperparameters",
    )


class ExecutionEvent(BaseModel):
    """Event emitted across lifecycle checkpoints and streaming responses."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(description="Unique identifier for the individual event")
    timestamp: datetime = Field(description="ISO timestamp of when the event was generated")
    node_name: str = Field(description="Workflow node or execution phase emitting the event")
    event_type: EventType = Field(description="Lifecycle type of the emitted event")
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual payload specific to the event type",
    )


class HumanApprovalResponse(BaseModel):
    """Feedback payload returned for human-in-the-loop review interrupts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(description="Target session identifier")
    thread_id: str = Field(description="Execution thread / checkpoint identifier")
    approved: bool = Field(description="Whether the requested execution step was approved")
    reason: str | None = Field(
        default=None,
        description="Optional justification or feedback accompanying approval decision",
    )
