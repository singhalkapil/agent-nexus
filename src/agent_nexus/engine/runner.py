"""Streaming execution runner for LangGraph agents.

Bridges LangGraph astream_events to Agent Nexus unified ExecutionEvent protocol,
handling node start/end, token streaming, human approval interrupts, and completion.
"""

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from agent_nexus.schemas.protocol import ExecutionEvent


class StreamingAgentRunner:
    """Orchestrates streaming event generation from a compiled LangGraph."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    async def astream_events(
        self,
        initial_state: Any,
        session_id: str,
        resume: bool = False,
    ) -> AsyncGenerator[ExecutionEvent, None]:
        """Stream events from the LangGraph execution.

        Yields ExecutionEvent instances for node lifecycles, token streaming,
        interrupt checkpoints, and execution completion.
        """
        config = {"configurable": {"thread_id": session_id}}

        # Internal graph names to filter out of node start/end events
        ignored_chain_names = {
            "LangGraph",
            "RunnableSequence",
            "RunnableParallel",
            "__start__",
            "__end__",
            "StateGraph",
        }

        try:
            input_val = initial_state
            if resume:
                # If resuming with an approval payload
                if isinstance(initial_state, Command):
                    input_val = initial_state
                elif isinstance(initial_state, dict):
                    input_val = Command(resume=initial_state)

            async for event in self.graph.astream_events(input_val, config=config, version="v2"):
                event_name = event.get("event", "")
                node_name = event.get("name", "graph")
                data = event.get("data", {})

                # 1. Node Start
                if event_name == "on_chain_start":
                    if node_name not in ignored_chain_names:
                        yield ExecutionEvent(
                            event_id=str(uuid.uuid4()),
                            timestamp=datetime.now(UTC),
                            node_name=node_name,
                            event_type="NODE_START",
                            payload={
                                "inputs": data.get("input", {}),
                                "thread_id": session_id,
                            },
                        )

                # 2. Token Stream (Chat Models or LLMs)
                elif event_name in {"on_chat_model_stream", "on_llm_stream"}:
                    chunk = data.get("chunk")
                    token_content = ""
                    if chunk is not None:
                        if hasattr(chunk, "content"):
                            token_content = str(chunk.content)
                        elif isinstance(chunk, dict):
                            token_content = str(chunk.get("content", ""))
                        else:
                            token_content = str(chunk)

                    if token_content:
                        yield ExecutionEvent(
                            event_id=str(uuid.uuid4()),
                            timestamp=datetime.now(UTC),
                            node_name=node_name,
                            event_type="TOKEN_STREAM",
                            payload={
                                "token": token_content,
                                "thread_id": session_id,
                            },
                        )

                # 3. Node End
                elif event_name == "on_chain_end":
                    if node_name not in ignored_chain_names:
                        yield ExecutionEvent(
                            event_id=str(uuid.uuid4()),
                            timestamp=datetime.now(UTC),
                            node_name=node_name,
                            event_type="NODE_END",
                            payload={
                                "outputs": data.get("output", {}),
                                "thread_id": session_id,
                            },
                        )

            # Check post-execution state for interrupts or completion
            state = self.graph.get_state(config)
            interrupts = [
                interrupt.value
                for task in (state.tasks or [])
                for interrupt in (getattr(task, "interrupts", []) or [])
            ]

            if interrupts:
                # Graph paused at an interrupt checkpoint
                active_node = state.next[0] if state.next else "approval_node"
                yield ExecutionEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(UTC),
                    node_name=active_node,
                    event_type="HUMAN_APPROVAL_REQUIRED",
                    payload={
                        "interrupts": interrupts,
                        "thread_id": session_id,
                        "session_id": session_id,
                        "current_state": state.values,
                    },
                )
            else:
                # Execution finished successfully
                yield ExecutionEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=datetime.now(UTC),
                    node_name="execution_engine",
                    event_type="EXECUTION_COMPLETE",
                    payload={
                        "final_state": state.values,
                        "thread_id": session_id,
                        "session_id": session_id,
                    },
                )

        except Exception as exc:
            yield ExecutionEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(UTC),
                node_name="execution_engine",
                event_type="ERROR",
                payload={
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "thread_id": session_id,
                },
            )

    async def resume_execution(
        self,
        thread_id: str,
        approval_decision: dict[str, Any],
    ) -> AsyncGenerator[ExecutionEvent, None]:
        """Resume an interrupted graph using the checkpointer and stream remaining events."""
        async for event in self.astream_events(
            initial_state=approval_decision,
            session_id=thread_id,
            resume=True,
        ):
            yield event
