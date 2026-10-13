"""Integration tests for LangGraph SSE Streaming Runner and FastAPI endpoints."""

import json
from typing import Any

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_nexus.api.main import app
from agent_nexus.api.routes import register_runner
from agent_nexus.engine.runner import StreamingAgentRunner
from agent_nexus.schemas.protocol import ExecutionEvent


def build_mock_graph():
    """Build a 3-node LangGraph state graph with an interrupt and token streaming."""

    def inspect_manifest(state: dict[str, Any]) -> dict[str, Any]:
        return {"manifest_inspected": True, "detour_cost": 6500}

    def detect_bottlenecks(state: dict[str, Any]) -> dict[str, Any]:
        # Triggers interrupt for detour cost > 5000
        decision = interrupt(
            {
                "question": "Detour cost exceeds $5,000 threshold. Approve reroute?",
                "cost": state.get("detour_cost", 0),
            }
        )
        return {"approval": decision}

    def reroute_freight(state: dict[str, Any]) -> dict[str, Any]:
        return {"rerouted": True, "status": "COMPLETED"}

    builder = StateGraph(dict)
    builder.add_node("inspect_manifest", inspect_manifest)
    builder.add_node("detect_bottlenecks", detect_bottlenecks)
    builder.add_node("reroute_freight", reroute_freight)

    builder.add_edge(START, "inspect_manifest")
    builder.add_edge("inspect_manifest", "detect_bottlenecks")
    builder.add_edge("detect_bottlenecks", "reroute_freight")
    builder.add_edge("reroute_freight", END)

    return builder.compile(checkpointer=MemorySaver())


class TestStreamingAgentRunner:
    """Tests for StreamingAgentRunner event translation and interrupt/resumption flow."""

    @pytest.mark.asyncio
    async def test_runner_emits_ordered_events_until_interrupt(self):
        graph = build_mock_graph()
        runner = StreamingAgentRunner(graph)
        session_id = "test-session-runner-1"

        events: list[ExecutionEvent] = []
        async for event in runner.astream_events(
            initial_state={"user_prompt": "Analyze Oakland delay"},
            session_id=session_id,
        ):
            events.append(event)

        event_types = [e.event_type for e in events]

        # Expect inspect_manifest start/end, then detect_bottlenecks start,
        # then HUMAN_APPROVAL_REQUIRED event
        assert "NODE_START" in event_types
        assert "NODE_END" in event_types
        assert event_types[-1] == "HUMAN_APPROVAL_REQUIRED"

        # Verify interrupt payload details
        interrupt_event = events[-1]
        assert interrupt_event.payload["thread_id"] == session_id
        assert "Detour cost exceeds" in str(interrupt_event.payload["interrupts"])

    @pytest.mark.asyncio
    async def test_runner_resume_execution_to_completion(self):
        graph = build_mock_graph()
        runner = StreamingAgentRunner(graph)
        session_id = "test-session-runner-2"

        # Phase 1: Run to interrupt
        phase1_events: list[ExecutionEvent] = []
        async for event in runner.astream_events(
            initial_state={"user_prompt": "Initial prompt"},
            session_id=session_id,
        ):
            phase1_events.append(event)

        assert phase1_events[-1].event_type == "HUMAN_APPROVAL_REQUIRED"

        # Phase 2: Resume with approval
        phase2_events: list[ExecutionEvent] = []
        async for event in runner.resume_execution(
            thread_id=session_id,
            approval_decision={"approved": True, "reason": "Authorized by Logistics Director"},
        ):
            phase2_events.append(event)

        phase2_types = [e.event_type for e in phase2_events]
        assert "NODE_START" in phase2_types
        assert "NODE_END" in phase2_types
        assert phase2_types[-1] == "EXECUTION_COMPLETE"
        assert phase2_events[-1].payload["final_state"]["rerouted"] is True

    @pytest.mark.asyncio
    async def test_runner_catches_node_exception_and_emits_error_event(self):
        def faulty_node(state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("Database connection timed out during execution")

        builder = StateGraph(dict)
        builder.add_node("faulty_node", faulty_node)
        builder.add_edge(START, "faulty_node")
        builder.add_edge("faulty_node", END)
        graph = builder.compile(checkpointer=MemorySaver())

        runner = StreamingAgentRunner(graph)
        events: list[ExecutionEvent] = []
        async for event in runner.astream_events(
            initial_state={"test": 1},
            session_id="fault-session",
        ):
            events.append(event)

        error_events = [e for e in events if e.event_type == "ERROR"]
        assert len(error_events) == 1
        assert "Database connection timed out" in error_events[0].payload["error"]


class TestFastAPIStreamingEndpoints:
    """Tests for FastAPI HTTP SSE streaming routes."""

    @pytest.fixture(autouse=True)
    def setup_mock_runner(self):
        graph = build_mock_graph()
        runner = StreamingAgentRunner(graph)
        register_runner("freight-reroute-agent", runner)
        return runner

    @pytest.mark.asyncio
    async def test_get_agents_endpoint(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/agents")
            assert resp.status_code == 200
            data = resp.json()
            agent_ids = {a["agent_id"] for a in data}
            assert "freight-reroute-agent" in agent_ids
            assert "cve-triage-agent" in agent_ids

    @pytest.mark.asyncio
    async def test_execute_endpoint_sse_stream(self):
        session_id = "test-api-session-1"
        form_data = {
            "prompt": "Evaluate container disruption in Long Beach",
            "agent_id": "freight-reroute-agent",
            "session_id": session_id,
        }

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/v1/execute", data=form_data)
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            lines = resp.text.splitlines()
            events: list[dict[str, Any]] = []
            for line in lines:
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])
                    events.append(payload)

            assert len(events) > 0
            event_types = [e["event_type"] for e in events]
            assert "NODE_START" in event_types
            assert "HUMAN_APPROVAL_REQUIRED" in event_types
            assert events[-1]["event_type"] == "HUMAN_APPROVAL_REQUIRED"

    @pytest.mark.asyncio
    async def test_approve_endpoint_sse_resumption(self):
        session_id = "test-api-session-2"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # 1. Execute until interrupt
            exec_resp = await client.post(
                "/api/v1/execute",
                data={
                    "prompt": "Reroute ship",
                    "agent_id": "freight-reroute-agent",
                    "session_id": session_id,
                },
            )
            assert exec_resp.status_code == 200

            # 2. Approve and stream completion
            approval_payload = {
                "session_id": session_id,
                "thread_id": session_id,
                "approved": True,
                "reason": "Expedited clearance granted",
            }
            appr_resp = await client.post(
                "/api/v1/approve?agent_id=freight-reroute-agent",
                json=approval_payload,
            )
            assert appr_resp.status_code == 200
            assert "text/event-stream" in appr_resp.headers["content-type"]

            lines = appr_resp.text.splitlines()
            events: list[dict[str, Any]] = []
            for line in lines:
                if line.startswith("data: "):
                    payload = json.loads(line[len("data: ") :])
                    events.append(payload)

            assert len(events) > 0
            assert events[-1]["event_type"] == "EXECUTION_COMPLETE"
            assert events[-1]["payload"]["final_state"]["rerouted"] is True
