"""End-to-end integration tests for Agent Nexus API endpoints and Static UI."""

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


def build_test_graph():
    """Create a sample 2-node graph with interrupt for endpoint verification."""

    def node_step(state: dict[str, Any]) -> dict[str, Any]:
        return {"processed": True}

    def approval_step(state: dict[str, Any]) -> dict[str, Any]:
        ans = interrupt({"alert": "Authorize detour cost of $5200"})
        return {"approved": ans, "completed": True}

    builder = StateGraph(dict)
    builder.add_node("node_step", node_step)
    builder.add_node("approval_step", approval_step)
    builder.add_edge(START, "node_step")
    builder.add_edge("node_step", "approval_step")
    builder.add_edge("approval_step", END)

    return builder.compile(checkpointer=MemorySaver())


class TestApiEndpointsAndUI:
    """Comprehensive test suite for API routes, multipart uploads, and UI serving."""

    @pytest.fixture(autouse=True)
    def setup_runners(self):
        graph = build_test_graph()
        runner = StreamingAgentRunner(graph)
        register_runner("freight-reroute-agent", runner)
        register_runner("cve-triage-agent", runner)
        return runner

    @pytest.mark.asyncio
    async def test_get_agents_catalog(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/agents")
            assert resp.status_code == 200
            data = resp.json()
            assert isinstance(data, list)
            agent_ids = {a["agent_id"] for a in data}
            assert "freight-reroute-agent" in agent_ids
            assert "cve-triage-agent" in agent_ids

    @pytest.mark.asyncio
    async def test_post_execute_with_multipart_attachment(self):
        session_id = "test-e2e-session-files"
        files = {
            "files": (
                "cargo_manifest.txt",
                b"Container C-101 destined for Oakland disrupted.",
                "text/plain",
            )
        }
        form_data = {
            "prompt": "Analyze disruption manifest",
            "agent_id": "freight-reroute-agent",
            "session_id": session_id,
        }

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/execute",
                data=form_data,
                files=files,
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            lines = resp.text.splitlines()
            events = [
                json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: ")
            ]
            assert len(events) > 0
            event_types = [e["event_type"] for e in events]
            assert "NODE_START" in event_types
            assert "HUMAN_APPROVAL_REQUIRED" in event_types

    @pytest.mark.asyncio
    async def test_post_approve_checkpoint_resumption(self):
        session_id = "test-e2e-approve-session"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # 1. Run until human approval required
            init_resp = await client.post(
                "/api/v1/execute",
                data={
                    "prompt": "Evaluate routing",
                    "agent_id": "freight-reroute-agent",
                    "session_id": session_id,
                },
            )
            assert init_resp.status_code == 200

            # 2. Submit approval payload
            approval_payload = {
                "session_id": session_id,
                "thread_id": session_id,
                "approved": True,
                "reason": "Passed risk threshold assessment",
            }
            appr_resp = await client.post(
                "/api/v1/approve?agent_id=freight-reroute-agent",
                json=approval_payload,
            )
            assert appr_resp.status_code == 200
            assert "text/event-stream" in appr_resp.headers["content-type"]

            lines = appr_resp.text.splitlines()
            events = [
                json.loads(line[len("data: ") :]) for line in lines if line.startswith("data: ")
            ]
            assert len(events) > 0
            assert events[-1]["event_type"] == "EXECUTION_COMPLETE"

    @pytest.mark.asyncio
    async def test_static_ui_mount_serves_html(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers.get("content-type", "")
            assert "Agent Nexus" in resp.text
            assert "Multi-Agent Gateway & Autonomous Streaming Orchestrator" in resp.text
            assert "Execution Timeline" in resp.text
