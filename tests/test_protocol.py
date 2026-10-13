"""Unit tests for Agent Nexus unified protocol schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agent_nexus.schemas.protocol import (
    AgentMetadata,
    ExecutionEvent,
    ExecutionRequest,
    HumanApprovalResponse,
)


class TestAgentMetadata:
    """Tests for AgentMetadata schema."""

    def test_valid_creation_and_defaults(self):
        metadata = AgentMetadata(
            agent_id="agent-pdf-summarizer",
            name="PDF Summarizer",
            description="Extracts and summarizes PDF documents",
        )
        assert metadata.agent_id == "agent-pdf-summarizer"
        assert metadata.name == "PDF Summarizer"
        assert metadata.description == "Extracts and summarizes PDF documents"
        assert metadata.input_types == []
        assert metadata.supported_files == []

    def test_immutability(self):
        metadata = AgentMetadata(
            agent_id="agent-01",
            name="Agent One",
            description="Test agent",
            input_types=["text/plain"],
            supported_files=[".txt"],
        )
        with pytest.raises(ValidationError):
            metadata.name = "Agent Two"  # type: ignore[misc]

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            AgentMetadata(
                agent_id="agent-01",
                name="Agent One",
                description="Test agent",
                unexpected_field="disallowed",  # type: ignore[call-arg]
            )

    def test_json_roundtrip(self):
        metadata = AgentMetadata(
            agent_id="agent-qa",
            name="QA Agent",
            description="Performs adversarial tests",
            input_types=["application/json"],
            supported_files=[".json"],
        )
        json_data = metadata.model_dump_json()
        deserialized = AgentMetadata.model_validate_json(json_data)
        assert deserialized == metadata


class TestExecutionRequest:
    """Tests for ExecutionRequest schema."""

    def test_valid_creation_and_defaults(self):
        req = ExecutionRequest(
            agent_id="agent-code",
            session_id="session-123",
            prompt="Refactor authentication handler",
        )
        assert req.agent_id == "agent-code"
        assert req.session_id == "session-123"
        assert req.prompt == "Refactor authentication handler"
        assert req.parameters == {}

    def test_immutability(self):
        req = ExecutionRequest(
            agent_id="agent-code",
            session_id="session-123",
            prompt="Test prompt",
            parameters={"temperature": 0.2},
        )
        with pytest.raises(ValidationError):
            req.prompt = "New prompt"  # type: ignore[misc]

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            ExecutionRequest(
                agent_id="agent-code",
                session_id="session-123",
                prompt="Prompt",
                invalid_extra=123,  # type: ignore[call-arg]
            )

    def test_json_roundtrip(self):
        req = ExecutionRequest(
            agent_id="agent-code",
            session_id="sess-abc",
            prompt="Analyze memory leak",
            parameters={"verbose": True, "top_k": 5},
        )
        json_str = req.model_dump_json()
        recovered = ExecutionRequest.model_validate_json(json_str)
        assert recovered == req


class TestExecutionEvent:
    """Tests for ExecutionEvent schema."""

    @pytest.mark.parametrize(
        "event_type",
        [
            "NODE_START",
            "NODE_END",
            "TOKEN_STREAM",
            "HUMAN_APPROVAL_REQUIRED",
            "EXECUTION_COMPLETE",
            "ERROR",
        ],
    )
    def test_valid_event_types(self, event_type):
        now = datetime.now(UTC)
        event = ExecutionEvent(
            event_id="evt-001",
            timestamp=now,
            node_name="planner_node",
            event_type=event_type,
            payload={"status": "in_progress"},
        )
        assert event.event_type == event_type
        assert event.node_name == "planner_node"

    def test_invalid_event_type_rejected(self):
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ExecutionEvent(
                event_id="evt-bad",
                timestamp=now,
                node_name="test_node",
                event_type="UNSUPPORTED_EVENT_TYPE",  # type: ignore[arg-type]
                payload={},
            )

    def test_immutability(self):
        now = datetime.now(UTC)
        event = ExecutionEvent(
            event_id="evt-002",
            timestamp=now,
            node_name="summarizer",
            event_type="NODE_START",
        )
        with pytest.raises(ValidationError):
            event.node_name = "different_node"  # type: ignore[misc]

    def test_extra_forbidden(self):
        now = datetime.now(UTC)
        with pytest.raises(ValidationError):
            ExecutionEvent(
                event_id="evt-003",
                timestamp=now,
                node_name="node",
                event_type="TOKEN_STREAM",
                unrecognized_arg="nope",  # type: ignore[call-arg]
            )

    def test_json_roundtrip(self):
        now = datetime.now(UTC)
        event = ExecutionEvent(
            event_id="evt-stream-42",
            timestamp=now,
            node_name="llm_generator",
            event_type="TOKEN_STREAM",
            payload={"delta": "Hello world", "index": 0},
        )
        json_str = event.model_dump_json()
        restored = ExecutionEvent.model_validate_json(json_str)
        assert restored == event


class TestHumanApprovalResponse:
    """Tests for HumanApprovalResponse schema."""

    def test_valid_creation(self):
        resp = HumanApprovalResponse(
            session_id="session-456",
            thread_id="thread-789",
            approved=True,
            reason="Verified safe execution",
        )
        assert resp.session_id == "session-456"
        assert resp.thread_id == "thread-789"
        assert resp.approved is True
        assert resp.reason == "Verified safe execution"

    def test_optional_reason_default_none(self):
        resp = HumanApprovalResponse(
            session_id="session-456",
            thread_id="thread-789",
            approved=False,
        )
        assert resp.reason is None

    def test_immutability(self):
        resp = HumanApprovalResponse(
            session_id="sess-1",
            thread_id="th-1",
            approved=True,
        )
        with pytest.raises(ValidationError):
            resp.approved = False  # type: ignore[misc]

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            HumanApprovalResponse(
                session_id="sess-1",
                thread_id="th-1",
                approved=True,
                extra_note="rejected",  # type: ignore[call-arg]
            )

    def test_json_roundtrip(self):
        resp = HumanApprovalResponse(
            session_id="session-human-1",
            thread_id="thread-gate-0",
            approved=False,
            reason="Blocked by compliance policies",
        )
        json_str = resp.model_dump_json()
        recovered = HumanApprovalResponse.model_validate_json(json_str)
        assert recovered == resp
