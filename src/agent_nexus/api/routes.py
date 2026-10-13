"""FastAPI routing endpoints for Agent Nexus streaming gateway."""

import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sse_starlette.sse import EventSourceResponse

from agent_nexus.engine.runner import StreamingAgentRunner
from agent_nexus.ingestion.parser import (
    DocumentIngestionError,
    DocumentIngestionService,
)
from agent_nexus.registry.catalog import (
    AgentNotFoundError,
    AgentRegistry,
    default_registry,
)
from agent_nexus.schemas.protocol import AgentMetadata, HumanApprovalResponse

router = APIRouter(prefix="/api/v1", tags=["Execution Gateway"])
ingestion_service = DocumentIngestionService()

# Global execution runners cache keyed by agent_id or thread_id
_graph_runners: dict[str, StreamingAgentRunner] = {}


def register_runner(agent_id: str, runner: StreamingAgentRunner) -> None:
    """Register or override a StreamingAgentRunner for an agent."""
    _graph_runners[agent_id] = runner


def get_runner(agent_id: str) -> StreamingAgentRunner:
    """Retrieve the runner bound to an agent."""
    if agent_id in _graph_runners:
        return _graph_runners[agent_id]

    # Check if registered agent has a graph factory
    try:
        agent_entry = default_registry.get_agent(agent_id)
        factory = agent_entry.get("graph_factory_fn")
        if factory:
            graph = factory()
            if graph is not None:
                runner = StreamingAgentRunner(graph)
                _graph_runners[agent_id] = runner
                return runner
    except Exception:
        pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"No execution graph registered for agent '{agent_id}'.",
    )


def set_registry(custom_registry: AgentRegistry) -> None:
    """Utility for dependency injection / testing."""
    global default_registry
    default_registry = custom_registry


@router.get("/agents", response_model=list[AgentMetadata])
async def list_agents() -> list[AgentMetadata]:
    """Retrieve metadata for all registered agents in the catalog."""
    return default_registry.list_registered_agents()


@router.post("/execute")
async def execute_agent(
    prompt: str = Form(...),
    agent_id: str = Form(...),
    session_id: str | None = Form(None),
    files: list[UploadFile] | None = File(None),
) -> EventSourceResponse:
    """Execute an agent workflow and stream standard Server-Sent Events (SSE)."""
    # 1. Validate agent exists in registry
    try:
        default_registry.get_agent(agent_id)
    except AgentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    # 2. Assign session ID if not provided
    active_session_id = session_id or str(uuid.uuid4())

    # 3. Process optional attachments
    parsed_attachments: list[dict[str, Any]] = []
    if files:
        for file in files:
            content_bytes = await file.read()
            if content_bytes:
                try:
                    doc = ingestion_service.parse_attachment(
                        filename=file.filename or "attachment",
                        content_bytes=content_bytes,
                        mime_type=file.content_type or "application/octet-stream",
                    )
                    parsed_attachments.append(doc)
                except DocumentIngestionError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed parsing attachment '{file.filename}': {exc}",
                    ) from exc

    # 4. Adapt inputs to agent's state structure
    initial_state = default_registry.adapt_inputs(
        agent_id=agent_id,
        prompt=prompt,
        attachments=parsed_attachments,
    )

    # 5. Retrieve runner
    runner = get_runner(agent_id)

    # 6. Stream JSON-serialized ExecutionEvents as SSE
    async def sse_event_generator() -> AsyncGenerator[str, None]:
        async for event in runner.astream_events(
            initial_state=initial_state,
            session_id=active_session_id,
        ):
            yield json.dumps(event.model_dump(mode="json"))

    return EventSourceResponse(sse_event_generator())


@router.post("/approve")
async def approve_step(
    payload: HumanApprovalResponse,
    agent_id: str | None = None,
) -> EventSourceResponse:
    """Resume execution of an interrupted agent thread with human decision."""
    target_agent_id = agent_id

    # If agent_id not provided, look in cached runners
    if not target_agent_id:
        if _graph_runners:
            target_agent_id = next(iter(_graph_runners.keys()))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No agent context available to resume thread.",
            )

    runner = get_runner(target_agent_id)

    approval_dict = {
        "approved": payload.approved,
        "reason": payload.reason,
        "session_id": payload.session_id,
        "thread_id": payload.thread_id,
    }

    async def sse_resume_generator() -> AsyncGenerator[str, None]:
        async for event in runner.resume_execution(
            thread_id=payload.thread_id,
            approval_decision=approval_dict,
        ):
            yield json.dumps(event.model_dump(mode="json"))

    return EventSourceResponse(sse_resume_generator())
