"""Main FastAPI application factory for Agent Nexus Gateway."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_nexus.api.routes import router as api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI gateway application."""
    app = FastAPI(
        title="Agent Nexus Gateway",
        description="Unified Event Protocol & Autonomous Multi-Agent Execution Engine",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy", "service": "agent-nexus-gateway"}

    return app


app = create_app()
