"""Main FastAPI application factory for Agent Nexus Gateway."""

from pathlib import Path

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app


app = create_app()
