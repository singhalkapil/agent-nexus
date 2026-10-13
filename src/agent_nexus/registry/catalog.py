"""Agent registry and input adaptation catalog.

Manages agent registration, metadata discovery, input state adaptation,
and graph factory binding.
"""

import re
from collections.abc import Callable
from typing import Any

from agent_nexus.schemas.protocol import AgentMetadata


class AgentRegistryError(Exception):
    """Base exception for agent registry operations."""


class AgentNotFoundError(AgentRegistryError, KeyError):
    """Raised when looking up an unregistered agent."""


class AgentRegistrationError(AgentRegistryError, ValueError):
    """Raised when registering an invalid or duplicate agent."""


def freight_reroute_adapter(prompt: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    """Input adapter for the Freight Disruption Agent.

    Extracts raw text from PDF/email attachments into raw_manifest_text
    and passes the prompt into user_prompt.
    """
    raw_manifest_text = ""
    manifest_sources: list[str] = []

    for attachment in attachments or []:
        filename = (attachment.get("filename") or "").lower()
        metadata = attachment.get("metadata", {})
        format_type = metadata.get("format", "")
        content = attachment.get("content", "")

        if (
            filename.endswith((".pdf", ".eml", ".txt"))
            or format_type in {"pdf", "email", "text"}
            or "manifest" in filename
        ):
            if content:
                if raw_manifest_text:
                    raw_manifest_text += "\n\n"
                raw_manifest_text += content
                manifest_sources.append(attachment.get("filename", "unnamed"))

    return {
        "user_prompt": prompt,
        "raw_manifest_text": raw_manifest_text,
        "manifest_sources": manifest_sources,
        "attachments": attachments,
    }


def cve_triage_adapter(prompt: str, attachments: list[dict[str, Any]]) -> dict[str, Any]:
    """Input adapter for the CVE Impact Triage Agent.

    Parses CVE IDs and target repository path from prompt or attachments.
    """
    cve_pattern = r"(CVE-\d{4}-\d{4,7})"
    cve_match = re.search(cve_pattern, prompt, re.IGNORECASE)
    cve_id = cve_match.group(1).upper() if cve_match else None

    # Search for repository paths (e.g. repo=/path, path: /path, or standard unix path tokens)
    repo_pattern = r"(?:repo(?:sitory)?|path)\s*[:=]\s*([^\s,;]+)"
    repo_match = re.search(repo_pattern, prompt, re.IGNORECASE)
    target_repo_path = repo_match.group(1) if repo_match else None

    if not target_repo_path:
        # Fallback to checking for words starting with / or ./ or github repo paths
        path_fallback = re.search(r"(\.?/[a-zA-Z0-9_\-\./]+)", prompt)
        if path_fallback:
            target_repo_path = path_fallback.group(1)

    return {
        "user_prompt": prompt,
        "cve_id": cve_id,
        "target_repo_path": target_repo_path,
        "attachments": attachments,
    }


class AgentRegistry:
    """Registry maintaining metadata, input adapters, and execution graph factories."""

    def __init__(self) -> None:
        self._agents: dict[str, dict[str, Any]] = {}
        self._register_default_agents()

    def register_agent(
        self,
        metadata: AgentMetadata,
        adapter_fn: Callable[[str, list[dict[str, Any]]], dict[str, Any]],
        graph_factory_fn: Callable[..., Any] | None = None,
    ) -> None:
        """Register a new agent in the catalog."""
        if not metadata or not metadata.agent_id:
            raise AgentRegistrationError("Agent metadata with non-empty agent_id is required.")

        self._agents[metadata.agent_id] = {
            "metadata": metadata,
            "adapter_fn": adapter_fn,
            "graph_factory_fn": graph_factory_fn or (lambda: None),
        }

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        """Retrieve an agent entry by its agent_id."""
        if agent_id not in self._agents:
            raise AgentNotFoundError(f"Agent '{agent_id}' is not registered.")
        return self._agents[agent_id]

    def list_registered_agents(self) -> list[AgentMetadata]:
        """List metadata for all currently registered agents."""
        return [entry["metadata"] for entry in self._agents.values()]

    def adapt_inputs(
        self,
        agent_id: str,
        prompt: str,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Adapt user prompt and attachments into the target agent's state format."""
        agent_entry = self.get_agent(agent_id)
        adapter_fn = agent_entry["adapter_fn"]
        return adapter_fn(prompt, attachments or [])

    def _register_default_agents(self) -> None:
        """Register the built-in standard agents required for Agent Nexus."""
        freight_metadata = AgentMetadata(
            agent_id="freight-reroute-agent",
            name="Freight Disruption Agent",
            description="Analyzes disruption manifests, emails, and cargo logistics for rerouting.",
            input_types=["text/plain", "application/pdf", "message/rfc822"],
            supported_files=[".pdf", ".eml", ".txt"],
        )
        self.register_agent(
            metadata=freight_metadata,
            adapter_fn=freight_reroute_adapter,
            graph_factory_fn=None,
        )

        cve_metadata = AgentMetadata(
            agent_id="cve-triage-agent",
            name="CVE Impact Triage Agent",
            description=(
                "Analyzes security vulnerabilities against source repositories and dependencies."
            ),
            input_types=["text/plain"],
            supported_files=[],
        )
        self.register_agent(
            metadata=cve_metadata,
            adapter_fn=cve_triage_adapter,
            graph_factory_fn=None,
        )


# Global default instance
default_registry = AgentRegistry()
