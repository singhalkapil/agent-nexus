"""Unit tests for DocumentIngestionService and AgentRegistry."""

import io
from email.message import EmailMessage

import pypdf
import pytest

from agent_nexus.ingestion.parser import (
    DocumentIngestionService,
    DocumentParsingError,
    FileTooLargeError,
    UnsupportedMimeTypeError,
)
from agent_nexus.registry.catalog import (
    AgentNotFoundError,
    AgentRegistry,
    cve_triage_adapter,
    freight_reroute_adapter,
)
from agent_nexus.schemas.protocol import AgentMetadata

RAW_EMAIL_FIXTURE = b"""From: dispatcher@globalfreight.com
To: operations@agentnexus.local
Subject: Critical Disruption Alert: Port Congestion
Date: Tue, 13 Oct 2026 14:30:00 +0000
Content-Type: text/plain; charset="utf-8"

Vessel Evergreen-9 has been held up at terminal 4 due to berth strikes.
All downstream freight requires immediate rerouting.
"""


def create_mock_pdf_bytes(text_lines: list[str] | None = None) -> bytes:
    """Helper to generate valid PDF byte stream with optional mocked text."""
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=612, height=792)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


class TestDocumentIngestionService:
    """Tests for document parsing, normalization, and security boundaries."""

    @pytest.fixture
    def service(self):
        return DocumentIngestionService(max_file_size=10 * 1024 * 1024)

    def test_pdf_parsing_success(self, service, monkeypatch):
        pdf_bytes = create_mock_pdf_bytes()

        # Mock page text extraction to guarantee deterministic, isolated verification
        def mock_extract_text(self, *args, **kwargs):
            return "Cargo Manifest Port of Oakland\nPage 1 of 2\nContainer ID: CT-9901"

        monkeypatch.setattr(pypdf.PageObject, "extract_text", mock_extract_text)

        result = service.parse_attachment(
            filename="manifest.pdf",
            content_bytes=pdf_bytes,
            mime_type="application/pdf",
        )
        assert result["filename"] == "manifest.pdf"
        assert "Cargo Manifest Port of Oakland" in result["content"]
        assert "Container ID: CT-9901" in result["content"]
        assert "Page 1 of 2" not in result["content"]  # Verified header/footer stripping
        assert result["metadata"]["format"] == "pdf"
        assert result["metadata"]["page_count"] == 1

    def test_pdf_parsing_malformed_raises_error(self, service):
        corrupted_bytes = b"NOT A VALID PDF CONTENT STREAM"
        with pytest.raises(DocumentParsingError):
            service.parse_attachment(
                filename="broken.pdf",
                content_bytes=corrupted_bytes,
                mime_type="application/pdf",
            )

    def test_email_parsing_rfc822(self, service):
        result = service.parse_attachment(
            filename="alert.eml",
            content_bytes=RAW_EMAIL_FIXTURE,
            mime_type="message/rfc822",
        )
        assert result["filename"] == "alert.eml"
        assert "Vessel Evergreen-9 has been held up" in result["content"]
        assert result["metadata"]["format"] == "email"
        assert result["metadata"]["subject"] == "Critical Disruption Alert: Port Congestion"
        assert result["metadata"]["sender"] == "dispatcher@globalfreight.com"

    def test_email_parsing_multipart(self, service):
        msg = EmailMessage()
        msg["Subject"] = "Multipart Log"
        msg["From"] = "system@nexus.ai"
        msg.set_content("Plain text body content for inspection.")

        result = service.parse_attachment(
            filename="notice.eml",
            content_bytes=msg.as_bytes(),
            mime_type="message/rfc822",
        )
        assert "Plain text body content for inspection." in result["content"]
        assert result["metadata"]["subject"] == "Multipart Log"

    def test_plain_text_normalization(self, service):
        raw_text = "  First line with spaces   \r\n\r\n\r\n   Second line   \n"
        result = service.parse_attachment(
            filename="notes.txt",
            content_bytes=raw_text.encode("utf-8"),
            mime_type="text/plain",
        )
        assert result["content"] == "First line with spaces\nSecond line"
        assert result["metadata"]["format"] == "text"

    def test_file_size_limit_enforced(self, service):
        oversized_data = b"X" * (10 * 1024 * 1024 + 1)
        with pytest.raises(FileTooLargeError):
            service.parse_attachment(
                filename="huge.txt",
                content_bytes=oversized_data,
                mime_type="text/plain",
            )

    def test_disallowed_mime_types_rejected(self, service):
        with pytest.raises(UnsupportedMimeTypeError):
            service.parse_attachment(
                filename="payload.bin",
                content_bytes=b"MZ\x90\x00",
                mime_type="application/x-dosexec",
            )

    def test_disallowed_file_extensions_rejected(self, service):
        with pytest.raises(UnsupportedMimeTypeError):
            service.parse_attachment(
                filename="run_script.sh",
                content_bytes=b"echo dangerous",
                mime_type="text/plain",
            )


class TestAgentRegistryAndAdapters:
    """Tests for catalog discovery and input adapters."""

    @pytest.fixture
    def registry(self):
        return AgentRegistry()

    def test_default_registered_agents_present(self, registry):
        agents = registry.list_registered_agents()
        agent_ids = {a.agent_id for a in agents}
        assert "freight-reroute-agent" in agent_ids
        assert "cve-triage-agent" in agent_ids

    def test_get_registered_agent(self, registry):
        agent = registry.get_agent("freight-reroute-agent")
        assert agent["metadata"].name == "Freight Disruption Agent"
        assert ".pdf" in agent["metadata"].supported_files

    def test_unregistered_agent_raises_not_found(self, registry):
        with pytest.raises(AgentNotFoundError):
            registry.get_agent("non-existent-agent")

        with pytest.raises(KeyError):  # AgentNotFoundError inherits from KeyError
            registry.adapt_inputs("non-existent-agent", "prompt", [])

    def test_freight_reroute_adapter_with_attachments(self):
        attachments = [
            {
                "filename": "manifest_dec.pdf",
                "content": "Port of Long Beach container congestion details",
                "metadata": {"format": "pdf"},
            },
            {
                "filename": "alert.eml",
                "content": "Storm causing 48h delays",
                "metadata": {"format": "email"},
            },
        ]
        prompt = "Assess container delays and provide reroute paths."
        adapted = freight_reroute_adapter(prompt, attachments)

        assert adapted["user_prompt"] == prompt
        assert "Port of Long Beach container congestion details" in adapted["raw_manifest_text"]
        assert "Storm causing 48h delays" in adapted["raw_manifest_text"]
        assert adapted["manifest_sources"] == ["manifest_dec.pdf", "alert.eml"]

    def test_freight_reroute_adapter_without_attachments(self):
        prompt = "Reroute ship Alpha from NY to Boston."
        adapted = freight_reroute_adapter(prompt, [])
        assert adapted["user_prompt"] == prompt
        assert adapted["raw_manifest_text"] == ""
        assert adapted["manifest_sources"] == []

    def test_cve_triage_adapter_extraction(self):
        prompt = (
            "Please triage vulnerability CVE-2024-3094 in repository path: /opt/nexus/crypto-lib"
        )
        adapted = cve_triage_adapter(prompt, [])

        assert adapted["cve_id"] == "CVE-2024-3094"
        assert adapted["target_repo_path"] == "/opt/nexus/crypto-lib"
        assert adapted["user_prompt"] == prompt

    def test_cve_triage_adapter_missing_fields_defaults_none(self):
        prompt = "Evaluate general security posture without specifics."
        adapted = cve_triage_adapter(prompt, [])

        assert adapted["cve_id"] is None
        assert adapted["target_repo_path"] is None
        assert adapted["user_prompt"] == prompt

    def test_registry_adapt_inputs_dispatch(self, registry):
        prompt = "Evaluate CVE-2023-4863 for repo: ./workspace/libwebp"
        adapted = registry.adapt_inputs("cve-triage-agent", prompt, [])

        assert adapted["cve_id"] == "CVE-2023-4863"
        assert adapted["target_repo_path"] == "./workspace/libwebp"

    def test_register_custom_agent(self, registry):
        custom_metadata = AgentMetadata(
            agent_id="custom-nlp-agent",
            name="Custom NLP Agent",
            description="Processes raw queries",
            input_types=["text/plain"],
            supported_files=[".txt"],
        )

        def custom_adapter(p, att):
            return {"query": p, "count": len(att)}

        registry.register_agent(custom_metadata, custom_adapter)
        agent = registry.get_agent("custom-nlp-agent")
        assert agent["metadata"].name == "Custom NLP Agent"

        adapted = registry.adapt_inputs("custom-nlp-agent", "hello", [{"file": "1"}])
        assert adapted == {"query": "hello", "count": 1}
