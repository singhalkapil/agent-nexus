"""Document ingestion service for multi-modal attachment parsing.

Provides secure parsing for PDF, RFC 822 emails, markdown, and plain text files
with strict size limits and boundary guards.
"""

import io
import re
import unicodedata
from email import policy
from email.parser import BytesParser
from typing import Any

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit

DISALLOWED_MIME_PATTERNS = [
    r"^application/x-dosexec",
    r"^application/x-executable",
    r"^application/x-sharedlib",
    r"^application/x-mach-binary",
    r"^application/x-msdownload",
    r"^application/x-sh",
    r"^text/x-shellscript",
    r"^application/x-bat",
]

DISALLOWED_EXTENSIONS = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".sh",
    ".bat",
    ".cmd",
    ".com",
    ".ps1",
    ".scr",
    ".vbs",
}


class DocumentIngestionError(Exception):
    """Base exception for document parsing failures."""


class FileTooLargeError(DocumentIngestionError, ValueError):
    """Raised when file exceeds the maximum allowed size."""


class UnsupportedMimeTypeError(DocumentIngestionError, ValueError):
    """Raised when file content or MIME type is rejected by security guards."""


class DocumentParsingError(DocumentIngestionError, ValueError):
    """Raised when document content is corrupted or unparseable."""


class DocumentIngestionService:
    """Service to normalize and parse document attachments safely."""

    def __init__(self, max_file_size: int = MAX_FILE_SIZE_BYTES) -> None:
        self.max_file_size = max_file_size

    def validate_security_boundaries(
        self, filename: str, content_bytes: bytes, mime_type: str
    ) -> None:
        """Validate payload size, extension, and MIME type."""
        if len(content_bytes) > self.max_file_size:
            raise FileTooLargeError(
                f"File size {len(content_bytes)} bytes exceeds limit of {self.max_file_size} bytes."
            )

        lower_filename = filename.lower()
        for ext in DISALLOWED_EXTENSIONS:
            if lower_filename.endswith(ext):
                raise UnsupportedMimeTypeError(
                    f"Forbidden file extension: '{ext}' for file '{filename}'."
                )

        normalized_mime = (mime_type or "").strip().lower()
        for pattern in DISALLOWED_MIME_PATTERNS:
            if re.match(pattern, normalized_mime):
                raise UnsupportedMimeTypeError(
                    f"Disallowed executable MIME type '{mime_type}' for file '{filename}'."
                )

    def normalize_text(self, text: str) -> str:
        """Apply Unicode normalization and whitespace trimming."""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        lines = [line.strip() for line in normalized.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned.strip()

    def parse_pdf(self, content_bytes: bytes) -> tuple[str, dict[str, Any]]:
        """Extract clean text and page count from PDF bytes stream."""
        try:
            stream = io.BytesIO(content_bytes)
            reader = PdfReader(stream)
            extracted_pages: list[str] = []
            for _, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                # Strip simple common page headers/footers like 'Page X of Y' or 'Page X'
                page_text_clean = re.sub(
                    r"(?i)(?:page\s+\d+(\s+of\s+\d+)?|\b\d+\s+of\s+\d+\b)",
                    "",
                    page_text,
                )
                cleaned = self.normalize_text(page_text_clean)
                if cleaned:
                    extracted_pages.append(cleaned)

            combined_text = "\n\n".join(extracted_pages)
            metadata = {
                "format": "pdf",
                "page_count": len(reader.pages),
                "is_encrypted": reader.is_encrypted,
            }
            return combined_text, metadata
        except (PdfReadError, Exception) as exc:
            raise DocumentParsingError(f"Failed to parse PDF document: {exc}") from exc

    def parse_email(self, content_bytes: bytes) -> tuple[str, dict[str, Any]]:
        """Parse RFC 822 email format extracting headers and plain text body."""
        try:
            msg = BytesParser(policy=policy.default).parsebytes(content_bytes)
            subject = str(msg.get("subject", "") or "")
            sender = str(msg.get("from", "") or "")
            date = str(msg.get("date", "") or "")

            body_parts: list[str] = []
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    if "attachment" in content_disposition:
                        continue
                    if content_type == "text/plain":
                        payload = part.get_payload(decode=True)
                        if payload:
                            body_parts.append(
                                payload.decode(
                                    part.get_content_charset() or "utf-8", errors="replace"
                                )
                            )
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    body_parts.append(
                        payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
                    )

            raw_body = "\n\n".join(body_parts)
            clean_body = self.normalize_text(raw_body)

            metadata = {
                "format": "email",
                "subject": subject,
                "sender": sender,
                "date": date,
            }
            return clean_body, metadata
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse RFC 822 email: {exc}") from exc

    def parse_attachment(
        self,
        filename: str,
        content_bytes: bytes,
        mime_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Parse attachment and return standardized document dict."""
        self.validate_security_boundaries(filename, content_bytes, mime_type)

        lower_filename = filename.lower()
        lower_mime = (mime_type or "").strip().lower()

        if lower_filename.endswith(".pdf") or "pdf" in lower_mime:
            content, meta = self.parse_pdf(content_bytes)
        elif (
            lower_filename.endswith(".eml")
            or lower_filename.endswith(".msg")
            or lower_mime in {"message/rfc822", "application/vnd.ms-outlook"}
        ):
            content, meta = self.parse_email(content_bytes)
        elif lower_mime.startswith("text/") or lower_filename.endswith(
            (".txt", ".md", ".csv", ".json", ".xml")
        ):
            try:
                raw_text = content_bytes.decode("utf-8", errors="replace")
            except Exception as exc:
                raise DocumentParsingError(f"Failed to decode text: {exc}") from exc
            content = self.normalize_text(raw_text)
            meta = {
                "format": "text",
                "encoding": "utf-8",
            }
        else:
            try:
                raw_text = content_bytes.decode("utf-8", errors="replace")
                content = self.normalize_text(raw_text)
                meta = {"format": "generic_text"}
            except Exception as exc:
                raise DocumentParsingError(
                    f"Unsupported format and unable to extract text for '{filename}': {exc}"
                ) from exc

        return {
            "filename": filename,
            "content": content,
            "metadata": meta,
        }
