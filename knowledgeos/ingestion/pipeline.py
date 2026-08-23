from __future__ import annotations

import csv
from dataclasses import dataclass, field
from html.parser import HTMLParser
import io
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..retrieval.hybrid import HybridRetriever
from ..types import KnowledgeChunk, SourceKind
from ..utils import chunk_text, merge_metadata, stable_hash


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title: str = ""
        self._capture_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._capture_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._capture_title = False

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self.title += data.strip()
        elif data.strip():
            self.parts.append(data.strip())


@dataclass(slots=True)
class IngestionReport:
    source: str
    chunks_created: int
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)


class IngestionPipeline:
    """Enterprise Document Ingestion Pipeline supporting Text, Markdown, PDF, DOCX, CSV, JSON, and Web."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def ingest_text(
        self,
        text: str,
        *,
        title: str,
        source_kind: SourceKind = SourceKind.INTERNAL,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionReport:
        metadata = metadata or {}
        chunk_pieces = chunk_text(text, max_tokens=180, overlap=30)
        chunk_ids: list[str] = []

        for index, piece in enumerate(chunk_pieces):
            chunk = self.retriever.add_chunk(
                piece,
                title=title,
                url=metadata.get("url"),
                source_kind=source_kind,
                metadata=merge_metadata(metadata, {"chunk_index": index, "total_chunks": len(chunk_pieces)}),
            )
            chunk_ids.append(chunk.id)

        return IngestionReport(
            source=title,
            chunks_created=len(chunk_pieces),
            metadata=metadata,
            chunk_ids=chunk_ids,
        )

    def ingest_file(self, path: str | Path, metadata: dict[str, Any] | None = None) -> IngestionReport:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        meta = metadata or {}
        suffix = file_path.suffix.lower()
        title = file_path.name

        if suffix == ".docx":
            text = self._read_docx(file_path)
        elif suffix == ".pdf":
            text = self._read_pdf(file_path)
        elif suffix == ".csv":
            text = self._read_csv(file_path)
        elif suffix == ".json":
            text = self._read_json(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        return self.ingest_text(
            text,
            title=title,
            source_kind=SourceKind.INTERNAL,
            metadata=merge_metadata(meta, {"path": str(file_path), "suffix": suffix, "size_bytes": file_path.stat().st_size}),
        )

    def ingest_url(self, url: str, metadata: dict[str, Any] | None = None) -> IngestionReport:
        meta = metadata or {}
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KnowledgeOS/0.1"})
        with urlopen(req, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()

        extracted_text, page_title = self._extract_text_from_web(raw, content_type)
        parsed = urlparse(url)
        title = page_title or f"{parsed.netloc}{parsed.path}"

        return self.ingest_text(
            extracted_text,
            title=title,
            source_kind=SourceKind.WEB,
            metadata=merge_metadata(meta, {"url": url, "content_type": content_type, "domain": parsed.netloc}),
        )

    def ingest_batch(self, items: Sequence[dict[str, Any]]) -> list[IngestionReport]:
        reports = []
        for item in items:
            reports.append(self.ingest_payload(item))
        return reports

    def ingest_payload(self, payload: dict[str, Any]) -> IngestionReport:
        meta = payload.get("metadata", {})
        if "text" in payload and payload["text"]:
            return self.ingest_text(
                payload["text"],
                title=payload.get("title", "untitled_document"),
                source_kind=SourceKind(payload.get("source_kind", SourceKind.INTERNAL.value)),
                metadata=meta,
            )
        if "path" in payload and payload["path"]:
            return self.ingest_file(payload["path"], metadata=meta)
        if "url" in payload and payload["url"]:
            return self.ingest_url(payload["url"], metadata=meta)
        raise ValueError("Payload must specify 'text', 'path', or 'url'")

    def _extract_text_from_web(self, raw: bytes, content_type: str) -> tuple[str, str]:
        if "html" in content_type.lower() or b"<html" in raw[:500].lower():
            parser = _HTMLTextExtractor()
            parser.feed(raw.decode("utf-8", errors="ignore"))
            return " ".join(parser.parts), parser.title
        return raw.decode("utf-8", errors="ignore"), ""

    def _read_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using multiple strategy fallbacks."""
        # Strategy 1: pypdf (fastest, works for most text-based PDFs)
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages: list[str] = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                # Filter out pages that are clearly binary garbage (mostly non-printable chars)
                printable_ratio = sum(1 for c in extracted if c.isprintable()) / max(len(extracted), 1)
                if extracted.strip() and printable_ratio > 0.7:
                    pages.append(extracted.strip())
            if pages:
                return "\n\n".join(pages)
        except Exception:
            pass

        # Strategy 2: pdfminer (better for complex layouts)
        try:
            from pdfminer.high_level import extract_text as pdfminer_extract
            text = pdfminer_extract(str(path))
            if text and text.strip():
                # Filter binary garbage
                printable_ratio = sum(1 for c in text if c.isprintable()) / max(len(text), 1)
                if printable_ratio > 0.7:
                    return text.strip()
        except Exception:
            pass

        # Strategy 3: pymupdf / fitz (most powerful, handles encrypted PDFs)
        try:
            import fitz  # type: ignore[import]
            doc = fitz.open(str(path))
            pages = []
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    pages.append(text.strip())
            doc.close()
            if pages:
                return "\n\n".join(pages)
        except Exception:
            pass

        return f"[PDF text extraction failed for '{path.name}'. The file may be scanned, image-based, or encrypted. Please convert to text or use a text-based PDF.]"

    def _read_pdf_bytes(self, data: bytes, filename: str = "upload.pdf") -> str:
        """Extract text from PDF bytes (used by Streamlit file uploader)."""
        import io as _io

        # Strategy 1: pypdf from bytes
        try:
            from pypdf import PdfReader
            reader = PdfReader(_io.BytesIO(data))
            pages: list[str] = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                printable_ratio = sum(1 for c in extracted if c.isprintable()) / max(len(extracted), 1)
                if extracted.strip() and printable_ratio > 0.7:
                    pages.append(extracted.strip())
            if pages:
                return "\n\n".join(pages)
        except Exception:
            pass

        # Strategy 2: pdfminer from bytes
        try:
            from pdfminer.high_level import extract_text_to_fp
            from pdfminer.layout import LAParams
            output = _io.StringIO()
            extract_text_to_fp(_io.BytesIO(data), output, laparams=LAParams())
            text = output.getvalue()
            if text and text.strip():
                printable_ratio = sum(1 for c in text if c.isprintable()) / max(len(text), 1)
                if printable_ratio > 0.7:
                    return text.strip()
        except Exception:
            pass

        # Strategy 3: pymupdf from bytes
        try:
            import fitz  # type: ignore[import]
            doc = fitz.open(stream=data, filetype="pdf")
            pages = []
            for page in doc:
                text = page.get_text("text")
                if text and text.strip():
                    pages.append(text.strip())
            doc.close()
            if pages:
                return "\n\n".join(pages)
        except Exception:
            pass

        return f"[PDF text extraction failed for '{filename}'. The file may be scanned or image-based.]"


    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document

            doc = Document(str(path))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _read_csv(self, path: Path) -> str:
        lines = []
        with open(path, mode="r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if headers:
                for row in reader:
                    row_parts = [f"{h}: {v}" for h, v in zip(headers, row) if v]
                    lines.append(", ".join(row_parts))
        return "\n".join(lines) if lines else path.read_text(encoding="utf-8", errors="ignore")

    def _read_json(self, path: Path) -> str:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        if isinstance(data, list):
            return "\n\n".join(json.dumps(item) if isinstance(item, dict) else str(item) for item in data)
        return json.dumps(data, indent=2)
