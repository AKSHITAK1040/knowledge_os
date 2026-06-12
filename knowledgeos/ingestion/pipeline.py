from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen
import io
import json

from ..retrieval.hybrid import HybridRetriever
from ..types import KnowledgeChunk, SourceKind
from ..utils import chunk_text, merge_metadata, stable_hash


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


@dataclass(slots=True)
class IngestionReport:
    source: str
    chunks_created: int
    metadata: dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    def ingest_text(self, text: str, *, title: str, source_kind: SourceKind = SourceKind.INTERNAL, metadata: dict[str, Any] | None = None) -> IngestionReport:
        metadata = metadata or {}
        chunk_count = 0
        for index, piece in enumerate(chunk_text(text)):
            self.retriever.add_chunk(piece, title=title, source_kind=source_kind, metadata=merge_metadata(metadata, {"chunk_index": index}))
            chunk_count += 1
        return IngestionReport(source=title, chunks_created=chunk_count, metadata=metadata)

    def ingest_file(self, path: str | Path) -> IngestionReport:
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            text = self._read_docx(file_path)
        elif suffix == ".pdf":
            text = self._read_pdf(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        return self.ingest_text(text, title=file_path.name, metadata={"path": str(file_path), "suffix": suffix})

    def ingest_url(self, url: str) -> IngestionReport:
        with urlopen(url, timeout=20) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
        text = self._extract_text_from_web(raw, content_type)
        parsed = urlparse(url)
        return self.ingest_text(text, title=parsed.netloc + parsed.path, source_kind=SourceKind.WEB, metadata={"url": url, "content_type": content_type})

    def ingest_payload(self, payload: dict[str, Any]) -> IngestionReport:
        if "text" in payload:
            return self.ingest_text(payload["text"], title=payload.get("title", "untitled"), metadata=payload.get("metadata", {}))
        if "path" in payload:
            return self.ingest_file(payload["path"])
        if "url" in payload:
            return self.ingest_url(payload["url"])
        raise ValueError("Payload must contain text, path, or url")

    def _extract_text_from_web(self, raw: bytes, content_type: str) -> str:
        if "html" in content_type.lower():
            parser = _HTMLTextExtractor()
            parser.feed(raw.decode("utf-8", errors="ignore"))
            return " ".join(parser.parts)
        return raw.decode("utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")

    def _read_docx(self, path: Path) -> str:
        try:
            from docx import Document

            document = Document(str(path))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")

