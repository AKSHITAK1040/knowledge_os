from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import quote_plus
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from ..types import Citation, SourceKind
from ..utils import summarize_text, stable_hash


@dataclass(slots=True)
class WebEvidence:
    title: str
    url: str
    excerpt: str
    source_kind: SourceKind = SourceKind.WEB

    def to_citation(self) -> Citation:
        return Citation(
            source_id=stable_hash(self.url),
            title=self.title,
            url=self.url,
            excerpt=self.excerpt,
            source_kind=self.source_kind,
        )


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[WebEvidence] = []
        self._current_url: str | None = None
        self._current_title: str = ""
        self._current_excerpt: str = ""
        self._capture_title = False
        self._capture_excerpt = False

    @staticmethod
    def _clean_url(raw_url: str | None) -> str:
        if not raw_url:
            return ""
        if "uddg=" in raw_url:
            from urllib.parse import parse_qs, unquote, urlparse

            parsed = urlparse(raw_url if raw_url.startswith("http") else f"https:{raw_url}")
            params = parse_qs(parsed.query)
            if "uddg" in params and params["uddg"]:
                return unquote(params["uddg"][0])
        if raw_url.startswith("//"):
            return f"https:{raw_url}"
        return raw_url

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value for key, value in attrs}
        if tag == "a" and attr_map.get("class", "").startswith("result__a"):
            self._current_url = self._clean_url(attr_map.get("href"))
            self._capture_title = True
        if tag == "a" and "result__snippet" in attr_map.get("class", ""):
            self._capture_excerpt = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._capture_title:
            self._capture_title = False
        if tag == "a" and self._capture_excerpt:
            self._capture_excerpt = False
            if self._current_url and self._current_title:
                self.links.append(
                    WebEvidence(
                        title=self._current_title.strip(),
                        url=self._current_url,
                        excerpt=summarize_text(self._current_excerpt.strip(), 40),
                    )
                )
                self._current_title = ""
                self._current_excerpt = ""

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._current_title += data
        if self._capture_excerpt:
            self._current_excerpt += data


class DuckDuckGoSearchClient:
    """Production DuckDuckGo search client using form POST and desktop headers."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"

    def search(self, query: str, max_results: int = 5) -> list[WebEvidence]:
        clean_query = query.strip()
        if not clean_query:
            return []

        from urllib.parse import urlencode
        from urllib.request import Request

        post_data = urlencode({"q": clean_query, "b": ""}).encode("utf-8")
        req = Request(
            "https://html.duckduckgo.com/html/",
            data=post_data,
            headers={
                "User-Agent": self.USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://html.duckduckgo.com/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

        try:
            with urlopen(req, timeout=15) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except Exception:
            return []

        parser = _DuckDuckGoParser()
        parser.feed(html)
        return parser.links[:max_results]


class AcademicSearchClient:
    """Production arXiv academic search client with HTTPS and error resilience."""

    USER_AGENT = "KnowledgeOS/1.0 (academic-research-agent; mailto:admin@knowledgeos.ai)"

    def search(self, query: str, max_results: int = 5) -> list[WebEvidence]:
        clean_query = query.strip()
        if not clean_query:
            return []

        from urllib.request import Request

        url = f"https://export.arxiv.org/api/query?search_query=all:{quote_plus(clean_query)}&start=0&max_results={max_results}"
        req = Request(url, headers={"User-Agent": self.USER_AGENT})
        try:
            with urlopen(req, timeout=15) as response:
                xml_text = response.read().decode("utf-8", errors="ignore")
            return self._parse_arxiv(xml_text)
        except Exception:
            return []

    def _parse_arxiv(self, xml_text: str) -> list[WebEvidence]:
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return []

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        evidences: list[WebEvidence] = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            # Clean up newlines in title
            title = " ".join(title.split())
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            summary = " ".join(summary.split())
            url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("rel") == "alternate":
                    url = link.attrib.get("href", "")
                    break
            if title and url:
                evidences.append(WebEvidence(title=title, url=url, excerpt=summarize_text(summary, 40), source_kind=SourceKind.ACADEMIC))
        return evidences

