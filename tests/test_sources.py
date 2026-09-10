from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from knowledgeos.research.sources import (
    AcademicSearchClient,
    DuckDuckGoSearchClient,
    WebEvidence,
    _DuckDuckGoParser,
)
from knowledgeos.types import SourceKind


SAMPLE_DDG_HTML = """
<!DOCTYPE html>
<html>
<body>
    <div class="result results_links results_links_deep web-result">
        <div class="links_main links_deep result__body">
            <h2 class="result__title">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2F&rut=1">Python Programming</a>
            </h2>
            <a class="result__snippet" href="#">Python is a high-level general-purpose programming language.</a>
        </div>
    </div>
</body>
</html>
"""

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>
      Attention Is All You Need
    </title>
    <summary>
      The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.
    </summary>
    <link rel="alternate" href="https://arxiv.org/abs/1706.03762" type="text/html"/>
  </entry>
</feed>
"""


class SourcesTests(unittest.TestCase):
    def test_ddg_parser_extracts_links_and_unquotes_uddg(self) -> None:
        parser = _DuckDuckGoParser()
        parser.feed(SAMPLE_DDG_HTML)
        self.assertEqual(len(parser.links), 1)
        self.assertEqual(parser.links[0].title, "Python Programming")
        self.assertEqual(parser.links[0].url, "https://www.python.org/")
        self.assertIn("Python is a high-level", parser.links[0].excerpt)

    def test_arxiv_parser_extracts_entries(self) -> None:
        client = AcademicSearchClient()
        evidences = client._parse_arxiv(SAMPLE_ARXIV_XML)
        self.assertEqual(len(evidences), 1)
        self.assertEqual(evidences[0].title, "Attention Is All You Need")
        self.assertEqual(evidences[0].url, "https://arxiv.org/abs/1706.03762")
        self.assertEqual(evidences[0].source_kind, SourceKind.ACADEMIC)

    @patch("knowledgeos.research.sources.urlopen")
    def test_ddg_search_client_uses_post_and_headers(self, mock_urlopen) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = SAMPLE_DDG_HTML.encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        client = DuckDuckGoSearchClient()
        results = client.search("python programming", max_results=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://www.python.org/")
        # Verify the request passed to urlopen was a POST with User-Agent
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_method(), "POST")
        self.assertIn("User-agent", req.headers)

    def test_empty_query_returns_empty_list(self) -> None:
        client = DuckDuckGoSearchClient()
        self.assertEqual(client.search(""), [])
        acad = AcademicSearchClient()
        self.assertEqual(acad.search("   "), [])


if __name__ == "__main__":
    unittest.main()
