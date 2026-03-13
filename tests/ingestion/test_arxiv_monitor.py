"""Tests for ArxivMonitor."""

from __future__ import annotations

import pytest
import httpx

from auto_researcher.config import ArxivConfig
from auto_researcher.ingestion.arxiv_monitor import ArxivMonitor


SAMPLE_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>  Test Paper on  LLMs  </title>
    <summary>  This paper studies large language models and their capabilities.  </summary>
    <published>2024-01-15T00:00:00Z</published>
    <updated>2024-01-16T00:00:00Z</updated>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <arxiv:primary_category term="cs.AI"/>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00001v1"/>
    <arxiv:doi>10.1234/test</arxiv:doi>
    <arxiv:journal_ref>Test Journal 2024</arxiv:journal_ref>
    <arxiv:comment>10 pages, 5 figures</arxiv:comment>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Another Paper on RL</title>
    <summary>Reinforcement learning paper abstract.</summary>
    <published>2024-01-14T00:00:00Z</published>
    <author><name>Charlie Brown</name></author>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.00002v1"/>
  </entry>
</feed>
"""

EMPTY_FEED = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
</feed>
"""


@pytest.fixture
def config():
    return ArxivConfig(
        categories=["cs.AI", "cs.LG"],
        max_results_per_query=10,
    )


@pytest.fixture
def monitor(config):
    return ArxivMonitor(config)


class TestParseFeed:
    def test_parses_two_entries(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert len(papers) == 2

    def test_parses_arxiv_id(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].arxiv_id == "2401.00001v1"
        assert papers[1].arxiv_id == "2401.00002v1"

    def test_normalizes_title_whitespace(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].metadata.title == "Test Paper on LLMs"

    def test_normalizes_abstract_whitespace(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert "  " not in papers[0].metadata.abstract

    def test_parses_authors(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].metadata.authors == ["Alice Smith", "Bob Jones"]
        assert papers[1].metadata.authors == ["Charlie Brown"]

    def test_parses_categories(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert "cs.AI" in papers[0].metadata.categories
        assert "cs.CL" in papers[0].metadata.categories

    def test_parses_pdf_url(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].metadata.pdf_url == "http://arxiv.org/pdf/2401.00001v1"

    def test_parses_doi(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].metadata.doi == "10.1234/test"

    def test_parses_journal_ref(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].metadata.journal_ref == "Test Journal 2024"

    def test_parses_comment(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].metadata.comment == "10 pages, 5 figures"

    def test_parses_dates(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[0].metadata.published.year == 2024
        assert papers[0].metadata.updated is not None
        assert papers[0].metadata.updated.day == 16

    def test_missing_optional_fields(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        assert papers[1].metadata.doi is None
        assert papers[1].metadata.journal_ref is None
        assert papers[1].metadata.comment is None
        # Second entry has no <updated> element in the feed
        assert papers[1].metadata.updated is None

    def test_empty_feed(self, monitor):
        papers = monitor._parse_feed(EMPTY_FEED)
        assert papers == []

    def test_entry_without_id_skipped(self, monitor):
        feed = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>No ID Paper</title>
    <summary>Abstract</summary>
    <published>2024-01-15T00:00:00Z</published>
  </entry>
</feed>
"""
        papers = monitor._parse_feed(feed)
        assert papers == []


class TestDuplicateDetection:
    def test_mark_processed_filters_duplicates(self, monitor):
        papers = monitor._parse_feed(SAMPLE_FEED)
        monitor.mark_processed(["2401.00001v1"])
        # Simulate what fetch_recent_papers does
        new_papers = [p for p in papers if p.arxiv_id not in monitor._processed_ids]
        assert len(new_papers) == 1
        assert new_papers[0].arxiv_id == "2401.00002v1"

    def test_load_processed_ids(self, monitor):
        monitor.load_processed_ids({"2401.00001v1", "2401.00002v1"})
        assert "2401.00001v1" in monitor._processed_ids
        assert "2401.00002v1" in monitor._processed_ids


class TestFetchRecentPapers:
    @pytest.mark.asyncio
    async def test_fetch_deduplicates_across_categories(self, monitor, monkeypatch):
        """Papers appearing in multiple categories should be deduplicated."""
        call_count = 0

        async def mock_fetch_category(self_inner, category):
            nonlocal call_count
            call_count += 1
            return monitor._parse_feed(SAMPLE_FEED)

        monkeypatch.setattr(ArxivMonitor, "_fetch_category", mock_fetch_category)
        papers = await monitor.fetch_recent_papers()
        # Called for each category but results deduplicated
        assert call_count == 2
        assert len(papers) == 2  # not 4

    @pytest.mark.asyncio
    async def test_fetch_skips_processed_ids(self, monitor, monkeypatch):
        async def mock_fetch_category(self_inner, category):
            return monitor._parse_feed(SAMPLE_FEED)

        monkeypatch.setattr(ArxivMonitor, "_fetch_category", mock_fetch_category)
        monitor.mark_processed(["2401.00001v1"])
        papers = await monitor.fetch_recent_papers()
        assert all(p.arxiv_id != "2401.00001v1" for p in papers)

    @pytest.mark.asyncio
    async def test_fetch_handles_category_error(self, monitor, monkeypatch):
        """If one category fails, others should still return results."""
        call_count = 0

        async def mock_fetch_category(self_inner, category):
            nonlocal call_count
            call_count += 1
            if category == "cs.AI":
                raise httpx.HTTPError("Network error")
            return monitor._parse_feed(SAMPLE_FEED)

        monkeypatch.setattr(ArxivMonitor, "_fetch_category", mock_fetch_category)
        papers = await monitor.fetch_recent_papers()
        assert len(papers) == 2  # only from cs.LG

    @pytest.mark.asyncio
    async def test_fetch_category_makes_correct_request(self, config, monkeypatch):
        monitor = ArxivMonitor(config)
        captured_params = {}

        async def mock_get(self_inner, url, params=None):
            captured_params.update(params or {})

            class MockResponse:
                status_code = 200
                text = EMPTY_FEED

                def raise_for_status(self):
                    pass

            return MockResponse()

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        await monitor._fetch_category("cs.AI")
        assert captured_params["search_query"] == "cat:cs.AI"
        assert captured_params["max_results"] == 10
        assert captured_params["sortBy"] == "submittedDate"
