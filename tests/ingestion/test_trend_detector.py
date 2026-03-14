"""Tests for TrendDetector."""

from __future__ import annotations

from datetime import datetime

import pytest

from auto_researcher.ingestion.trend_detector import (
    TopicTrend,
    TrendDetector,
    TrendReport,
)
from auto_researcher.models import Paper, PaperMetadata


def make_paper(
    arxiv_id: str = "2401.00001",
    title: str = "Test Paper",
    abstract: str = "Machine learning study",
    categories: list[str] | None = None,
    published: datetime | None = None,
) -> Paper:
    return Paper(
        metadata=PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            authors=["Author"],
            abstract=abstract,
            categories=categories or ["cs.AI"],
            published=published or datetime(2024, 1, 15),
        ),
    )


class TestComputeTrendMetrics:
    def test_growing_topic(self):
        td = TrendDetector()
        topic = TopicTrend(
            topic_id=0,
            keywords=["ml"],
            paper_count=10,
            monthly_counts={"2024-01": 2, "2024-02": 4, "2024-03": 6, "2024-04": 8},
        )
        td._compute_trend_metrics(topic)
        assert topic.growth_rate > 0

    def test_declining_topic(self):
        td = TrendDetector()
        topic = TopicTrend(
            topic_id=0,
            keywords=["old"],
            paper_count=10,
            monthly_counts={"2024-01": 8, "2024-02": 6, "2024-03": 4, "2024-04": 2},
        )
        td._compute_trend_metrics(topic)
        assert topic.growth_rate < 0

    def test_single_month_zero_growth(self):
        td = TrendDetector()
        topic = TopicTrend(
            topic_id=0,
            keywords=["x"],
            paper_count=5,
            monthly_counts={"2024-01": 5},
        )
        td._compute_trend_metrics(topic)
        assert topic.growth_rate == 0.0
        assert topic.acceleration == 0.0

    def test_two_months_no_acceleration(self):
        td = TrendDetector()
        topic = TopicTrend(
            topic_id=0,
            keywords=["x"],
            paper_count=5,
            monthly_counts={"2024-01": 2, "2024-02": 4},
        )
        td._compute_trend_metrics(topic)
        assert topic.acceleration == 0.0

    def test_acceleration_computed_three_months(self):
        td = TrendDetector()
        topic = TopicTrend(
            topic_id=0,
            keywords=["x"],
            paper_count=10,
            monthly_counts={"2024-01": 1, "2024-02": 2, "2024-03": 5},
        )
        td._compute_trend_metrics(topic)
        # acceleration should be positive (accelerating growth)
        assert topic.acceleration > 0


class TestEstimateHypePosition:
    def test_emerging(self):
        td = TrendDetector(min_topic_size=3)
        topic = TopicTrend(topic_id=0, keywords=[], paper_count=5, growth_rate=0.5, acceleration=0.1)
        td._estimate_hype_position(topic)
        assert topic.hype_position == "emerging"

    def test_peak(self):
        td = TrendDetector(min_topic_size=3)
        topic = TopicTrend(topic_id=0, keywords=[], paper_count=5, growth_rate=0.2, acceleration=-0.1)
        td._estimate_hype_position(topic)
        assert topic.hype_position == "peak"

    def test_declining(self):
        td = TrendDetector(min_topic_size=3)
        topic = TopicTrend(topic_id=0, keywords=[], paper_count=5, growth_rate=-0.2, acceleration=-0.1)
        td._estimate_hype_position(topic)
        assert topic.hype_position == "declining"

    def test_trough(self):
        td = TrendDetector(min_topic_size=3)
        topic = TopicTrend(topic_id=0, keywords=[], paper_count=5, growth_rate=-0.2, acceleration=0.1)
        td._estimate_hype_position(topic)
        assert topic.hype_position == "trough"

    def test_plateau(self):
        td = TrendDetector(min_topic_size=3)
        topic = TopicTrend(topic_id=0, keywords=[], paper_count=5, growth_rate=0.05, acceleration=0.0)
        td._estimate_hype_position(topic)
        assert topic.hype_position == "plateau"

    def test_unknown_small_topic(self):
        td = TrendDetector(min_topic_size=10)
        topic = TopicTrend(topic_id=0, keywords=[], paper_count=3, growth_rate=0.5, acceleration=0.1)
        td._estimate_hype_position(topic)
        assert topic.hype_position == "unknown"


class TestDetectEmergingCombinations:
    def test_new_category_combination(self):
        td = TrendDetector()
        # Add historical papers first
        historical = [make_paper(arxiv_id="old1", categories=["cs.AI", "cs.LG"])]
        td._paper_history.extend(historical)

        recent = [make_paper(arxiv_id="new1", categories=["cs.AI", "cs.CL"])]
        td._paper_history.extend(recent)

        combos = td._detect_emerging_combinations(recent)
        assert len(combos) == 1
        assert set(combos[0]["categories"]) == {"cs.AI", "cs.CL"}

    def test_existing_combination_not_flagged(self):
        td = TrendDetector()
        historical = [make_paper(arxiv_id="old1", categories=["cs.AI", "cs.CL"])]
        td._paper_history.extend(historical)

        recent = [make_paper(arxiv_id="new1", categories=["cs.AI", "cs.CL"])]
        td._paper_history.extend(recent)

        combos = td._detect_emerging_combinations(recent)
        assert len(combos) == 0

    def test_single_category_no_combo(self):
        td = TrendDetector()
        recent = [make_paper(arxiv_id="new1", categories=["cs.AI"])]
        td._paper_history.extend(recent)
        combos = td._detect_emerging_combinations(recent)
        assert len(combos) == 0


class TestAnalyze:
    @pytest.mark.asyncio
    async def test_too_few_papers_returns_empty(self):
        td = TrendDetector(min_topic_size=5)
        papers = [make_paper(arxiv_id=f"p{i}") for i in range(3)]
        report = await td.analyze(papers)
        assert isinstance(report, TrendReport)
        assert report.trending_topics == []
        assert report.declining_topics == []

    @pytest.mark.asyncio
    async def test_accumulates_paper_history(self):
        td = TrendDetector(min_topic_size=100)  # high threshold to skip modeling
        batch1 = [make_paper(arxiv_id=f"a{i}") for i in range(3)]
        batch2 = [make_paper(arxiv_id=f"b{i}") for i in range(3)]
        await td.analyze(batch1)
        await td.analyze(batch2)
        assert len(td._paper_history) == 6

    @pytest.mark.asyncio
    async def test_total_papers_analyzed_in_report(self):
        td = TrendDetector(min_topic_size=100)
        papers = [make_paper(arxiv_id=f"p{i}") for i in range(5)]
        report = await td.analyze(papers)
        assert report.total_papers_analyzed == 5
