"""Research trend detection via topic modeling and time-series analysis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from auto_researcher.models import Paper
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TopicTrend:
    """A detected research topic with trend information."""

    topic_id: int
    keywords: list[str]
    representative_titles: list[str] = field(default_factory=list)
    paper_count: int = 0
    monthly_counts: dict[str, int] = field(default_factory=dict)
    growth_rate: float = 0.0  # positive = growing, negative = declining
    acceleration: float = 0.0  # second derivative
    hype_position: str = "unknown"  # "emerging", "peak", "trough", "plateau", "declining"


@dataclass
class TrendReport:
    """Summary of detected research trends."""

    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    trending_topics: list[TopicTrend] = field(default_factory=list)
    declining_topics: list[TopicTrend] = field(default_factory=list)
    emerging_combinations: list[dict[str, list[str]]] = field(default_factory=list)
    total_papers_analyzed: int = 0


class TrendDetector:
    """Detects research trends from paper metadata using topic modeling."""

    def __init__(self, min_topic_size: int = 5, n_topics: int | None = None) -> None:
        self._min_topic_size = min_topic_size
        self._n_topics = n_topics
        self._topic_model = None
        self._paper_history: list[Paper] = []
        self._topic_time_series: dict[int, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    async def analyze(self, papers: list[Paper]) -> TrendReport:
        """Analyze a batch of papers for trends.

        Accumulates papers over time for better topic modeling.
        """
        self._paper_history.extend(papers)
        logger.info("trend_analysis_start", total_papers=len(self._paper_history))

        topics = self._fit_topics()
        self._update_time_series(papers, topics)

        trending = []
        declining = []
        for topic in topics:
            self._compute_trend_metrics(topic)
            self._estimate_hype_position(topic)
            if topic.growth_rate > 0.1 and topic.paper_count >= self._min_topic_size:
                trending.append(topic)
            elif topic.growth_rate < -0.1 and topic.paper_count >= self._min_topic_size:
                declining.append(topic)

        trending.sort(key=lambda t: t.growth_rate, reverse=True)
        declining.sort(key=lambda t: t.growth_rate)

        emerging_combos = self._detect_emerging_combinations(papers)

        report = TrendReport(
            trending_topics=trending,
            declining_topics=declining,
            emerging_combinations=emerging_combos,
            total_papers_analyzed=len(self._paper_history),
        )
        logger.info(
            "trend_analysis_complete",
            trending=len(trending),
            declining=len(declining),
            emerging_combos=len(emerging_combos),
        )
        return report

    def _fit_topics(self) -> list[TopicTrend]:
        """Fit topic model on accumulated paper abstracts."""
        texts = [p.metadata.abstract for p in self._paper_history if p.metadata.abstract]
        if len(texts) < self._min_topic_size:
            return []

        try:
            return self._fit_with_sklearn(texts)
        except Exception:
            logger.exception("topic_modeling_failed")
            return []

    def _fit_with_sklearn(self, texts: list[str]) -> list[TopicTrend]:
        """Topic modeling using sklearn NMF + TF-IDF as a lightweight fallback."""
        from sklearn.decomposition import NMF
        from sklearn.feature_extraction.text import TfidfVectorizer

        n_topics = self._n_topics or max(5, len(texts) // 20)
        n_topics = min(n_topics, len(texts) - 1)

        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            max_df=0.95,
            min_df=2,
        )
        tfidf = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        model = NMF(n_components=n_topics, random_state=42, max_iter=300)
        doc_topics = model.fit_transform(tfidf)

        topics: list[TopicTrend] = []
        for topic_idx in range(n_topics):
            top_word_indices = model.components_[topic_idx].argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_word_indices]

            # Find representative papers for this topic
            topic_scores = doc_topics[:, topic_idx]
            top_doc_indices = topic_scores.argsort()[-3:][::-1]
            rep_titles = [
                self._paper_history[i].metadata.title
                for i in top_doc_indices
                if topic_scores[i] > 0.01
            ]

            # Count papers assigned to this topic
            assigned = sum(1 for scores in doc_topics if scores[topic_idx] == scores.max() and scores.max() > 0.01)

            topics.append(
                TopicTrend(
                    topic_id=topic_idx,
                    keywords=keywords,
                    representative_titles=rep_titles,
                    paper_count=assigned,
                )
            )

        # Store doc-topic assignments for time series
        self._last_doc_topics = doc_topics
        return topics

    def _update_time_series(
        self, new_papers: list[Paper], topics: list[TopicTrend]
    ) -> None:
        """Update topic time series with new papers."""
        if not hasattr(self, "_last_doc_topics") or self._last_doc_topics is None:
            return

        # Only process the newly added papers (last N in history)
        offset = len(self._paper_history) - len(new_papers)
        for i, paper in enumerate(new_papers):
            doc_idx = offset + i
            if doc_idx >= len(self._last_doc_topics):
                break
            scores = self._last_doc_topics[doc_idx]
            dominant_topic = int(scores.argmax())
            month_key = paper.metadata.published.strftime("%Y-%m")
            self._topic_time_series[dominant_topic][month_key] += 1

        # Update topic objects with monthly counts
        for topic in topics:
            topic.monthly_counts = dict(self._topic_time_series.get(topic.topic_id, {}))

    def _compute_trend_metrics(self, topic: TopicTrend) -> None:
        """Compute growth rate and acceleration for a topic."""
        counts = topic.monthly_counts
        if len(counts) < 2:
            topic.growth_rate = 0.0
            topic.acceleration = 0.0
            return

        sorted_months = sorted(counts.keys())
        values = [counts[m] for m in sorted_months]

        # Simple linear regression for growth rate
        n = len(values)
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator > 0 and y_mean > 0:
            slope = numerator / denominator
            topic.growth_rate = slope / y_mean  # normalize by mean
        else:
            topic.growth_rate = 0.0

        # Acceleration: change in growth rate (second derivative)
        if n >= 3:
            first_half = values[: n // 2]
            second_half = values[n // 2 :]
            rate_1 = (first_half[-1] - first_half[0]) / max(len(first_half), 1)
            rate_2 = (second_half[-1] - second_half[0]) / max(len(second_half), 1)
            topic.acceleration = rate_2 - rate_1
        else:
            topic.acceleration = 0.0

    def _estimate_hype_position(self, topic: TopicTrend) -> None:
        """Estimate where a topic is in the hype cycle."""
        gr = topic.growth_rate
        acc = topic.acceleration

        if topic.paper_count < self._min_topic_size:
            topic.hype_position = "unknown"
        elif gr > 0.3 and acc > 0:
            topic.hype_position = "emerging"
        elif gr > 0.1 and acc < 0:
            topic.hype_position = "peak"
        elif gr < -0.1 and acc < 0:
            topic.hype_position = "declining"
        elif gr < -0.1 and acc > 0:
            topic.hype_position = "trough"
        elif abs(gr) <= 0.1:
            topic.hype_position = "plateau"
        else:
            topic.hype_position = "unknown"

    def _detect_emerging_combinations(
        self, recent_papers: list[Paper]
    ) -> list[dict[str, list[str]]]:
        """Detect new co-occurrences of categories that haven't appeared together before."""
        # Build historical co-occurrence set (excluding recent)
        historical = self._paper_history[: -len(recent_papers)] if len(recent_papers) < len(self._paper_history) else []
        historical_pairs: set[tuple[str, str]] = set()
        for paper in historical:
            cats = sorted(paper.metadata.categories)
            for i in range(len(cats)):
                for j in range(i + 1, len(cats)):
                    historical_pairs.add((cats[i], cats[j]))

        # Find new combinations in recent papers
        new_combos: list[dict[str, list[str]]] = []
        seen: set[tuple[str, str]] = set()
        for paper in recent_papers:
            cats = sorted(paper.metadata.categories)
            for i in range(len(cats)):
                for j in range(i + 1, len(cats)):
                    pair = (cats[i], cats[j])
                    if pair not in historical_pairs and pair not in seen:
                        seen.add(pair)
                        new_combos.append(
                            {
                                "categories": list(pair),
                                "example_titles": [paper.metadata.title],
                            }
                        )

        return new_combos
