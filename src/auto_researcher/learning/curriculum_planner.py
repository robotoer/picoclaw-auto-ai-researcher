"""ZPD-based curriculum planning for autonomous learning."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from auto_researcher.config import CurriculumConfig
from auto_researcher.models.memory import MetaMemoryEntry
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

TOPIC_ASSESSMENT_PROMPT = """\
Assess the prerequisite overlap between the system's current knowledge and this topic.

Current competencies (topic: competence 0-1):
{competencies}

Target topic: {topic}
Topic description: {description}

Respond with JSON:
{{
    "prerequisite_overlap": <float 0-1>,
    "prerequisites": ["<topic>"],
    "difficulty": <float 0-1>,
    "estimated_learning_effort": <float 0-1>
}}
"""


class TopicCandidate:
    """A candidate topic for the next learning step."""

    def __init__(
        self,
        topic: str,
        description: str = "",
        prerequisite_overlap: float = 0.5,
        field_momentum: float = 0.5,
        gap_density: float = 0.5,
        strategic_value: float = 0.5,
        difficulty: float = 0.5,
    ) -> None:
        self.topic = topic
        self.description = description
        self.prerequisite_overlap = prerequisite_overlap
        self.field_momentum = field_momentum
        self.gap_density = gap_density
        self.strategic_value = strategic_value
        self.difficulty = difficulty

    def zpd_score(self, min_overlap: float, max_overlap: float) -> float:
        """Score based on Zone of Proximal Development: prefer 40-80% overlap."""
        if self.prerequisite_overlap < min_overlap:
            return self.prerequisite_overlap / min_overlap * 0.5
        if self.prerequisite_overlap > max_overlap:
            overshoot = (self.prerequisite_overlap - max_overlap) / (1.0 - max_overlap)
            return max(0.0, 1.0 - overshoot)
        # In the sweet spot
        midpoint = (min_overlap + max_overlap) / 2
        distance = abs(self.prerequisite_overlap - midpoint) / (max_overlap - min_overlap) * 2
        return 1.0 - distance * 0.3


class CurriculumPlanner:
    """Plans learning curriculum using Zone of Proximal Development theory.

    State = knowledge graph (competencies)
    Action = choose topic
    Reward = knowledge gain
    """

    def __init__(self, llm: LLMClient, config: CurriculumConfig) -> None:
        self._llm = llm
        self._config = config
        self._competencies: dict[str, MetaMemoryEntry] = {}
        self._learning_history: list[tuple[str, float]] = []  # (topic, reward)

    def load_competencies(self, entries: list[MetaMemoryEntry]) -> None:
        """Load current knowledge state."""
        for entry in entries:
            self._competencies[entry.topic] = entry

    def get_competence(self, topic: str) -> float:
        entry = self._competencies.get(topic)
        return entry.competence_level if entry else 0.0

    async def select_next_topics(
        self,
        candidates: list[TopicCandidate],
        n: int = 3,
    ) -> list[TopicCandidate]:
        """Select the top-n topics to learn next based on ZPD and strategic value."""
        scored: list[tuple[float, TopicCandidate]] = []

        for candidate in candidates:
            # Assess prerequisite overlap if not already set
            if candidate.prerequisite_overlap == 0.5 and self._competencies:
                overlap = await self._assess_overlap(candidate)
                candidate.prerequisite_overlap = overlap

            zpd = candidate.zpd_score(self._config.zpd_min_overlap, self._config.zpd_max_overlap)
            combined = (
                zpd
                + self._config.field_momentum_weight * candidate.field_momentum
                + self._config.gap_density_weight * candidate.gap_density
                + self._config.strategic_value_weight * candidate.strategic_value
            )
            scored.append((combined, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [c for _, c in scored[:n]]

        logger.info(
            "curriculum_topics_selected",
            topics=[c.topic for c in selected],
            scores=[s for s, _ in scored[:n]],
        )
        return selected

    async def _assess_overlap(self, candidate: TopicCandidate) -> float:
        """Use LLM to assess prerequisite overlap with current knowledge."""
        competency_text = "\n".join(
            f"- {topic}: {entry.competence_level:.2f}"
            for topic, entry in sorted(self._competencies.items(), key=lambda x: x[1].competence_level, reverse=True)[:20]
        )
        prompt = TOPIC_ASSESSMENT_PROMPT.format(
            competencies=competency_text,
            topic=candidate.topic,
            description=candidate.description or candidate.topic,
        )
        try:
            result = await self._llm.generate_structured(
                prompt=prompt,
                system="You are an expert in AI/ML knowledge assessment.",
                temperature=0.3,
            )
            return max(0.0, min(1.0, float(result.get("prerequisite_overlap", 0.5))))
        except Exception:
            logger.exception("overlap_assessment_failed", topic=candidate.topic)
            return 0.5

    def record_learning_outcome(self, topic: str, knowledge_gain: float) -> None:
        """Record the outcome of a learning episode (RL reward signal)."""
        self._learning_history.append((topic, knowledge_gain))

        # Update competence level
        if topic in self._competencies:
            entry = self._competencies[topic]
            entry.competence_level = min(1.0, entry.competence_level + knowledge_gain * 0.3)
            entry.last_assessed = datetime.utcnow()
        else:
            self._competencies[topic] = MetaMemoryEntry(
                topic=topic,
                competence_level=min(1.0, knowledge_gain * 0.3),
            )

        logger.info("learning_recorded", topic=topic, gain=knowledge_gain)

    def get_knowledge_frontier(self, threshold: float = 0.3) -> list[str]:
        """Get topics at the frontier of current knowledge (partially learned)."""
        return [
            topic for topic, entry in self._competencies.items()
            if threshold <= entry.competence_level < 0.8
        ]

    def suggest_review_topics(self, decay_days: int = 30) -> list[str]:
        """Suggest topics that may need review based on time since last assessment."""
        now = datetime.utcnow()
        review_needed = []
        for topic, entry in self._competencies.items():
            days_since = (now - entry.last_assessed).days
            if days_since > decay_days and entry.competence_level > 0.3:
                review_needed.append(topic)
        return review_needed
