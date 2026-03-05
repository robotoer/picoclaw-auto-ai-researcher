"""ZPD-based curriculum planning for autonomous learning."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
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
            entry.last_assessed = datetime.now(UTC)
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
        now = datetime.now(UTC)
        review_needed = []
        for topic, entry in self._competencies.items():
            days_since = (now - entry.last_assessed).days
            if days_since > decay_days and entry.competence_level > 0.3:
                review_needed.append(topic)
        return review_needed


class ThompsonSampler:
    """Thompson Sampling for topic selection using Beta posteriors."""

    def __init__(self) -> None:
        self._alphas: dict[str, float] = {}
        self._betas: dict[str, float] = {}

    def update(self, topic: str, reward: float) -> None:
        """Update the posterior for a topic based on observed reward (0-1)."""
        if topic not in self._alphas:
            self._alphas[topic] = 1.0
            self._betas[topic] = 1.0
        self._alphas[topic] += reward
        self._betas[topic] += 1.0 - reward

    def sample(self, topics: list[str]) -> str:
        """Sample from posterior and return the topic with highest sampled value."""
        best_topic = topics[0]
        best_val = -1.0
        for topic in topics:
            alpha = self._alphas.get(topic, 1.0)
            beta = self._betas.get(topic, 1.0)
            val = random.betavariate(alpha, beta)
            if val > best_val:
                best_val = val
                best_topic = topic
        return best_topic

    def get_posterior(self, topic: str) -> tuple[float, float]:
        """Return (alpha, beta) for a topic's posterior."""
        return (self._alphas.get(topic, 1.0), self._betas.get(topic, 1.0))


class UCBSelector:
    """Upper Confidence Bound topic selection."""

    def __init__(self, exploration_weight: float = 2.0) -> None:
        self._counts: dict[str, int] = {}
        self._total_rewards: dict[str, float] = {}
        self._total_selections: int = 0
        self._exploration_weight = exploration_weight

    def update(self, topic: str, reward: float) -> None:
        self._counts[topic] = self._counts.get(topic, 0) + 1
        self._total_rewards[topic] = self._total_rewards.get(topic, 0.0) + reward
        self._total_selections += 1

    def select(self, topics: list[str]) -> str:
        """Select topic with highest UCB score."""
        # First try any unexplored topic
        for topic in topics:
            if topic not in self._counts:
                return topic

        best_topic = topics[0]
        best_ucb = -1.0
        for topic in topics:
            count = self._counts[topic]
            mean_reward = self._total_rewards[topic] / count
            exploration = self._exploration_weight * math.sqrt(
                math.log(self._total_selections) / count
            )
            ucb = mean_reward + exploration
            if ucb > best_ucb:
                best_ucb = ucb
                best_topic = topic
        return best_topic

    def get_ucb_score(self, topic: str) -> float | None:
        """Return the current UCB score for a topic, or None if unexplored."""
        if topic not in self._counts or self._total_selections == 0:
            return None
        count = self._counts[topic]
        mean_reward = self._total_rewards[topic] / count
        exploration = self._exploration_weight * math.sqrt(
            math.log(self._total_selections) / count
        )
        return mean_reward + exploration


class MetaRLEpisode:
    """Record of a single meta-RL episode."""

    def __init__(self, topics: list[str], rewards: list[float], eval_score: float) -> None:
        self.topics = topics
        self.rewards = rewards
        self.eval_score = eval_score


class MetaRLTrainer:
    """Meta-RL training loop for curriculum policy optimization.

    Episodes: learning phase (agent studies curriculum) -> evaluation phase (tested on held-out tasks).
    The curriculum policy is updated to maximize evaluation performance.
    """

    def __init__(
        self,
        planner: CurriculumPlanner,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
    ) -> None:
        self._planner = planner
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._episodes: list[MetaRLEpisode] = []
        self._policy_params: dict[str, float] = {
            "field_momentum_weight": planner._config.field_momentum_weight,
            "gap_density_weight": planner._config.gap_density_weight,
            "strategic_value_weight": planner._config.strategic_value_weight,
        }
        self._thompson = ThompsonSampler()
        self._ucb = UCBSelector()

    async def run_episode(
        self,
        candidates: list[TopicCandidate],
        learn_fn: Any,
        eval_fn: Any,
        steps: int = 3,
    ) -> MetaRLEpisode:
        """Run one meta-RL episode.

        Args:
            candidates: Available topics to learn from.
            learn_fn: async callable(topic) -> reward (float 0-1).
            eval_fn: async callable() -> eval_score (float 0-1).
            steps: Number of learning steps per episode.
        """
        selected = await self._planner.select_next_topics(candidates, n=steps)
        rewards: list[float] = []
        topics_studied: list[str] = []

        for candidate in selected:
            reward = await learn_fn(candidate.topic)
            rewards.append(reward)
            topics_studied.append(candidate.topic)
            self._planner.record_learning_outcome(candidate.topic, reward)
            self._thompson.update(candidate.topic, reward)
            self._ucb.update(candidate.topic, reward)

        eval_score = await eval_fn()
        episode = MetaRLEpisode(topics_studied, rewards, eval_score)
        self._episodes.append(episode)

        self._update_policy(episode)

        logger.info(
            "meta_rl_episode_complete",
            episode_num=len(self._episodes),
            topics=topics_studied,
            eval_score=eval_score,
        )
        return episode

    def _update_policy(self, episode: MetaRLEpisode) -> None:
        """Update policy parameters based on episode outcome."""
        if len(self._episodes) < 2:
            return

        prev = self._episodes[-2]
        improvement = episode.eval_score - prev.eval_score

        for key in self._policy_params:
            grad = improvement * self._learning_rate
            self._policy_params[key] = max(0.01, min(1.0, self._policy_params[key] + grad))

        self._planner._config.field_momentum_weight = self._policy_params["field_momentum_weight"]
        self._planner._config.gap_density_weight = self._policy_params["gap_density_weight"]
        self._planner._config.strategic_value_weight = self._policy_params["strategic_value_weight"]

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def policy_params(self) -> dict[str, float]:
        return dict(self._policy_params)

    @property
    def training_history(self) -> list[dict[str, Any]]:
        return [
            {
                "episode": i + 1,
                "topics": ep.topics,
                "rewards": ep.rewards,
                "eval_score": ep.eval_score,
            }
            for i, ep in enumerate(self._episodes)
        ]

    @property
    def thompson_sampler(self) -> ThompsonSampler:
        return self._thompson

    @property
    def ucb_selector(self) -> UCBSelector:
        return self._ucb
