"""Interest-Weighted Policy Gradient reward computation."""

from __future__ import annotations

import math

from auto_researcher.config import IWPGWeights
from auto_researcher.models.reward import IWPGReward
from auto_researcher.models.research_thread import ResearchThread
from auto_researcher.utils.llm import LLMClient
from auto_researcher.utils.logging import get_logger

logger = get_logger(__name__)

UTILITY_PROMPT = """\
Estimate the future adoption and practical utility of this research on a 0-1 scale.
Consider: How likely is this to be used by practitioners within 1 year? 5 years?

Research: {title}
Content: {content}

Respond with JSON: {{"utility": <float 0-1>, "reasoning": "<string>"}}
"""


class IWPGScorer:
    """Computes Interest-Weighted Policy Gradient rewards for research outputs."""

    def __init__(self, llm: LLMClient, weights: IWPGWeights) -> None:
        self._llm = llm
        self._weights = weights
        self._weight_history: list[IWPGWeights] = [weights]
        self._literature_embeddings: list[list[float]] = []
        self._reward_history: list[tuple[IWPGReward, float]] = []

    def load_literature_embeddings(self, embeddings: list[list[float]]) -> None:
        """Load existing literature embeddings for novelty computation."""
        self._literature_embeddings = embeddings

    async def compute_reward(
        self,
        thread: ResearchThread,
        embedding: list[float] | None = None,
    ) -> IWPGReward:
        """Compute the full IWPG reward for a research thread."""
        novelty = self._compute_novelty(embedding)
        surprise = self._compute_surprise(thread)
        utility = await self._compute_utility(thread)
        reproducibility = self._compute_reproducibility(thread)
        redundancy = self._compute_redundancy(embedding)
        complexity_cost = self._compute_complexity_cost(thread)

        reward = IWPGReward(
            novelty=novelty,
            surprise=surprise,
            utility=utility,
            reproducibility=reproducibility,
            redundancy=redundancy,
            complexity_cost=complexity_cost,
        )

        total = reward.total(
            alpha=self._weights.novelty,
            beta=self._weights.surprise,
            gamma=self._weights.utility,
            delta=self._weights.reproducibility,
            epsilon=self._weights.redundancy_penalty,
            zeta=self._weights.complexity_cost,
        )
        self._reward_history.append((reward, total))

        logger.info("iwpg_computed", thread_id=thread.id, total=total)
        return reward

    def total_reward(self, reward: IWPGReward) -> float:
        return reward.total(
            alpha=self._weights.novelty,
            beta=self._weights.surprise,
            gamma=self._weights.utility,
            delta=self._weights.reproducibility,
            epsilon=self._weights.redundancy_penalty,
            zeta=self._weights.complexity_cost,
        )

    def _compute_novelty(self, embedding: list[float] | None) -> float:
        """Novelty = 1 - cosine similarity to k-NN in literature."""
        if embedding is None or not self._literature_embeddings:
            return 0.5

        k = min(5, len(self._literature_embeddings))
        similarities = [
            self._cosine_similarity(embedding, lit_emb)
            for lit_emb in self._literature_embeddings
        ]
        similarities.sort(reverse=True)
        top_k_sim = sum(similarities[:k]) / k
        return max(0.0, min(1.0, 1.0 - top_k_sim))

    def _compute_surprise(self, thread: ResearchThread) -> float:
        """Surprise based on how much results deviated from predictions."""
        if not thread.result_ids:
            return 0.5
        # Use number of results and revision count as proxy for unexpected findings
        result_factor = min(1.0, len(thread.result_ids) * 0.3)
        revision_factor = min(1.0, thread.revision_count * 0.2)
        return min(1.0, 0.3 + result_factor * 0.4 + revision_factor * 0.3)

    async def _compute_utility(self, thread: ResearchThread) -> float:
        """Estimate future-discounted adoption using LLM."""
        content = thread.draft_sections.get("abstract", thread.title)
        try:
            result = await self._llm.generate_structured(
                prompt=UTILITY_PROMPT.format(title=thread.title, content=content[:3000]),
                system="You are a research impact analyst. Return only valid JSON.",
                temperature=0.3,
            )
            return max(0.0, min(1.0, float(result.get("utility", 0.5))))
        except Exception:
            logger.exception("utility_estimation_failed")
            return 0.5

    def _compute_reproducibility(self, thread: ResearchThread) -> float:
        """Estimate reproducibility from methodology completeness."""
        score = 0.3  # base
        if "methodology" in thread.draft_sections:
            score += 0.3
        if thread.experiment_ids:
            score += 0.2
        if thread.result_ids:
            score += 0.2
        return min(1.0, score)

    def _compute_redundancy(self, embedding: list[float] | None) -> float:
        """Redundancy penalty: high if too similar to existing work."""
        if embedding is None or not self._literature_embeddings:
            return 0.0
        max_sim = max(
            self._cosine_similarity(embedding, lit_emb)
            for lit_emb in self._literature_embeddings
        )
        # Penalize if max similarity > 0.9
        if max_sim > 0.9:
            return min(1.0, (max_sim - 0.9) * 10.0)
        return 0.0

    def _compute_complexity_cost(self, thread: ResearchThread) -> float:
        """Complexity cost based on compute usage relative to budget."""
        if thread.compute_budget <= 0:
            return 1.0
        ratio = thread.compute_used / thread.compute_budget
        return min(1.0, ratio)

    def update_weights(self, new_weights: IWPGWeights) -> None:
        """Update IWPG weights (meta-RL outer loop)."""
        self._weights = new_weights
        self._weight_history.append(new_weights)
        logger.info("iwpg_weights_updated")

    def meta_rl_weight_update(self, feedback_scores: list[float]) -> IWPGWeights:
        """Meta-RL weight learning: adjust weights based on observed outcomes.

        Uses simple gradient-free optimization: increase weights for dimensions
        correlated with high feedback, decrease for low feedback.
        """
        if len(self._reward_history) < 3 or len(feedback_scores) < 3:
            return self._weights

        # Use the most recent entries matching feedback_scores length
        recent = self._reward_history[-len(feedback_scores):]
        dim_names = ["novelty", "surprise", "utility", "reproducibility", "redundancy_penalty", "complexity_cost"]
        dim_reward_keys = ["novelty", "surprise", "utility", "reproducibility", "redundancy", "complexity_cost"]

        adjustments = {}
        for dim_name, reward_key in zip(dim_names, dim_reward_keys):
            dim_values = [getattr(r[0], reward_key) for r in recent]
            if not dim_values:
                continue
            # Correlation between dimension value and feedback
            n = len(dim_values)
            mean_d = sum(dim_values) / n
            mean_f = sum(feedback_scores[:n]) / n
            cov = sum((d - mean_d) * (f - mean_f) for d, f in zip(dim_values, feedback_scores[:n])) / n
            var_d = sum((d - mean_d) ** 2 for d in dim_values) / n
            var_f = sum((f - mean_f) ** 2 for f in feedback_scores[:n]) / n
            denom = math.sqrt(var_d * var_f) if var_d > 0 and var_f > 0 else 1.0
            correlation = cov / denom
            current_weight = getattr(self._weights, dim_name)
            adjustments[dim_name] = max(0.05, min(0.40, current_weight + 0.02 * correlation))

        new_weights = IWPGWeights(**adjustments)
        self.update_weights(new_weights)
        return new_weights

    def surrogate_rewards(self, reward: IWPGReward) -> dict[str, float]:
        """Compute surrogate rewards at multiple timescales."""
        immediate = reward.novelty * 0.5 + reward.surprise * 0.5
        short_term = reward.utility * 0.6 + reward.reproducibility * 0.4
        medium_term = self.total_reward(reward)
        long_term = (reward.utility * 0.4 + reward.novelty * 0.3
                     + (1.0 - reward.redundancy) * 0.3)
        return {
            "immediate": immediate,
            "short_term": short_term,
            "medium_term": medium_term,
            "long_term": long_term,
        }

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
