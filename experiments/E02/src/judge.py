"""LLM judge and expert proxy evaluation pipeline for E02: LLM Judge Reliability.

Rates research hypotheses using multiple LLMs via the OpenRouter API across
four evaluation formats (absolute, absolute_cot, pairwise, pairwise_cot).
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import (
    AbsoluteRating,
    Dimension,
    Hypothesis,
    PairwiseRating,
    QualityTier,
    RatingSession,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIMENSIONS = list(Dimension)

LLM_JUDGE_MODELS = [
    "anthropic/claude-opus-4-6",
    "openai/gpt-4o",
    "google/gemini-2.0-flash-001",
]

EXPERT_PROXY_MODEL = "anthropic/claude-opus-4"

N_POSITION_BIAS_RUNS = 3

# ---------------------------------------------------------------------------
# Dimension rubrics
# ---------------------------------------------------------------------------

DIMENSION_RUBRICS = {
    Dimension.NOVELTY: (
        "Novelty: How original and non-obvious is the hypothesis?\n"
        "  1 - Very poor: Restates well-known facts or existing results; "
        "no new angle.\n"
        "  2 - Poor: Minor variation on existing work with no meaningful twist.\n"
        "  3 - Below average: Some originality but largely derivative.\n"
        "  4 - Moderate: Has merit and offers a somewhat new angle, but "
        "similar ideas exist.\n"
        "  5 - Above average: Clearly original combination of ideas; "
        "non-obvious connection.\n"
        "  6 - Very good: Introduces a genuinely new perspective or mechanism.\n"
        "  7 - Excellent: Highly novel; if true, would open a new line of "
        "research."
    ),
    Dimension.FEASIBILITY: (
        "Feasibility: How practically testable is the hypothesis with "
        "current methods?\n"
        "  1 - Very poor: Requires technology that does not exist; "
        "untestable in principle.\n"
        "  2 - Poor: Would require unrealistic resources or decades of work.\n"
        "  3 - Below average: Testable in theory but with major practical "
        "barriers.\n"
        "  4 - Moderate: Testable but requires significant effort or "
        "specialized resources.\n"
        "  5 - Above average: Clearly testable with existing methods and "
        "reasonable resources.\n"
        "  6 - Very good: Straightforward to test with standard tools and "
        "datasets.\n"
        "  7 - Excellent: Could be tested quickly with readily available "
        "resources; clear experimental design."
    ),
    Dimension.IMPORTANCE: (
        "Importance: How significant would confirming or refuting this "
        "hypothesis be?\n"
        "  1 - Very poor: Trivial or irrelevant; no one would care about "
        "the result.\n"
        "  2 - Poor: Marginal relevance; would confirm what is already assumed.\n"
        "  3 - Below average: Some interest to a narrow sub-community.\n"
        "  4 - Moderate: Meaningful contribution but limited broader impact.\n"
        "  5 - Above average: Would advance understanding in a notable way.\n"
        "  6 - Very good: High impact; would change how researchers approach "
        "the problem.\n"
        "  7 - Excellent: Potentially paradigm-shifting; broad implications "
        "across fields."
    ),
    Dimension.CLARITY: (
        "Clarity: How clearly and precisely is the hypothesis stated?\n"
        "  1 - Very poor: Incoherent, circular reasoning, or contradictory.\n"
        "  2 - Poor: Vague and ambiguous; multiple interpretations possible.\n"
        "  3 - Below average: Understandable but imprecise in key terms.\n"
        "  4 - Moderate: Reasonably clear but could be sharper; some "
        "ambiguity remains.\n"
        "  5 - Above average: Well-stated with minor room for improvement.\n"
        "  6 - Very good: Precise and unambiguous; easy to evaluate.\n"
        "  7 - Excellent: Crystal clear; all terms defined, no ambiguity, "
        "immediately testable as stated."
    ),
    Dimension.SPECIFICITY: (
        "Specificity: How specific and falsifiable are the predictions?\n"
        "  1 - Very poor: Unfalsifiable; no concrete prediction.\n"
        "  2 - Poor: Too vague to design an experiment around.\n"
        "  3 - Below average: Loosely specified; hard to determine what "
        "would count as evidence.\n"
        "  4 - Moderate: Identifies the general phenomenon but lacks precise "
        "predictions.\n"
        "  5 - Above average: Specifies measurable outcomes; some boundary "
        "conditions unclear.\n"
        "  6 - Very good: Clear predictions with defined success/failure "
        "criteria.\n"
        "  7 - Excellent: Precise quantitative or qualitative predictions; "
        "unambiguous falsification criteria."
    ),
}

FULL_RUBRIC = "\n\n".join(DIMENSION_RUBRICS[d] for d in DIMENSIONS)

# ---------------------------------------------------------------------------
# Expert proxy personas
# ---------------------------------------------------------------------------

EXPERT_PERSONAS = [
    {
        "id": "expert_theorist",
        "system": (
            "You are a senior ML researcher with 15+ years of experience "
            "focused on theoretical foundations of machine learning. You value "
            "mathematical rigor, provable guarantees, and connections to "
            "established theory. You are skeptical of purely empirical claims "
            "without theoretical grounding. Evaluate hypotheses from this "
            "perspective."
        ),
    },
    {
        "id": "expert_applied",
        "system": (
            "You are an applied AI researcher who has shipped multiple "
            "production ML systems. You value practical impact, scalability, "
            "and real-world applicability. You care about whether a hypothesis "
            "would lead to systems that actually work better in practice, not "
            "just theoretical elegance. Evaluate hypotheses from this "
            "perspective."
        ),
    },
    {
        "id": "expert_manager",
        "system": (
            "You are a research program manager at a major AI lab. You "
            "evaluate research directions for strategic fit, feasibility "
            "within resource constraints, and clarity of communication. You "
            "value hypotheses that are well-scoped, clearly articulated, and "
            "could realistically be executed by a small team in 6-12 months. "
            "Evaluate hypotheses from this perspective."
        ),
    },
]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ABSOLUTE_SYSTEM = """\
You are an expert research evaluator. You will rate research hypotheses on \
five quality dimensions using a 1-7 Likert scale.

{rubric}

{persona}
"""

ABSOLUTE_PROMPT_NO_COT = """\
Rate the following hypothesis on all five dimensions (novelty, feasibility, \
importance, clarity, specificity) using the 1-7 scale from the rubric.

Hypothesis ID: {hypothesis_id}
Hypothesis: {hypothesis_text}
Rationale: {hypothesis_rationale}

Respond with ONLY a JSON object in this exact format (no other text):
{{
  "novelty": <int 1-7>,
  "feasibility": <int 1-7>,
  "importance": <int 1-7>,
  "clarity": <int 1-7>,
  "specificity": <int 1-7>
}}
"""

ABSOLUTE_PROMPT_COT = """\
Rate the following hypothesis on all five dimensions (novelty, feasibility, \
importance, clarity, specificity) using the 1-7 scale from the rubric.

Hypothesis ID: {hypothesis_id}
Hypothesis: {hypothesis_text}
Rationale: {hypothesis_rationale}

First, reason step-by-step about the hypothesis quality on each dimension. \
Then provide your final scores.

Respond in this exact format:

REASONING:
<your step-by-step reasoning for each dimension>

SCORES:
{{
  "novelty": <int 1-7>,
  "feasibility": <int 1-7>,
  "importance": <int 1-7>,
  "clarity": <int 1-7>,
  "specificity": <int 1-7>
}}
"""

PAIRWISE_SYSTEM = """\
You are an expert research evaluator. You will compare pairs of research \
hypotheses on five quality dimensions.

{rubric}

{persona}
"""

PAIRWISE_PROMPT_NO_COT = """\
Compare the following two hypotheses on all five dimensions (novelty, \
feasibility, importance, clarity, specificity). For each dimension, indicate \
which hypothesis is better: "a", "b", or "tie".

Hypothesis A:
  ID: {id_a}
  Text: {text_a}
  Rationale: {rationale_a}

Hypothesis B:
  ID: {id_b}
  Text: {text_b}
  Rationale: {rationale_b}

Respond with ONLY a JSON object in this exact format (no other text):
{{
  "novelty": "<a|b|tie>",
  "feasibility": "<a|b|tie>",
  "importance": "<a|b|tie>",
  "clarity": "<a|b|tie>",
  "specificity": "<a|b|tie>"
}}
"""

PAIRWISE_PROMPT_COT = """\
Compare the following two hypotheses on all five dimensions (novelty, \
feasibility, importance, clarity, specificity). For each dimension, indicate \
which hypothesis is better: "a", "b", or "tie".

Hypothesis A:
  ID: {id_a}
  Text: {text_a}
  Rationale: {rationale_a}

Hypothesis B:
  ID: {id_b}
  Text: {text_b}
  Rationale: {rationale_b}

First, reason step-by-step about each dimension comparing the two hypotheses. \
Then provide your final verdicts.

Respond in this exact format:

REASONING:
<your step-by-step reasoning for each dimension>

VERDICTS:
{{
  "novelty": "<a|b|tie>",
  "feasibility": "<a|b|tie>",
  "importance": "<a|b|tie>",
  "clarity": "<a|b|tie>",
  "specificity": "<a|b|tie>"
}}
"""


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def _call_llm(
    system: str,
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    """Call an LLM via an OpenAI-compatible API (default: OpenRouter)."""
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _extract_json_object(raw: str) -> dict | None:
    """Extract the first JSON object from raw LLM output."""
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("json_parse_failed", raw=text[:300])
        return None


def _parse_absolute_response(
    raw: str,
    hypothesis_id: str,
    rater_id: str,
    rater_type: str,
    chain_of_thought: bool,
) -> list[AbsoluteRating]:
    """Parse absolute scoring response into a list of AbsoluteRating objects."""
    reasoning = ""
    if chain_of_thought:
        # Extract reasoning from CoT response.
        scores_marker = "SCORES:"
        idx = raw.find(scores_marker)
        if idx != -1:
            reasoning = raw[:idx].replace("REASONING:", "").strip()
            raw_scores = raw[idx + len(scores_marker) :]
        else:
            # Fallback: try to parse the whole thing as JSON.
            reasoning = ""
            raw_scores = raw
    else:
        raw_scores = raw

    scores = _extract_json_object(raw_scores)
    if scores is None:
        # Last-ditch: try the entire raw string.
        scores = _extract_json_object(raw)
    if scores is None:
        logger.warning(
            "absolute_parse_failed",
            hypothesis_id=hypothesis_id,
            rater_id=rater_id,
            raw=raw[:300],
        )
        return []

    ratings: list[AbsoluteRating] = []
    for dim in DIMENSIONS:
        score_val = scores.get(dim.value)
        if score_val is None:
            logger.warning(
                "missing_dimension_score",
                dimension=dim.value,
                hypothesis_id=hypothesis_id,
            )
            continue
        try:
            score_int = int(score_val)
            score_int = max(1, min(7, score_int))
        except (ValueError, TypeError):
            logger.warning(
                "invalid_score_value",
                dimension=dim.value,
                value=score_val,
            )
            continue
        ratings.append(
            AbsoluteRating(
                hypothesis_id=hypothesis_id,
                rater_id=rater_id,
                rater_type=rater_type,
                dimension=dim,
                score=score_int,
                reasoning=reasoning,
            )
        )
    return ratings


def _parse_pairwise_response(
    raw: str,
    hypothesis_a_id: str,
    hypothesis_b_id: str,
    rater_id: str,
    chain_of_thought: bool,
) -> list[PairwiseRating]:
    """Parse pairwise comparison response into a list of PairwiseRating objects."""
    reasoning = ""
    if chain_of_thought:
        verdicts_marker = "VERDICTS:"
        idx = raw.find(verdicts_marker)
        if idx != -1:
            reasoning = raw[:idx].replace("REASONING:", "").strip()
            raw_verdicts = raw[idx + len(verdicts_marker) :]
        else:
            reasoning = ""
            raw_verdicts = raw
    else:
        raw_verdicts = raw

    verdicts = _extract_json_object(raw_verdicts)
    if verdicts is None:
        verdicts = _extract_json_object(raw)
    if verdicts is None:
        logger.warning(
            "pairwise_parse_failed",
            hypothesis_a_id=hypothesis_a_id,
            hypothesis_b_id=hypothesis_b_id,
            rater_id=rater_id,
            raw=raw[:300],
        )
        return []

    ratings: list[PairwiseRating] = []
    valid_winners = {"a", "b", "tie"}
    for dim in DIMENSIONS:
        winner_val = verdicts.get(dim.value)
        if winner_val is None:
            logger.warning(
                "missing_dimension_verdict",
                dimension=dim.value,
            )
            continue
        winner_str = str(winner_val).strip().lower()
        if winner_str not in valid_winners:
            logger.warning(
                "invalid_winner_value",
                dimension=dim.value,
                value=winner_val,
            )
            continue
        ratings.append(
            PairwiseRating(
                hypothesis_a_id=hypothesis_a_id,
                hypothesis_b_id=hypothesis_b_id,
                rater_id=rater_id,
                dimension=dim,
                winner=winner_str,
                reasoning=reasoning,
            )
        )
    return ratings


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------


async def rate_hypotheses_absolute(
    hypotheses: list[Hypothesis],
    model_name: str,
    api_key: str,
    base_url: str,
    rater_id: str,
    chain_of_thought: bool = False,
    presentation_order: list[str] | None = None,
) -> RatingSession:
    """Rate all hypotheses on all 5 dimensions using absolute 1-7 Likert scoring.

    Args:
        hypotheses: Hypotheses to rate.
        model_name: OpenRouter model identifier.
        api_key: API key for the provider.
        base_url: API base URL.
        rater_id: Unique identifier for this rater/session.
        chain_of_thought: Whether to request step-by-step reasoning.
        presentation_order: Optional ordering of hypothesis IDs. If None,
            a random shuffle is used.

    Returns:
        A RatingSession containing all absolute ratings.
    """
    fmt = "absolute_cot" if chain_of_thought else "absolute"
    rater_type: str = "expert_proxy" if "expert" in rater_id else "llm_judge"

    # Determine persona text (empty for plain LLM judges).
    persona = ""
    for ep in EXPERT_PERSONAS:
        if ep["id"] == rater_id:
            persona = ep["system"]
            break

    system = ABSOLUTE_SYSTEM.format(rubric=FULL_RUBRIC, persona=persona)
    prompt_template = ABSOLUTE_PROMPT_COT if chain_of_thought else ABSOLUTE_PROMPT_NO_COT

    # Determine presentation order.
    if presentation_order is not None:
        id_to_hyp = {h.id: h for h in hypotheses}
        ordered = [id_to_hyp[hid] for hid in presentation_order if hid in id_to_hyp]
    else:
        ordered = list(hypotheses)
        random.shuffle(ordered)

    order_ids = [h.id for h in ordered]

    session = RatingSession(
        rater_id=rater_id,
        rater_type=rater_type,
        model_name=model_name,
        format=fmt,
        presentation_order=order_ids,
    )

    for hyp in ordered:
        prompt = prompt_template.format(
            hypothesis_id=hyp.id,
            hypothesis_text=hyp.text,
            hypothesis_rationale=hyp.rationale,
        )
        logger.info(
            "rating_absolute",
            hypothesis_id=hyp.id,
            rater_id=rater_id,
            model=model_name,
            cot=chain_of_thought,
        )
        try:
            raw = await _call_llm(
                system=system,
                prompt=prompt,
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            )
            ratings = _parse_absolute_response(
                raw=raw,
                hypothesis_id=hyp.id,
                rater_id=rater_id,
                rater_type=rater_type,
                chain_of_thought=chain_of_thought,
            )
            session.ratings.extend(ratings)
            logger.info(
                "rated_hypothesis",
                hypothesis_id=hyp.id,
                n_dims=len(ratings),
            )
        except Exception:
            logger.exception(
                "absolute_rating_failed",
                hypothesis_id=hyp.id,
                rater_id=rater_id,
            )

        await asyncio.sleep(1.0)

    return session


async def rate_hypotheses_pairwise(
    pairs: list[tuple[Hypothesis, Hypothesis]],
    model_name: str,
    api_key: str,
    base_url: str,
    rater_id: str,
    chain_of_thought: bool = False,
) -> RatingSession:
    """Rate hypothesis pairs on all 5 dimensions.

    Args:
        pairs: List of (hypothesis_a, hypothesis_b) tuples.
        model_name: OpenRouter model identifier.
        api_key: API key for the provider.
        base_url: API base URL.
        rater_id: Unique identifier for this rater/session.
        chain_of_thought: Whether to request step-by-step reasoning.

    Returns:
        A RatingSession containing all pairwise ratings.
    """
    fmt = "pairwise_cot" if chain_of_thought else "pairwise"
    rater_type: str = "expert_proxy" if "expert" in rater_id else "llm_judge"

    persona = ""
    for ep in EXPERT_PERSONAS:
        if ep["id"] == rater_id:
            persona = ep["system"]
            break

    system = PAIRWISE_SYSTEM.format(rubric=FULL_RUBRIC, persona=persona)
    prompt_template = PAIRWISE_PROMPT_COT if chain_of_thought else PAIRWISE_PROMPT_NO_COT

    session = RatingSession(
        rater_id=rater_id,
        rater_type=rater_type,
        model_name=model_name,
        format=fmt,
    )

    for hyp_a, hyp_b in pairs:
        prompt = prompt_template.format(
            id_a=hyp_a.id,
            text_a=hyp_a.text,
            rationale_a=hyp_a.rationale,
            id_b=hyp_b.id,
            text_b=hyp_b.text,
            rationale_b=hyp_b.rationale,
        )
        logger.info(
            "rating_pairwise",
            pair=f"{hyp_a.id}_vs_{hyp_b.id}",
            rater_id=rater_id,
            model=model_name,
            cot=chain_of_thought,
        )
        try:
            raw = await _call_llm(
                system=system,
                prompt=prompt,
                model=model_name,
                api_key=api_key,
                base_url=base_url,
            )
            ratings = _parse_pairwise_response(
                raw=raw,
                hypothesis_a_id=hyp_a.id,
                hypothesis_b_id=hyp_b.id,
                rater_id=rater_id,
                chain_of_thought=chain_of_thought,
            )
            session.pairwise_ratings.extend(ratings)
            logger.info(
                "rated_pair",
                pair=f"{hyp_a.id}_vs_{hyp_b.id}",
                n_dims=len(ratings),
            )
        except Exception:
            logger.exception(
                "pairwise_rating_failed",
                pair=f"{hyp_a.id}_vs_{hyp_b.id}",
                rater_id=rater_id,
            )

        await asyncio.sleep(1.0)

    return session


# ---------------------------------------------------------------------------
# High-level pipelines
# ---------------------------------------------------------------------------


async def run_expert_proxy_ratings(
    hypotheses: list[Hypothesis],
    api_key: str,
    base_url: str,
    model: str = "anthropic/claude-opus-4",
) -> list[RatingSession]:
    """Run 3 expert proxy raters (different personas) using a strong model.

    Each expert rates all hypotheses using absolute scoring with CoT.

    Args:
        hypotheses: Hypotheses to evaluate.
        api_key: API key.
        base_url: API base URL.
        model: Model to use for expert proxies.

    Returns:
        List of RatingSession objects, one per expert persona.
    """
    sessions: list[RatingSession] = []

    for persona in EXPERT_PERSONAS:
        logger.info(
            "running_expert_proxy",
            persona_id=persona["id"],
            model=model,
            n_hypotheses=len(hypotheses),
        )
        session = await rate_hypotheses_absolute(
            hypotheses=hypotheses,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            rater_id=persona["id"],
            chain_of_thought=True,
        )
        sessions.append(session)

    return sessions


def select_pairwise_pairs(
    hypotheses: list[Hypothesis],
    n_pairs: int = 20,
) -> list[tuple[Hypothesis, Hypothesis]]:
    """Select hypothesis pairs spanning the quality range.

    Strategy: group hypotheses by quality tier, then create pairs that
    compare across tiers (to maximize discriminative signal) as well as
    within tiers (to test fine-grained discrimination).
    """
    by_tier: dict[QualityTier, list[Hypothesis]] = {}
    for h in hypotheses:
        by_tier.setdefault(h.quality_tier, []).append(h)

    pairs: list[tuple[Hypothesis, Hypothesis]] = []
    tiers = list(by_tier.keys())

    # Cross-tier pairs.
    for i in range(len(tiers)):
        for j in range(i + 1, len(tiers)):
            for ha in by_tier[tiers[i]]:
                for hb in by_tier[tiers[j]]:
                    pairs.append((ha, hb))

    # Within-tier pairs.
    for tier_hyps in by_tier.values():
        for i in range(len(tier_hyps)):
            for j in range(i + 1, len(tier_hyps)):
                pairs.append((tier_hyps[i], tier_hyps[j]))

    # Shuffle and limit to n_pairs.
    random.shuffle(pairs)
    return pairs[:n_pairs]


async def run_llm_judge_ratings(
    hypotheses: list[Hypothesis],
    pairs: list[tuple[Hypothesis, Hypothesis]],
    models: list[str],
    api_key: str,
    base_url: str,
) -> list[RatingSession]:
    """Run all LLM judge evaluations across formats and models.

    For each model, runs:
    - Absolute scoring (no CoT) x N_POSITION_BIAS_RUNS orderings
    - Absolute scoring with CoT x 1
    - Pairwise comparison (no CoT) x 1

    Args:
        hypotheses: Hypotheses to evaluate.
        pairs: Pre-selected hypothesis pairs for pairwise comparison.
        models: List of model identifiers to use.
        api_key: API key.
        base_url: API base URL.

    Returns:
        List of all RatingSession objects.
    """
    sessions: list[RatingSession] = []

    for model in models:
        safe_model = model.replace("/", "_")

        # Absolute scoring with position bias testing (multiple orderings).
        for run_idx in range(N_POSITION_BIAS_RUNS):
            order = [h.id for h in hypotheses]
            random.shuffle(order)
            rater_id = f"judge_{safe_model}_absolute_run{run_idx}"

            logger.info(
                "running_llm_judge_absolute",
                model=model,
                run=run_idx,
                cot=False,
            )
            session = await rate_hypotheses_absolute(
                hypotheses=hypotheses,
                model_name=model,
                api_key=api_key,
                base_url=base_url,
                rater_id=rater_id,
                chain_of_thought=False,
                presentation_order=order,
            )
            sessions.append(session)

        # Absolute scoring with CoT (single run).
        rater_id_cot = f"judge_{safe_model}_absolute_cot"
        logger.info(
            "running_llm_judge_absolute_cot",
            model=model,
        )
        session_cot = await rate_hypotheses_absolute(
            hypotheses=hypotheses,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            rater_id=rater_id_cot,
            chain_of_thought=True,
        )
        sessions.append(session_cot)

        # Pairwise comparison (no CoT).
        rater_id_pw = f"judge_{safe_model}_pairwise"
        logger.info(
            "running_llm_judge_pairwise",
            model=model,
            n_pairs=len(pairs),
        )
        session_pw = await rate_hypotheses_pairwise(
            pairs=pairs,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            rater_id=rater_id_pw,
            chain_of_thought=False,
        )
        sessions.append(session_pw)

    return sessions


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_session(session: RatingSession, output_dir: Path) -> Path:
    """Save a RatingSession to a JSON file in the output directory.

    The filename is derived from the rater_id and format.

    Args:
        session: The rating session to persist.
        output_dir: Directory where the file will be written.

    Returns:
        Path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_rater = session.rater_id.replace("/", "_")
    filename = f"{safe_rater}_{session.format}.json"
    out_path = output_dir / filename
    out_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    logger.info("session_saved", path=str(out_path), rater_id=session.rater_id)
    return out_path


async def run_full_evaluation(
    hypotheses: list[Hypothesis],
    api_key: str,
    base_url: str,
    output_dir: Path,
    llm_models: list[str] | None = None,
    expert_model: str = EXPERT_PROXY_MODEL,
    n_pairwise_pairs: int = 20,
) -> list[RatingSession]:
    """Run the complete evaluation pipeline (expert proxies + LLM judges).

    Args:
        hypotheses: All hypotheses to evaluate.
        api_key: OpenRouter API key.
        base_url: OpenRouter base URL.
        output_dir: Directory for output JSON files.
        llm_models: Models for LLM judges. Defaults to LLM_JUDGE_MODELS.
        expert_model: Model for expert proxy raters.
        n_pairwise_pairs: Number of hypothesis pairs for pairwise evaluation.

    Returns:
        All RatingSession objects produced.
    """
    if llm_models is None:
        llm_models = LLM_JUDGE_MODELS

    all_sessions: list[RatingSession] = []

    # Phase 1: Expert proxy ratings.
    logger.info("phase_expert_proxies", n_hypotheses=len(hypotheses))
    expert_sessions = await run_expert_proxy_ratings(
        hypotheses=hypotheses,
        api_key=api_key,
        base_url=base_url,
        model=expert_model,
    )
    for s in expert_sessions:
        save_session(s, output_dir)
    all_sessions.extend(expert_sessions)

    # Phase 2: LLM judge ratings.
    pairs = select_pairwise_pairs(hypotheses, n_pairs=n_pairwise_pairs)
    logger.info(
        "phase_llm_judges",
        n_models=len(llm_models),
        n_pairs=len(pairs),
    )
    judge_sessions = await run_llm_judge_ratings(
        hypotheses=hypotheses,
        pairs=pairs,
        models=llm_models,
        api_key=api_key,
        base_url=base_url,
    )
    for s in judge_sessions:
        save_session(s, output_dir)
    all_sessions.extend(judge_sessions)

    logger.info(
        "evaluation_complete",
        n_sessions=len(all_sessions),
        output_dir=str(output_dir),
    )
    return all_sessions
