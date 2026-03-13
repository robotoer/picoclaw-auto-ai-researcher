"""Generate stratified research hypotheses across quality tiers and source models."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .models import Hypothesis, QualityTier

logger = structlog.get_logger(__name__)

SOURCE_MODELS: list[dict[str, str | int]] = [
    {"name": "anthropic/claude-sonnet-4", "count": 17},
    {"name": "openai/gpt-4o", "count": 17},
    {"name": "google/gemini-2.0-flash-001", "count": 16},
]

TIER_PROMPTS: dict[QualityTier, str] = {
    QualityTier.TRIVIAL: """\
Generate {n} research hypotheses about AI/ML that are TRIVIAL.
These should be obvious, well-established facts restated as hypotheses. They \
should not propose anything new or surprising — any ML practitioner would \
immediately agree without needing evidence.

Examples of the flavor we want:
- "Larger language models achieve higher accuracy on standard NLP benchmarks."
- "Training neural networks for more epochs reduces training loss."

For each hypothesis provide:
- "text": the hypothesis statement (1-2 sentences)
- "rationale": a 2-3 sentence rationale explaining why this hypothesis is plausible

Respond with a JSON array of objects with keys "text" and "rationale".""",
    QualityTier.INCREMENTAL: """\
Generate {n} research hypotheses about AI/ML that are INCREMENTAL.
These should be small, sensible extensions of known results — the kind of \
hypothesis that could appear in a solid workshop paper. They should be testable \
and non-obvious but not paradigm-shifting.

Examples of the flavor we want:
- "Combining LoRA with gradient checkpointing reduces peak GPU memory by at \
least 30% compared to LoRA alone during fine-tuning of 7B-parameter models."
- "Curriculum learning ordered by perplexity improves few-shot performance on \
multilingual benchmarks relative to random ordering."

For each hypothesis provide:
- "text": the hypothesis statement (1-2 sentences)
- "rationale": a 2-3 sentence rationale explaining why this hypothesis is plausible

Respond with a JSON array of objects with keys "text" and "rationale".""",
    QualityTier.AMBITIOUS: """\
Generate {n} research hypotheses about AI/ML that are AMBITIOUS and NOVEL.
These should be creative, potentially high-impact research directions that have \
not yet been thoroughly explored. They should be specific enough to be testable \
but bold enough to be surprising if confirmed.

Examples of the flavor we want:
- "Transformer models trained on code repositories implicitly learn formal \
verification strategies that can be extracted and applied to prove correctness \
of unseen programs without additional training."
- "A self-supervised objective based on predicting the Kolmogorov complexity of \
token subsequences produces representations that transfer better to scientific \
reasoning tasks than next-token prediction."

For each hypothesis provide:
- "text": the hypothesis statement (1-2 sentences)
- "rationale": a 2-3 sentence rationale explaining why this hypothesis is plausible

Respond with a JSON array of objects with keys "text" and "rationale".""",
    QualityTier.FLAWED: """\
Generate {n} research hypotheses about AI/ML that are FLAWED.
Each hypothesis should contain a clear logical error. Types of flaws to include: \
circular reasoning, unfalsifiable claims, self-contradictions, conflating \
correlation with causation, or undefined/unmeasurable terms. The hypotheses \
should sound superficially plausible to a casual reader but fall apart under \
scrutiny.

Examples of the flavor we want:
- "A model is intelligent if and only if it behaves in a way that an intelligent \
system would behave." (circular)
- "Scaling language models will eventually solve all reasoning tasks, because any \
failure can be attributed to insufficient scale." (unfalsifiable)

For each hypothesis provide:
- "text": the hypothesis statement (1-2 sentences)
- "rationale": a 2-3 sentence rationale (write it as if you believe the \
hypothesis is sound — the flaw should be embedded, not called out)

Respond with a JSON array of objects with keys "text" and "rationale".""",
    QualityTier.MIXED: """\
Generate {n} research hypotheses about AI/ML that are MIXED quality.
Each should contain a genuinely interesting core idea combined with a significant \
weakness — such as vague operationalization, questionable assumptions, scope that \
is too broad, or a confound that undermines the conclusion. A reviewer would say \
"interesting direction but needs major revision."

Examples of the flavor we want:
- "Reinforcement learning from human feedback eliminates all harmful biases in \
language models, as measured by standard toxicity benchmarks." (interesting \
direction but overclaims and uses narrow metrics)
- "Sparse mixture-of-experts models are more energy efficient than dense models \
for all downstream tasks." (plausible core but overgeneralizes)

For each hypothesis provide:
- "text": the hypothesis statement (1-2 sentences)
- "rationale": a 2-3 sentence rationale (present it favorably — the weakness \
should be embedded in the hypothesis itself, not pointed out in the rationale)

Respond with a JSON array of objects with keys "text" and "rationale".""",
}

SYSTEM_PROMPT = """\
You are an expert AI/ML researcher generating research hypotheses for a study \
on LLM evaluation. Produce exactly the number of hypotheses requested. Each \
hypothesis must be distinct and specific. Respond only with the JSON array."""


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def _call_model(
    model: str,
    prompt: str,
    api_key: str,
    base_url: str,
) -> list[dict[str, str]]:
    """Call an LLM via the OpenAI-compatible API and parse a JSON array response."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        response.raise_for_status()

    body = response.json()
    raw_text = body["choices"][0]["message"]["content"]
    return _parse_hypotheses(raw_text)


def _parse_hypotheses(raw_text: str) -> list[dict[str, str]]:
    """Extract a JSON array of hypothesis dicts from raw LLM output."""
    text = raw_text.strip()
    start_idx = text.find("[")
    end_idx = text.rfind("]")
    if start_idx == -1 or end_idx == -1:
        logger.warning("no_json_array_in_response", raw=text[:200])
        return []

    try:
        items = json.loads(text[start_idx : end_idx + 1])
    except json.JSONDecodeError:
        logger.warning("json_parse_failed", raw=text[:200])
        return []

    parsed: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict) and "text" in item and "rationale" in item:
            parsed.append({"text": item["text"], "rationale": item["rationale"]})
        else:
            logger.warning("hypothesis_parse_error", item=str(item)[:200])
    return parsed


def _build_assignments(
    tiers: list[QualityTier],
    models: list[dict[str, str | int]],
    per_tier: int = 10,
) -> list[dict[str, str | int]]:
    """Distribute hypothesis generation tasks across models and tiers.

    Returns a list of task dicts with keys: model, tier, count.
    """
    assignments: list[dict[str, str | int]] = []

    for tier in tiers:
        remaining = per_tier
        model_counts: list[int] = []
        for i, m in enumerate(models):
            share = remaining // (len(models) - i)
            model_counts.append(share)
            remaining -= share

        for m, count in zip(models, model_counts):
            if count > 0:
                assignments.append(
                    {"model": m["name"], "tier": tier.value, "count": count}
                )

    return assignments


async def generate_hypotheses(
    output_dir: Path,
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
) -> list[Hypothesis]:
    """Generate 50 research hypotheses stratified across quality tiers and models.

    Args:
        output_dir: Directory to write ``hypotheses.json``.
        api_key: OpenRouter API key.
        base_url: API base URL.

    Returns:
        List of 50 :class:`Hypothesis` objects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tiers = list(QualityTier)
    assignments = _build_assignments(tiers, SOURCE_MODELS, per_tier=10)

    logger.info(
        "generating_hypotheses",
        n_assignments=len(assignments),
        tiers=[t.value for t in tiers],
    )

    all_hypotheses: list[Hypothesis] = []
    hypothesis_counter = 0

    for assignment in assignments:
        model = str(assignment["model"])
        tier = QualityTier(str(assignment["tier"]))
        count = int(assignment["count"])

        prompt = TIER_PROMPTS[tier].format(n=count)

        logger.info(
            "calling_model",
            model=model,
            tier=tier.value,
            requested=count,
        )

        try:
            raw_hypotheses = await _call_model(
                model=model,
                prompt=prompt,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception:
            logger.exception(
                "generation_failed",
                model=model,
                tier=tier.value,
            )
            continue

        for h in raw_hypotheses[:count]:
            hypothesis_counter += 1
            all_hypotheses.append(
                Hypothesis(
                    id=f"H{hypothesis_counter:02d}",
                    text=h["text"],
                    rationale=h["rationale"],
                    quality_tier=tier,
                    source_model=model,
                )
            )

        logger.info(
            "batch_complete",
            model=model,
            tier=tier.value,
            generated=min(len(raw_hypotheses), count),
        )

        await asyncio.sleep(1.0)

    out_path = output_dir / "hypotheses.json"
    out_path.write_text(
        json.dumps(
            [h.model_dump(mode="json") for h in all_hypotheses],
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "hypotheses_saved",
        path=str(out_path),
        total=len(all_hypotheses),
        by_tier={
            t.value: sum(1 for h in all_hypotheses if h.quality_tier == t)
            for t in tiers
        },
        by_model={
            str(m["name"]): sum(
                1 for h in all_hypotheses if h.source_model == m["name"]
            )
            for m in SOURCE_MODELS
        },
    )

    return all_hypotheses
