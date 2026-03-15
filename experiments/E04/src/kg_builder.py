"""Knowledge graph construction and contradiction detection for E04.

Builds a KG from extracted claims, applies multi-layer hallucination prevention
filtering, detects contradictions against ground truth, performs spot-check
verification, and tracks growth curves.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict

import httpx
import structlog
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import (
    Claim,
    ContradictionResult,
    ExtractionResult,
    KGEdge,
    KGNode,
    KnowledgeGraph,
    Paper,
    SpotCheckResult,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_NOUN_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"  # capitalised phrases
    r"|[a-z]+(?:\s+[a-z]+){0,3})"  # short lowercase phrases
)


def _hash_id(*parts: str) -> str:
    """Create a deterministic short hash from string parts."""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def _extract_noun_phrases(text: str) -> list[str]:
    """Extract candidate noun phrases from text via simple heuristic."""
    # Grab capitalised phrases and short term sequences
    matches = _NOUN_PHRASE_RE.findall(text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        m_lower = m.lower()
        if m_lower not in seen and len(m_lower) > 2:
            seen.add(m_lower)
            result.append(m)
    return result


# ---------------------------------------------------------------------------
# TF-IDF similarity
# ---------------------------------------------------------------------------


def compute_claim_similarity(claim_a: str, claim_b: str) -> float:
    """Compute semantic similarity between two claims using TF-IDF cosine similarity.

    Uses sklearn TfidfVectorizer.  Returns a float in [0, 1].
    """
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([claim_a, claim_b])
    except ValueError:
        # Happens when both strings are empty / stop-words only
        return 0.0
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0]
    return float(sim)


def _pairwise_similarity_matrix(texts: list[str]) -> list[list[float]]:
    """Return an NxN similarity matrix for a list of texts."""
    if not texts:
        return []
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(texts)
    except ValueError:
        return [[0.0] * len(texts) for _ in texts]
    sim = cosine_similarity(tfidf)
    return sim.tolist()


# ---------------------------------------------------------------------------
# Knowledge-graph construction
# ---------------------------------------------------------------------------


def build_kg_from_claims(claims: list[Claim]) -> KnowledgeGraph:
    """Build a knowledge graph from a list of claims.

    Each claim becomes an edge.  Subject and object entities are extracted as
    nodes using simple noun-phrase extraction.  The claim text itself serves as
    the edge relation label.
    """
    kg = KnowledgeGraph()

    for claim in claims:
        # Store claim
        kg.claims[claim.claim_id] = claim

        # Extract noun phrases to use as subject / object nodes
        phrases = _extract_noun_phrases(claim.text)
        if len(phrases) < 2:
            # Fall back: split on common relation words
            parts = re.split(r"\b(?:is|are|was|were|has|have|can|uses|improves|outperforms)\b", claim.text, maxsplit=1)
            phrases = [p.strip() for p in parts if p.strip()]

        if len(phrases) < 2:
            # Still not enough – use whole text as a single self-loop node
            node_id = _hash_id(claim.text)
            if node_id not in kg.nodes:
                kg.nodes[node_id] = KGNode(
                    node_id=node_id,
                    label=claim.text[:120],
                    node_type="concept",
                )
            edge_id = _hash_id("edge", claim.claim_id)
            kg.edges[edge_id] = KGEdge(
                edge_id=edge_id,
                source_node=node_id,
                target_node=node_id,
                relation=claim.text,
                claim_id=claim.claim_id,
                paper_id=claim.paper_id,
                confidence=claim.confidence,
            )
            continue

        # Use first phrase as subject, last as object
        subj_label = phrases[0]
        obj_label = phrases[-1]

        subj_id = _hash_id("node", subj_label.lower())
        obj_id = _hash_id("node", obj_label.lower())

        if subj_id not in kg.nodes:
            kg.nodes[subj_id] = KGNode(
                node_id=subj_id,
                label=subj_label,
                node_type="concept",
            )
        if obj_id not in kg.nodes:
            kg.nodes[obj_id] = KGNode(
                node_id=obj_id,
                label=obj_label,
                node_type="concept",
            )

        edge_id = _hash_id("edge", claim.claim_id)
        kg.edges[edge_id] = KGEdge(
            edge_id=edge_id,
            source_node=subj_id,
            target_node=obj_id,
            relation=claim.text,
            claim_id=claim.claim_id,
            paper_id=claim.paper_id,
            confidence=claim.confidence,
        )

    logger.info(
        "kg_built",
        n_nodes=len(kg.nodes),
        n_edges=len(kg.edges),
        n_claims=len(kg.claims),
    )
    return kg


# ---------------------------------------------------------------------------
# Layer 1: Multi-extractor voting
# ---------------------------------------------------------------------------


def layer1_multi_extractor_voting(
    extraction_results: list[ExtractionResult],
    similarity_threshold: float = 0.85,
    min_agreement: int = 2,
) -> list[Claim]:
    """Layer 1: Accept claims that >= min_agreement extractors agree on.

    For each claim from each extractor, check if a semantically similar claim
    (cosine similarity >= threshold) exists from at least (min_agreement - 1)
    other extractors.  Return the deduplicated set of agreed-upon claims.
    """
    # Group claims by paper
    claims_by_paper: dict[str, list[tuple[str, Claim]]] = defaultdict(list)
    for er in extraction_results:
        for claim in er.claims:
            claims_by_paper[claim.paper_id].append((er.extractor_model, claim))

    accepted: list[Claim] = []

    for paper_id, extractor_claims in claims_by_paper.items():
        # Collect all texts and their extractor labels
        texts = [c.text for _, c in extractor_claims]
        extractors = [e for e, _ in extractor_claims]
        claims_list = [c for _, c in extractor_claims]

        if not texts:
            continue

        sim_matrix = _pairwise_similarity_matrix(texts)
        used: set[int] = set()

        for i in range(len(texts)):
            if i in used:
                continue

            # Find all claims similar enough to claim i
            similar_indices = [i]
            for j in range(i + 1, len(texts)):
                if j in used:
                    continue
                if sim_matrix[i][j] >= similarity_threshold:
                    similar_indices.append(j)

            # Count distinct extractors in the similar group
            distinct_extractors = {extractors[idx] for idx in similar_indices}

            if len(distinct_extractors) >= min_agreement:
                # Accept the first claim as representative
                accepted.append(claims_list[i])
                used.update(similar_indices)

    logger.info(
        "layer1_voting_complete",
        n_input=sum(len(er.claims) for er in extraction_results),
        n_accepted=len(accepted),
    )
    return accepted


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
async def _llm_call(
    client: httpx.AsyncClient,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
) -> str:
    """Make a chat completion call to an OpenRouter-compatible API."""
    response = await client.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _parse_nli_json(text: str) -> dict:
    """Best-effort parse of LLM NLI response into a dict."""
    # Try to find JSON in the response
    json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    # Fallback: treat as neutral
    return {"relationship": "neutral", "confidence": 0.5, "reasoning": text[:200]}


# ---------------------------------------------------------------------------
# Layer 2: Temporal consistency (ground truth contradiction check)
# ---------------------------------------------------------------------------

_NLI_SYSTEM_PROMPT = (
    "You are an expert at natural language inference for scientific claims. "
    "Respond ONLY with a JSON object."
)

_NLI_USER_TEMPLATE = """Given two claims, determine the relationship:
Claim A: {claim_a}
Claim B: {claim_b}

Respond with JSON: {{"relationship": "entailment"|"neutral"|"contradiction", "confidence": 0.0-1.0, "reasoning": "..."}}"""


async def layer2_temporal_consistency(
    claims: list[Claim],
    ground_truth_claims: list[Claim],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "anthropic/claude-sonnet-4-6",
) -> tuple[list[Claim], list[Claim]]:
    """Layer 2: Check claims against ground truth for contradictions.

    Uses LLM-based NLI to judge entailment / neutral / contradiction.
    Returns (accepted_claims, flagged_claims).
    Flagged claims are those that contradict established ground truth.
    """
    if not ground_truth_claims:
        logger.info("layer2_skipped", reason="no_ground_truth")
        return claims, []

    gt_texts = [c.text for c in ground_truth_claims]
    gt_sim_matrix_cache: list[list[float]] | None = None

    accepted: list[Claim] = []
    flagged: list[Claim] = []

    async with httpx.AsyncClient() as client:
        for claim in claims:
            # Find top-3 most similar ground-truth claims via TF-IDF
            all_texts = gt_texts + [claim.text]
            vectorizer = TfidfVectorizer()
            try:
                tfidf = vectorizer.fit_transform(all_texts)
            except ValueError:
                accepted.append(claim)
                continue

            sims = cosine_similarity(tfidf[-1:], tfidf[:-1])[0]
            top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]

            is_flagged = False
            for idx in top_indices:
                if sims[idx] < 0.1:
                    # Too dissimilar to bother checking
                    continue

                gt_claim = ground_truth_claims[idx]
                user_prompt = _NLI_USER_TEMPLATE.format(
                    claim_a=gt_claim.text,
                    claim_b=claim.text,
                )

                try:
                    raw = await _llm_call(
                        client, api_key, base_url, model,
                        _NLI_SYSTEM_PROMPT, user_prompt,
                    )
                    result = _parse_nli_json(raw)
                except Exception:
                    logger.warning("layer2_llm_error", claim_id=claim.claim_id)
                    continue

                if result.get("relationship") == "contradiction" and result.get("confidence", 0) >= 0.7:
                    is_flagged = True
                    logger.info(
                        "layer2_contradiction_found",
                        claim_id=claim.claim_id,
                        gt_claim_id=gt_claim.claim_id,
                        confidence=result.get("confidence"),
                    )
                    break

            if is_flagged:
                flagged.append(claim)
            else:
                accepted.append(claim)

    logger.info(
        "layer2_complete",
        n_input=len(claims),
        n_accepted=len(accepted),
        n_flagged=len(flagged),
    )
    return accepted, flagged


# ---------------------------------------------------------------------------
# Layer 3: Source verification
# ---------------------------------------------------------------------------

_VERIFY_SYSTEM_PROMPT = (
    "You are a meticulous scientific fact-checker. "
    "Determine whether a claim is supported by the given paper abstract. "
    "Respond ONLY with a JSON object."
)

_VERIFY_USER_TEMPLATE = """Paper title: {title}
Paper abstract: {abstract}

Claim: {claim_text}

Is the claim supported by the paper abstract?
Respond with JSON: {{"supported": true|false, "confidence": 0.0-1.0, "reasoning": "..."}}"""


async def layer3_source_verification(
    flagged_claims: list[Claim],
    papers: dict[str, Paper],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "anthropic/claude-sonnet-4-6",
) -> tuple[list[Claim], list[Claim]]:
    """Layer 3: Verify flagged claims against source paper text.

    For each flagged claim, check if the claim is actually supported by
    the paper's abstract.  Returns (verified_claims, rejected_claims).
    """
    verified: list[Claim] = []
    rejected: list[Claim] = []

    async with httpx.AsyncClient() as client:
        for claim in flagged_claims:
            paper = papers.get(claim.paper_id)
            if paper is None or not paper.abstract:
                # Cannot verify without source — reject conservatively
                logger.warning("layer3_no_source", claim_id=claim.claim_id, paper_id=claim.paper_id)
                rejected.append(claim)
                continue

            user_prompt = _VERIFY_USER_TEMPLATE.format(
                title=paper.title,
                abstract=paper.abstract,
                claim_text=claim.text,
            )

            try:
                raw = await _llm_call(
                    client, api_key, base_url, model,
                    _VERIFY_SYSTEM_PROMPT, user_prompt,
                )
                result = _parse_nli_json(raw)
            except Exception:
                logger.warning("layer3_llm_error", claim_id=claim.claim_id)
                rejected.append(claim)
                continue

            if result.get("supported", False) and result.get("confidence", 0) >= 0.6:
                verified.append(claim)
                logger.info(
                    "layer3_verified",
                    claim_id=claim.claim_id,
                    confidence=result.get("confidence"),
                )
            else:
                rejected.append(claim)
                logger.info(
                    "layer3_rejected",
                    claim_id=claim.claim_id,
                    reasoning=result.get("reasoning", "")[:120],
                )

    logger.info(
        "layer3_complete",
        n_input=len(flagged_claims),
        n_verified=len(verified),
        n_rejected=len(rejected),
    )
    return verified, rejected


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------


async def detect_contradictions(
    kg_claims: list[Claim],
    ground_truth_claims: list[Claim],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "anthropic/claude-sonnet-4-6",
) -> list[ContradictionResult]:
    """Check all KG claims against ground truth for contradictions.

    For efficiency, first filter to candidate pairs using TF-IDF similarity,
    then use LLM-based NLI for top candidates.
    """
    if not kg_claims or not ground_truth_claims:
        return []

    kg_texts = [c.text for c in kg_claims]
    gt_texts = [c.text for c in ground_truth_claims]

    # Build TF-IDF similarity between KG claims and ground truth
    all_texts = kg_texts + gt_texts
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform(all_texts)
    except ValueError:
        return []

    kg_tfidf = tfidf[: len(kg_texts)]
    gt_tfidf = tfidf[len(kg_texts) :]
    sim = cosine_similarity(kg_tfidf, gt_tfidf)

    # Collect candidate pairs where similarity > 0.15
    candidates: list[tuple[int, int, float]] = []
    for i in range(len(kg_texts)):
        for j in range(len(gt_texts)):
            if sim[i][j] >= 0.15:
                candidates.append((i, j, float(sim[i][j])))

    # Sort by similarity descending, take top candidates
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:200]  # cap for cost control

    results: list[ContradictionResult] = []

    async with httpx.AsyncClient() as client:
        for kg_idx, gt_idx, tfidf_sim in candidates:
            kg_claim = kg_claims[kg_idx]
            gt_claim = ground_truth_claims[gt_idx]

            user_prompt = _NLI_USER_TEMPLATE.format(
                claim_a=gt_claim.text,
                claim_b=kg_claim.text,
            )

            try:
                raw = await _llm_call(
                    client, api_key, base_url, model,
                    _NLI_SYSTEM_PROMPT, user_prompt,
                )
                nli = _parse_nli_json(raw)
            except Exception:
                logger.warning(
                    "contradiction_llm_error",
                    kg_claim_id=kg_claim.claim_id,
                    gt_claim_id=gt_claim.claim_id,
                )
                continue

            relationship = nli.get("relationship", "neutral")
            confidence = float(nli.get("confidence", 0.5))

            is_contradiction = relationship == "contradiction" and confidence >= 0.7

            results.append(
                ContradictionResult(
                    claim_id=kg_claim.claim_id,
                    ground_truth_claim_id=gt_claim.claim_id,
                    entailment_score=confidence if relationship == "entailment" else 0.0,
                    contradiction_score=confidence if relationship == "contradiction" else 0.0,
                    is_contradiction=is_contradiction,
                    reasoning=nli.get("reasoning", ""),
                )
            )

    n_contradictions = sum(1 for r in results if r.is_contradiction)
    logger.info(
        "contradiction_detection_complete",
        n_candidates=len(candidates),
        n_checked=len(results),
        n_contradictions=n_contradictions,
    )
    return results


# ---------------------------------------------------------------------------
# Spot-check verification
# ---------------------------------------------------------------------------

_SPOT_CHECK_SYSTEM_PROMPT = (
    "You are a meticulous scientific fact-checker performing a spot-check audit. "
    "Categorise the claim as one of: correct, partially_supported, unsupported, fabricated_source. "
    "Respond ONLY with a JSON object."
)

_SPOT_CHECK_USER_TEMPLATE = """Paper title: {title}
Paper abstract: {abstract}

Claim (attributed to this paper): {claim_text}

Categorise this claim:
- "correct": fully supported by the abstract
- "partially_supported": some aspects supported, others not verifiable from abstract alone
- "unsupported": not supported by the abstract but not fabricated
- "fabricated_source": the claim attributes information to this paper that clearly does not come from it

Respond with JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}}"""


async def spot_check_claims(
    claims: list[Claim],
    papers: dict[str, Paper],
    api_key: str,
    base_url: str = "https://openrouter.ai/api/v1",
    model: str = "anthropic/claude-opus-4",
    n_samples: int = 50,
) -> list[SpotCheckResult]:
    """Randomly sample and verify claims against source papers.

    LLM checks whether the claim is supported by the paper abstract.
    Categories: correct, partially_supported, unsupported, fabricated_source
    """
    # Filter to claims whose paper we have
    verifiable = [c for c in claims if c.paper_id in papers and papers[c.paper_id].abstract]
    if not verifiable:
        logger.warning("spot_check_no_verifiable_claims")
        return []

    sample_size = min(n_samples, len(verifiable))
    sampled = random.sample(verifiable, sample_size)

    results: list[SpotCheckResult] = []

    async with httpx.AsyncClient() as client:
        for claim in sampled:
            paper = papers[claim.paper_id]

            user_prompt = _SPOT_CHECK_USER_TEMPLATE.format(
                title=paper.title,
                abstract=paper.abstract,
                claim_text=claim.text,
            )

            try:
                raw = await _llm_call(
                    client, api_key, base_url, model,
                    _SPOT_CHECK_SYSTEM_PROMPT, user_prompt,
                )
                parsed = _parse_nli_json(raw)
            except Exception:
                logger.warning("spot_check_llm_error", claim_id=claim.claim_id)
                results.append(
                    SpotCheckResult(
                        claim_id=claim.claim_id,
                        paper_id=claim.paper_id,
                        category="error",
                        reasoning="LLM call failed",
                    )
                )
                continue

            category = parsed.get("category", "unsupported")
            valid_categories = {"correct", "partially_supported", "unsupported", "fabricated_source"}
            if category not in valid_categories:
                category = "unsupported"

            results.append(
                SpotCheckResult(
                    claim_id=claim.claim_id,
                    paper_id=claim.paper_id,
                    category=category,
                    reasoning=parsed.get("reasoning", ""),
                )
            )

    # Log summary
    category_counts: dict[str, int] = defaultdict(int)
    for r in results:
        category_counts[r.category] += 1

    logger.info(
        "spot_check_complete",
        n_sampled=len(sampled),
        n_results=len(results),
        category_counts=dict(category_counts),
    )
    return results


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------


def compute_duplicate_rate(claims: list[Claim], threshold: float = 0.95) -> float:
    """Compute near-duplicate rate using TF-IDF cosine similarity.

    Returns the fraction of claims that are near-duplicates of an earlier claim.
    """
    if len(claims) < 2:
        return 0.0

    texts = [c.text for c in claims]
    sim_matrix = _pairwise_similarity_matrix(texts)
    if not sim_matrix:
        return 0.0

    n_duplicates = 0
    for i in range(1, len(texts)):
        for j in range(i):
            if sim_matrix[i][j] >= threshold:
                n_duplicates += 1
                break  # only count each claim as duplicate once

    return n_duplicates / len(claims)


def compute_provenance_completeness(claims: list[Claim]) -> float:
    """Fraction of claims with full provenance (paper_id + section + quote)."""
    if not claims:
        return 0.0

    complete = sum(
        1
        for c in claims
        if c.paper_id and c.section and c.quote
    )
    return complete / len(claims)
