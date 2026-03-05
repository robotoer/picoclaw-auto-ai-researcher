# 08 — Hallucination Cascade Prevention

> *How do we prevent a single false claim from corrupting the entire knowledge graph?*

---

## 1. Problem Statement

The picoclaw-auto-ai-researcher ingests papers, extracts structured claims via LLM, and stores them in a knowledge graph (Neo4j). Future hypotheses and experiments build on these claims. If the LLM hallucinates a false claim, it enters the graph as if it were true. Downstream reasoning then treats it as a premise, producing derived claims, hypotheses, and experimental designs that inherit and amplify the error. This is a **hallucination cascade**: a single bad seed that corrupts an expanding subgraph of conclusions.

The cascade is dangerous because:
- **Silent propagation**: No error signal is generated when a false claim is used as a premise.
- **Confidence laundering**: A low-confidence hallucinated claim can appear to gain support when multiple derived claims reference it.
- **Exponential fan-out**: Each derived claim can itself become a premise for further derivations.

This document presents a multi-layered defense combining information-theoretic monitoring, creative verification heuristics, a cascade prevention architecture, and a formal confidence propagation framework.

---

## 2. Information-Theoretic Approaches

### 2.1 Shannon Entropy Monitoring

**Intuition**: The distribution of claims across topics and relation types should reflect the natural structure of the scientific literature. A hallucination injection distorts this distribution — either by concentrating claims in a narrow area (entropy drop) or by introducing contradictions across many areas (entropy spike).

**Formulation**: Let the knowledge graph at time *t* contain claims partitioned into *k* topic categories. Define the claim distribution as:

```
p_i(t) = n_i(t) / N(t)
```

where `n_i(t)` is the number of claims in category *i* and `N(t)` is the total claim count. The Shannon entropy is:

```
H(t) = -sum_{i=1}^{k} p_i(t) * log2(p_i(t))
```

**Anomaly detection**: Maintain a rolling window of `H(t)` values over the last *W* ingestion batches. Compute the z-score:

```
z_H(t) = (H(t) - mu_H) / sigma_H
```

where `mu_H` and `sigma_H` are the mean and standard deviation of `H` over the window. Flag an anomaly when `|z_H(t)| > 2.5`.

- **Low entropy anomaly** (`z_H < -2.5`): Too many claims concentrated in one area. Possible systematic hallucination about a single topic.
- **High entropy anomaly** (`z_H > 2.5`): Unusual spread of contradictory claims. Possible incoherent hallucination across topics.

**Granularity**: Apply this at multiple levels — global topic distribution, per-topic relation-type distribution, and per-entity degree distribution. A hallucinated entity that suddenly becomes a hub (high in-degree) will show up as a local entropy anomaly.

**Recommended parameters**: Window size *W* = 50 batches; threshold = 2.5 sigma; recompute after each ingestion batch.

### 2.2 KL Divergence for Drift Detection

**Intuition**: The knowledge graph evolves gradually as new papers are ingested. A sudden shift in the claim distribution signals that something unusual entered the system — either a genuine paradigm shift or a hallucination injection.

**Formulation**: After ingesting batch *b* at time *t*, compute the KL divergence between the new claim distribution and the prior:

```
D_KL(P_t || P_{t-1}) = sum_{i=1}^{k} p_i(t) * log2(p_i(t) / p_i(t-1))
```

Use Laplace smoothing to avoid division by zero for categories absent in *P_{t-1}*:

```
p_i'(t) = (n_i(t) + alpha) / (N(t) + k * alpha),  alpha = 1
```

**Distinguishing paradigm shifts from hallucination**: A genuine paradigm shift has the following signatures that hallucination lacks:

| Signal | Paradigm Shift | Hallucination |
|--------|---------------|---------------|
| Source diversity | Multiple independent papers | One paper or one extraction run |
| Temporal consistency | Claims appear gradually over weeks | Claims appear in a single batch |
| Citation backing | New claims cite existing established work | New claims lack citation grounding |
| Relation coherence | New claims form internally consistent subgraph | New claims may contradict each other |

**Decision rule**: If `D_KL > tau` (threshold), examine the batch that caused the drift. If >80% of the divergence-contributing claims come from a single source or extraction run, flag as potential hallucination. If claims come from 3+ independent sources with citation grounding, accept as genuine shift.

**Recommended thresholds**: `tau = 0.1` for alert, `tau = 0.3` for quarantine. These should be calibrated on the first 3 months of operation by observing normal drift rates.

**Recent research support**: Halperin (2025) proposes prompt-response semantic divergence metrics using KL divergence and Jensen-Shannon divergence to detect faithfulness hallucinations. The practical score `PS = JSD(P_prompt, P_response) + W_1(P_prompt, P_response)` combines distributional divergence with Wasserstein distance for robust detection.

### 2.3 Mutual Information for Source Independence

**Intuition**: A claim corroborated by multiple independent sources is more trustworthy than one supported by a single source or by sources that paraphrase each other. Mutual information quantifies the statistical dependence between sources — high MI means the sources are not truly independent.

**Formulation**: For two sources A and B that both support claim C, compute:

```
I(A; B | C) = H(A | C) + H(B | C) - H(A, B | C)
```

In practice, estimate source independence using:

1. **Citation overlap**: If papers A and B share >50% of their reference lists, they are not independent sources. Compute Jaccard similarity of citation lists: `J(refs_A, refs_B) = |refs_A ∩ refs_B| / |refs_A ∪ refs_B|`.

2. **Author overlap**: Shared authors reduce independence. `author_independence = 1 - |authors_A ∩ authors_B| / |authors_A ∪ authors_B|`.

3. **Textual similarity**: If the passages from which the claims were extracted have cosine similarity > 0.85 (using SPECTER embeddings), one source may be paraphrasing the other.

**Effective independence score**:

```
independence(A, B) = (1 - J_citation) * author_independence * (1 - max(0, cos_sim - 0.5) / 0.5)
```

A claim requires a minimum **effective independent corroboration count** (EICC):

```
EICC(claim) = sum_{pairs (A,B) supporting claim} independence(A, B)
```

**Threshold**: Claims with `EICC < 1.5` should not be used as premises for downstream reasoning without additional verification. Claims with `EICC >= 3.0` can be treated as well-established.

### 2.4 Minimum Description Length (MDL) Principle

**Intuition**: A legitimate claim fits naturally into the existing knowledge structure — it connects to known entities, uses established relation types, and is consistent with nearby claims. A hallucinated claim is "surprising" in a bad way: it requires more bits to encode relative to the existing graph because it doesn't fit the learned patterns.

**Formulation**: Model the knowledge graph as a probabilistic model `M` that assigns a code length to each claim. The code length of a new claim `c = (e1, r, e2, conditions)` is:

```
L(c | M) = -log2 P_M(c)
```

where `P_M(c)` is decomposed as:

```
P_M(c) = P(e1) * P(r | e1) * P(e2 | e1, r) * P(conditions | e1, r, e2)
```

Each factor is estimated from the existing graph:
- `P(e1)` = normalized frequency of entity e1 in existing claims
- `P(r | e1)` = fraction of e1's claims using relation r
- `P(e2 | e1, r)` = fraction of (e1, r, *) claims that have e2 as target
- `P(conditions | ...)` = estimated from conditional text similarity to existing conditions

**Anomaly scoring**: Compute the description length for each new claim. Claims in the top 5th percentile of description length (most surprising) are flagged for manual review or additional verification.

**Proxy for Kolmogorov complexity**: Since true Kolmogorov complexity is uncomputable, we use the graph-based probabilistic model as a practical proxy. An alternative proxy: compute the minimum number of graph edits (node additions, edge additions, type changes) needed to make the claim consistent with the existing graph. More edits = higher effective complexity = more suspicious.

**Connection to SEKA (2024)**: The SEKA framework for unsupervised KG anomaly detection uses similar principles — identifying triples that don't fit the statistical patterns of the existing graph. Our MDL approach formalizes this intuition.

### 2.5 Error-Correcting Codes Analogy

**Intuition**: In communications, error-correcting codes add redundancy so that the receiver can detect and correct errors even through a noisy channel. We can treat the LLM extraction process as a noisy channel and the knowledge graph as the received message, then apply redundancy requirements.

**Verification path redundancy**: For any claim used in downstream reasoning, require *N* independent verification paths:

```
verification_paths(claim) = {
    path_1: source_paper → extraction → claim
    path_2: independent_paper → extraction → corroborating_claim
    path_3: benchmark_data → direct_verification
    ...
}
```

**Hamming-distance analogy**: Define a "claim distance" metric between two claims as:

```
d(c1, c2) = w_e * entity_distance + w_r * relation_distance + w_c * condition_distance
```

Two claims are "corroborating" if `d(c1, c2) < epsilon` (they say approximately the same thing from different sources). A claim is "verifiable" if it has at least *t* corroborating claims from independent sources, analogous to a code with minimum distance *t+1*.

**Tiered redundancy requirements**:

| Claim usage level | Min verification paths | Rationale |
|---|---|---|
| Stored in KG only | 1 (extraction only) | Low risk: not used downstream |
| Used as premise for hypothesis | 2 | Medium risk: affects reasoning |
| Used as basis for experiment design | 3 | High risk: wastes compute on false premise |
| Cited in published output | 4 | Critical: affects external trust |

**Parity check analogy**: Periodically select random subgraphs and verify internal consistency. If a subgraph contains claims A→B, B→C, and A→C, check that the transitive closure is consistent. Inconsistencies indicate at least one claim is erroneous — analogous to a parity check failure.

---

## 3. Creative Non-Information-Theory Approaches

### 3.1 Temporal Consistency Checking

**Principle**: Scientific claims have temporal structure. A paper published in 2023 cannot report results about a method first published in 2024. Temporal violations are strong hallucination signals.

**Implementation**:

1. For every extracted claim `(entity_1, relation, entity_2)`, look up the publication dates of the entities:
   - `date(entity_1)` = earliest paper that introduces or defines entity_1
   - `date(entity_2)` = earliest paper that introduces or defines entity_2
   - `date(claim_source)` = publication date of the paper from which the claim was extracted

2. **Temporal rules**:
   - If `relation` is "outperforms", "improves upon", or "extends", then both entities must exist before `date(claim_source)`.
   - If `relation` is "introduces" or "proposes", then `date(entity_1)` should approximately equal `date(claim_source)`.
   - If the claim references a benchmark result, the benchmark must exist before the claim date.

3. **Violation scoring**: Each temporal violation adds a penalty to the claim's suspicion score. Hard violations (method referenced before it exists) trigger immediate quarantine. Soft violations (method referenced unusually early) add a flag for review.

**Data source**: Semantic Scholar API provides publication dates. For entities not in Semantic Scholar, use the earliest mention date in the knowledge graph.

### 3.2 Citation Graph Cross-Validation

**Principle**: If a paper claims "Method X outperforms Method Y on Dataset Z", the paper should cite the papers introducing X, Y, and Z. Missing citations are a red flag — the LLM may have hallucinated a relationship that the paper doesn't actually discuss.

**Implementation**:

1. Extract the citation list from each ingested paper (available from Semantic Scholar or parsed from the PDF).
2. For each extracted claim, identify the papers that introduce the referenced entities.
3. Check whether those papers appear in the source paper's citation list.

**Scoring**:

```
citation_grounding(claim) = |cited_entity_papers ∩ source_citations| / |cited_entity_papers|
```

- `citation_grounding = 1.0`: All referenced entities' origin papers are cited. Strong grounding.
- `citation_grounding >= 0.5`: Most references present. Acceptable.
- `citation_grounding < 0.3`: Most entity-origin papers not cited. High suspicion of hallucination.

**Exception handling**: Review papers and surveys may reference methods without citing every origin paper. Weight this check lower for papers classified as surveys.

### 3.3 Multi-Extractor Consensus

**Principle**: Run the same paper through multiple extraction pipelines and accept only claims that appear in the majority. This is the LLM equivalent of error-correcting codes — redundant extraction reduces the probability of accepting a hallucinated claim.

**Implementation**:

1. For each paper, run *M* extraction passes using:
   - Different LLMs (e.g., Claude, GPT-4, Llama)
   - Same LLM with different temperatures (0.0, 0.3, 0.7)
   - Same LLM with different extraction prompts (structured, free-form, chain-of-thought)

2. Align claims across extractors using semantic similarity (SPECTER embeddings, cosine threshold > 0.80).

3. Accept a claim only if it appears in at least `ceil(M/2) + 1` extraction runs (strict majority).

**Error probability**: If each extractor independently hallucinates a specific claim with probability *p*, the probability that a strict majority of *M* extractors all hallucinate the same claim is:

```
P(false_consensus) = sum_{k=ceil(M/2)+1}^{M} C(M,k) * p^k * (1-p)^(M-k)
```

For *M* = 5 and *p* = 0.05 (5% hallucination rate per extractor), `P(false_consensus) ≈ 0.00003` — a 1600x reduction in hallucination pass-through rate.

**Cost optimization**: Running 5 extractors on every paper is expensive. Use a tiered approach:
- **Tier 1**: Single extraction for initial ingestion (fast, cheap).
- **Tier 2**: Triple extraction for claims that will be used as premises (triggered on demand).
- **Tier 3**: Five-way extraction for claims entering published outputs (rare, high stakes).

**Recent research support**: The Confidence-Informed Self-Consistency (CISC) method (2025) shows that adding self-assessment confidence scores to each extraction and using weighted majority vote outperforms naive majority voting, reducing required paths by 40%. The ICE (Iterative Consensus Enhancement) framework demonstrates that diminishing returns typically set in beyond 3-5 diverse extractors.

### 3.4 Semantic Embedding Consistency

**Principle**: A legitimately extracted claim should be semantically close to the source text from which it was extracted. If the claim's embedding is far from the source paragraph's embedding, the extraction likely hallucinated content not present in the text.

**Implementation**:

1. For each extracted claim, record the source paragraph (the text span from which it was extracted).
2. Compute embeddings using SPECTER or a domain-specific scientific embedding model:
   - `emb_claim` = embedding of the claim tuple rendered as text: "entity_1 relation entity_2 under conditions"
   - `emb_source` = embedding of the source paragraph
3. Compute cosine similarity: `sim = cos(emb_claim, emb_source)`.

**Thresholds**:
- `sim >= 0.75`: Consistent. Claim likely grounded in source text.
- `0.50 <= sim < 0.75`: Borderline. Flag for review.
- `sim < 0.50`: Inconsistent. High probability of hallucination.

**Critical caveat — "The Semantic Illusion"**: Recent research (arxiv:2512.15068) demonstrates that embedding-based detection has fundamental limits. Semantically plausible hallucinations can preserve high cosine similarity with source text while introducing factual errors invisible to embeddings. Cosine similarity catches only "lazy" hallucinations (Type 1 confabulations) and has unacceptable false-positive rates on sophisticated errors.

**Mitigation**: Never use embedding consistency as the sole verification method. Use it as one signal in an ensemble. Pair it with citation graph cross-validation and multi-extractor consensus for robust detection. Embedding consistency is a fast pre-filter, not a final arbiter.

### 3.5 Ground Truth Anchoring via Benchmarks

**Principle**: For claims about model performance ("X achieves 95.2% accuracy on ImageNet"), cross-check against known benchmark leaderboards that have verified results.

**Implementation**:

1. Identify performance claims during extraction (relation types: "achieves", "scores", "outperforms on").
2. Query Papers With Code API for the relevant benchmark leaderboard.
3. Compare the claimed performance value against the verified leaderboard entry.

**Verification rules**:
- If the claimed value matches the leaderboard entry (within tolerance): verified, high confidence.
- If the claimed value is within 2% of the leaderboard entry: acceptable, may reflect different evaluation protocol.
- If the claimed value differs by >5% from the leaderboard entry: flag as suspicious.
- If no leaderboard entry exists for the claimed model-benchmark pair: unverifiable via this method (not necessarily wrong).

**Scope**: This method only applies to quantitative performance claims. Qualitative claims ("X is more interpretable than Y") cannot be verified this way. Estimated coverage: 15-25% of all extracted claims in the ML domain.

---

## 4. Cascade Prevention Architecture

### 4.1 Four-Layer Defense

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 4: Continuous Monitoring & Rollback                      │
│  Background process. Periodic subgraph consistency checks.      │
│  Provenance tracking. Cascade impact analysis. Rollback.        │
└─────────────────────────────────────────────────────────────────┘
         ▲
┌────────┴────────────────────────────────────────────────────────┐
│  LAYER 3: Integration-Time Verification                         │
│  When a claim is used as premise for hypothesis or experiment.  │
│  Triggered on demand. Multi-extractor consensus. EICC check.   │
└─────────────────────────────────────────────────────────────────┘
         ▲
┌────────┴────────────────────────────────────────────────────────┐
│  LAYER 2: Post-Extraction Batch Verification                    │
│  After each ingestion batch. Entropy monitoring. KL drift.      │
│  Temporal consistency. Citation cross-validation.               │
└─────────────────────────────────────────────────────────────────┘
         ▲
┌────────┴────────────────────────────────────────────────────────┐
│  LAYER 1: Extraction-Time Verification                          │
│  During claim extraction. Semantic embedding consistency.       │
│  MDL anomaly scoring. Basic sanity checks.                      │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Layer 1: Extraction-Time Verification

**When**: During LLM claim extraction, before the claim enters the knowledge graph.

**Checks**:
1. **Semantic embedding consistency** (Section 3.4): `sim(claim, source) >= 0.50`.
2. **MDL anomaly score** (Section 2.4): `L(claim | M)` not in top 5th percentile.
3. **Basic structural sanity**:
   - Entities are non-empty strings.
   - Relation type is from the allowed vocabulary.
   - Confidence score is in [0, 1].
   - Conditions field is parseable.
4. **Duplicate detection**: Check if a semantically equivalent claim already exists (cosine similarity > 0.95 with existing claims).

**Outcome**: Claims that fail any check are tagged with a warning flag but still stored (with `status = "unverified"`). Claims that pass all checks enter as `status = "provisional"`.

**Latency budget**: <500ms per claim. All checks use pre-computed embeddings and cached graph statistics.

### 4.3 Layer 2: Post-Extraction Batch Verification

**When**: After each ingestion batch completes (typically after processing one paper or a group of related papers).

**Checks**:
1. **Shannon entropy monitoring** (Section 2.1): Compute `z_H(t)` for this batch.
2. **KL divergence drift** (Section 2.2): Compute `D_KL(P_t || P_{t-1})`.
3. **Temporal consistency** (Section 3.1): Verify all claims in the batch respect temporal ordering.
4. **Citation graph cross-validation** (Section 3.2): Verify `citation_grounding >= 0.3` for all claims.

**Outcome**: If batch-level anomalies are detected (entropy or KL thresholds exceeded), the entire batch is quarantined pending review. Individual claims that fail temporal or citation checks are downgraded to `status = "suspicious"`.

**Latency budget**: <30 seconds per batch. Semantic Scholar API calls are cached.

### 4.4 Layer 3: Integration-Time Verification

**When**: A claim is about to be used as a premise for hypothesis generation, experiment design, or synthesis.

**Checks**:
1. **Multi-extractor consensus** (Section 3.3): Run Tier 2 (triple extraction) if not already done. Require majority agreement.
2. **Effective independent corroboration count** (Section 2.3): `EICC >= 1.5`.
3. **Ground truth anchoring** (Section 3.5): If the claim is a performance claim, verify against benchmarks.
4. **Confidence propagation check** (Section 5): Ensure the derived conclusion's confidence does not exceed the framework's bounds.

**Outcome**: Claims that fail integration-time verification are not used as premises. The system must find alternative premises or flag the research direction as under-supported.

**Latency budget**: <5 minutes per claim (allows for multi-extractor runs).

### 4.5 Layer 4: Continuous Monitoring & Rollback

**When**: Background process running continuously.

**Components**:

#### 4.5.1 Provenance Tracking

Every claim in the knowledge graph stores:
```
{
  claim_id: uuid,
  source_paper_id: str,
  extraction_method: str,
  extraction_timestamp: datetime,
  extractor_model: str,
  extractor_prompt_hash: str,
  source_paragraph_hash: str,
  verification_status: enum("unverified", "provisional", "verified", "suspicious", "quarantined"),
  verification_history: [
    {timestamp, method, result, details}
  ],
  downstream_dependents: [claim_id, ...],  // claims derived from this one
  upstream_premises: [claim_id, ...],       // claims this one was derived from
  confidence: float,
  confidence_provenance: str  // how confidence was computed
}
```

#### 4.5.2 Cascade Impact Analysis

When a claim is flagged as suspicious or quarantined:

1. **Forward trace**: Traverse `downstream_dependents` recursively to find all claims that depend on the flagged claim.
2. **Impact score**: `impact = |downstream_dependents_transitive|` — the number of claims that would be affected if this claim is retracted.
3. **Confidence recalculation**: Recompute confidence for all downstream claims with the flagged claim removed from the premise set.

#### 4.5.3 Subgraph Consistency Audits

Run nightly:
1. Sample 100 random subgraphs of size 10-50 claims.
2. Check internal consistency: no contradictory claims, transitive relations hold, temporal ordering valid.
3. Flag inconsistent subgraphs for investigation.

#### 4.5.4 Quarantine and Rollback

**Quarantine**: A flagged claim and all its downstream dependents are marked as `quarantined`. Quarantined claims are excluded from all downstream reasoning but not deleted.

**Rollback procedure**:
1. Identify the root cause claim (the original hallucination).
2. Forward-trace all downstream dependents.
3. Mark all as quarantined.
4. Re-extract claims from the source paper using a different extractor.
5. If re-extraction confirms the original claim: restore from quarantine.
6. If re-extraction contradicts: delete the original claim and all dependents that were solely derived from it. Dependents with alternative support paths are recalculated.

**Soft delete**: Never hard-delete claims. Mark as `status = "retracted"` with a reason. This preserves audit trail and enables analysis of hallucination patterns over time.

---

## 5. Confidence Propagation Framework

### 5.1 The Problem

If claim A has confidence 0.9 and is used to derive hypothesis B, what is B's maximum confidence? Naive approaches (multiply confidences, take the minimum) either decay too aggressively or not enough. We need a principled framework.

### 5.2 Formal Model

Model the knowledge graph as a directed acyclic graph (DAG) of reasoning, where each node is a claim with a confidence score and edges represent "used as premise for" relationships.

**Rule 1: Conjunction decay.** If a conclusion *C* depends on premises *P_1, P_2, ..., P_n*, then:

```
conf(C) <= min(conf(P_1), conf(P_2), ..., conf(P_n)) * decay(n)
```

where `decay(n)` accounts for the risk of reasoning errors in combining *n* premises:

```
decay(n) = 1 / (1 + lambda * (n - 1))
```

with `lambda = 0.1`. This gives:
- 1 premise: decay = 1.0 (no penalty)
- 2 premises: decay = 0.91
- 5 premises: decay = 0.71
- 10 premises: decay = 0.53

**Rule 2: Chain attenuation.** For a chain of reasoning A → B → C → D, confidence attenuates at each step:

```
conf(D) <= conf(A) * gamma^depth
```

where `depth` is the chain length and `gamma = 0.95` is the per-step attenuation factor.

For a chain of depth 3 starting from confidence 0.9:
```
conf(D) <= 0.9 * 0.95^3 = 0.77
```

For depth 10:
```
conf(X) <= 0.9 * 0.95^10 = 0.54
```

This ensures that long chains of reasoning cannot maintain artificially high confidence.

**Rule 3: Corroboration boost.** If a claim is supported by *k* independent lines of evidence with confidences *c_1, c_2, ..., c_k*, the combined confidence is:

```
conf_combined = 1 - prod_{i=1}^{k} (1 - c_i * independence_i)
```

where `independence_i` is the effective independence score from Section 2.3 (0 to 1).

Example: Two independent sources with confidence 0.7 each:
```
conf_combined = 1 - (1 - 0.7)^2 = 1 - 0.09 = 0.91
```

Three independent sources with confidence 0.6 each:
```
conf_combined = 1 - (1 - 0.6)^3 = 1 - 0.064 = 0.936
```

**Rule 4: Hard ceiling.** No derived claim can have confidence > 0.95, regardless of how much evidence supports it. This reflects irreducible uncertainty in automated reasoning.

**Rule 5: Minimum threshold for use.** Claims below a confidence threshold cannot be used as premises:

| Usage | Min confidence |
|---|---|
| Stored in KG | 0.10 |
| Used in hypothesis generation | 0.50 |
| Used in experiment design | 0.70 |
| Cited in published output | 0.85 |

### 5.3 Confidence Recalculation

When any premise's confidence changes (due to new evidence, verification, or retraction), all downstream claims must have their confidence recalculated. This is implemented as a topological sort traversal of the reasoning DAG:

1. Sort claims in topological order (premises before conclusions).
2. For each claim, recompute confidence using Rules 1-4 with current premise confidences.
3. If any claim drops below the usage threshold for its current usage level, flag it for review.

**Performance**: Topological sort is O(V + E) where V = claims, E = dependency edges. For a graph of 100K claims, this completes in seconds.

### 5.4 Connection to Recent UQ Research

The framework aligns with recent work on uncertainty quantification in LLM reasoning chains (survey: arxiv:2503.15850, KDD 2025 tutorial). Key insights from that literature:

- **CoT-UQ** integrates chain-of-thought reasoning with response-level uncertainty, similar to our chain attenuation rule.
- **Tree of Uncertain Thoughts** extends Tree of Thoughts by quantifying uncertainties in intermediate reasoning steps using Monte Carlo Dropout.
- **UAG** monitors token probability at each generation step, dynamically retracting to more reliable states when high uncertainty is detected — analogous to our quarantine mechanism.
- The survey identifies **system-level uncertainty** as a critical gap: "errors in early steps lead to cascading failures, especially with misplaced confidence." Our framework directly addresses this with chain attenuation and hard ceilings.

---

## 6. Implementation Priority

### Phase 0 (immediate, low cost):
- Provenance tracking schema (Section 4.5.1)
- Confidence propagation framework (Section 5)
- Semantic embedding consistency check (Section 3.4)
- Basic structural sanity checks (Section 4.2)

### Phase 1 (first month, medium cost):
- Shannon entropy monitoring (Section 2.1)
- KL divergence drift detection (Section 2.2)
- Temporal consistency checking (Section 3.1)
- Quarantine and rollback mechanism (Section 4.5.4)

### Phase 2 (second month, higher cost):
- Citation graph cross-validation via Semantic Scholar (Section 3.2)
- MDL anomaly scoring (Section 2.4)
- Ground truth anchoring via Papers With Code (Section 3.5)

### Phase 3 (third month, highest cost):
- Multi-extractor consensus (Section 3.3)
- Mutual information source independence (Section 2.3)
- Continuous subgraph consistency audits (Section 4.5.3)

---

## 7. Key Metrics to Track

| Metric | Target | Measurement |
|---|---|---|
| Hallucination pass-through rate | <1% of claims | Sample-based human audit (monthly) |
| False quarantine rate | <5% of flagged claims | Review of quarantined claims that turn out correct |
| Cascade depth (max) | <5 levels | Monitor reasoning DAG depth |
| Mean confidence at depth 5 | <0.60 | Computed from confidence propagation |
| Verification latency (Layer 1) | <500ms | Instrumentation |
| Verification latency (Layer 3) | <5 min | Instrumentation |
| EICC distribution | Median >= 2.0 | Computed from source independence scores |

---

## 8. References and Sources

- Farquhar et al. (2024). "Detecting hallucinations in large language models using semantic entropy." *Nature*. [Link](https://www.nature.com/articles/s41586-024-07421-0)
- Kossen et al. (2024). "Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs." [arXiv:2406.15927](https://arxiv.org/abs/2406.15927)
- Halperin (2025). "Prompt-Response Semantic Divergence Metrics for Faithfulness Hallucination and Misalignment Detection." [arXiv:2508.10192](https://arxiv.org/html/2508.10192)
- Tonmoy et al. (2025). "Hallucination Detection and Mitigation in Large Language Models." [arXiv:2601.09929](https://arxiv.org/pdf/2601.09929)
- Bayat et al. (2025). "Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons." [arXiv:2512.05156](https://arxiv.org/abs/2512.05156)
- Xiao et al. (2025). "Uncertainty Quantification and Confidence Calibration in LLMs: A Survey." [arXiv:2503.15850](https://arxiv.org/abs/2503.15850)
- Niwa et al. (2025). "The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems." [arXiv:2512.15068](https://arxiv.org/abs/2512.15068)
- Manakul et al. (2025). "Confidence Improves Self-Consistency in LLMs." [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1030/)
- Alivanistos et al. (2024). "Knowledge Graphs, Large Language Models, and Hallucinations: An NLP Perspective." [arXiv:2411.14258](https://arxiv.org/abs/2411.14258)
- Amazon Science (2024). "GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework." [Link](https://www.amazon.science/publications/grapheval-a-knowledge-graph-based-llm-hallucination-evaluation-framework)
- Agrawal et al. (2024). "Can Knowledge Graphs Reduce Hallucinations in LLMs? A Survey." [NAACL 2024](https://aclanthology.org/2024.naacl-long.219.pdf)
- Ni et al. (2024). "Reliable Knowledge Graph Reasoning with Uncertainty Quantification." [ACM CIKM 2024](https://dl.acm.org/doi/pdf/10.1145/3627673.3680266)
- Schneider et al. (2024). "Anomaly Detection and Classification in Knowledge Graphs." [arXiv:2412.04780](https://arxiv.org/abs/2412.04780)
- RLKGE (2025). "Trustworthiness Measurement for Knowledge Graph Triples Based on Reinforcement Learning." [ACM Web Conference 2025](https://dl.acm.org/citation.cfm?id=3313586)
- KnowGraph (2024). "Knowledge-Enabled Anomaly Detection via Logical Reasoning on Graph Data." [ACM CCS 2024](https://dl.acm.org/doi/10.1145/3658644.3690354)
- KARMA framework. "Multi-agent LLMs for automated KG enrichment through entity discovery, relation extraction, and conflict resolution."
- Fact Verification in Knowledge Graphs Using LLMs (2025). [ACM SIGIR 2025](https://dl.acm.org/doi/10.1145/3726302.3730142)

---

*← Back to [Architecture and Roadmap](07-architecture-and-roadmap.md)*
