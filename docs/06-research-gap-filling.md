# 06 — Research Gap Filling

> *How do we systematically map what we don't know, and build an agent that turns white space into knowledge?*

---

## 1. What Constitutes a Research Gap?

A "research gap" is not simply a topic nobody has studied. The concept is richer and multi-dimensional:

### 1.1 Types of Research Gaps

| Gap Type | Description | Example |
|----------|-------------|---------|
| **Empirical gap** | A question has been theorized about but not tested | "Nobody has systematically measured how much context window length affects planning quality in agentic tasks" |
| **Population gap** | A phenomenon studied in one context hasn't been studied in another | "RL from human feedback studied for text; not for formal mathematical reasoning" |
| **Methodological gap** | Better methods exist in a related field that haven't been applied here | "Bayesian experimental design widely used in drug discovery; rarely in neural architecture search" |
| **Contradictory findings gap** | Two studies reach conflicting conclusions; the discrepancy is unexplained | "Study A finds scaling laws hold; Study B finds they plateau — the conditions differ but nobody has studied which conditions determine which outcome" |
| **Negative results gap** | An expected relationship doesn't hold; nobody has explained why | "Large models don't improve on certain types of logical reasoning — why not?" |
| **Combination gap** | Two well-studied techniques haven't been combined | "Mixture of Experts architectures + Constitutional AI has very little published work" |
| **Operationalization gap** | A concept is widely discussed but poorly operationalized | "'Alignment' is discussed everywhere but has many incompatible formal definitions" |
| **Replication gap** | Influential results that haven't been independently replicated | "Many RLHF results published with proprietary data and no public replication" |
| **Scale gap** | Results established at one scale unknown to hold at another | "Token efficiency of sparse transformers studied at 7B; not at 70B+" |

### 1.2 The "Gap Quality" Spectrum

Not all gaps are equally worth filling. Gap quality depends on:

1. **Importance:** How much does resolving this gap matter for downstream science or applications?
2. **Tractability:** Can the gap be filled given available tools and resources?
3. **Novelty:** Has this exact question been addressed under different terminology (the semantic gap problem)?
4. **Timeliness:** Is the gap likely to be filled by others soon regardless?
5. **Foundational impact:** Does filling this gap unlock other gaps?

An ideal gap target scores high on all five. A well-designed autonomous researcher should focus on "high leverage" gaps — those where a single clear result reshapes the landscape around it.

---

## 2. Methods for Gap Identification

### 2.1 Citation Analysis

Citation patterns reveal gaps implicitly. The absence of citations between two sets of papers that should be connected is itself informative.

**Techniques:**

**Bibliographic coupling:** Two papers A and B are bibliographically coupled if they cite many of the same papers (they address the same problem). A cluster of highly coupled papers with few connections to another cluster suggests a potential bridge gap.

**Structural holes (Burt, 1992):** In social network theory, structural holes are positions in a network where no one currently bridges two groups. In citation networks, structural holes between research communities represent opportunities for cross-pollination research (see [05](05-ai-research-community-workflow.md)).

**Co-citation temporal analysis:** If papers A and B were frequently co-cited from 2019-2021 but the co-citation rate dropped, something resolved the connection or the two directions diverged — both are interesting signals.

**Reference list analysis for completeness:** Given paper X on topic T, and a known set of important papers on T, check which important papers X doesn't cite. This identifies gaps in the author's knowledge (and by extension, in the community's cross-pollination).

### 2.2 Topic Modeling for Gap Detection

Topic models (LDA, NMF, BERTopic) applied to large paper corpora produce a "topic landscape" where each topic is a cluster of semantically related papers. Gaps appear as:

- **Underrepresented topic combinations:** Topics that often co-appear in the abstract but rarely have dedicated papers.
- **Topic deserts:** Regions of the embedding space with low paper density but high logical proximity to populated areas.
- **Temporal voids:** Topics that were heavily researched, then abandoned, but the open questions from that era remain unresolved.

**BERTopic** is particularly useful because it produces coherent, human-interpretable topic labels, enabling the agent to reason about topic gaps at the semantic level.

### 2.3 Negative Results Databases

Existing resources:
- **PLOS ONE** has a "publication bias" mandate to publish negative results.
- **F1000Research** publishes null results.
- **The Journal of Negative Results in Biomedicine** (dormant but archived).
- **arXiv "negative results" papers** (identifiable via keyword extraction).

For AI specifically, negative results are often buried in appendices, ablation studies, or discussion sections. An agent that systematically extracts "we tried X but it didn't work" statements from full paper text would accumulate a valuable negative results database.

**Using this database:**
- Before proposing an experiment, check whether it (or a near-equivalent) has already failed.
- Treat persistent failures on a specific approach as evidence that the approach is fundamentally limited in that context.
- Identify cases where multiple groups independently failed at the same thing — this is a genuine hard problem worth investigating.

### 2.4 Expert Elicitation at Scale

Human researchers have intuitive knowledge about what's missing in their field that isn't captured in published papers. Large-scale surveys (like the "AI Research Desiderata" exercises in various communities) reveal this tacit knowledge.

An autonomous agent can systematically extract this knowledge by:
- Mining "open problems" sections in survey papers.
- Parsing workshop call-for-papers (researchers write these to describe what they want to see).
- Extracting "future work" sections from published papers.
- Parsing grant applications (less accessible but highly informative when available).

---

## 3. Identifying Promising Unexplored Combinations

### 3.1 The Combination Space Framework

Many productive research results come from combining existing ideas in new ways. The combination space can be represented as a multi-dimensional tensor:

```
Combination(technique_A, technique_B, ..., domain, objective, evaluation)
```

For example: `Combination(sparse attention, curriculum learning, multi-step RL, atari, sample efficiency, data efficiency metrics)` — has this been studied? If not, and if theoretical reasons suggest it should work, it's a gap worth filling.

The size of this space is enormous (combinatorial), so the agent must be intelligent about which combinations to prioritize.

### 3.2 Compatibility Scoring

Not all technique combinations are compatible. A compatibility score can be estimated by:

1. **Mechanistic compatibility:** Do the mechanisms of techniques A and B interact well? (e.g., techniques that both modify the loss function may interfere with each other).
2. **Empirical compatibility:** Find papers where A and B were used together (even in different contexts) and whether that combination worked.
3. **Theoretical compatibility:** Is there a reason A and B should be complementary? (e.g., one reduces variance while the other reduces bias).

High compatibility score + low existing coverage = high-priority combination gap.

### 3.3 "First-Order Neighbor" Exploration

A productive heuristic for gap finding: take any well-established result and ask "what is the nearest unexplored question?" This systematically generates incremental but valuable research:

- If "technique A improves metric M on dataset D" is established:
  - Does technique A improve M on dataset D' (different distribution)?
  - Does technique A improve metric M' (related but different metric)?
  - Does technique B (analogous to A) also improve M?
  - Why does A improve M? (mechanism investigation)
  - Under what conditions does A fail to improve M?

These "first-order neighbors" are manageable research projects that together fill in the space around any significant finding.

---

## 4. Evaluation Metrics for Novelty and Usefulness

### 4.1 Novelty Metrics

**Semantic novelty:** Distance in embedding space from the k-nearest existing papers. Computed using SPECTER or SciBERT embeddings. A result is novel if its core claim vector is far from all existing paper vectors.

**Combinatorial novelty:** The first time a specific combination of techniques/concepts appears in the literature. Can be measured as: is this combination present in the n-gram or concept co-occurrence statistics of the existing corpus?

**Temporal novelty:** A result that would have been publishable (if it existed) 3 years ago but wasn't published. This captures "slow gaps" — ideas that were technically feasible but weren't pursued.

**Surprise score:** How much does the result violate predictions made by existing models/theories? (See the compression-progress formulation in [02](02-reinforcement-learning-profiles.md).)

### 4.2 Usefulness Metrics

**Downstream citability:** Will researchers in adjacent areas be able to build on this? Estimated by: how many papers in adjacent areas could have cited this result (if it existed) based on their reference lists?

**Engineering utility:** Does this result translate into a specific improvement in system performance? Estimated by: does the paper include an artifact (code, dataset, model) that others can directly use?

**Problem resolution rate:** What fraction of the outstanding questions in a sub-field does this result answer? A result that answers the central open question is more useful than one that answers a side question.

**Pedagogical clarity:** Does this result simplify the understanding of an existing phenomenon? A cleaner proof or a more interpretable explanation is useful even if the underlying claim was already known.

### 4.3 The Novelty-Usefulness Tradeoff

There is a fundamental tension: highly novel results are often less immediately useful (they open new directions rather than closing existing ones), while highly useful results are often less novel (they apply established methods to new domains).

The ideal research portfolio balances:
- **Breakthrough work:** High novelty, uncertain usefulness, potentially transformative.
- **Development work:** Moderate novelty, high usefulness, clearly valuable.
- **Application work:** Low novelty, immediate utility, important for adoption.

An autonomous research agent should maintain a portfolio across all three types, calibrating the balance based on the research agenda.

---

## 5. Novel Proposals

### 5.1 The Gap Map of the Intelligence Space

**Core Proposal:** Build and continuously maintain a comprehensive "Gap Map" — a structured topological representation of the intelligence research space that makes gaps explicit, navigable, and actionable.

**Structure:**
- **Nodes:** Concepts, techniques, datasets, benchmarks, research questions, hypotheses.
- **Edges (positive):** "builds on," "improves upon," "is evaluated on," "requires," "enables."
- **Edges (negative — the key innovation):** "should connect but doesn't," "has been tried and failed (with conditions)," "is theoretically related but empirically unexplored."
- **Regions:** Clusters of densely-covered territory (mature sub-fields) and sparse territory (emerging areas).
- **Frontiers:** The boundary between well-covered and sparse regions — the most productive place for new work.

**Maintenance pipeline:**
```
Daily ArXiv digest
    │
    ▼
Entity & relation extraction (new nodes & positive edges)
    │
    ▼
Contradiction detection (new negative edges: "refutes")
    │
    ▼
Coverage density update (which regions just got denser)
    │
    ▼
Frontier recalculation (which regions are now at the boundary)
    │
    ▼
Gap ranking update (expected value of filling each gap)
    │
    ▼
Research agenda notification (agent selects next gap to fill)
```

**Visualization:** The Gap Map can be rendered as an interactive graph where node size represents the volume of coverage, edge color represents relationship type, and red regions represent identified gaps. This visualization is itself a research contribution — it makes the structure of the field legible.

### 5.2 The Gap Filling Protocol

Once a gap is identified and selected, the agent follows a systematic protocol:

```
1. CHARACTERIZE the gap precisely
   - What is the question being asked?
   - What do we know about the adjacent territory?
   - What are the leading hypotheses?
   - What would count as filling the gap?

2. ASSESS tractability
   - What experiments would answer the question?
   - What data/compute is required?
   - Are there existing datasets, code, or models to leverage?

3. DESIGN the minimal filling experiment
   - What is the smallest experiment that provides clear signal?
   - What are the conditions, controls, and metrics?

4. EXECUTE (or simulate execution)
   - Run the experiment using available tools.
   - If full execution is not possible, run proxy experiments.

5. EVALUATE the result
   - Does the result clearly fill the gap?
   - What new gaps does it open?
   - What are the limitations and caveats?

6. PUBLISH and update the Gap Map
   - Write up the result.
   - Update the Gap Map with the new coverage.
   - Identify and log the newly opened gaps.
```

### 5.3 Systematic Exploration Strategies for the Intelligence Space

Beyond reactive gap-filling (responding to identified gaps), the agent should pursue **proactive exploration** of the intelligence space:

**Strategy 1 — Grid Search over Technique Combinations:**
Define a grid of independent axes (e.g., architecture × training objective × data regime × evaluation task). Systematically fill in the grid, prioritizing intersections with high theoretical interest and low existing coverage.

**Strategy 2 — Perturbation Analysis:**
Take any published result and systematically perturb its conditions: change the dataset, the model size, the evaluation metric, the training regime. Which perturbations preserve the result? Which destroy it? The conditions that matter are the most informative.

**Strategy 3 — Boundary Pushing:**
For every capability that a model is reported to lack, design the minimal experiment that tests whether the capability exists under different conditions. Many "LLMs can't do X" results turn out to be "LLMs can't do X with vanilla prompting" — the capability exists but requires specific conditions.

**Strategy 4 — Historical Dead Ends Revisited:**
Many research directions were abandoned not because they were fundamentally limited, but because they were computationally intractable at the time, or because the field's attention moved elsewhere before they were fully explored. With dramatically increased compute and improved tools, these deserve revisitation.

---

## 6. Operationalizing the Gap Map

### 6.1 Data Sources

| Source | Information | Update Frequency |
|--------|-------------|-----------------|
| ArXiv CS/ML | New papers, new claims | Daily |
| Semantic Scholar | Citation counts, author networks | Weekly |
| Papers With Code | Benchmark results, code links | Daily |
| OpenReview | Peer review data, reject reasons | Per venue cycle |
| Hugging Face Hub | Model/dataset adoption | Daily |
| GitHub starred repos | Community interest signals | Weekly |
| Twitter/X ML community | Early trend signals, discussions | Real-time |

### 6.2 Gap Map Schema (Simplified)

```json
{
  "node": {
    "id": "sparse_attention_rl",
    "type": "technique_combination",
    "coverage_score": 0.12,
    "last_updated": "2024-11-15",
    "adjacent_nodes": ["sparse_attention", "sample_efficiency_rl"],
    "gaps": [
      {
        "gap_id": "gap_sparse_attn_rl_curriculum",
        "description": "Effect of sparse attention on curriculum learning efficiency in RL",
        "estimated_importance": 0.7,
        "estimated_tractability": 0.8,
        "status": "open"
      }
    ]
  }
}
```

### 6.3 Gap Prioritization Algorithm

```python
def gap_priority_score(gap):
    importance = estimate_downstream_impact(gap)
    tractability = estimate_experiment_feasibility(gap)
    novelty = 1.0 - estimate_existing_coverage(gap)
    urgency = estimate_time_to_being_filled_by_others(gap)
    
    return (importance * tractability * novelty) / (1 + urgency)
    # High urgency discounts priority — if others are about to publish this, 
    # either move fast or let them and focus elsewhere
```

---

*Next: [07 — Architecture and Roadmap](07-architecture-and-roadmap.md)*
