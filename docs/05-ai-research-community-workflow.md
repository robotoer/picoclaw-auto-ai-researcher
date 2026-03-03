# 05 — AI Research Community Workflow

> *If we want to automate scientific research, we first need to understand how it actually works.*

---

## 1. The Research Lifecycle

Scientific research is not a clean pipeline — it is a messy, iterative, socially embedded process. But for the purposes of designing an autonomous research agent, we can describe a canonical lifecycle:

```
Research Question
      │
      ▼
Literature Review ──► Identify Gaps / Contradictions
      │
      ▼
Hypothesis Formation
      │
      ▼
Experimental Design
      │
      ▼
Execution (Simulation / Computation / Data Analysis)
      │
      ▼
Result Interpretation
      │
      ▼
Manuscript Preparation
      │
      ▼
Peer Review ──► Revision → Revision → ...
      │
      ▼
Publication
      │
      ▼
Community Uptake (Citation, Replication, Extension, Critique)
      │
      ▼
[Feeds back into the research question of others]
```

Each stage has distinct information requirements, decision-making processes, and failure modes. An autonomous research system must be competent at all of them.

---

## 2. The Research Question — Where It All Begins

### 2.1 How Researchers Choose Questions

Research questions emerge from several sources:

- **Anomalies and surprises:** A result that doesn't fit existing theory demands explanation.
- **Gap identification:** Systematic literature review reveals unexplored intersections.
- **Technology pull:** A new experimental tool or compute capability enables questions that were previously intractable.
- **Societal demand:** Pressure from funders, industry, or society directs research toward specific problems.
- **Taste and aesthetics:** Researchers' intuitions about what's "interesting" or "beautiful" — difficult to formalize but highly influential.

### 2.2 Automating Question Generation

An autonomous research agent can generate research questions from:

1. **Contradiction mining:** Find papers that reach conflicting conclusions on the same question; the resolution is a research question.
2. **Combination search:** Find two sub-fields that share structural similarities but have not been connected; ask whether methods from one apply to the other.
3. **Failure mode analysis:** Find papers that report negative results or acknowledged limitations; ask why those limitations exist and whether they can be overcome.
4. **Trend extrapolation:** Identify accelerating research directions and ask what the next step is.
5. **Analogy transfer:** Find a solved problem in a related field and ask whether the solution transfers.

The quality of generated research questions can be estimated by: (a) how much community effort is already implicitly directed at the question, (b) the expected information gain from answering it, and (c) feasibility given available tools.

---

## 3. Literature Review: The Information Retrieval Problem

### 3.1 The Anatomy of a Literature Review

A thorough literature review does more than find relevant papers — it:

- Maps the intellectual landscape: who has worked on what, with what methods.
- Identifies competing theories and the evidence for each.
- Traces the history of ideas to understand where the field came from and why it moved in certain directions.
- Distinguishes "settled science" from "active debate" from "open territory."

### 3.2 Discovery Mechanisms

How do human researchers find papers?

- **Keyword search:** Google Scholar, Semantic Scholar, ArXiv search.
- **Citation chasing:** Forward (papers that cite X) and backward (papers that X cites).
- **Author tracking:** Follow the work of known experts.
- **Conference proceedings:** NeurIPS, ICML, ICLR, ACL proceedings as curated sets.
- **Social signals:** Twitter/X discussions, blog posts, recommendation from colleagues.
- **Alert services:** Semantic Scholar Alerts, ArXiv daily digest.

An autonomous research agent should replicate all of these. Citation chasing is particularly powerful and underutilized by simple keyword-search agents. The citation graph is a structured representation of intellectual influence that enables:

- **Co-citation analysis:** Papers frequently cited together tend to address the same problem.
- **Bibliographic coupling:** Papers that cite the same sources are likely related.
- **PageRank-like influence scores:** Identify foundational papers and key recent contributions.

### 3.3 The Semantic Gap Problem

Simple keyword search misses papers that use different terminology for the same concept. "Attention" in transformers is structurally related to "soft addressing" in memory networks and "value function approximation" in RL — but keyword search would not find these connections.

**Solutions:**
- Embedding-based semantic search (e.g., SPECTER embeddings for scientific papers from AllenAI).
- Concept normalization: map terms to a shared ontology (e.g., OBO Foundry for biology; CS ontologies are less mature).
- LLM-mediated query expansion: given a query, an LLM generates multiple alternative phrasings.

---

## 4. Community Mechanisms for Gap Discovery

### 4.1 Survey Papers and Benchmarks

Survey papers (by authors like Lilian Weng, Sebastian Ruder, and others in AI) serve as community gap-detection documents — they explicitly identify open problems. A research agent should:

1. Identify high-quality survey papers in its domains.
2. Extract the "open problems" and "future directions" sections.
3. Track whether those problems have been addressed in subsequent literature.
4. Prioritize unaddressed open problems.

Benchmarks serve a related function: a benchmark that no current method solves well is an explicit community invitation to improve.

### 4.2 Workshop Culture

Workshops at major venues (NeurIPS, ICML, ICLR) are where emerging topics get their first community airing before they are mature enough for main conference papers. Workshop papers and proceedings often contain the most forward-looking and speculative ideas.

An autonomous research agent should monitor workshop proceedings and identify recurring themes that haven't yet produced main-track papers — these are early indicators of emerging gaps.

### 4.3 Negative Results and Replication Failures

The research community systematically underpublishes negative results. Journals of Negative Results exist but are marginalized. This creates a "file drawer problem": many negative findings sit in researchers' notebooks, meaning the community keeps re-running failed experiments.

**Opportunity for an autonomous research agent:**
- Build a database of negative results from methods sections ("we tried X but it didn't work because...").
- When designing experiments, check this database to avoid known dead ends.
- Publish negative results that are novel and informative — this fills an underserved niche.

---

## 5. ArXiv, Citation Graphs, and Research Trend Detection

### 5.1 ArXiv as a Real-Time Research Observatory

ArXiv is not just a preprint server — it is a continuous signal about where the research community's attention is directed. Key analyses:

- **Submission volume by category over time:** Identifies explosive growth areas (e.g., the transformer wave from 2017-2020, the LLM fine-tuning explosion from 2022-2024).
- **Cross-category submissions:** Papers submitted to multiple categories indicate interdisciplinary convergence.
- **Author network evolution:** New collaborations between researchers from different sub-fields signal emerging bridges between communities.

### 5.2 Citation Graph Analysis

The citation graph is among the richest sources of structured research intelligence. Key analyses:

| Analysis | Method | Insight |
|---------|--------|---------|
| Betweenness centrality | Graph algorithms | Papers that bridge sub-fields |
| Co-citation clustering | Spectral clustering | Identifies research communities |
| Citation velocity | Time-series analysis | Early identification of "breakthrough" papers |
| Reference list overlap | Jaccard similarity | Near-duplicate research detection |
| Author collaboration graph | Community detection | Research team structures and alliances |
| Knowledge flow | Directed graph analysis | How ideas propagate between sub-fields |

### 5.3 Research Trend Detection Pipeline

An autonomous research agent should run a continuous trend detection pipeline:

```python
# Pseudocode for trend detection
for each day:
    new_papers = fetch_arxiv_daily_digest()
    
    for paper in new_papers:
        topics = extract_topics(paper)  # LDA, BERTopic, or LLM extraction
        update_topic_timeseries(topics)
    
    # Identify trending topics
    trending = [t for t in topics 
                if growth_rate(t, window=30) > threshold
                and absolute_volume(t) > min_papers]
    
    # Identify declining topics
    declining = [t for t in topics 
                 if growth_rate(t, window=90) < -threshold]
    
    # Identify emerging combinations
    new_combinations = find_new_topic_co-occurrences(new_papers, 
                                                      existing_combinations)
    
    # Update research agenda
    update_agenda(trending, declining, new_combinations)
```

### 5.4 The Hype Cycle and Research Investment Strategy

Gartner's Hype Cycle describes a common pattern: technology trigger → peak of inflated expectations → trough of disillusionment → slope of enlightenment → plateau of productivity. Research follows a similar pattern.

An autonomous research agent should recognize where different research directions are on this cycle:
- **Peak of inflated expectations:** Many incremental papers, fierce competition, diminishing returns per paper. Strategy: seek contrarian angles or wait for the trough.
- **Trough of disillusionment:** Few papers, previous results being questioned. Strategy: investigate the real limitations; this is where important negative results live.
- **Slope of enlightenment:** Realistic understanding of capabilities emerging. Strategy: connect with adjacent fields, build on the solid core.

---

## 6. Peer Review Simulation

### 6.1 The Function of Peer Review

Peer review serves multiple functions:
- **Quality filter:** Identify and reject fatally flawed work.
- **Improvement mechanism:** Reviews improve the quality of accepted work through revision.
- **Community norm enforcement:** Maintains standards around methodology, statistical rigor, and contribution clarity.
- **Credibility signal:** Accepted papers carry a (imperfect) quality signal.

### 6.2 Automated Peer Review

Research on automated peer review includes:
- **ASAP-Review (Yuan et al., 2021):** LLM trained to predict review scores and generate review text.
- **ReviewRobot (Liu et al., 2023):** Multi-aspect paper quality assessment.
- **OpenReview data:** A valuable resource — thousands of papers with multi-round reviews, author responses, and final decisions.

For an autonomous research agent, simulated peer review serves several functions:
1. **Pre-submission quality check:** Before "publishing" a result, run it through simulated reviewers and revise.
2. **Training signal:** Use acceptance/rejection predictions as part of the research quality reward model.
3. **Learning from rejection patterns:** Track which types of contributions consistently receive which types of critiques.

### 6.3 The Peer Review Simulation Loop

```
Draft output
      │
      ▼
Reviewer Agent 1 ──► "Methodology is unclear in section 3"
Reviewer Agent 2 ──► "Missing comparison with baseline X"
Reviewer Agent 3 ──► "Results don't support claim Y"
      │
      ▼
Author Agent responds to each review
      │
      ▼
Area Chair Agent makes accept/reject/revise decision
      │
      ▼
[If revise: loop back to Author Agent]
[If accept: proceed to publication]
[If reject: analyze reasons, decide to revise substantially or move to different venue]
```

---

## 7. Novel Proposals: Automated Gap-Filling, Trend-Following, and Counter-Narrative Research

### 7.1 Systematic Gap-Filling Strategy

Rather than waiting to stumble across gaps, an autonomous research agent can pursue **systematic gap mapping and filling**:

1. **Enumerate the space:** Define the axes of relevant variation in a research area (e.g., for RL: environment type × objective type × model architecture × learning algorithm).
2. **Map coverage:** Which combinations have been studied? (Citation graph + topic modeling).
3. **Rank uncovered regions:** Score by expected value = prior probability of positive result × scientific importance.
4. **Fill gaps sequentially:** Address uncovered regions in order of expected value.

This is analogous to how drug discovery uses systematic combinatorial chemistry — exhaustively exploring combinations rather than relying on serendipity.

### 7.2 Trend-Following with Contrarian Timing

A naive trend-following strategy — publish in the most popular areas — leads to crowded, low-impact work. A smarter strategy:

- **Follow emerging trends early** (when the field is growing fast but papers are few → high novelty, less competition).
- **Exit mature trends early** (when citation growth slows → diminishing returns).
- **Identify what the trend *implies* but hasn't yet addressed** — the second-order research questions that arise from a trending topic.

### 7.3 Counter-Narrative Research

Some of the most impactful research challenges prevailing assumptions. An autonomous research agent with enough context can systematically look for:

- **Widely-cited papers whose results don't replicate** (many of these exist in CV and NLP).
- **Consensus views that rest on a small number of papers** (a single study with high citation count is a fragile consensus).
- **Benchmarks that the community optimizes for but that don't correlate well with real-world performance** (a rich target: GLUE, ImageNet, etc. have known limitations).

Counter-narrative research is high-risk (reviewers resist challenges to consensus) but high-impact when successful. An autonomous agent is better positioned than a human researcher to pursue this: it doesn't face career risk from challenging established figures.

### 7.4 Cross-Pollination Research

Some of the most impactful AI papers are successful transfers of ideas across fields:
- Attention mechanism from NLP → Computer Vision (ViT).
- Variational inference from statistics → Deep learning (VAE).
- Contrastive learning from self-supervised vision → NLP (CLIP-like models).
- Actor-critic RL → LLM alignment (PPO in RLHF).

An autonomous research agent can systematically seek such transfers:
1. For each major technique in field A, ask: "Does field B have an analogous problem this technique might address?"
2. For each major unsolved problem in field C, ask: "Has this been solved in field D under different terminology?"

---

## 8. Modeling the Research Community as a Multi-Player Game

The research community is not a cooperative system optimizing for truth — it is a competitive ecosystem where researchers compete for credit, funding, and prestige. Understanding this is essential for an autonomous research agent that must interact with the community:

- **Priority disputes:** Being first matters enormously. An agent that is slow to publish will see its ideas claimed by others.
- **Citation norms:** Citing the "right" papers is politically as well as intellectually important.
- **Venue prestige hierarchy:** Publishing in NeurIPS/ICLR/ICML vs. ArXiv-only vs. second-tier conferences has very different impact.
- **Author prestige effects:** Papers from high-prestige labs receive more scrutiny and more citations regardless of quality (Matthew effect).

An autonomous research agent should model these dynamics to maximize the real-world impact of its outputs — not to game the system, but to ensure that genuinely good work is communicated in ways the community can receive and build upon.

---

*Next: [06 — Research Gap Filling](06-research-gap-filling.md)*
