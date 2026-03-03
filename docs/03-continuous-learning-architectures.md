# 03 — Continuous Learning Architectures

> *How does an autonomous research agent grow smarter from its own work without forgetting what it already knows?*

---

## 1. The Continuous Learning Imperative

A static model is a liability in a field that moves as fast as AI research. A model trained on papers through 2023 is already outdated about entire subfields that emerged or transformed in 2024. A research agent that cannot update its knowledge in real-time is not autonomous — it is an expensive search index.

Continuous learning (also called lifelong learning, incremental learning, or online learning) refers to the ability of a system to accumulate new knowledge over time without:
1. **Catastrophic forgetting:** Overwriting previously learned knowledge when learning new things.
2. **Plasticity-stability trade-off collapse:** Becoming either too rigid to learn new things or too fluid to retain old ones.
3. **Computational explosion:** Requiring a full retraining cycle for every new paper absorbed.

The state of the art in continuous learning is promising but not yet solved. This document surveys the landscape and proposes architectures suited to autonomous research agents.

---

## 2. Catastrophic Forgetting and Mitigation Strategies

Catastrophic forgetting (McCloskey & Cohen, 1989; French, 1999) is the tendency of neural networks to abruptly lose performance on previously learned tasks when trained on new data. The phenomenon arises because gradient descent updates weights globally — learning "arXiv paper A says X" may update the same weights that encoded "textbook chapter B says Y," corrupting the earlier knowledge.

### 2.1 Elastic Weight Consolidation (EWC)

EWC (Kirkpatrick et al., 2017 — DeepMind) addresses catastrophic forgetting by adding a quadratic penalty to the loss function that resists changes to weights that were important for previous tasks. The importance of each weight is estimated via the diagonal of the Fisher information matrix.

**Formula:** `L(θ) = L_new(θ) + Σ_i (λ/2) F_i (θ_i - θ*_i)²`

Where `F_i` is the Fisher information for parameter `i`, `θ*_i` is the value after the previous task, and `λ` controls the rigidity-plasticity tradeoff.

**Limitation for research agents:** EWC was designed for task sequences, not continuous streams. Computing Fisher information is expensive, and the approximation degrades when many tasks have been learned. For an agent absorbing thousands of papers, this doesn't scale directly.

**Adaptation:** Sparse EWC — only protect the top-k% most important weights, ignoring the rest. This reduces memory and compute while preserving the core benefit.

### 2.2 Progressive Neural Networks

Progressive Neural Networks (Rusu et al., 2016 — DeepMind) avoid forgetting by freezing previously learned columns of the network and adding new columns for new tasks, with lateral connections allowing the new columns to leverage existing features.

**Advantage:** Perfect retention of prior knowledge; new capabilities don't interfere with old ones.
**Disadvantage:** Network grows unboundedly with new tasks. For a research agent absorbing literature indefinitely, this is impractical without pruning/compression.

**Adaptation:** Combine progressive networks with periodic knowledge distillation (see Section 4) to compress accumulated columns back into a compact representation.

### 2.3 Experience Replay

Replay (Robins, 1995; Rolnick et al., 2019) maintains a buffer of past training examples and interleaves them with new training data. This prevents the optimizer from forgetting by continually rehearsing old knowledge alongside new.

**Variants:**
- **Random replay:** Sample uniformly from a fixed-size buffer.
- **Prioritized replay:** Oversample examples with high loss (i.e., those the current model finds most surprising).
- **Generative replay (Shin et al., 2017):** Train a generative model of past data; replay from the generator rather than storing raw examples. This is more memory-efficient.

**For research agents:** A generative replay buffer trained on absorbed paper abstracts can provide a continuous rehearsal signal. The generator is itself updated as new papers arrive, preventing the rehearsal distribution from becoming stale.

### 2.4 Memory-Augmented Networks

Differentiable External Memory (NTM — Graves et al., 2014; Memory Networks — Weston et al., 2015) externalizes long-term storage into a differentiable memory matrix. The network learns addressing operations that allow it to store and retrieve information without modifying its weights.

For a research agent, this is a compelling direction: the model's *weights* encode general reasoning capabilities, while domain-specific knowledge is stored in *external memory* that can be updated without any retraining. New papers are encoded and written to memory; retrieval at inference time grounds the agent's responses in current knowledge.

This is the theoretical foundation behind Retrieval-Augmented Generation (RAG), but with the key difference that a full differentiable memory allows end-to-end optimization of the write and read operations.

---

## 3. Online Learning from ArXiv

### 3.1 The ArXiv Fire Hose

ArXiv currently receives ~15,000 new submissions per month in cs.AI, cs.LG, and stat.ML alone. A research agent must decide:

1. **Which papers to read at all** (relevance filtering).
2. **How deeply to read each** (abstract only, section-level, full text).
3. **What to extract** (claims, methods, results, open problems).
4. **How to integrate with existing knowledge** (confirm, contradict, extend, or supersede).

### 3.2 Relevance Filtering with Active Learning

Rather than processing all papers indiscriminately, the agent maintains a relevance model that predicts the "expected epistemic value" of each paper given the current research agenda. Active learning principles apply: the agent should prioritize papers where its uncertainty about the content (given title/abstract) is highest, and where the topic intersects its current gaps.

**Implementation:**
- Fine-tune a text classifier on the agent's past reading history: papers flagged as "high-value" by the agent vs. those skimmed and discarded.
- Use uncertainty sampling: prefer papers where the classifier is least confident.
- Rebalance periodically to avoid the classifier becoming overly narrow.

### 3.3 Incremental Knowledge Graph Updates

As new papers are processed, the agent updates its knowledge graph:

1. **Entity extraction:** Identify named models, datasets, metrics, authors, institutions.
2. **Relation extraction:** Identify "outperforms," "is a variant of," "refutes," "is applied to."
3. **Conflict detection:** Flag new claims that contradict existing graph edges.
4. **Provenance tracking:** Every graph edge records its source paper(s) and confidence.

Conflict detection is critical: when a paper reports that method A outperforms method B on benchmark C, but the existing graph says the opposite, the agent should not silently overwrite — it should flag the contradiction, investigate the experimental conditions, and update its confidence appropriately.

### 3.4 Temporal Weighting

Knowledge has a shelf life. A result published in 2019 about the state-of-the-art on a benchmark may be entirely superseded by 2024. The agent should maintain temporal weight decay for facts: older facts have lower confidence weight, and the agent is more willing to revise them when new evidence arrives.

**Formal model:** Each knowledge edge has a half-life parameter `τ` that is topic-dependent. Facts about benchmark leaderboards decay fast (τ ~ 6 months). Facts about theoretical properties of algorithms decay slowly (τ ~ 5 years). The agent learns these half-life parameters from the empirical rate at which past facts are contradicted.

---

## 4. Knowledge Consolidation and Distillation

### 4.1 Why Consolidation is Necessary

An agent that simply accumulates raw information will become progressively slower and more expensive to query. Consolidation compresses accumulated knowledge into more efficient representations, analogous to how human sleep consolidates episodic memories into semantic knowledge (the hippocampal-neocortical consolidation hypothesis).

### 4.2 Knowledge Distillation for Compression

Knowledge distillation (Hinton et al., 2015) trains a compact student model to match the output distribution of a larger teacher. Applied to research agents:

1. **Teacher:** The full agent with its accumulated external memory, fine-tuned weights, and experience.
2. **Student:** A compact version that has internalized the most important patterns.
3. **Training signal:** The student is trained to reproduce the teacher's outputs on a representative set of research tasks.
4. **Deployment:** The student handles routine queries; the teacher (with its richer memory) handles complex tasks.

Periodic distillation prevents unbounded memory growth while preserving the most important learned patterns.

### 4.3 Concept Merging and Deduplication

The knowledge graph will accumulate many near-duplicate concepts (e.g., "large language model," "LLM," "foundation model," "GPT-style model"). A consolidation step periodically:

1. Clusters semantically similar nodes.
2. Merges them into a canonical concept.
3. Redirects edges from obsolete nodes to canonical ones.
4. Flags cases where merging is ambiguous for human review.

This is analogous to ontology alignment in knowledge engineering, but automated and continuously running.

### 4.4 Hypothesis Consolidation

Research hypotheses accumulated over time need to be periodically reviewed:

- Hypotheses confirmed by multiple independent experiments should be "promoted" to established facts with higher confidence.
- Contradicted hypotheses should be demoted or moved to a "falsified" store.
- Hypotheses that remain unverified for a long time should be flagged as "stale candidates" and prioritized for experimental testing.

---

## 5. Lifelong Learning Benchmarks and Evaluation

### 5.1 Standard Benchmarks

| Benchmark | Domain | What it Tests |
|-----------|--------|---------------|
| Split-MNIST / Split-CIFAR | Vision | Retention across task sequences |
| Permuted MNIST | Vision | Catastrophic forgetting |
| CORe50 (Lomonaco et al., 2017) | Vision | Continuous object recognition |
| CLEAR (Lin et al., 2021) | Vision + NLP | Temporal distribution shift |
| Dynabench | NLP | Dynamic adversarial benchmarks |
| StreamQA | NLP | Continual question answering over news |

### 5.2 Metrics for Research Agents

Standard benchmarks don't capture the research-specific requirements. Proposed metrics:

| Metric | Definition |
|--------|-----------|
| **Knowledge Retention Rate (KRR)** | Fraction of facts learned before time T that are correctly recalled at time T+Δ |
| **Novelty Sensitivity** | Ability to recognize and flag claims that contradict the current knowledge graph |
| **Update Efficiency** | Average compute cost per new fact integrated |
| **Contradiction Resolution Rate** | Fraction of detected contradictions correctly resolved (by investigating experimental conditions) |
| **Knowledge Age Distribution** | Distribution of ages of facts in the graph — a healthy agent has a mix of old stable knowledge and fresh recent knowledge |
| **Forward Transfer** | Does learning topic A improve performance on topic B? |
| **Backward Transfer** | Does learning new things about topic A improve recall of old facts about topic A? |

---

## 6. Novel Proposal: Self-Directed Curriculum for Research Topics

### 6.1 The Curriculum Problem

A human PhD student doesn't learn everything at random order — they work through foundational material first, then build toward specialized knowledge, guided by a supervisor and their own sense of what's important. A research agent needs an analogous curriculum.

The challenge is that the curriculum itself should be emergent: the agent should determine what to learn next based on its current state, not follow a fixed syllabus.

### 6.2 Zone of Proximal Development (ZPD) Curriculum

Inspired by Vygotsky's ZPD, the agent should prioritize learning material that is:
- **Not already mastered** (the agent's confidence on related topics is high → skip).
- **Not too far beyond current capability** (the agent has zero context for the topic → defer until prerequisites are learned).
- **At the productive edge** where existing knowledge provides scaffolding for new learning.

**Implementation:**
1. Maintain a topic graph with nodes labeled by current competence level (0–1 scale).
2. For each candidate paper or topic, estimate: (a) prerequisite overlap with current knowledge, (b) expected competence gain from studying it.
3. Prioritize topics where prerequisite overlap is 40–80% (not too easy, not too hard) weighted by importance to the research agenda.

### 6.3 Research Importance Weighting

Not all topics deserve equal attention. The curriculum is weighted by:

- **Field momentum:** Topics with accelerating publication rates deserve more attention.
- **Gap density:** Topics underrepresented in current knowledge but important to the research agenda.
- **Strategic value:** Topics that are prerequisites for many downstream research directions.

### 6.4 Curriculum as Exploration Policy

Formalizing this as an RL problem:
- **State:** Current knowledge graph with competence labels.
- **Action:** Choose the next topic/paper to study.
- **Reward:** Knowledge gain measured by improvement on downstream tasks (e.g., can the agent now generate better hypotheses? Solve harder problems?).

The curriculum policy is trained via meta-RL: episodes consist of a learning phase (agent studies a curriculum) followed by an evaluation phase (agent is tested on a held-out set of research tasks). The curriculum policy is updated to maximize evaluation performance.

---

## 7. Architectural Recommendations

For an autonomous research agent, we recommend a **hybrid architecture** that combines:

| Component | Technology | Role |
|-----------|-----------|------|
| Fast retrieval layer | RAG + vector DB (Qdrant) | Ground responses in current literature |
| Parametric knowledge | Fine-tuned LLM | General reasoning + deeply absorbed knowledge |
| Episodic memory | Structured log with temporal metadata | Recent research history |
| Knowledge graph | Neo4j + GNN encoder | Concept relationships, provenance |
| Consolidation service | Runs nightly | Merges, deduplicates, distills |
| Curriculum planner | RL policy | Directs learning toward productive topics |

This architecture separates concerns cleanly: updating the retrieval layer is cheap (add documents, re-embed); updating the knowledge graph is moderate (extract entities, reconcile); updating the parametric weights is expensive (fine-tuning) and should happen only periodically when a large body of new knowledge has accumulated.

---

*Next: [04 — Mixture of Agents Systems](04-mixture-of-agents-systems.md)*
