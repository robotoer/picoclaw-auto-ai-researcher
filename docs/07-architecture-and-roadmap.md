# 07 — Architecture and Roadmap

> *How do all the pieces fit together, and what does the path to full autonomy look like?*

---

## 1. System Overview

The picoclaw-auto-ai-researcher is a multi-agent, continuously-learning system for automated scientific research. It ingests the scientific literature in real time, identifies research opportunities, designs and runs experiments (initially simulated, eventually real), synthesizes results, and produces research outputs that the community can build upon.

The system is designed around three core loops:

```
┌─────────────────────────────────────────────────────────────────┐
│  OUTER LOOP (Weekly/Monthly): Strategic Research Planning       │
│  Select research directions, update the Gap Map,                │
│  allocate resources across sub-projects                         │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│  MIDDLE LOOP (Daily): Research Execution                        │
│  Run experiments, integrate literature, generate hypotheses,    │
│  draft outputs, run simulated peer review                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│  INNER LOOP (Real-time): Knowledge Ingestion                    │
│  Process new papers, update knowledge graph,                    │
│  detect contradictions, update Gap Map                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Diagram

```
                        ┌──────────────────────────────────────┐
                        │           ORCHESTRATOR               │
                        │   (Strategic planning, task routing, │
                        │    resource management)              │
                        └──────────┬───────────────────────────┘
                                   │
          ┌────────────────────────┼──────────────────────────┐
          │                        │                          │
          ▼                        ▼                          ▼
┌─────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  INGESTION &    │    │  RESEARCH         │    │  OUTPUT &         │
│  KNOWLEDGE      │    │  GENERATION       │    │  EVALUATION       │
│  LAYER          │    │  LAYER            │    │  LAYER            │
│                 │    │                   │    │                   │
│ ArXiv Monitor   │    │ Hypothesis Gen.   │    │ Science Comm.     │
│ PDF Extractor   │    │ Experiment Design │    │ Peer Review Sim.  │
│ Claim Extractor │    │ Literature Analyst│    │ Impact Predictor  │
│ KG Updater      │    │ Synthesizer       │    │ Publication Mgr.  │
│ Trend Detector  │    │ Critic            │    │ Citation Tracker  │
└────────┬────────┘    └────────┬──────────┘    └────────┬──────────┘
         │                      │                        │
         ▼                      ▼                        ▼
┌────────────────────────────────────────────────────────────────┐
│                    SHARED INFRASTRUCTURE                       │
│                                                                │
│  Knowledge Graph (Neo4j)     │  Vector Store (Qdrant)         │
│  Episodic Memory (log + RAG) │  Gap Map (custom graph DB)     │
│  Reward Model(s)             │  Experiment Runner (sandbox)   │
│  Model Registry              │  Curriculum Planner            │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipelines

### 3.1 Ingestion Pipeline

```
Source (ArXiv API, Semantic Scholar, Papers With Code)
    │
    ▼
Relevance Filter (fine-tuned classifier → score 0-1)
    │
    ├── score < 0.3 → discard (log title for trend statistics)
    ├── score 0.3-0.7 → abstract-level processing
    └── score > 0.7 → full paper processing
                          │
                          ▼
              PDF/LaTeX extraction
              (tables, figures, code, equations preserved)
                          │
                          ▼
              Structured claim extraction
              (entity, relation, entity, confidence, conditions)
                          │
                          ▼
              Knowledge Graph update
              (add new nodes/edges, flag contradictions)
                          │
                          ▼
              Gap Map update
              (adjust coverage scores, recalculate frontiers)
                          │
                          ▼
              Episodic memory write
              (store paper summary with metadata)
```

**Key design decisions:**
- Relevance scoring is adaptive: as the research agenda changes, the relevance model is retrained.
- Full-paper processing is compute-intensive; the 0.7 threshold is a knob that trades coverage for cost.
- Contradiction detection is critical and runs on every new fact inserted: new claim vs. existing claims → compute semantic similarity → flag conflicts for investigation.

### 3.2 Research Generation Pipeline

```
Gap Map query → ranked list of research opportunities
    │
    ▼
Opportunity selector (Orchestrator applies IWPG scoring)
    │
    ▼
Research thread initialization
    │
    ├── Literature Analyst: build knowledge context for this gap
    │
    ├── Hypothesis Generator: generate candidate hypotheses
    │
    ├── Critic: attack each hypothesis (2-3 rounds of debate)
    │
    └── Surviving hypotheses → Experiment Designer
                                        │
                                        ▼
                              Experiment execution
                              (code sandbox, simulation,
                               or proxy experiment design)
                                        │
                                        ▼
                              Result interpretation
                              (Statistician Agent)
                                        │
                                        ▼
                              Gap Map update + Synthesizer
                                        │
                                        ▼
                              Science Communicator → Draft output
                                        │
                                        ▼
                              Simulated peer review (3 agents)
                                        │
                              ┌─────────┴──────────┐
                              │                    │
                          Accept                Revise
                              │                    │
                              ▼                    └──► Revision loop
                        Publish output                  (max 3 rounds)
```

### 3.3 Learning Pipeline

```
Published outputs + community feedback
    │
    ▼
Reward signal computation (IWPG formula)
    │
    ▼
Reward model update (periodically, using new (output, score) pairs)
    │
    ▼
Policy update (RL fine-tuning of orchestration policy)
    │
    ▼
Curriculum planner update (adjust topic weights)
    │
    ▼
Knowledge consolidation (nightly):
    ├── Knowledge graph deduplication
    ├── Confidence decay for stale facts
    ├── Student model distillation
    └── Hypothesis promotion/demotion
```

---

## 4. The "Interesting" Metric — How to Define and Optimize for It

### 4.1 Why "Interesting" is Hard to Define

"Interesting" is ultimately a social phenomenon — it depends on the audience, the context, and the historical moment. A result that is interesting in 2024 may be trivial in 2026. A result that is interesting to RL researchers may be irrelevant to NLP researchers.

Nevertheless, we can decompose "interesting" into operationalizable components:

### 4.2 The SUNFIRE Framework

We propose the **SUNFIRE** framework for operationalizing research interestingness:

| Dimension | Symbol | Description | Proxy Metric |
|-----------|--------|-------------|--------------|
| **Surprise** | S | Violates existing expectations | Prediction error of prior belief model on key claims |
| **Usefulness** | U | Enables downstream work | Estimated forward-citation potential |
| **Novelty** | N | Hasn't been said before | Semantic distance from existing literature |
| **Feasibility** | F | Can be reproduced by others | Code availability + required compute estimate |
| **Impact breadth** | I | Affects many sub-fields | Number of adjacent communities that could use this |
| **Rigor** | R | Well-supported conclusions | Experimental controls, statistical power, ablations |
| **Elegance** | E | Parsimonious explanation | Complexity of the explanation vs. amount it explains |

**Composite score:**
```
Interesting(output) = w_S·S + w_U·U + w_N·N + w_F·F + w_I·I + w_R·R + w_E·E
```

Where weights are learned via meta-RL (outer loop) by observing which outputs receive the best community reception over time.

### 4.3 Anti-Gaming Safeguards

The system must be protected against optimizing for proxy metrics at the expense of the true objective:

- **Novelty hacking:** Generating genuinely new but meaningless combinations of words. *Safeguard:* Novelty must be paired with a non-trivial feasibility score; purely linguistic novelty doesn't count.
- **Surprise manipulation:** Making false claims to maximize prediction error. *Safeguard:* Claims must be grounded to verifiable evidence; ungrounded claims receive zero surprise credit.
- **Community gaming:** Flooding the community with low-quality outputs to inflate citation counts. *Safeguard:* Quality filters and rate limits on output publication.

---

## 5. Proposed Phases and Milestones

### Phase 0: Foundation (Months 1–3)

**Goal:** Basic infrastructure, no autonomy yet.

**Deliverables:**
- [ ] ArXiv ingestion pipeline (daily digest → structured extraction)
- [ ] Knowledge graph schema and initial seed population
- [ ] Vector store with SPECTER embeddings of ArXiv CS.AI/CS.LG corpus
- [ ] Gap Map v0.1 (topic clusters + basic coverage scoring)
- [ ] Literature Analyst agent (can answer questions about a paper corpus)
- [ ] Basic evaluation suite for research output quality

**Success Criteria:** System can accurately answer questions about recent AI papers better than a vanilla GPT-4 query.

---

### Phase 1: Hypothesis Generation (Months 4–6)

**Goal:** Agent can generate and evaluate research hypotheses without executing experiments.

**Deliverables:**
- [ ] Hypothesis Generator agent with structured output format
- [ ] Critic agent with adversarial debate loop
- [ ] Simulated peer review (3-agent panel)
- [ ] SUNFIRE reward model v0.1
- [ ] Curriculum planner v0.1 (topic selection)
- [ ] First 50 published research hypotheses with documentation

**Success Criteria:** A sample of 10 generated hypotheses is rated by human researchers as "plausible and interesting" at rate ≥ 60%.

---

### Phase 2: Experiment Execution (Months 7–12)

**Goal:** Agent can design and run computational experiments to test its hypotheses.

**Deliverables:**
- [ ] Secure code execution sandbox (sandboxed Python + ML libraries)
- [ ] Experiment Designer agent
- [ ] Integration with Hugging Face Hub (model loading, dataset access)
- [ ] Statistician agent (result interpretation, statistical testing)
- [ ] Automated experiment logging and reproducibility packaging
- [ ] First 10 complete research outputs (hypothesis → experiment → result → write-up)

**Success Criteria:** At least 3 of 10 first research outputs are considered "publishable quality" by independent human review. At least 1 is submitted to a venue.

---

### Phase 3: Continuous Learning Loop (Months 13–18)

**Goal:** System improves its research capability from its own outputs.

**Deliverables:**
- [ ] Reward model training from community feedback (citations, reviews)
- [ ] RL fine-tuning pipeline for orchestration policy
- [ ] Continuous learning architecture (EWC + replay + periodic distillation)
- [ ] Self-directed curriculum with ZPD-based topic selection
- [ ] Gap Map v1.0 (full intelligence-space topology)
- [ ] Meta-learning for rapid adaptation to new sub-fields

**Success Criteria:** System's output quality improves measurably (SUNFIRE score) over a 3-month RL training window. System identifies and fills at least 5 "genuine" research gaps (confirmed by domain experts).

---

### Phase 4: Multi-Agent Ensemble (Months 19–24)

**Goal:** Full multi-agent architecture with specialization and emergent collaboration.

**Deliverables:**
- [ ] All specialized agents (Literature Analyst, Hypothesis Generator, Experiment Designer, Critic, Synthesizer, Science Communicator)
- [ ] Agent orchestration framework with conflict resolution
- [ ] Modular expert spawning (auto-create new specialists for new sub-fields)
- [ ] Cross-domain Synthesizer producing genuine interdisciplinary insights
- [ ] Published peer-reviewed paper from the autonomous system

**Success Criteria:** System publishes at least one peer-reviewed paper at a recognized venue, with the autonomous system listed as a co-author (or the paper is accepted before disclosure of its origin, establishing blind quality).

---

### Phase 5: Full Autonomy (Month 24+)

**Goal:** System operates with minimal human intervention, producing a steady stream of research outputs, continuously expanding the Gap Map, and improving its own capabilities.

**Long-term milestones:**
- System identifies a paradigm-shifting research direction before the human community recognizes it.
- System produces a collaborative research program with human researchers in a specific sub-field.
- System's outputs are cited by human researchers without knowing the source was autonomous.

---

## 6. Risk Analysis and Safeguards

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hallucination cascade (false claims propagate through KG) | High | High | Every claim tagged with source; confidence decay; adversarial critique |
| Reward hacking (gaming SUNFIRE metrics) | Medium | High | Regular human audits; adversarial red-teaming of reward model |
| Catastrophic forgetting | Medium | Medium | EWC + replay + distillation architecture |
| Compounding planning errors | High | Medium | Checkpoint and rollback; human review of strategic decisions |
| Compute cost explosion | Medium | High | Budget-aware planning; cost-per-insight tracking |
| Context window exhaustion on long projects | High | Medium | Hierarchical compression; external memory |

### 6.2 Epistemic Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Confirmation bias** | System reinforces its priors instead of genuinely testing hypotheses | Adversarial critic; diversity in hypothesis generation |
| **Availability bias** | System overweights recent/popular papers | Time-weighted retrieval; explicit coverage of older literature |
| **Narrow expertise** | System becomes expert in a narrow area and misses cross-domain insights | Curriculum planner with diversity bonus; Synthesizer agent |
| **False consensus** | Multi-agent system converges on shared errors | Information asymmetry; enforced disagreement protocols |

### 6.3 Societal and Ethical Risks

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Research pollution** | Flooding venues with low-quality machine-generated papers | Strict quality gates; transparent disclosure of AI origin |
| **Credit displacement** | Autonomous system displaces human researchers | Framing as collaborative tool; shared credit models |
| **Dual-use research** | System discovers dangerous results autonomously | Hardcoded topic exclusions; human review for sensitive areas |
| **Research monoculture** | Autonomous system homogenizes research directions | Diversity objectives; human strategic oversight |

### 6.4 Safeguard Architecture

Every output from the system passes through three gates before external publication:

1. **Automated quality gate:** SUNFIRE score above threshold; grounding check; plagiarism check.
2. **Automated safety gate:** Topic exclusion check; dual-use flag check; factual consistency check.
3. **Human review gate (initially):** A human researcher reviews outputs above certain risk thresholds. As the system matures and demonstrates reliability, this gate becomes sampled (spot-check) rather than exhaustive.

---

## 7. Concrete Next Steps and Experiments

### 7.1 Immediate Experiments (Weeks 1–4)

**Experiment 1: Baseline Literature Agent**
- Build a RAG system over the ArXiv CS.AI corpus (last 12 months).
- Evaluate on a set of 50 questions about recent papers (ground truth from human expert).
- Metric: F1 score on factual recall + relevance of retrieved context.

**Experiment 2: Gap Detection Baseline**
- Run BERTopic over 10,000 recent AI papers.
- Manually review the output for topic coverage gaps.
- Compare against a human expert's assessment of field gaps.
- Metric: precision and recall of gap detection vs. expert opinion.

**Experiment 3: Hypothesis Quality Study**
- Generate 20 hypotheses using GPT-4 with chain-of-thought + literature context.
- Have 3 domain experts rate each on novelty, plausibility, and interest.
- Establish a baseline for SUNFIRE metric calibration.
- Metric: inter-rater agreement; correlation between SUNFIRE score and expert ratings.

**Experiment 4: Critic Effectiveness**
- Take 10 published papers from 2024 and have the Critic Agent review them.
- Compare critic feedback to actual peer reviews (available on OpenReview).
- Metric: overlap with real reviewer concerns; novel issues identified that reviewers missed.

### 7.2 Medium-Term Milestones (Months 1–3)

- **M1.1:** ArXiv ingestion pipeline processing 500+ papers/day with <5% extraction error rate.
- **M1.2:** Knowledge graph with 100K+ entities and 500K+ edges, 95% grounded to verified sources.
- **M1.3:** Gap Map v0.1 covering 50 AI sub-topics with quantified coverage scores.
- **M1.4:** SUNFIRE reward model trained on OpenReview data achieving >0.7 correlation with reviewer scores.
- **M1.5:** Curriculum planner selecting topics that improve literature-agent performance faster than random topic selection.

### 7.3 Key Design Questions to Resolve Early

1. **What base model to use?** GPT-4o vs. Claude 3.5 vs. open source (Llama 3.1, Qwen 2.5). Open source preferred for cost and fine-tuning flexibility.
2. **How to structure the knowledge graph?** Pure property graph (Neo4j) vs. RDF triple store vs. custom schema. Property graph recommended for flexibility.
3. **How to handle compute costs for experiments?** Cloud GPU budget per research thread; automatic scaling based on expected return.
4. **What is the minimum viable "interesting" research output?** A reproducible negative result with clear implications is more valuable than a speculative positive claim.

---

## 8. The Long View

The picoclaw-auto-ai-researcher, if successful, is not just a tool for producing more papers faster. It is a prototype for a new kind of scientific institution — one that can:

- Operate continuously, across all timezones, on all relevant sub-fields simultaneously.
- Maintain a coherent research agenda while adapting to new results in real time.
- Scale horizontally by spinning up new specialized agents as the frontier expands.
- Measure its own performance objectively and improve continuously.

The human researcher's role in this system shifts from *doing* research to *directing* research: setting the high-level mission, reviewing strategic outputs, calibrating values (what "interesting" means), and ensuring the system operates safely. This is not a replacement of human scientific creativity — it is an amplification of it, allowing human researchers to operate at a higher level of abstraction while the system handles the vast, important work of systematic exploration.

The frontier of the intelligence space is vast. We have barely scratched the surface. A system like this — curious, tireless, self-improving, and relentlessly gap-seeking — could accelerate the exploration of that space by an order of magnitude.

---

## Appendix: Key References

| System/Paper | Authors | Year | Relevance |
|-------------|---------|------|-----------|
| AutoGPT | Significant Labs | 2023 | Autonomous agent baseline |
| BabyAGI | Nakajima | 2023 | Task-queue planning |
| Voyager | Wang et al. (NVIDIA) | 2023 | Skill accumulation, curriculum |
| ReAct | Yao et al. | 2022 | Reasoning + acting |
| Reflexion | Shinn et al. | 2023 | Self-critique, episodic memory |
| Tree of Thoughts | Yao et al. | 2023 | Deliberative planning |
| RLHF / InstructGPT | Ouyang et al. (OpenAI) | 2022 | RL from human feedback |
| Constitutional AI | Bai et al. (Anthropic) | 2022 | Principle-guided self-refinement |
| RLAIF | Lee et al. (Google) | 2023 | RL from AI feedback |
| EWC | Kirkpatrick et al. (DeepMind) | 2017 | Catastrophic forgetting |
| Switch Transformer | Fedus et al. (Google) | 2021 | Mixture of Experts |
| AI Safety via Debate | Irving et al. | 2018 | Debate as alignment/evaluation |
| Self-Refine | Madaan et al. | 2023 | Iterative improvement |
| STaR | Zelikman et al. | 2022 | Self-taught reasoning |
| BERTopic | Grootendorst | 2022 | Topic modeling for gap detection |
| SPECTER | Cohan et al. (AllenAI) | 2020 | Scientific paper embeddings |
| AutoGen | Wu et al. (Microsoft) | 2023 | Multi-agent framework |
| Mixture-of-Agents | Wang et al. | 2024 | Ensemble aggregation |

---

*← Back to [README](../README.md)*
