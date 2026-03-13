# Experiments

> Systematic validation of the autonomous AI researcher's components and loops, ordered from foundational building blocks to full system-level hypotheses.

## Experiment Dependency Graph

```
Layer 1: Foundational Capabilities (no dependencies — run first)
  E01 Claim Extraction ──┐
  E02 LLM Judge ─────────┼──┐
  E03 Novelty Metrics ───┤  │
  E04 KG Consistency ────┘  │
                             │
Layer 2: Component Validation (depend on Layer 1)
  E05 Gap Map ───────────────┤  (needs E01, E04)
  E06 SUNFIRE Validation ────┤  (needs E02, E03)
  E07 Reflexion Memory ──────┤  (needs E02, E06)
  E08 Critic Agent ──────────┘  (needs E01, E02)
                             │
Layer 3: Loop Mechanics (depend on Layer 2)
  E09 ReAct-Reflexion ───────┤  (needs E06, E07, E08)
  E10 Evolutionary Search ───┤  (needs E06)
  E11 Self-Play Debate ──────┤  (needs E08)
  E12 Self-Training ─────────┘  (needs E06)
                             │
Layer 4: System-Level Hypotheses (depend on Layer 3)
  E13 Full ReAct-Reflexion Loop  (needs E01-E09)
  E14 Evolutionary Loop         (needs E06, E09, E10, E13)
  E15 Loop Comparison           (needs E09-E14)
```

## Parallelization

Within each layer, experiments can run in parallel:

- **Layer 1:** E01, E02, E03, E04 are fully independent — run all 4 simultaneously
- **Layer 2:** E05 and E08 can start once E01+E04 or E01+E02 complete; E06 needs E02+E03; E07 needs E06
- **Layer 3:** E10, E11, E12 can start once their dependencies complete; E09 needs E07+E08
- **Layer 4:** Sequential — E13 first, then E14, then E15

## Experiment Index

### Layer 1 — Foundational Capabilities

| ID | Title | Hypothesis | Status | Key Metric |
|----|-------|-----------|--------|------------|
| [E01](E01-claim-extraction-accuracy.md) | Claim Extraction Accuracy | LLMs extract claims at ≥80% precision, ≥70% recall | **Completed** | F1 score |
| [E02](E02-llm-judge-reliability.md) | LLM Judge Reliability | LLM judges achieve Cohen's kappa ≥0.4 with experts | **Completed** | Cohen's kappa |
| [E03](E03-semantic-novelty-measurement.md) | Semantic Novelty Measurement | Novelty metrics achieve AUC ≥0.70 vs human labels | **Completed** | AUC-ROC |
| [E04](E04-knowledge-graph-consistency.md) | KG Consistency Under Updates | KG maintains ≥90% consistency after 500 papers | Planned | Contradiction rate |

### Layer 2 — Component Validation

| ID | Title | Hypothesis | Status | Key Metric |
|----|-------|-----------|--------|------------|
| [E05](E05-gap-map-detection-accuracy.md) | Gap Map Detection Accuracy | Gap Map achieves ≥60% precision, ≥40% recall vs experts | Planned | Precision/Recall |
| [E06](E06-sunfire-scoring-validation.md) | SUNFIRE Scoring Validation | SUNFIRE correlates with expert quality at ρ ≥ 0.5 | Planned | Spearman ρ |
| [E07](E07-reflexion-memory-impact.md) | Reflexion Memory Impact | Reflexion improves output quality ≥15% over 10 iterations | Planned | SUNFIRE delta |
| [E08](E08-critic-agent-effectiveness.md) | Critic Agent Effectiveness | Critic catches ≥70% errors with ≤20% false positive rate | Planned | TPR / FPR |

### Layer 3 — Loop Mechanics

| ID | Title | Hypothesis | Status | Key Metric |
|----|-------|-----------|--------|------------|
| [E09](E09-react-reflexion-iteration-dynamics.md) | ReAct-Reflexion Dynamics | Quality improves ~log(iteration) over 20 iterations | Planned | Mann-Kendall trend |
| [E10](E10-evolutionary-strategy-search.md) | Evolutionary Strategy Search | MAP-Elites achieves ≥30% QD-score gain in 200 gens | Planned | QD-Score |
| [E11](E11-self-play-debate-dynamics.md) | Self-Play Debate | Debate produces ≥25% quality gain over single-pass | Planned | Expert quality rating |
| [E12](E12-self-training-stability.md) | Self-Training Stability | Top-10% filtering + external data prevents collapse | Planned | Generation quality trajectory |

### Layer 4 — System-Level Hypotheses

| ID | Title | Hypothesis | Status | Key Metric |
|----|-------|-----------|--------|------------|
| [E13](E13-full-react-reflexion-research-loop.md) | Full ReAct-Reflexion Loop | Loop produces workshop-quality output in 50 iterations | Planned | Expert rating ≥ 4/7 |
| [E14](E14-evolutionary-research-loop.md) | Evolutionary Research Loop | Evolved strategies outperform hand-designed by ≥40% | Planned | SUNFIRE improvement |
| [E15](E15-loop-comparison-and-composition.md) | Loop Comparison & Composition | Composition outperforms single loops by ≥25% | Planned | Quality per dollar |

## Results Summary

### E01: Claim Extraction Accuracy — Completed

**Hypothesis NOT SUPPORTED.** Best model (Claude Sonnet 4) achieves precision 0.803 (passes ≥0.80), but recall 0.660 (fails ≥0.70) and hallucination rate 0.213 (fails ≤0.10).

| Model | F1 | Precision | Recall | Halluc. Rate |
|-------|-----|-----------|--------|--------------|
| Claude Sonnet 4 | **0.720** | **0.803** | **0.660** | **0.213** |
| Claude Opus 4.6 | 0.612 | 0.703 | 0.551 | 0.323 |
| Claude Sonnet 4.6 | 0.589 | 0.649 | 0.605 | 0.388 |
| GPT-4o | 0.518 | 0.773 | 0.418 | 0.265 |
| Claude 3.5 Haiku | 0.386 | 0.646 | 0.296 | 0.381 |

**Key findings:**
- Model capability does not monotonically predict extraction accuracy
- Sonnet 4 outperforms both newer (Sonnet 4.6) and more capable (Opus 4.6) models
- All models have hallucination rates >20%, requiring downstream verification (informs E02+)
- Precision passes threshold but recall and hallucination rate do not
- Paper: [experiments/E01/paper/main.pdf](E01/paper/main.pdf)

### E02: LLM Judge Reliability — Completed

**Hypothesis SUPPORTED.** All five dimensions exceed kappa ≥ 0.4 threshold (5/5 pass), and Spearman ρ = 0.739 (passes ≥ 0.5). No position bias or self-preference bias detected.

| Model | Novelty κ | Feasibility κ | Importance κ | Clarity κ | Specificity κ | Spearman ρ |
|-------|----------|--------------|-------------|---------|-------------|-----------|
| Claude Sonnet 4.6 | **0.894** | 0.841 | 0.821 | **0.763** | **0.808** | **0.860** |
| Claude Opus 4.6 | 0.817 | 0.828 | **0.828** | 0.656 | 0.794 | 0.843 |
| Gemini 2.0 Flash | 0.712 | **0.870** | 0.600 | 0.517 | 0.466 | 0.798 |
| GPT-4o | 0.492 | 0.764 | 0.333 | 0.423 | 0.399 | 0.723 |

**Key findings:**
- Claude Sonnet 4.6 is the most reliable judge, approaching expert-expert baseline agreement
- Feasibility is the easiest dimension to judge; clarity and specificity are hardest
- No position bias (d=0.029) or self-preference bias detected
- CoT does not improve agreement (d=-0.112) — non-CoT absolute scoring is recommended
- LLM judges validated as reliable quality signal for all downstream optimization loops
- Paper: [experiments/E02/paper/main.pdf](E02/paper/main.pdf)

### E03: Semantic Novelty Measurement — Completed

**Hypothesis PARTIALLY SUPPORTED.** H1 passes (AUC = 0.943 for LLM judgment), but H2 fails (best Spearman ρ with citation = 0.183, below 0.30 threshold). Overall NOT SUPPORTED because both criteria required.

| Metric | AUC | Spearman ρ (citation) | Spearman ρ (human) |
|--------|-----|----------------------|-------------------|
| LLM judgment | **0.943** | 0.183 | **0.713** |
| Combined | **0.943** | 0.060 | 0.551 |
| Embedding distance | 0.891 | -0.144 | 0.270 |
| Topic distance | 0.768 | -0.180 | 0.173 |
| Atypical references | 0.578 | 0.015 | 0.136 |

**Key findings:**
- LLM novelty judgment is the best metric (AUC=0.943, correlates strongly with human scores ρ=0.713)
- Novelty and citation impact are essentially independent — this is informative, not a failure
- Atypical reference combinations (Uzzi method) perform poorly with 100-paper corpus
- Inter-rater agreement fair (Fleiss' κ=0.339), confirming novelty is inherently subjective
- For E06 SUNFIRE: use LLM judgment as novelty signal, track impact separately
- Paper: [experiments/E03/paper/main.pdf](E03/paper/main.pdf)

## Estimated Total Cost

| Layer | API Cost | Human Time | Calendar Time |
|-------|----------|-----------|---------------|
| Layer 1 | $180-360 | 28-54 hrs | 2-3 weeks |
| Layer 2 | $200-500 | 24-48 hrs | 3-5 weeks |
| Layer 3 | $800-3000 | 20-40 hrs | 4-8 weeks |
| Layer 4 | $5500-14000 | 60-90 hrs | 10-18 weeks |
| **Total** | **$6680-17860** | **132-232 hrs** | **~4-8 months** |

## How Results Flow Forward

When an experiment completes:

1. Move experiment file to `experiments-completed/` with full paper write-up
2. Update this README with results summary and key findings
3. Update downstream experiment files with any new information
4. Update relevant `docs/` files with empirical findings
5. If results invalidate assumptions, add new experiments or revise existing ones
