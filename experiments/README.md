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
| [E01](E01-claim-extraction-accuracy.md) | Claim Extraction Accuracy | LLMs extract claims at ≥80% precision, ≥70% recall | Planned | F1 score |
| [E02](E02-llm-judge-reliability.md) | LLM Judge Reliability | LLM judges achieve Cohen's kappa ≥0.4 with experts | Planned | Cohen's kappa |
| [E03](E03-semantic-novelty-measurement.md) | Semantic Novelty Measurement | Novelty metrics achieve AUC ≥0.70 vs human labels | Planned | AUC-ROC |
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

*No experiments completed yet. This section will be updated as results come in.*

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
