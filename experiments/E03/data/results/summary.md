# E03: Semantic Novelty Measurement — Results Summary

Generated: 2026-03-13T20:34:30.528016+00:00
Papers: 100  |  Metrics: 5

## AUC-ROC per Metric

| Metric | AUC | 95% CI | Spearman ρ (citation) | Spearman ρ (human) | P@10 | P@20 |
|--------|-----|--------|----------------------|-------------------|------|------|
| atypical_references | 0.578 | [0.369, 0.742] | 0.015 (p=0.8797) | 0.136 (p=0.1723) | 0.000 | 0.000 |
| combined | 0.943 | [0.888, 0.986] | 0.060 (p=0.5561) | 0.551 (p=0.0001) | 0.300 | 0.200 |
| embedding_distance | 0.891 | [0.758, 0.976] | -0.144 (p=0.1549) | 0.270 (p=0.0067) | 0.300 | 0.150 |
| llm_judgment | 0.943 | [0.873, 1.000] | 0.183 (p=0.0681) | 0.713 (p=0.0001) | 0.200 | 0.200 |
| topic_distance | 0.768 | [0.376, 0.969] | -0.180 (p=0.0755) | 0.173 (p=0.0897) | 0.200 | 0.150 |

## Hypothesis Evaluation

- **H1** AUC >= 0.70: **PASS** (best AUC = 0.943)
- **H2** Spearman ρ >= 0.30 with citation: **FAIL** (best ρ = 0.183)
- **Overall: NOT SUPPORTED**

## Inter-Rater Agreement

Fleiss' kappa: 0.339 (3 raters)
