# E04: Knowledge Graph Consistency — Results Summary

Generated: 2026-03-15T00:45:18.153131+00:00
Conditions: 4

## Contradiction Rate by Condition

| Condition | Papers | Claims | Contradictions | Rate | Halluc. Rate | Provenance |
|-----------|--------|--------|----------------|------|-------------|------------|
| No Filter | 500 | 4466 | 6 | 0.0013 | 0.0000 | 1.000 |
| Layer 1 | 500 | 2530 | 10 | 0.0040 | 0.0000 | 1.000 |
| Layer 1+2 | 500 | 2479 | 2 | 0.0008 | 0.0000 | 1.000 |
| Layer 1+2+3 | 500 | 2530 | 10 | 0.0040 | 0.0400 | 1.000 |

## Hypothesis Evaluation

- **h1_no_filter_baseline**: contradiction_rate <= 10% — **PASS** (value=0.0013434841021047917)
- **h2_layer1_2_rate**: contradiction_rate <= 2% — **PASS** (value=0.0008067769261799112)
- **h3_all_layers_rate**: contradiction_rate <= 0.5% — **PASS** (value=0.003952569169960474)
- **h4_hallucination_rate**: hallucination_rate <= 5% — **PASS** (value=0.04)
- **h5_provenance**: provenance_completeness >= 95% — **PASS** (value=1.0)
- **h6_growth_stable**: contradiction rate slope non-positive with all layers — **FAIL** (value=True)
- **Overall: NOT SUPPORTED**

## Chi-Squared Test (Across All Conditions)

- χ² = 10.064679169641428, p = 0.018024264702813722
- Significant: True

## Extractor Independence

- Effective independent extractors: 3.0
- Observed consensus error rate: 1.0
- Theoretical rate (independence): 1.0
