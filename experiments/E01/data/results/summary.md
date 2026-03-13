# E01: Claim Extraction Accuracy — Results Summary

Generated: 2026-03-13T02:59:42.085336+00:00
Papers: 18  |  Models: 4

## Per-Model Results

| Model | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Halluc. Rate |
|-------|--------------------|-----------------|-------------|--------------|
| anthropic/claude-3.5-haiku | 0.646 [0.580, 0.712] | 0.296 [0.240, 0.358] | 0.386 [0.324, 0.450] | 0.381 |
| anthropic/claude-opus-4-6 | 0.704 [0.648, 0.757] | 0.552 [0.495, 0.610] | 0.613 [0.558, 0.667] | 0.321 |
| anthropic/claude-sonnet-4 | 0.796 [0.748, 0.842] | 0.655 [0.609, 0.700] | 0.714 [0.670, 0.757] | 0.219 |
| openai/gpt-4o | 0.773 [0.722, 0.823] | 0.418 [0.364, 0.473] | 0.518 [0.461, 0.573] | 0.265 |

## Hypothesis Evaluation

- Best model: **anthropic/claude-sonnet-4** (F1=0.714)
- Precision >= 0.80: **FAIL** (0.796)
- Recall >= 0.70: **FAIL** (0.654)
- Hallucination rate <= 0.10: **FAIL** (0.219)
- **Hypothesis NOT SUPPORTED**

## Statistical Tests

- anthropic/claude-3.5-haiku_vs_anthropic/claude-opus-4-6: {'statistic': 23.361111, 'p_value': 1e-06, 'significant': True}
- anthropic/claude-3.5-haiku_vs_anthropic/claude-sonnet-4: {'statistic': 29.166667, 'p_value': 0.0, 'significant': True}
- anthropic/claude-3.5-haiku_vs_openai/gpt-4o: {'statistic': 12.96, 'p_value': 0.000318, 'significant': True}
- anthropic/claude-opus-4-6_vs_anthropic/claude-sonnet-4: {'statistic': 2.083333, 'p_value': 0.148915, 'significant': False}
- anthropic/claude-opus-4-6_vs_openai/gpt-4o: {'statistic': 9.090909, 'p_value': 0.002569, 'significant': True}
- anthropic/claude-sonnet-4_vs_openai/gpt-4o: {'statistic': 12.190476, 'p_value': 0.00048, 'significant': True}
- anthropic/claude-3.5-haiku_vs_anthropic/claude-opus-4-6_cohens_d: -1.0221
- anthropic/claude-3.5-haiku_vs_anthropic/claude-sonnet-4_cohens_d: -1.6213
- anthropic/claude-3.5-haiku_vs_openai/gpt-4o_cohens_d: -0.5948
- anthropic/claude-opus-4-6_vs_anthropic/claude-sonnet-4_cohens_d: -0.5369
- anthropic/claude-opus-4-6_vs_openai/gpt-4o_cohens_d: 0.4519
- anthropic/claude-sonnet-4_vs_openai/gpt-4o_cohens_d: 1.0383
