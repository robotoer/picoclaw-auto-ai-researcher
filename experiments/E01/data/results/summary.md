# E01: Claim Extraction Accuracy — Results Summary

Generated: 2026-03-13T03:23:02.973947+00:00
Papers: 18  |  Models: 5

## Per-Model Results

| Model | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Halluc. Rate |
|-------|--------------------|-----------------|-------------|--------------|
| anthropic/claude-3.5-haiku | 0.646 [0.580, 0.712] | 0.296 [0.240, 0.358] | 0.386 [0.324, 0.450] | 0.381 |
| anthropic/claude-opus-4-6 | 0.703 [0.647, 0.756] | 0.551 [0.494, 0.609] | 0.612 [0.556, 0.667] | 0.323 |
| anthropic/claude-sonnet-4-6 | 0.649 [0.589, 0.703] | 0.605 [0.559, 0.651] | 0.589 [0.534, 0.639] | 0.388 |
| anthropic/claude-sonnet-4 | 0.803 [0.756, 0.848] | 0.660 [0.615, 0.705] | 0.720 [0.677, 0.762] | 0.213 |
| openai/gpt-4o | 0.773 [0.722, 0.823] | 0.418 [0.364, 0.473] | 0.518 [0.461, 0.573] | 0.265 |

## Hypothesis Evaluation

- Best model: **anthropic/claude-sonnet-4** (F1=0.720)
- Precision >= 0.80: **PASS** (0.803)
- Recall >= 0.70: **FAIL** (0.660)
- Hallucination rate <= 0.10: **FAIL** (0.213)
- **Hypothesis NOT SUPPORTED**

## Statistical Tests

- anthropic/claude-3.5-haiku_vs_anthropic/claude-opus-4-6: {'statistic': 22.4, 'p_value': 2e-06, 'significant': True}
- anthropic/claude-3.5-haiku_vs_anthropic/claude-sonnet-4-6: {'statistic': 24.324324, 'p_value': 1e-06, 'significant': True}
- anthropic/claude-3.5-haiku_vs_anthropic/claude-sonnet-4: {'statistic': 31.609756, 'p_value': 0.0, 'significant': True}
- anthropic/claude-3.5-haiku_vs_openai/gpt-4o: {'statistic': 12.96, 'p_value': 0.000318, 'significant': True}
- anthropic/claude-opus-4-6_vs_anthropic/claude-sonnet-4-6: {'statistic': 0.071429, 'p_value': 0.789268, 'significant': False}
- anthropic/claude-opus-4-6_vs_anthropic/claude-sonnet-4: {'statistic': 3.5, 'p_value': 0.061369, 'significant': False}
- anthropic/claude-opus-4-6_vs_openai/gpt-4o: {'statistic': 8.1, 'p_value': 0.004427, 'significant': True}
- anthropic/claude-sonnet-4-6_vs_anthropic/claude-sonnet-4: {'statistic': 1.5625, 'p_value': 0.2113, 'significant': False}
- anthropic/claude-sonnet-4-6_vs_openai/gpt-4o: {'statistic': 6.05, 'p_value': 0.013906, 'significant': True}
- anthropic/claude-sonnet-4_vs_openai/gpt-4o: {'statistic': 13.136364, 'p_value': 0.00029, 'significant': True}
- anthropic/claude-3.5-haiku_vs_anthropic/claude-opus-4-6_cohens_d: -1.0166
- anthropic/claude-3.5-haiku_vs_anthropic/claude-sonnet-4-6_cohens_d: -0.9377
- anthropic/claude-3.5-haiku_vs_anthropic/claude-sonnet-4_cohens_d: -1.6659
- anthropic/claude-3.5-haiku_vs_openai/gpt-4o_cohens_d: -0.5948
- anthropic/claude-opus-4-6_vs_anthropic/claude-sonnet-4-6_cohens_d: 0.1116
- anthropic/claude-opus-4-6_vs_anthropic/claude-sonnet-4_cohens_d: -0.5789
- anthropic/claude-opus-4-6_vs_openai/gpt-4o_cohens_d: 0.4466
- anthropic/claude-sonnet-4-6_vs_anthropic/claude-sonnet-4_cohens_d: -0.727
- anthropic/claude-sonnet-4-6_vs_openai/gpt-4o_cohens_d: 0.3479
- anthropic/claude-sonnet-4_vs_openai/gpt-4o_cohens_d: 1.0811
