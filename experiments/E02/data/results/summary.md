# E02: LLM Judge Reliability — Results Summary

Generated: 2026-03-13T06:21:00.017217+00:00
Hypotheses: 50  |  Judge Models: 4

## Inter-Rater Agreement: Weighted Cohen's Kappa (Aggregate)

| Dimension | Mean Kappa |
|-----------|-----------|
| Novelty | 0.729 |
| Feasibility | 0.826 |
| Importance | 0.646 |
| Clarity | 0.590 |
| Specificity | 0.617 |

## Per-Model Kappa by Dimension

| Model | Novelty | Feasibility | Importance | Clarity | Specificity |
|-------|--------|--------|--------|--------|--------|
| anthropic/claude-opus-4-6 | 0.817 | 0.828 | 0.828 | 0.656 | 0.794 |
| anthropic/claude-sonnet-4-6 | 0.894 | 0.841 | 0.821 | 0.763 | 0.808 |
| google/gemini-2.0-flash-001 | 0.712 | 0.870 | 0.600 | 0.517 | 0.466 |
| openai/gpt-4o | 0.492 | 0.764 | 0.333 | 0.423 | 0.399 |

## Human-Human Baseline (Expert Proxy Agreement)

| Dimension | Mean Kappa |
|-----------|-----------|
| Novelty | 0.897 |
| Feasibility | 0.882 |
| Importance | 0.808 |
| Clarity | 0.730 |
| Specificity | 0.821 |

## Overall Correlation: Spearman's Rho

**Aggregate**: rho = 0.739, p = 0.0001

| Model | Spearman Rho | p-value |
|-------|-------------|---------|
| anthropic/claude-opus-4-6 | 0.843 | 0.0001 |
| anthropic/claude-sonnet-4-6 | 0.860 | 0.0001 |
| google/gemini-2.0-flash-001 | 0.798 | 0.0001 |
| openai/gpt-4o | 0.723 | 0.0001 |

## Hypothesis Evaluation

- **H1** Kappa >= 0.40 on at least 3/5 dimensions: **PASS** (5/5 pass)
- **H2** Spearman rho >= 0.50: **PASS** (rho = 0.739)
- **CoT effect**: Does not improve agreement (Cohen's d = -0.112)
- **Overall: SUPPORTED**

## Bias Tests

**Position bias**: d = 0.029, significant = False, biased (d >= 0.3 AND sig) = False

### Self-Preference Bias (per session)

| Session | Effect Size | Significant | Biased |
|---------|------------|-------------|--------|
| judge_anthropic_claude-opus-4-6_absolute_cot | -0.007 | No | No |
| judge_anthropic_claude-opus-4-6_absolute_run0 | -0.033 | No | No |
| judge_anthropic_claude-opus-4-6_absolute_run1 | -0.018 | No | No |
| judge_anthropic_claude-opus-4-6_absolute_run2 | -0.033 | No | No |
| judge_anthropic_claude-sonnet-4-6_absolute_cot | 0.000 | No | No |
| judge_anthropic_claude-sonnet-4-6_absolute_run0 | -0.027 | No | No |
| judge_anthropic_claude-sonnet-4-6_absolute_run1 | -0.033 | No | No |
| judge_anthropic_claude-sonnet-4-6_absolute_run2 | -0.024 | No | No |
| judge_google_gemini-2.0-flash-001_absolute_cot | 0.149 | No | No |
| judge_google_gemini-2.0-flash-001_absolute_run0 | 0.200 | No | No |
| judge_google_gemini-2.0-flash-001_absolute_run1 | 0.174 | No | No |
| judge_google_gemini-2.0-flash-001_absolute_run2 | 0.198 | No | No |
| judge_openai_gpt-4o_absolute_cot | 0.025 | No | No |
| judge_openai_gpt-4o_absolute_run0 | -0.063 | No | No |
| judge_openai_gpt-4o_absolute_run1 | -0.055 | No | No |
| judge_openai_gpt-4o_absolute_run2 | -0.056 | No | No |

## Format Comparison

Absolute mean kappa: 0.681
Absolute sessions: 16
Pairwise sessions: 4

## CoT Effect

CoT mean kappa: 0.666
No-CoT mean kappa: 0.687
Cohen's d: -0.112
