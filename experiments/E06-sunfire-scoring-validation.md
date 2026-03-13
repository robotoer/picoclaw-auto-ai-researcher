# E06: SUNFIRE Scoring Validation

> **Layer 2 — Component Validation**

## Status: Planned

## Hypothesis

SUNFIRE composite scores (Surprise, Usefulness, Novelty, Feasibility, Impact, Rigor, Elegance) correlate with human expert quality judgments at Spearman rho >= 0.5, and learned meta-RL weights outperform uniform weights in predicting expert overall quality ratings.

This hypothesis is falsifiable: if all weighting schemes achieve rho < 0.3, SUNFIRE is not measuring what we think it measures and must be replaced or fundamentally restructured.

## Why This Matters

SUNFIRE is the primary reward signal for all optimization loops in the system. The ReAct-Reflexion loop (E09), evolutionary search (E10), self-play debate (E11), and full research loops (E13, E14) all use SUNFIRE scores to evaluate and select research outputs. If SUNFIRE does not correlate with actual research quality, every loop optimizes for the wrong objective — a direct instance of Goodhart's Law (doc 12 trap #2). This is perhaps the highest-leverage validation in the entire experiment plan: a broken reward signal corrupts everything downstream.

## Background & Prior Work

**LLM-as-Judge reliability and biases.** Zheng et al. (2023) found that GPT-4 judge agreement with humans reaches 85%, exceeding human-human agreement at 81%, but documented systematic biases: position bias (preferring the first response), verbosity bias (preferring longer responses), and self-preference bias (an LLM preferring its own outputs). Cohen's kappa was as low as 0.21 on specialized tasks. Since SUNFIRE dimensions are scored by LLM judges, each dimension may inherit these biases, and the composite score may amplify them if biases are correlated across dimensions.

**AI Scientist evaluation calibration.** Lu et al. (2024) reported automated reviewer accuracy of 0.65 (vs. 0.66 human) and correlation r=0.18 with average human scores (vs. r=0.14 human-human). The F1 for accept/reject decisions was 0.57 (vs. 0.49 human). These results suggest that LLM scoring can approximate aggregate human accuracy but has poor item-level correlation — exactly the failure mode that would make SUNFIRE unreliable as a per-output reward signal.

**Novelty measurement.** Uzzi et al. (2013) demonstrated that atypical combinations of prior work predict 2x citation impact, and Arts & Veugelers (2020) operationalized novelty via cosine distance of embeddings. The Disruption Index provides a complementary structural measure. These external novelty metrics provide independent validation targets for SUNFIRE's Novelty and Surprise dimensions specifically.

**FunSearch evaluation design.** Romera-Paredes et al. (2023) used an automated evaluator (mathematical proof checker) as a hallucination guard during program search, demonstrating that tight evaluator-reality coupling is essential for search-based discovery. SUNFIRE lacks such a formal ground truth, making human validation critical.

**Meta-RL for reward shaping.** The meta-RL weight learning approach (doc 07) assumes that dimension weights can be optimized from feedback. This is only useful if the dimensions themselves carry signal — hence the need to validate individual dimension correlations before optimizing weights.

## Methodology

### Design

Between-groups comparison of 80 research outputs rated by both SUNFIRE (automated) and human experts, with three weighting schemes compared. The design is fully crossed: every output receives both automated and human ratings.

Independent variables: weighting scheme (uniform, doc 07 defaults, learned/grid-search weights), output quality tier (published paper, workshop paper, AI-generated hypothesis, deliberately flawed).
Dependent variables: Spearman rho with expert overall quality, per-dimension correlations, ranking agreement.

### Data Requirements

- **Research outputs (80 total):**
  - 20 published papers accepted at top venues (NeurIPS, ICML, ICLR, ACL, AAAI — representing high quality)
  - 20 workshop papers or preprints with mixed reviews (representing medium quality)
  - 20 AI-generated research hypotheses (from E02 or generated fresh — representing variable quality)
  - 20 deliberately flawed outputs: 10 with factual errors, 5 with logical flaws, 5 with trivial/incremental contributions (representing low quality)
- **Expert raters:** 3 domain experts (PhD-level researchers in AI/ML with >= 5 publications). Each rates all 80 outputs.
- **Rating instrument:** Each output rated on 7 SUNFIRE dimensions (1-7 Likert scale matching SUNFIRE range) plus overall quality (1-7). Dimension definitions provided to experts match SUNFIRE documentation exactly. Rating order randomized per expert to mitigate position effects.

### Procedure

1. **Output collection.** Assemble the 80 outputs. For published and workshop papers, extract the abstract + introduction + key results section (standardized to ~1,500 words) to equalize length across categories and reduce verbosity bias. For AI-generated hypotheses, use outputs from E02 or generate 20 new hypotheses on diverse AI topics. For flawed outputs, create by modifying real papers with specific planted errors documented in a separate log.
2. **SUNFIRE scoring.** Run the full SUNFIRE pipeline on all 80 outputs. Record each dimension score (S, U, N, F, I, R, E) and the composite score under each weighting scheme. Run 3 times per output to measure scoring stability (coefficient of variation).
3. **Expert rating.** Present outputs to each expert in randomized order, blinded to source category. Each expert rates all 7 dimensions plus overall quality. Collect ratings over a 2-week period. Experts may not discuss ratings with each other.
4. **Inter-rater reliability.** Compute ICC (intraclass correlation coefficient, two-way random, absolute agreement) across the 3 experts for each dimension and overall quality. If ICC < 0.5 for overall quality, the expert ratings are too unreliable to serve as ground truth; consider recruiting additional experts.
5. **Correlation analysis.** For each weighting scheme, compute Spearman rho between SUNFIRE composite and expert mean overall quality. Compute per-dimension Spearman rho between SUNFIRE dimension scores and corresponding expert dimension ratings.
6. **Weight scheme comparison.** Compare the three weighting schemes using Steiger's test for dependent correlations (all correlations use the same expert ratings). If meta-RL weights are unavailable, simulate by grid search over the 7-dimensional weight simplex (1,000 random weight vectors, select the one maximizing leave-one-out cross-validated Spearman rho).
7. **Ranking analysis.** Compute Kendall's tau between SUNFIRE-ranked and expert-ranked outputs. Identify the top-10 and bottom-10 by each method; measure overlap.
8. **Calibration analysis.** Bin SUNFIRE scores into quartiles and check whether mean expert ratings increase monotonically across bins. Plot calibration curves.
9. **Dimension redundancy analysis.** Compute the 7x7 correlation matrix among SUNFIRE dimensions. Identify pairs with r > 0.8 (potentially redundant). Run principal component analysis to determine the effective dimensionality of the SUNFIRE space.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| Composite-quality correlation | Spearman rho between SUNFIRE composite and expert mean overall quality | Bootstrap 95% CI (10,000 resamples), permutation test for significance | Rank-based, no distributional assumptions, appropriate for ordinal ratings |
| Per-dimension correlation | Spearman rho between each SUNFIRE dimension and corresponding expert dimension rating | Bootstrap 95% CI per dimension | Same justification; 7 tests with Holm-Bonferroni correction |
| Weight scheme comparison | Difference in Spearman rho between weighting schemes | Steiger's test for dependent correlations | Appropriate for comparing correlations from the same sample |
| Ranking agreement | Kendall's tau between SUNFIRE and expert rankings | Permutation test | Non-parametric test for ranking concordance |
| Inter-rater reliability | ICC(2,k) across 3 experts | Point estimate with 95% CI | Standard for multi-rater reliability with random raters |
| Scoring stability | Coefficient of variation across 3 SUNFIRE runs per output | Descriptive (mean, max CV) | Measures automated scoring reproducibility |
| Calibration | Monotonicity of expert mean quality across SUNFIRE quartiles | Jonckheere-Terpstra test for ordered alternatives | Tests whether higher SUNFIRE scores correspond to higher quality |
| Dimension redundancy | Pairwise Spearman rho among SUNFIRE dimensions, PCA variance explained | Descriptive (correlation matrix, scree plot) | Identifies whether 7 dimensions are necessary or reducible |

**Effect size reporting:** Report Cohen's d for SUNFIRE score differences between quality tiers (published vs. flawed). A meaningful separation requires d >= 0.8 (large effect).

## Success Criteria

- Best weighting scheme achieves Spearman rho >= 0.5 with expert mean overall quality (bootstrap 95% CI lower bound >= 0.35).
- At least 5 of 7 SUNFIRE dimensions individually correlate at rho >= 0.3 with their corresponding expert dimension ratings.
- Learned/optimized weights outperform uniform weights by at least delta-rho >= 0.1 (Steiger's test p < 0.05).
- SUNFIRE scores clearly separate quality tiers: published paper scores significantly higher than flawed outputs (Mann-Whitney U, p < 0.01, Cohen's d >= 0.8).
- Scoring stability: mean CV across outputs < 0.10.

## Failure Criteria

- All weighting schemes achieve rho < 0.3 with expert overall quality. This indicates SUNFIRE is not measuring research quality in a way that aligns with expert judgment, and the entire reward signal must be reconsidered. Downstream loops (E09-E14) cannot proceed with an unvalidated reward.
- Fewer than 3 of 7 dimensions individually correlate at rho >= 0.2. This indicates most dimensions are noise, and SUNFIRE should be simplified to the validated subset.
- SUNFIRE cannot distinguish published papers from deliberately flawed outputs (Mann-Whitney U p > 0.05). This is a catastrophic failure — the metric cannot detect known-bad research.
- ICC among experts < 0.4 for overall quality. The expert ground truth is unreliable; recruit more experts or revise the rating protocol before re-running.

**If the experiment fails:** Investigate which dimensions carry signal and which are noise. Consider (a) reducing SUNFIRE to validated dimensions only, (b) replacing LLM-scored dimensions with computable proxies (e.g., citation-based novelty for the N dimension), (c) calibrating SUNFIRE against a larger pool of human ratings, or (d) adopting a simpler binary quality signal (pass/fail) instead of a continuous composite.

## Estimated Cost & Timeline

- **API calls:** ~$150-300 (80 outputs x 7 dimensions x 3 runs, plus embedding costs)
- **Expert rating:** ~$750-1,500 (3 experts, ~8 hours each rating 80 outputs at ~6 min/output)
- **Output preparation:** ~8 hours (assembling corpus, creating flawed outputs, standardizing format)
- **Analysis and write-up:** ~12-16 hours (correlation analysis, calibration plots, PCA, write-up)
- **Calendar time:** 3-4 weeks (expert rating requires scheduling buffer)

## Dependencies

- **E02** (LLM Judge Reliability): Establishes baseline reliability of LLM-based scoring, which SUNFIRE dimensions use. If E02 shows LLM judges are unreliable, SUNFIRE dimension scores inherit that unreliability.
- **E03** (Semantic Novelty Measurement): Validates the novelty measurement approach that underlies SUNFIRE's Novelty and Surprise dimensions.

## Informs

- **E09** (ReAct-Reflexion Loop): Uses SUNFIRE as the reward signal for iteration.
- **E10** (Evolutionary Strategy Search): Uses SUNFIRE for fitness evaluation.
- **E11** (Self-Play Debate): Uses SUNFIRE to judge debate outcomes.
- **E12** (Self-Training Stability): Monitors SUNFIRE score trajectories for stability.
- **E13** (Full Research Loop): End-to-end loop optimizes SUNFIRE.
- **E14** (Evolutionary Research Loop): End-to-end loop optimizes SUNFIRE.
- **E15** (Loop Comparison): Compares loops on SUNFIRE scores — meaningless if SUNFIRE is invalid.

## References

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv preprint arXiv:2408.06292*.
- Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A. (2023). Mathematical discoveries from program search with large language models. *Nature*, 625, 468-475.
- Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. (2013). Atypical combinations and scientific impact. *Science*, 342(6157), 468-472.
- Arts, S., & Veugelers, R. (2020). Technology familiarity, recombinant novelty, and breakthrough invention. *Industrial and Corporate Change*, 24(6), 1215-1246.
