# E12: Self-Training Stability

> **Layer 3 — Loop Mechanics**

## Status: Planned

## Hypothesis

STaR-style self-training on research outputs maintains or improves quality for >= 5 generations when using aggressive top-percentile filtering (top 10%) and 30% external data mixing, but degrades without these safeguards.

This hypothesis is falsifiable: if condition (d) (top-10% filtering + 30% external data) does not maintain >= 95% of generation-0 quality through generation 5, or if condition (a) (no filtering) does not show measurable degradation by generation 3, or if all conditions show equivalent trajectories (safeguards make no difference), the hypothesis is rejected.

## Why This Matters

Doc 11 Path C proposes STaR-style self-training as a capability growth mechanism: the agent generates research outputs, filters for the best, and trains on its own best work to improve. Zelikman et al. (2022) showed this works for reasoning — self-generated rationales bootstrap reasoning capability with diminishing returns of approximately 1/n per generation, plateauing after 3-5 iterations.

However, Shumailov et al. (2024) demonstrated that AI models collapse when trained on recursively generated data: quality degrades progressively as the tails of the output distribution are lost across generations. Doc 12 identifies Self-Training Collapse as a MEDIUM-HIGH danger intractability trap. The tension between STaR's promise and model collapse risk is unresolved. We need to know exactly when collapse begins, how fast it progresses, and whether the proposed mitigations (top-percentile filtering and external data mixing) actually prevent it. Without this experiment, Path C implementation cannot be responsibly attempted.

## Background & Prior Work

**STaR (Self-Taught Reasoner).** Zelikman et al. (2022) introduced a bootstrap loop where a language model generates rationales for reasoning tasks, filters for rationales that lead to correct answers, and fine-tunes on the successful rationales. This improved reasoning performance with diminishing returns: approximately 1/n improvement per generation, plateauing after 3-5 iterations. The key insight is that self-generated training data can improve capability — but STaR relies on a verifiable correctness signal (answer correctness) for filtering.

**Model collapse.** Shumailov et al. (2024) showed that training on recursively self-generated data causes progressive quality degradation across generations. The mechanism is distributional: each generation of self-training loses information from the tails of the output distribution, causing the model to converge toward the mode. Over multiple generations, this produces increasingly homogeneous, low-variance outputs — "model collapse." The finding was demonstrated on both language models and image models, and the degradation was consistent across model sizes.

**Reward model overoptimization.** Gao et al. (2023) established scaling laws showing that the optimal number of optimization steps against a reward model scales as sqrt(reward_model_size). Beyond this point, optimizing the proxy (reward model score) actively degrades true performance. The implication for self-training is that even with a quality filter (SUNFIRE), over-optimization will eventually degrade quality. The KL penalty — constraining the trained model to stay close to the base model — was shown to help.

**External data as a stabilizer.** Shumailov et al. (2024) noted that maintaining access to original (non-generated) training data prevents collapse. The mechanism is straightforward: external data preserves distributional coverage that self-generated data progressively loses. The question is how much external data is needed to stabilize the loop.

**Goodhart's Law.** Goodhart (1984) and Manheim & Garrabrant (2019) formalized the divergence between proxy optimization and true objective optimization. In self-training, the quality filter (SUNFIRE) is the proxy. If the model learns to produce outputs that score high on SUNFIRE without genuine quality, the self-training loop amplifies this misalignment across generations.

## Methodology

### Design

Longitudinal study tracking output quality across 10 self-training generations under 5 filtering/mixing conditions. Total: 5 conditions x 10 generations x 1000 outputs per generation = 50,000 generated outputs.

**Independent variables:**
- Condition (filtering and mixing strategy):
  - (a) No filtering — train on all 1000 outputs from each generation
  - (b) Top-50% filtering — train on the 500 highest-SUNFIRE-scoring outputs
  - (c) Top-10% filtering — train on the 100 highest-SUNFIRE-scoring outputs
  - (d) Top-10% filtering + 30% external data — train on 100 top outputs mixed with 43 real paper excerpts (maintaining 70/30 ratio)
  - (e) Top-10% filtering + 30% external data + diversity bonus — same as (d) but filtering penalizes similarity to the training set (cosine distance penalty)
- Generation number (0-10)

**Dependent variables:** Mean SUNFIRE score, SUNFIRE score variance, output diversity, repetition rate, embedding space coverage, mode collapse indicators.

### Data Requirements

- **Base task:** Research hypothesis generation on AI safety subtopics. Use 10 specific subtopics (e.g., scalable oversight, deceptive alignment, reward hacking, distributional shift, interpretability methods, emergent capabilities, value alignment, corrigibility, mesa-optimization, cooperative AI) to enable per-topic analysis.
- **Base model:** Either (a) a fine-tuned 7B parameter open-source model (e.g., Llama-3-8B or Mistral-7B, fine-tuned on ~500 human-written research hypotheses) or (b) a prompted frontier model with controlled sampling. Option (a) is preferred for genuine self-training; option (b) is a proxy if fine-tuning infrastructure is unavailable.
- **Generation volume:** 1000 hypotheses per generation per condition. Each hypothesis: 100-300 tokens.
- **External data pool:** 500 real research hypothesis excerpts drawn from published AI safety papers (extracted from abstracts, introduction sections, and "future work" sections). Stratified across the 10 subtopics. This pool remains fixed across all generations.
- **Human evaluation:** Expert panel (2 evaluators) rates a stratified sample of 50 outputs per condition at generations 0, 3, 5, 7, and 10 (50 x 5 conditions x 5 generations = 1,250 ratings per evaluator). Used to validate that SUNFIRE trends reflect genuine quality changes.
- **Embedding model:** Sentence-transformer for diversity and coverage metrics. Fixed model across all generations.

### Procedure

1. **Base model preparation.** Fine-tune the base model on 500 human-written research hypotheses (or establish the prompting baseline for the frontier model proxy). Generate 1000 hypotheses as generation 0 output. Score all with SUNFIRE. This is the shared starting point for all conditions.
2. **Self-training loop (per condition, 10 generations):**
   - **Generate:** Produce 1000 hypotheses using the current model/prompt
   - **Score:** Score all 1000 with SUNFIRE
   - **Filter:** Apply condition-specific filtering:
     - (a): No filter, use all 1000
     - (b): Keep top 500 by SUNFIRE
     - (c): Keep top 100 by SUNFIRE
     - (d): Keep top 100 by SUNFIRE, add 43 external examples (30% of total)
     - (e): Keep top 100 by SUNFIRE with diversity penalty (subtract 0.2 x max_cosine_similarity_to_training_set from SUNFIRE score before ranking), add 43 external examples
   - **Train:** Fine-tune model on the filtered set (or update the few-shot prompt with filtered examples for the frontier model proxy)
   - **Record:** All 1000 outputs, all SUNFIRE scores, the filtered training set, model checkpoint
3. **Per-generation metrics computation.** After each generation, compute: mean SUNFIRE, SUNFIRE variance, pairwise cosine distance matrix (sample 200 for efficiency), n-gram overlap with previous generation, embedding convex hull volume (approximated via PCA to 50 dimensions), and top-3 template frequency (most common structural patterns).
4. **Collapse detection.** At each generation, compute the mode collapse indicator: if the top-3 most frequent structural templates account for > 50% of outputs, flag as potential collapse.
5. **Human evaluation checkpoints.** At generations 0, 3, 5, 7, and 10: randomly sample 50 outputs per condition, present to 2 expert evaluators in randomized blinded order. Evaluators rate each on a 7-point scale covering novelty, correctness, specificity, and significance.
6. **Shumailov replication check.** Specifically analyze condition (a) for the progressive degradation pattern described in Shumailov et al. (2024): decreasing variance, decreasing tail coverage, increasing modal concentration.
7. **Cross-condition comparison at generation 5.** The primary comparison point. If condition (d) has maintained quality but condition (a) has degraded, the safeguards are validated.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| Quality trajectory | Mean SUNFIRE score per generation per condition | Repeated-measures ANOVA (generation x condition), Mauchly's sphericity test, Greenhouse-Geisser correction if needed | Detects main effects of generation (degradation over time) and condition (safeguard effectiveness), plus interaction (differential degradation rates) |
| Per-condition trend | SUNFIRE trajectory within each condition | Mann-Kendall trend test per condition | Non-parametric trend detection; identifies which conditions degrade vs stabilize |
| Degradation onset | First generation where mean SUNFIRE drops > 5% from generation 0 | Descriptive comparison across conditions | Identifies how quickly collapse begins under each regime |
| Condition comparison at gen 5 | SUNFIRE distributions across conditions at generation 5 | Kruskal-Wallis test, post-hoc Dunn's with Bonferroni | Non-parametric comparison at the primary checkpoint |
| Diversity trajectory | Mean pairwise cosine distance over generations | Mann-Kendall trend test per condition | Detects diversity loss (a key model collapse signature) |
| Mode collapse indicator | Top-3 template frequency over generations | Descriptive (plot trajectory), threshold detection (>50% = collapse) | Direct measure of output homogenization |
| Embedding coverage | Convex hull volume in PCA-reduced space over generations | Descriptive (plot trajectory) | Measures distributional coverage loss |
| SUNFIRE-human correlation | Spearman correlation between SUNFIRE and human ratings at checkpoints | Spearman rank correlation per condition per generation | Validates that SUNFIRE trends reflect genuine quality changes, not Goodhart drift |
| Repetition rate | Mean 4-gram overlap between consecutive generations | Descriptive (plot trajectory) | Detects surface-level repetition across generations |
| Human inter-rater reliability | Agreement between 2 evaluators | ICC (two-way mixed, absolute agreement) | Ensures human evaluation is reliable enough to validate SUNFIRE |

**Effect size reporting:** Report partial eta-squared for ANOVA effects, rank-biserial correlation for pairwise comparisons.

## Success Criteria

- Condition (d) (top-10% + 30% external) maintains >= 95% of generation-0 mean SUNFIRE score through generation 5 (Mann-Kendall trend test not significant for negative trend, p >= 0.05).
- Condition (a) (no filtering) shows measurable degradation: mean SUNFIRE drops > 10% by generation 3, with significant negative Mann-Kendall trend (p < 0.05).
- Clear separation between safeguarded and unsafeguarded conditions: significant generation x condition interaction in repeated-measures ANOVA (p < 0.05).
- Human ratings at generation 5 confirm SUNFIRE trends: Spearman rho >= 0.5 between SUNFIRE and human ratings for condition (d).
- Diversity metrics (pairwise cosine distance, embedding coverage) remain within 80% of generation-0 values for condition (d) through generation 5.

## Failure Criteria

- All conditions degrade equally, including (d) and (e). This would indicate that the proposed safeguards (filtering + external data) are insufficient to prevent collapse, and Path C requires fundamentally different mitigation strategies.
- No condition degrades through generation 10. This would indicate that model collapse is not a real risk at this scale/task, and the safeguards are unnecessary overhead. While this is a positive finding for system design, it falsifies the hypothesis that safeguards are needed.
- SUNFIRE scores remain stable but human ratings degrade. This would indicate a Goodhart effect: the model learns to optimize the SUNFIRE proxy while genuine quality declines, meaning SUNFIRE is an unreliable training signal for self-training.
- Condition (e) (diversity bonus) performs worse than condition (d). This would indicate that the diversity penalty interferes with quality selection, and the simplest safeguard (filtering + mixing) is sufficient.

**If the experiment fails:** If all conditions degrade: investigate whether (1) the top-10% threshold is too aggressive or too lenient, (2) the 30% external ratio is insufficient (try 50%, 70%), (3) SUNFIRE is too noisy to serve as a reliable quality filter, or (4) the task itself (hypothesis generation) is inherently unstable under self-training. Consider alternative stabilization mechanisms: KL penalties (Gao et al., 2023), periodic resets to the base model, or ensembled quality filters.

## Estimated Cost & Timeline

- **Fine-tuning:** ~$200-500 per condition (10 fine-tuning runs per condition on a 7B model; or ~$100-200 for prompted frontier model proxy)
- **Generation:** ~$500-1,000 (50,000 total outputs across all conditions and generations; plus SUNFIRE scoring)
- **Total compute (5 conditions):** ~$2,000-5,000 (fine-tuning is the major cost; prompted proxy reduces this significantly)
- **Human evaluation:** ~40-60 hours across 2 experts (1,250 ratings each at ~2-3 min per rating)
- **Analysis and write-up:** ~12-16 hours
- **Calendar time:** 6-8 weeks (fine-tuning pipeline setup: 1-2 weeks; sequential generation runs: 2-3 weeks; human evaluation: 2 weeks; analysis: 1 week)
- **Infrastructure requirement:** GPU access for fine-tuning (Tier 1+ budget). Alternatively, a prompted frontier model proxy can be used at lower cost but with less ecological validity.

## Dependencies

- **E06** (SUNFIRE Calibration): SUNFIRE must be validated as a quality signal before it can be used as a training filter. If SUNFIRE is uncorrelated with true quality, filtering on SUNFIRE is meaningless.
- **E02** (LLM-as-Judge Reliability): Evaluation reliability bounds the usefulness of SUNFIRE as a self-training signal.
- **Infrastructure:** Requires fine-tuning infrastructure (GPU access, training pipeline). If unavailable, a prompted frontier model proxy can be substituted with reduced ecological validity.

## Informs

- **Path C implementation timeline:** If condition (d) succeeds, Path C is viable with the specified safeguards. If it fails, Path C requires redesign or deprioritization.
- **Doc 12 intractability trap analysis:** Validates or refutes the MEDIUM-HIGH danger rating for Self-Training Collapse. Provides empirical degradation curves to calibrate risk.
- **Safeguard parameter selection:** The experiment identifies the minimum safeguards needed (filtering threshold, external data ratio, diversity penalty) and whether more aggressive safeguards improve stability.
- **Generation budget planning:** The degradation onset point per condition provides a principled maximum generation count for production self-training loops.

## References

- Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). STaR: Bootstrapping Reasoning With Reasoning. *Advances in Neural Information Processing Systems (NeurIPS 2022)*.
- Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature*, 631, 755-759.
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. *International Conference on Machine Learning (ICML 2023)*.
- Goodhart, C. A. E. (1984). Problems of Monetary Management: The U.K. Experience. In *Monetary Theory and Practice*. Macmillan.
- Manheim, D., & Garrabrant, S. (2019). Categorizing Variants of Goodhart's Law. *arXiv preprint arXiv:1803.04585*.
- Shinn, N., Cassano, F., Gopinath, A., Shakkottai, K., Labash, A., & Karthik, B. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
