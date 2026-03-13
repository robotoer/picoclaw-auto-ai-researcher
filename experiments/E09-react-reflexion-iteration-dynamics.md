# E09: ReAct-Reflexion Iteration Dynamics

> **Layer 3 — Loop Mechanics**

## Status: Planned

## Hypothesis

A ReAct-Reflexion research agent shows statistically significant improvement in research output quality (SUNFIRE score) over 20 consecutive research iterations on a focused subdomain, with diminishing but positive returns following approximately log(iteration) scaling.

This hypothesis is falsifiable: if no condition shows a significant positive trend (Mann-Kendall p < 0.05) over 20 iterations, or if SUNFIRE improvements are not confirmed by human expert ratings (Spearman rho < 0.5), the hypothesis is rejected.

## Why This Matters

Doc 09 Loop 1 proposes ReAct-Reflexion as the core execution loop for the autonomous researcher. Reflexion demonstrated +20% on HotPotQA over 5 iterations and reached 91% pass@1 on HumanEval (Shinn et al., 2023), but research tasks are far more open-ended than QA or code generation. The original Reflexion work used up to 12 trial iterations on tasks with clear success signals. Research lacks such signals — quality is assessed by noisy proxies.

We need to establish three things: (1) whether reflection actually helps on research-type tasks, (2) how many iterations before plateau, and (3) whether quality genuinely improves or whether the agent merely learns to optimize the score metric while actual quality stagnates or degrades (Goodhart's Law; Manheim & Garrabrant, 2019). Without this experiment, building the full Loop 1 system risks investing in a mechanism that may not transfer from structured benchmarks to open-ended research.

## Background & Prior Work

**ReAct.** Yao et al. (2023) introduced interleaved reasoning and acting traces for language model agents, achieving +34% on ALFWorld and +10% on WebShop over chain-of-thought and act-only baselines. The key insight is that reasoning traces ground actions in context, while actions ground reasoning in observations. However, ReAct alone has no memory across episodes — each task attempt is independent.

**Reflexion.** Shinn et al. (2023) added verbal reinforcement learning to ReAct: after each episode, the agent generates a natural language reflection on what went wrong and stores it for future retrieval. This produced 91% pass@1 on HumanEval (+11% over GPT-4 baseline), +22% on AlfWorld, and +20% on HotPotQA. Critically, improvements showed diminishing returns across iterations, with most gains in the first 3-5 trials and a plateau by iteration 12.

**Diminishing returns in self-improvement.** Zelikman et al. (2022) observed that STaR self-training shows approximately 1/n improvement per generation, plateauing after 3-5 iterations. This suggests a general pattern: iterative self-improvement yields logarithmic rather than linear returns, consistent with easy gains being captured first.

**Goodhart's Law.** Goodhart (1984) and Manheim & Garrabrant (2019) formalized the risk that optimizing a proxy metric diverges from optimizing the true objective. In our context, SUNFIRE scores could improve while actual research quality stagnates if the agent learns to produce outputs that score well on SUNFIRE's specific rubric without genuine intellectual progress.

**Reward model overoptimization.** Gao et al. (2023) quantified this risk: performance on the true objective initially improves with optimization against a reward model, then degrades. The optimal number of optimization steps scales as sqrt(reward_model_size). This provides a quantitative framework for predicting when Goodhart effects emerge.

## Methodology

### Design

Within-subjects repeated-measures design with 3 conditions, 3 topics, and 20 iterations per condition-topic pair. Total: 3 conditions x 3 topics x 20 iterations = 180 research outputs.

**Independent variables:**
- Condition (between-iteration treatment):
  - (a) ReAct only — no stored reflections, each iteration is independent
  - (b) ReAct + Reflexion — stored verbal reflections retrieved at each iteration
  - (c) ReAct + Reflexion + compound memory — episodic memories (specific past experiences) + strategy memories (general principles extracted from reflections)
- Iteration number (1-20)

**Dependent variables:** SUNFIRE score, human expert rating (1-7), semantic novelty, repetition rate, reflection quality.

**Random effect:** Research subtopic (3 topics), treated as a random effect to generalize beyond specific topics.

### Data Requirements

- **Research subtopics:** 3 well-defined AI research subtopics with known state-of-the-art, selected to span difficulty levels. Candidates: (1) prompt engineering techniques for reasoning, (2) LLM watermarking methods, (3) efficient fine-tuning approaches. Final selection based on availability of ground-truth literature for evaluation.
- **Per iteration output:** One complete research artifact consisting of literature synthesis, hypothesis, supporting argument, and critique response. Expected length: 500-2000 tokens.
- **Human evaluation:** 2 domain experts rate iterations 1, 5, 10, 15, and 20 for all condition-topic pairs (2 experts x 5 checkpoints x 3 conditions x 3 topics = 90 ratings per expert). Inter-rater reliability measured via ICC.
- **Embedding model:** Sentence-transformer (all-MiniLM-L6-v2 or equivalent) for computing semantic novelty.
- **Prior iteration corpus:** All previous outputs within a condition-topic pair, used for computing novelty and repetition metrics.

### Procedure

1. **Topic selection.** Select 3 AI research subtopics. For each, compile a reference set of 20-30 key papers representing the current state-of-the-art.
2. **Baseline calibration.** Run 5 independent single-shot generations per topic (no iteration) to establish baseline SUNFIRE distribution. This provides the reference point for measuring improvement.
3. **Iteration runs.** For each condition-topic pair, execute 20 consecutive iterations:
   - Each iteration: literature search (simulated via reference set) → hypothesis generation → self-critique → reflection generation (conditions b, c) → memory storage (conditions b, c) → next iteration
   - Condition (a): No reflection stored; each iteration receives only the task prompt and reference set
   - Condition (b): Reflection from prior iteration stored and retrieved via recency; up to 3 most recent reflections included in prompt
   - Condition (c): Same as (b) plus episodic memories (specific past outputs flagged as high/low quality) and strategy memories (general principles extracted every 5 iterations)
4. **Automated scoring.** Score every output with SUNFIRE. Record all sub-scores.
5. **Human evaluation.** At iterations 1, 5, 10, 15, 20: two domain experts rate each output on a 7-point scale covering novelty, correctness, significance, and clarity. Experts are blinded to condition and iteration number. Present outputs in randomized order.
6. **Novelty and repetition analysis.** For each iteration, compute: (a) mean cosine distance to all prior iteration outputs (semantic novelty), (b) 4-gram overlap rate with all prior outputs (repetition), (c) unique concept count (extracted via NER/keyword extraction).
7. **Reflection quality audit.** For conditions (b) and (c), manually evaluate a sample of 30 reflections (10 per topic): does the reflection identify a genuine problem? Is the identified problem addressed in the next iteration?
8. **Memory utilization tracking.** For condition (c), log which memories are retrieved at each iteration and whether retrieved memories are actually incorporated into the output.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| Quality trend | SUNFIRE score as a function of iteration | Mixed-effects model: SUNFIRE ~ log(iteration) + condition + (1\|topic) | Log model captures expected diminishing returns; random intercept for topic accounts for difficulty variation |
| Trend significance | Monotonic trend in SUNFIRE over iterations | Mann-Kendall trend test per condition | Non-parametric, robust to non-normality, appropriate for time-series trend detection |
| Condition comparison | Pairwise quality difference at iteration 20 | Wilcoxon signed-rank test (paired by topic) | Paired non-parametric test; small sample (3 topics) makes parametric tests unreliable |
| SUNFIRE-human agreement | Correlation between SUNFIRE and human ratings | Spearman rank correlation at checkpoint iterations | Non-parametric correlation; tests whether SUNFIRE improvement reflects genuine quality gain |
| Human inter-rater reliability | Agreement between 2 expert raters | Intraclass Correlation Coefficient (ICC, two-way mixed, absolute agreement) | Standard reliability measure for ordinal ratings |
| Semantic novelty trend | Mean embedding distance from prior outputs over iterations | Mann-Kendall trend test | Detects whether novelty declines (agent repeating itself) or is maintained |
| Repetition rate trend | 4-gram overlap with prior outputs over iterations | Descriptive (plot trajectory) | Complements novelty metric; detects surface-level repetition |
| Model fit comparison | log(iteration) vs linear vs plateau model | AIC/BIC comparison | Determines which trajectory shape best describes improvement dynamics |

**Effect size reporting:** Report Cohen's d for condition differences and R-squared for the fitted trajectory model.

## Success Criteria

- Condition (c) (ReAct + Reflexion + compound memory) shows a statistically significant positive Mann-Kendall trend (p < 0.05) in SUNFIRE score over 20 iterations.
- SUNFIRE score improvement from iteration 1 to iteration 20 is >= 20% in condition (c).
- Human expert ratings confirm SUNFIRE improvement is genuine: Spearman rho >= 0.5 between SUNFIRE scores and human ratings at the 5 checkpoint iterations.
- Log(iteration) model provides better fit (lower AIC) than linear model, confirming diminishing returns pattern.
- Condition (c) outperforms condition (a) at iteration 20 (Wilcoxon p < 0.05).

## Failure Criteria

- No condition shows a significant positive Mann-Kendall trend (p >= 0.05). This would indicate that iterative research does not reliably improve quality, undermining the core Loop 1 mechanism.
- SUNFIRE scores improve but human ratings do not correlate (Spearman rho < 0.3). This would indicate a Goodhart effect — the agent optimizes the metric without genuine quality improvement.
- Semantic novelty shows a significant negative trend (agent is repeating itself with cosmetic variation rather than producing genuinely new ideas).
- No significant difference between conditions (a), (b), and (c). This would indicate that reflection and memory add no value over independent attempts.

**If the experiment fails:** Investigate whether failure is due to (1) reflection quality (reflections are too generic to be useful), (2) memory retrieval (relevant reflections aren't retrieved), (3) task formulation (research subtopics are too broad for measurable progress), or (4) evaluation noise (SUNFIRE variance is too high to detect real trends). Consider narrower task definitions or alternative reflection mechanisms.

## Estimated Cost & Timeline

- **API calls:** ~$200-400 (180 research iterations x 3-5 API calls per iteration for generation, critique, reflection; plus SUNFIRE scoring)
- **Human evaluation:** ~16-24 hours across 2 experts (90 outputs each at ~5-8 min per rating)
- **Reflection audit:** ~4-6 hours (30 reflections, detailed manual analysis)
- **Analysis and write-up:** ~8-12 hours
- **Calendar time:** 3-4 weeks (human evaluation is the bottleneck; iteration runs can complete in 2-3 days)

## Dependencies

- **E02** (LLM-as-Judge Reliability): Evaluation methodology must be validated before trusting SUNFIRE as the primary automated metric.
- **E06** (SUNFIRE Calibration): SUNFIRE scoring must be calibrated and its correlation with human judgment established.
- **E07** (Reflexion Basic Benefit): Basic reflexion benefit on simpler tasks should be confirmed before testing on research tasks.
- **E08** (Critic Effectiveness): The self-critique component must produce useful feedback for reflection to have material to work with.

## Informs

- **E13** (Full ReAct-Reflexion Loop): Determines the expected improvement trajectory and optimal iteration count for the complete loop.
- **Path B implementation:** Establishes whether ReAct-Reflexion is viable as the core improvement mechanism, and what memory architecture is needed.
- **Iteration budget planning:** The fitted log(iteration) model provides a principled basis for deciding how many iterations to run in production (point of diminishing returns).

## References

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *International Conference on Learning Representations (ICLR 2023)*.
- Shinn, N., Cassano, F., Gopinath, A., Shakkottai, K., Labash, A., & Karthik, B. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Zelikman, E., Wu, Y., Mu, J., & Goodman, N. D. (2022). STaR: Bootstrapping Reasoning With Reasoning. *Advances in Neural Information Processing Systems (NeurIPS 2022)*.
- Goodhart, C. A. E. (1984). Problems of Monetary Management: The U.K. Experience. In *Monetary Theory and Practice*. Macmillan.
- Manheim, D., & Garrabrant, S. (2019). Categorizing Variants of Goodhart's Law. *arXiv preprint arXiv:1803.04585*.
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. *International Conference on Machine Learning (ICML 2023)*.
