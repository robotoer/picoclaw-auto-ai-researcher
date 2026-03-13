# E15: Loop Comparison and Composition

> **Layer 4 — System-Level Hypotheses**

## Status: Planned

## Hypothesis

Composing multiple loop mechanisms — ReAct-Reflexion, evolutionary strategy search, and self-play debate — produces research outputs that are >= 25% higher quality than any single loop mechanism alone, as measured by both SUNFIRE scores and independent expert ratings, when controlling for total compute budget.

This hypothesis is falsifiable: if the 3-loop composition does not significantly outperform all single-loop conditions on expert ratings at p < 0.01, or if the composition's quality-per-dollar is worse than 3x that of the best single loop, the hypothesis is rejected.

## Why This Matters

Doc 09 proposes a 6-layer composed architecture as the target system design. The central bet of this entire project is that composition produces emergent capability — that the whole is greater than the sum of its parts. But composition also adds complexity, latency, and cost. Without rigorous measurement, there is no way to distinguish genuine emergent benefit from mere overhead.

This experiment is the ultimate test of the bitter lesson thesis (Sutton, 2019) as applied to research automation: does more search (via composing multiple search mechanisms) beat more engineering (via a single well-designed loop)? If composition works, it validates the full 6-layer architecture and justifies the engineering investment. If it fails, the project should adopt the simplest effective loop and invest in scaling it rather than composing additional mechanisms.

This is also the only experiment that directly compares all loop types head-to-head under equal compute budgets, making it the definitive input to the final architecture decision described in doc 07.

## Background & Prior Work

**ReAct.** Yao et al. (2023, ICLR) introduced interleaved reasoning and acting for language model agents. E13 validates this as the inner loop of the research pipeline, establishing the single-loop baseline.

**Reflexion.** Shinn et al. (2023, NeurIPS) demonstrated verbal self-reflection for iterative improvement, achieving 91% pass@1 on HumanEval. E13 combines ReAct with Reflexion as the foundational loop mechanism.

**MAP-Elites and evolutionary search.** Mouret & Clune (2015) introduced quality-diversity optimization. FunSearch (Romera-Paredes et al., 2023, Nature) and AlphaEvolve (DeepMind, 2025) demonstrated that LLM-guided evolution discovers novel solutions across diverse domains. E14 validates evolutionary strategy search as a standalone loop mechanism.

**AI Scientist.** Lu et al. (2024) achieved automated reviewer accuracy of 0.65, and AI Scientist v2 produced papers exceeding the average human acceptance threshold. These results demonstrate that composed pipelines (literature search + hypothesis + experiment + writing + review) can achieve publishable quality, but the specific contribution of composition vs. simply having a good base model is unclear.

**Self-play and debate.** E11 tests self-play debate as a mechanism for improving research output quality through adversarial critique. The debate loop provides a complementary mechanism to reflexion: reflexion improves by reflecting on past failures, while debate improves by adversarial pressure from a second agent.

**Scaling laws.** FunSearch shows log-linear improvement with evaluations, but it is unknown whether this scaling is additive across composition. If composing loops provides merely additive benefits, the overhead may not justify the complexity. If benefits are superlinear (emergent), composition is strongly motivated.

**Reward overoptimization.** Gao et al. (2023) showed diminishing and eventually negative returns from proxy reward optimization. Composing multiple loops that all optimize SUNFIRE score could exacerbate Goodhart effects, making it essential to validate with human expert ratings, not just automated scores.

**Goodhart's Law.** Manheim & Garrabrant (2019) identified four variants. Composition may provide natural Goodhart robustness if different loops optimize different aspects of quality, creating a form of multi-objective optimization. Alternatively, composition may amplify Goodhart effects if all loops exploit the same proxy weaknesses. E15 distinguishes these possibilities.

**Self-training collapse.** Shumailov et al. (2024, Nature) showed recursive self-training causes model collapse. Composed loops with multiple feedback pathways may be more or less susceptible to this — multiple diverse signals could prevent collapse, or multiple recursive pathways could accelerate it.

## Methodology

### Design

Five-condition comparison experiment with equal compute budgets. All conditions are evaluated on the same 5 research tasks with both automated (SUNFIRE) and human expert ratings. The design enables both pairwise comparisons and omnibus tests across all conditions, plus ablation analysis of the 3-loop composition.

**Conditions:**
- (a) ReAct-Reflexion only — the E13 loop
- (b) Evolutionary search only — the E14 loop
- (c) Self-play debate only — the E11 loop
- (d) ReAct-Reflexion + Debate — 2-loop composition
- (e) ReAct-Reflexion + Debate + Evolutionary search — 3-loop composition

Independent variables: loop condition (5 levels), research task (5 levels).
Dependent variables: SUNFIRE scores, expert quality ratings (1-7 on novelty/rigor/significance), output diversity, novel insight count, compute efficiency, failure rate.
Control: equal compute budget across conditions (measured in API tokens/dollars). Each condition receives the same total API budget, ensuring that any quality differences are due to how compute is used, not how much.

### Data Requirements

- **Research tasks:** 5 AI research tasks spanning distinct subdomains, consistent with tasks used in E13 and E14 for comparability. Tasks are designed to vary in difficulty and openness (from well-scoped technical questions to broad open problems).
- **Seed data:** Each task begins with the same seed knowledge graph and gap map, identical across all 5 conditions.
- **Compute budget:** Determined by the average cost of 100 iterations in E13, multiplied by 5 conditions. Each condition receives exactly 1/5 of the total budget.
- **Expert panel:** 3 experts (PhD-level researchers), each evaluating all outputs from all conditions across all tasks. Evaluation is fully blinded — experts see only the research output, not which condition produced it.
- **E13, E14, E11 outputs:** Where possible, outputs from prior experiments on overlapping tasks are reused rather than regenerated, ensuring consistency. Where tasks differ, new runs are executed under the standardized budget.

### Procedure

1. **Task and budget standardization:** Define 5 research tasks with identical seed data. Compute the per-condition API budget based on E13 cost data. Verify that all conditions can produce meaningful output within the budget.

2. **Condition execution (per condition, per task):**

   **(a) ReAct-Reflexion only:** Run the E13 loop for as many iterations as the budget allows (~100 iterations). Collect all outputs.

   **(b) Evolutionary search only:** Run the E14 evolutionary loop for as many generations as the budget allows. Use SUNFIRE fitness. Collect the top strategy's outputs.

   **(c) Self-play debate only:** Run the E11 debate loop for as many rounds as the budget allows. Two LLM agents alternate proposing and critiquing hypotheses. Collect all outputs.

   **(d) ReAct-Reflexion + Debate (2-loop composition):** Each ReAct-Reflexion iteration is followed by a debate round where a second agent critiques and the original agent defends or revises. The debate outcome feeds into the reflexion memory. Budget is split approximately 60/40 between reflexion iterations and debate rounds.

   **(e) ReAct-Reflexion + Debate + Evolutionary search (3-loop composition):** The inner loop is condition (d). The outer loop evolves the strategy controlling condition (d) using MAP-Elites. Budget is split approximately 50/30/20 between reflexion, debate, and evolutionary overhead.

3. **Output collection:** For each condition-task pair, collect: (i) the single best output by SUNFIRE score, (ii) the top-3 outputs by SUNFIRE score, (iii) all outputs for diversity analysis.

4. **Human evaluation:** All best outputs from all condition-task pairs (5 conditions x 5 tasks = 25 outputs) are pooled and randomized. Each expert rates each output on novelty (1-7), rigor (1-7), and significance (1-7). Experts also identify "genuinely novel insights" in each output (count and brief description) and flag any outputs that appear to be reward-hacked (high surface quality but substantively hollow).

5. **Ablation analysis (condition e only):** For the 3-loop composition, measure the marginal contribution of each component by selectively disabling it:
   - Disable debate: run ReAct-Reflexion + Evolutionary only.
   - Disable evolution: run ReAct-Reflexion + Debate only (identical to condition d).
   - Disable reflexion memory: run ReAct + Debate + Evolutionary without episodic memory.
   Compare each ablated variant against the full composition on the same tasks.

6. **Equal-compute normalization:** For each condition, compute quality-per-dollar as (mean expert rating) / (total API cost). This enables comparison of whether composition delivers proportional returns on the additional orchestration overhead.

### Metrics & Statistical Tests

| Metric | Description | Test |
|--------|-------------|------|
| Expert quality rating | Mean of novelty + rigor + significance (1-7) per output | Friedman test across 5 conditions, post-hoc Wilcoxon with Holm correction |
| Condition (e) vs best single loop | 3-loop composition vs max of (a), (b), (c) | Wilcoxon signed-rank (paired by task), p < 0.01 threshold |
| Effect size | Magnitude of quality difference between composed and single loops | Cliff's delta with bootstrap 95% CI |
| Quality-per-dollar | Mean expert rating divided by API cost | Descriptive comparison, bootstrap 95% CI on ratio |
| SUNFIRE scores | Automated quality scores across all conditions | Friedman test, Spearman correlation with expert ratings per condition |
| Output diversity | Pairwise cosine distance of output embeddings within each condition | Kruskal-Wallis across conditions |
| Novel insight count | Expert-identified genuinely new ideas per output | Poisson regression with condition as predictor |
| Failure rate | Proportion of iterations producing unusable output | Exact binomial CI per condition, chi-squared comparison |
| Ablation contribution | Marginal quality loss from removing each component in condition (e) | Paired Wilcoxon on task-level ratings, full vs ablated |
| SUNFIRE-expert agreement | Per-condition correlation between SUNFIRE and expert ratings | Spearman correlation, DeLong test comparing AUCs if binarized |
| Reward hacking detection | Expert-flagged hollow outputs per condition | Fisher's exact test comparing rates across conditions |

## Success Criteria

- Condition (e) — 3-loop composition — outperforms all single-loop conditions (a, b, c) by >= 25% on mean expert ratings (Friedman test p < 0.01, post-hoc Wilcoxon with Holm correction p < 0.01 for each pairwise comparison).
- Ablation analysis shows positive interaction effects: each component of the composition contributes measurably (removing any one component reduces quality by a statistically significant amount, p < 0.05).
- Quality-per-dollar for condition (e) is within 2x of the best single-loop condition, indicating that the composition overhead is manageable relative to the quality gain.
- Novel insight count is highest for condition (e), with at least 30% more expert-identified novel ideas than the best single loop.

## Failure Criteria

- No statistically significant difference in expert ratings between composed and single-loop conditions (Friedman p > 0.05), indicating that composition adds complexity without benefit.
- Quality-per-dollar for composition is > 3x worse than the best single loop, indicating that overhead outweighs any quality benefit.
- Ablation shows that one component dominates — removing it collapses quality while removing the others has no effect — indicating that the "composition" is really just one effective loop with dead weight.
- Experts flag a significantly higher rate of reward-hacked outputs in composed conditions, indicating that composition amplifies Goodhart effects.
- Condition (e) shows lower output diversity than single loops, indicating that composition leads to convergent mode collapse rather than emergent exploration.

## Estimated Cost & Timeline

| Component | Estimate |
|-----------|----------|
| API costs (5 conditions x 5 tasks x 100 iterations equivalent) | $3,000-8,000 |
| Expert evaluation (3 experts, ~8-13 hours each) | 25-40 hours expert time |
| Infrastructure and engineering (composition orchestration) | 2-3 weeks setup |
| Condition execution | 4-6 weeks |
| Ablation runs | 1-2 weeks |
| Human evaluation period | 1-2 weeks |
| Analysis and reporting | 1-2 weeks |
| **Total calendar time** | **8-14 weeks** |

## Dependencies

| Experiment | What E15 needs from it |
|------------|----------------------|
| E09 | Iteration dynamics characterization for budget estimation and convergence expectations |
| E10 | Scaling behavior to predict quality-vs-compute curves for each condition |
| E11 | Validated self-play debate mechanism, baseline debate-only performance |
| E13 | Validated ReAct-Reflexion loop, baseline performance data, best strategy configuration |
| E14 | Validated evolutionary loop, baseline evolutionary performance, best evolved strategy |

## Informs

- **Final architecture decision** (doc 07): whether to build the full 6-layer composed system, adopt a simpler 2-loop composition, or use the best single loop.
- **Scaling budget allocation** (doc 10): the quality-per-dollar curves determine how to allocate compute in production deployment.
- **Whether to pursue full 6-layer composition** or stick with a simpler architecture that captures most of the benefit at lower complexity and cost.
- **Component interaction model:** which pairs or triples of loops synergize and which are redundant, informing future system design beyond the immediate project.
- **Goodhart robustness:** whether composition naturally mitigates or amplifies proxy metric gaming, informing evaluation system design.
- **Bitter lesson quantification:** concrete evidence on how much additional search (via composition) buys in the research automation domain, calibrating expectations for future scaling.

## References

- Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.
- Mouret, J.-B. & Clune, J. (2015). Illuminating Search Spaces by Mapping Elites.
- Romera-Paredes, B. et al. (2023). Mathematical Discoveries from Program Search with Large Language Models. *Nature*.
- DeepMind (2025). AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms.
- Lu, C. et al. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.
- Shumailov, I. et al. (2024). AI Models Collapse When Trained on Recursively Generated Data. *Nature*.
- Gao, L. et al. (2023). Scaling Laws for Reward Model Overoptimization.
- Manheim, D. & Garrabrant, S. (2019). Categorizing Variants of Goodhart's Law.
- Sutton, R. (2019). The Bitter Lesson.
- Karpathy autoresearch architecture (2024). 3-file architecture, 5-min time budget.
- SUNFIRE framework (internal doc 07).
