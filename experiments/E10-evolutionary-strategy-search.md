# E10: Evolutionary Strategy Search

> **Layer 3 — Loop Mechanics**

## Status: Planned

## Hypothesis

MAP-Elites evolutionary search over research strategy programs finds strategies that outperform the initial seed strategies by >= 30% in SUNFIRE score within 200 generations, achieving >= 40% coverage of the strategy behavior space.

This hypothesis is falsifiable: if QD-score does not increase >= 30% from initialization to generation 200, or if coverage remains below 40%, or if the best evolved elite does not outperform the best seed strategy by >= 20%, the hypothesis is rejected.

## Why This Matters

Doc 09 Loop 3 and doc 11 Path A propose evolutionary program search as a key self-improvement mechanism, directly inspired by FunSearch (Romera-Paredes et al., 2023) and AlphaEvolve (DeepMind, 2025). Those systems achieved remarkable results — FunSearch improved cap set bounds from 496 to 512, and AlphaEvolve matched SOTA on 75% of 50+ open problems and improved on 20%. However, both operated on mathematical problems with perfect automated evaluation (correctness is verifiable). Research strategies are evaluated by noisy proxies (SUNFIRE), introducing a fundamental difference: fitness landscape noise may prevent evolutionary search from making reliable progress.

If evolution cannot find good strategies under noisy evaluation, Path A is non-viable. If it can, we learn how many generations are needed, what mutation operators work, and whether the discovered strategies are genuinely diverse or converge to a single template.

## Background & Prior Work

**MAP-Elites.** Mouret & Clune (2015) introduced MAP-Elites as a quality-diversity algorithm that maintains an archive of solutions mapped across user-defined behavior dimensions. Unlike traditional evolutionary algorithms that optimize a single objective, MAP-Elites seeks to fill an archive with the best-performing solution in each cell of a discretized behavior space. Standard metrics include QD-Score (sum of all elite fitnesses), Coverage (percentage of occupied cells), and Best Performance (fitness of the single best elite). Fontaine et al. (2021) extended this with differentiable quality-diversity methods.

**FunSearch.** Romera-Paredes et al. (2023) used LLMs as mutation operators in an evolutionary search over programs, discovering new mathematical constructions. Key finding: millions of samples were required, and the approach relied on a perfect automated evaluator (mathematical correctness). The LLM-as-mutator paradigm is the direct inspiration for our strategy evolution approach.

**AlphaEvolve.** DeepMind (2025) improved on FunSearch by requiring thousands of samples rather than millions, matching SOTA on 75% of 50+ problems and improving on 20%. The efficiency gain came from better prompt engineering and ensemble evaluation. This suggests that LLM-based evolution can be practical at moderate compute budgets.

**Noisy fitness evaluation.** Evolutionary search under noisy fitness is a well-studied problem. Standard mitigations include re-evaluation (averaging fitness over multiple evaluations), increased population size, and statistical selection operators. Jin & Branke (2005) provide a comprehensive survey. The key risk is that noise causes the archive to fill with lucky evaluations rather than genuinely good solutions.

**Reward model overoptimization.** Gao et al. (2023) showed that optimizing against a learned reward model initially improves true performance but eventually degrades it. This is directly relevant: SUNFIRE is our "reward model," and evolved strategies may learn to exploit SUNFIRE-specific patterns rather than producing genuinely better research.

## Methodology

### Design

Evolutionary search using MAP-Elites with LLM-based mutation, compared against two baselines: random search (same compute budget) and single-strategy hill climbing.

**Independent variables:**
- Search method: (a) MAP-Elites with LLM mutation, (b) random search (sample random strategies, evaluate, keep best), (c) hill climbing (mutate best-so-far, keep if better)
- Generation number (1-200)

**Dependent variables:** QD-Score, Coverage, Best Performance, strategy diversity, convergence rate.

### Data Requirements

- **Strategy representation:** Research strategies encoded as short programs (50-200 lines of pseudocode or Python) specifying: topic selection heuristic (keyword-based, citation-based, gap-based), literature search depth (number of papers, search breadth), hypothesis generation style (analogical, compositional, contrastive, extrapolative), critique depth (surface check, deep analysis, adversarial), iteration count (1-10 research refinement cycles).
- **Behavior descriptors:** Two axes defining the MAP-Elites archive grid:
  - Axis 1: Exploration vs exploitation balance (0-1 continuous, discretized to 10 bins). Measured by ratio of novel topic exploration to depth on known topics.
  - Axis 2: Depth vs breadth of literature coverage (0-1 continuous, discretized to 10 bins). Measured by ratio of papers cited to topics covered.
  - Total archive capacity: 100 cells (10 x 10 grid).
- **Seed strategies:** 10 hand-designed strategies spanning the behavior space, including: breadth-first survey strategy, depth-first specialist strategy, analogy-based cross-pollination strategy, contrarian strategy (seeks to challenge consensus), incremental strategy (small extensions of known results), and 5 variations.
- **Evaluation tasks:** 3 research tasks used for fitness evaluation. Each strategy is run on all 3 tasks, fitness = mean SUNFIRE score across tasks. Tasks are fixed across all generations to ensure comparability.
- **Compute budget:** Each strategy evaluation requires 3 task runs x ~5 API calls per run = 15 API calls. At 200 generations with ~10 mutations per generation, total = ~30,000 API calls for MAP-Elites. Baselines use the same total budget.

### Procedure

1. **Strategy space definition.** Formalize the strategy program template with mutable parameters and structural elements. Define the behavior descriptor computation from strategy execution traces.
2. **Seed initialization.** Design 10 seed strategies, evaluate each on the 3 research tasks, compute behavior descriptors, and place in the MAP-Elites archive.
3. **MAP-Elites loop (200 generations):**
   - Select a random occupied cell from the archive
   - Use the LLM to generate a mutated variant of the selected strategy (prompt includes the strategy code and instructions to modify one or more components)
   - Evaluate the mutant on all 3 research tasks, compute mean SUNFIRE score (fitness) and behavior descriptors
   - If the mutant's cell in the archive is empty, or if the mutant's fitness exceeds the current occupant's fitness, place the mutant in the archive
   - Log: generation number, mutation type, parent fitness, offspring fitness, cell placement, archive state
4. **Baseline: random search.** Generate 2000 random strategies (same total compute as MAP-Elites: 200 generations x 10 mutations). Evaluate each. Track best-so-far and archive occupancy if placed in the same grid.
5. **Baseline: hill climbing.** Start from the best seed. Each generation, generate 10 mutations, evaluate, keep the best if it improves. 200 generations. Track best-so-far.
6. **Re-evaluation.** After evolution completes, re-evaluate the top 10 elites from each method on 5 new research tasks (not used during evolution) to test generalization. Re-evaluate each elite 3 times to measure fitness variance.
7. **Mutation analysis.** Categorize all mutations (structural: add/remove/reorder steps; parametric: change thresholds/counts; semantic: change strategy logic). Compute success rate by mutation category.
8. **Strategy convergence analysis.** Compute pairwise edit distance between all elites in the final archive. Identify strategy clusters and common motifs.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| QD-Score trajectory | Sum of all elite fitnesses over generations | Bootstrap 95% CI on improvement from gen 0 to gen 200 | Non-parametric CI avoids distributional assumptions on QD-score |
| Coverage trajectory | Fraction of archive cells occupied over generations | Descriptive (plot with 95% CI from 5 independent runs) | Coverage is a proportion; CI from replicate runs |
| Best Performance trajectory | Fitness of best elite over generations | Descriptive (plot with 95% CI from 5 independent runs) | Tracks optimization progress |
| Generation comparison | Elite fitnesses at gen 1 vs 50 vs 100 vs 200 | Kruskal-Wallis test, post-hoc Dunn's test with Bonferroni | Non-parametric comparison of elite fitness distributions across time points |
| Method comparison | Best fitness: MAP-Elites vs random vs hill climbing | Kruskal-Wallis test across 5 replicate runs | Non-parametric; 5 replicates per method |
| Generalization | Re-evaluation fitness vs training fitness | Paired Wilcoxon signed-rank test | Detects overfitting to training tasks |
| Mutation productivity | Success rate by mutation category | Chi-squared test of independence | Tests whether mutation type affects success probability |

**Effect size reporting:** Report rank-biserial correlation for Wilcoxon tests, epsilon-squared for Kruskal-Wallis.

## Success Criteria

- QD-Score increases >= 30% from initialization (generation 0) to generation 200, with bootstrap 95% CI excluding 0% improvement.
- Coverage reaches >= 40% of the 100-cell archive (>= 40 occupied cells).
- Best elite in MAP-Elites outperforms the best seed strategy by >= 20% in SUNFIRE score.
- MAP-Elites outperforms random search and hill climbing on QD-Score (Kruskal-Wallis p < 0.05).
- Generalization: re-evaluated fitness on held-out tasks is within 20% of training fitness (strategies are not overfit to training tasks).

## Failure Criteria

- QD-Score plateaus within 20 generations (no significant trend after generation 20). This would indicate that LLM mutations are too random or that the fitness landscape is too noisy for evolutionary progress.
- Best elite does not outperform the best seed strategy. This would mean evolution adds nothing over hand-design, undermining the entire Path A approach.
- MAP-Elites performs no better than random search. This would indicate that the structured search adds no value and the fitness landscape lacks exploitable structure.
- Generalization gap > 50% (strategies are heavily overfit to training tasks). This would indicate SUNFIRE exploitation (Goodhart effect) rather than genuine strategy quality.

**If the experiment fails:** Investigate whether failure is due to (1) noisy fitness (re-evaluate with more samples per strategy), (2) mutation quality (LLM mutations are too random; try more constrained mutation operators), (3) behavior descriptor design (axes don't capture meaningful variation), or (4) strategy representation (programs are too complex for effective mutation). Consider simpler strategy representations or ensembled fitness evaluation.

## Estimated Cost & Timeline

- **API calls:** ~$500-800 (MAP-Elites: ~30,000 calls; random baseline: ~30,000 calls; hill climbing: ~30,000 calls; re-evaluation: ~2,000 calls. At ~$0.005-0.01 per call)
- **5 replicate runs:** Multiply base cost by 5 for statistical power = $2,500-4,000 total
- **Human analysis:** ~8-12 hours for mutation analysis and strategy interpretation
- **Analysis and write-up:** ~8-12 hours
- **Calendar time:** 4-6 weeks (compute-bound; API rate limits may extend timeline. Replicate runs can be parallelized.)

## Dependencies

- **E06** (SUNFIRE Calibration): SUNFIRE must be a sufficiently reliable fitness signal for evolution to make progress. If SUNFIRE noise is too high, fitness-based selection becomes random selection.
- **E02** (LLM-as-Judge Reliability): Evaluation reliability bounds the signal-to-noise ratio of fitness evaluations.

## Informs

- **E14** (Evolutionary Loop): Full evolutionary loop design depends on knowing whether MAP-Elites works for strategy search and what parameters (generation count, population size, mutation operators) are effective.
- **Path A implementation:** Directly determines feasibility and expected compute cost of evolutionary self-improvement.
- **Mutation operator design:** Analysis of productive vs unproductive mutations informs how to prompt the LLM for strategy modification.

## References

- Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites. *arXiv preprint arXiv:1504.04909*.
- Fontaine, M. C., Togelius, J., Nikolaidis, S., & Hoover, A. K. (2021). Differentiable Quality Diversity. *Advances in Neural Information Processing Systems (NeurIPS 2021)*.
- Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A. (2023). Mathematical discoveries from program search with large language models. *Nature*, 625, 468-475.
- DeepMind. (2025). AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms. *Google DeepMind Technical Report*.
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. *International Conference on Machine Learning (ICML 2023)*.
- Jin, Y., & Branke, J. (2005). Evolutionary optimization in uncertain environments — a survey. *IEEE Transactions on Evolutionary Computation*, 9(3), 303-317.
- Goodhart, C. A. E. (1984). Problems of Monetary Management: The U.K. Experience. In *Monetary Theory and Practice*. Macmillan.
- Manheim, D., & Garrabrant, S. (2019). Categorizing Variants of Goodhart's Law. *arXiv preprint arXiv:1803.04585*.
