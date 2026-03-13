# E14: Evolutionary Research Loop

> **Layer 4 — System-Level Hypotheses**

## Status: Planned

## Hypothesis

An evolutionary research loop — MAP-Elites over research strategies combined with ReAct execution and SUNFIRE fitness evaluation — discovers research strategies that outperform the best hand-designed strategy by >= 40% in SUNFIRE score and produces higher-quality outputs than the pure ReAct-Reflexion loop (E13) by >= 15%, within 500 generations.

This hypothesis is falsifiable: if the best evolved strategy does not exceed the best seed strategy by >= 40% in SUNFIRE score, or if human experts do not prefer evolved outputs over E13 outputs at p < 0.05, the hypothesis is rejected.

## Why This Matters

This experiment tests doc 09 Loop 3 and doc 11 Path A — the hypothesis that evolutionary search over the space of research strategies is more effective than iterating within a single fixed strategy. FunSearch (Romera-Paredes et al., 2023) proved that LLM-guided evolutionary search discovers novel mathematical constructions, and AlphaEvolve (DeepMind, 2025) matched state-of-the-art on 75% and improved on 20% of 50+ open problems. The question is whether this evolutionary advantage transfers from well-defined mathematical and algorithmic domains to the open-ended domain of research strategy design.

This experiment also directly tests the bitter lesson (Sutton, 2019): does more search (via evolution over strategies) beat more engineering (via a single carefully designed ReAct-Reflexion loop)? If evolutionary search works, it suggests the system can self-improve its own research methodology — a meta-learning capability that would make the system fundamentally more scalable than a hand-designed pipeline.

If evolution fails to improve over the baseline, it indicates that research strategy design requires human insight that LLM-guided mutation cannot replicate, and the project should prioritize Path B (deeper reflexion within a fixed strategy) over Path A (evolutionary strategy search).

## Background & Prior Work

**MAP-Elites.** Mouret & Clune (2015) introduced MAP-Elites as a quality-diversity algorithm that maintains an archive of high-performing solutions across a grid of behavior descriptors. The key metrics are QD-Score (sum of fitness across filled cells), Coverage (fraction of cells filled), and Best Performance (fitness of the best solution). MAP-Elites has been shown to discover diverse, high-quality solutions in domains where a single optimum is insufficient — exactly the situation in research strategy design, where different strategies may excel in different research contexts.

**FunSearch.** Romera-Paredes et al. (2023, Nature) demonstrated that pairing LLMs with evolutionary search produces programs that surpass the best known results on combinatorial problems (cap set problem, online bin packing). FunSearch showed log-linear improvement with number of evaluations, suggesting that sufficient compute can yield continued gains. E14 tests whether this scaling pattern holds for research strategies, which are more complex and less clearly evaluable than mathematical constructions.

**AlphaEvolve.** DeepMind (2025) scaled the evolutionary approach to 50+ open problems, matching SOTA on 75% and improving on 20%. This establishes the generality of LLM-guided evolution beyond a single domain. However, all AlphaEvolve problems have clear automated evaluation — the key challenge for E14 is that SUNFIRE scoring is a proxy metric that may not perfectly capture research quality.

**ReAct and Reflexion.** Yao et al. (2023, ICLR) and Shinn et al. (2023, NeurIPS) provide the execution framework within each strategy evaluation. Each candidate strategy is executed using a ReAct agent with Reflexion, making the evolutionary search a meta-level optimization over the parameters and structure of the inner loop.

**Reward overoptimization.** Gao et al. (2023) showed that optimizing too aggressively against a proxy reward leads to degraded true performance. This is a critical risk for E14: evolved strategies that maximize SUNFIRE score may exploit weaknesses in the scoring function rather than genuinely improving research quality. Human expert validation is essential to detect this failure mode.

**Goodhart's Law.** Manheim & Garrabrant (2019) categorized four variants of Goodhart's Law (regressional, extremal, causal, adversarial), all of which are potential failure modes when evolving strategies against a proxy metric. E14 must monitor for extremal Goodhart effects specifically, where strategies that score well on SUNFIRE do so by gaming measurable dimensions while neglecting unmeasured aspects of quality.

**Self-training collapse.** Shumailov et al. (2024, Nature) documented model collapse from recursive self-training. In E14, evolved strategies generate outputs that are scored by SUNFIRE, which is itself LLM-based. This recursive structure risks a form of evolutionary collapse where strategies converge on outputs that satisfy the LLM scorer but lack genuine quality.

## Methodology

### Design

Evolutionary experiment using MAP-Elites with LLM-guided mutation and crossover. Research strategies are represented as programs (~100-200 lines) that control the full research iteration pipeline. Strategies are evaluated by executing them on a fixed set of research tasks and measuring SUNFIRE fitness. The experiment runs for 500 generations with human expert validation of top evolved outputs.

Independent variables: generation number (0-500), strategy genotype (program code).
Dependent variables: SUNFIRE fitness, QD-Score, Coverage, Best Performance, human expert quality ratings.
Behavior descriptor axes: (1) exploration-exploitation ratio (topic diversity across tasks), (2) depth-breadth ratio (papers-per-topic vs topics-per-iteration).
Control: seed strategies from E13 best-performing configuration plus 19 hand-designed variants.

### Data Requirements

- **Research tasks:** 5 AI research tasks spanning different subdomains, drawn from the same task set used in E13 for direct comparison. Each task specifies a research area, seed papers, and an initial gap map.
- **Seed strategies (20 total):** The best-performing strategy from E13, plus 19 variants covering different configurations of: topic selection policy (gap-priority, diversity-weighted, recency-biased), search depth (1-3 hops in citation graph), hypothesis generation prompt template (structured, free-form, analogical), critique style (strict, lenient, adversarial), iteration structure (linear, tree-branching, portfolio), memory management (FIFO, relevance-ranked, diversity-filtered).
- **Strategy representation:** Each strategy is a Python-like program of 100-200 lines defining the control flow and parameters of one research iteration. Strategies are human-readable and LLM-mutable.
- **Evaluation infrastructure:** Each strategy is evaluated by executing it on all 5 research tasks and averaging the SUNFIRE scores. Each evaluation requires approximately 5 API calls (1 per task).
- **Expert panel:** 5 experts for final validation, same caliber as E13 (PhD-level researchers with relevant publications).
- **E13 outputs:** Top-5 outputs from E13 serve as the comparison baseline for human evaluation.

### Procedure

1. **Initialization (generation 0):** Seed the MAP-Elites archive with 20 hand-designed strategies. Evaluate each on all 5 research tasks. Compute behavior descriptors and place in archive grid.

2. **Per generation (x500):**
   a. Select 2 parent strategies from the archive (tournament selection within occupied cells).
   b. Generate offspring via LLM-guided mutation: prompt the LLM with the parent strategy code and instructions to modify one or more components (e.g., "Make the topic selection more exploratory" or "Add a step that cross-references hypotheses with recent failures"). Mutation types: point mutation (change a parameter), structural mutation (add/remove/reorder steps), crossover (combine elements from both parents).
   c. Evaluate the offspring strategy on all 5 research tasks. Compute average SUNFIRE fitness.
   d. Compute behavior descriptors for the offspring.
   e. If the offspring occupies an empty cell or has higher fitness than the current occupant, insert it into the archive. Otherwise, discard.
   f. Log: strategy code, fitness, behavior descriptors, task-level scores, archive state.

3. **Monitoring (every 50 generations):** Record QD-Score, Coverage, Best Performance. Cluster all archived strategies by code similarity (embedding-based). Identify emergent strategy motifs — recurring structural patterns in high-fitness strategies.

4. **Human validation (after generation 500):** Select the top-5 evolved strategies by SUNFIRE fitness. Execute each on 3 new research tasks (not used during evolution) to test generalization. Collect outputs and include them in a blinded evaluation pool alongside the top-5 outputs from E13. Experts rate each output on a 1-7 scale (novelty, rigor, significance) and provide binary "worth developing" judgments.

5. **Analysis:**
   a. Plot QD-Score, Coverage, Best Performance trajectories over 500 generations.
   b. Compare evolved vs seed strategy fitness with bootstrap 95% CI.
   c. Compare evolved vs E13 outputs with Mann-Whitney U on expert ratings.
   d. Cluster evolved strategies and characterize what makes high-fitness strategies qualitatively different from seeds.
   e. Track compute efficiency: quality per dollar across generations.
   f. Analyze for Goodhart effects: do strategies with high SUNFIRE scores but low expert ratings exist? What do they exploit?

### Metrics & Statistical Tests

| Metric | Description | Test |
|--------|-------------|------|
| QD-Score trajectory | Sum of fitness across all filled archive cells over generations | Visualization, Mann-Kendall trend test |
| Coverage trajectory | Fraction of behavior descriptor grid cells filled | Visualization, convergence analysis |
| Best Performance trajectory | Maximum fitness in archive over generations | Visualization, bootstrap 95% CI at generation 500 |
| Evolved vs seed fitness | SUNFIRE score of best evolved vs best seed strategy | Bootstrap 95% CI on difference, one-tailed test for >= 40% improvement |
| Evolved vs E13 expert ratings | Human quality ratings of evolved outputs vs E13 outputs | Wilcoxon signed-rank test (paired by expert), Mann-Whitney U |
| Expert preference | Blinded preference between evolved and E13 outputs | Exact binomial test on preference counts |
| Strategy cluster analysis | Structural similarity of high-fitness strategies | Hierarchical clustering on strategy embeddings, Kruskal-Wallis across clusters |
| Generalization | Performance of evolved strategies on held-out tasks | Paired comparison of training vs held-out task fitness |
| Compute efficiency | Quality per API dollar across generations | Cumulative cost curve, marginal quality per dollar |
| Goodhart detection | Correlation between SUNFIRE and expert ratings | Spearman correlation, identification of high-SUNFIRE/low-expert outliers |

## Success Criteria

- Best evolved strategy outperforms best seed strategy by >= 40% in SUNFIRE score (bootstrap 95% CI excludes 0%).
- QD-Score shows sustained increase throughout 500 generations (not plateauing before generation 100).
- Human experts prefer evolved strategy outputs over E13 outputs at p < 0.05 (Wilcoxon signed-rank).
- Coverage >= 50% of the behavior descriptor grid, indicating diverse strategy discovery.
- Evolved strategies are qualitatively different from seeds — they contain structural innovations, not just parameter variations.

## Failure Criteria

- Evolution stalls within the first 50 generations (QD-Score plateaus, no new archive insertions).
- Best evolved strategy does not exceed best seed by >= 40% in SUNFIRE score.
- Evolved strategies are not meaningfully different from seeds (only surface-level rewording or minor parameter changes, as determined by strategy cluster analysis).
- Experts do not prefer evolved outputs over E13 outputs (p > 0.10), indicating that SUNFIRE optimization does not transfer to genuine quality improvement (Goodhart failure).
- Evolved strategies show signs of reward hacking — high SUNFIRE scores driven by exploiting scoring function weaknesses rather than genuine research quality.

## Estimated Cost & Timeline

| Component | Estimate |
|-----------|----------|
| API costs (500 generations x 5 evaluations, plus mutations) | $2,000-5,000 |
| Expert evaluation (5 experts, ~3-4 hours each) | 15-20 hours expert time |
| Infrastructure and engineering | 1-2 weeks setup |
| Evolution execution (500 generations) | 4-8 weeks |
| Human validation and analysis | 2 weeks |
| **Total calendar time** | **6-12 weeks** |

## Dependencies

| Experiment | What E14 needs from it |
|------------|----------------------|
| E06 | Validated SUNFIRE scoring with known correlation to expert judgment |
| E09 | Iteration dynamics characterization (expected improvement rates, plateau detection) |
| E10 | Scaling behavior data to predict compute requirements |
| E13 | Baseline ReAct-Reflexion performance, best strategy as top seed, comparison outputs |

## Informs

- **Path A vs Path B decision** (doc 11): determines whether evolutionary strategy search (Path A) should be prioritized over deeper reflexion within a fixed strategy (Path B).
- **Doc 09 Loop 3 viability:** validates or invalidates the evolutionary loop architecture.
- **Scaling decisions** (doc 10): if evolution works, the quality-vs-compute scaling curve informs budget allocation for production runs.
- **E15 design:** the best evolved strategy becomes a condition in the loop comparison experiment.
- **Goodhart robustness:** empirical data on whether SUNFIRE is robust enough to serve as an evolutionary fitness function, or whether it needs additional guardrails.
- **Bitter lesson validation:** concrete evidence on whether more search beats more engineering in the research strategy domain.

## References

- Mouret, J.-B. & Clune, J. (2015). Illuminating Search Spaces by Mapping Elites.
- Romera-Paredes, B. et al. (2023). Mathematical Discoveries from Program Search with Large Language Models. *Nature*.
- DeepMind (2025). AlphaEvolve: A Gemini-Powered Coding Agent for Designing Advanced Algorithms.
- Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.
- Gao, L. et al. (2023). Scaling Laws for Reward Model Overoptimization.
- Manheim, D. & Garrabrant, S. (2019). Categorizing Variants of Goodhart's Law.
- Shumailov, I. et al. (2024). AI Models Collapse When Trained on Recursively Generated Data. *Nature*.
- Sutton, R. (2019). The Bitter Lesson.
- Lu, C. et al. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.
- SUNFIRE framework (internal doc 07).
