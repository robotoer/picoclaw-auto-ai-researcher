# E13: Full ReAct-Reflexion Research Loop

> **Layer 4 — System-Level Hypotheses**

## Status: Planned

## Hypothesis

A complete ReAct-Reflexion research loop — topic selection, literature search, hypothesis generation, critique, experiment design, reflection, memory update, repeat — produces research outputs that independent experts rate as "potentially publishable at a workshop" (>= 4/7 on a quality scale) within 50 iterations, starting from a cold start on a chosen subdomain.

This hypothesis is falsifiable: if no subdomain produces an output rated >= 4/7 by a majority of experts after 50 iterations, or if output quality shows no statistically significant improvement over iterations (flat or declining Mann-Kendall trend), the hypothesis is rejected.

## Why This Matters

This is the minimal viable autonomous research loop (doc 11 section 3). It composes results from E01 (claim extraction), E04 (knowledge graph construction), E05 (gap map detection), E07 (reflexion-based improvement), E08 (critic filtering), and E09 (iteration dynamics) into a single end-to-end system. If this composed loop cannot produce workshop-quality output, the entire project premise — that LLM-driven autonomous research loops can generate meaningful scientific contributions — needs fundamental revision. Every subsequent system-level experiment (E14, E15) and the full architecture described in doc 07 Phase 2+ depends on this loop achieving a baseline level of competence.

This experiment also establishes the concrete baseline against which evolutionary (E14) and composed (E15) approaches will be compared, making it the foundational measurement for all architecture decisions.

## Background & Prior Work

**ReAct.** Yao et al. (2023, ICLR) introduced the ReAct paradigm for synergizing reasoning and acting in language models. The interleaved thought-action-observation traces provide interpretable decision-making and grounded reasoning, which are prerequisites for a research loop that must decide what to investigate, execute searches, and interpret results in sequence.

**Reflexion.** Shinn et al. (2023, NeurIPS) demonstrated that LLM agents can improve through verbal self-reflection stored in episodic memory. Reflexion agents achieved 91% pass@1 on HumanEval (vs. 80% for baseline), showing that reflection-based memory can drive meaningful improvement across iterations. The key question for E13 is whether this improvement transfers from well-defined coding tasks to the open-ended domain of research hypothesis generation.

**AI Scientist.** Lu et al. (2024) demonstrated automated end-to-end scientific discovery, with an automated reviewer achieving human-level accuracy (0.65). AI Scientist v2 achieved papers exceeding the average human acceptance threshold, establishing that LLM-driven research pipelines can in principle produce publishable-quality output. However, AI Scientist operates on a narrower scope (ML experiments with code execution); E13 tests whether a similar quality level is achievable for hypothesis-stage research without code execution.

**Karpathy autoresearch.** The 3-file architecture with a 5-minute time budget demonstrated that approximately 100 experiments per night is feasible, establishing a practical throughput baseline. E13 adopts a similar philosophy of rapid iteration but focuses on research ideation quality rather than code experiment execution.

**SUNFIRE framework.** The SUNFIRE scoring system (doc 07) provides a multi-dimensional evaluation of research outputs covering Surprise, Usefulness, Novelty, Feasibility, Impact, Rigor, and Elegance. E13 uses SUNFIRE as the automated quality metric, with human expert ratings as the ground truth validation.

**Self-training collapse.** Shumailov et al. (2024, Nature) showed that models trained on their own outputs can degenerate. This motivates careful monitoring of hypothesis diversity and quality trajectories across iterations — a loop that feeds its own outputs back risks mode collapse into repetitive or degrading ideas.

**Scaling laws.** FunSearch (Romera-Paredes et al., 2023, Nature) demonstrated log-linear improvement with number of evaluations. If similar scaling holds for research loops, 50 iterations may be sufficient to observe meaningful quality gains, but we must test this empirically.

## Methodology

### Design

Multi-subdomain longitudinal study with human expert evaluation at fixed checkpoints. Three AI research subdomains are run through the full loop for 50 iterations each, with quality assessed at iterations 1, 10, 25, and 50. A human baseline comparison provides an anchor for interpreting absolute quality levels.

Independent variables: iteration number (1-50), subdomain (3 levels).
Dependent variables: SUNFIRE score (automated), expert quality ratings (1-7 scale on novelty, rigor, significance), binary "worth developing further" judgment, hypothesis diversity, Gap Map evolution metrics.
Control: human researcher baseline on the same subdomains with equivalent time investment.

### Data Requirements

- **Subdomains:** 3 AI research subdomains with active open questions: (1) efficient inference for mixture-of-experts architectures, (2) reward model robustness under distribution shift, (3) multi-agent coordination for code generation. Selected for having sufficient recent literature, well-defined open problems, and distinct enough characteristics to test generalization.
- **Seed knowledge graph:** 200 papers per subdomain (600 total) from arXiv cs.AI, cs.LG, cs.CL, and cs.MA, published 2022-2025. Papers selected by relevance ranking from Semantic Scholar API. Each paper processed through the E01 claim extraction pipeline and integrated into the knowledge graph (E04).
- **Initial Gap Map:** Generated from the seed KG using the E05 gap detection pipeline, providing the starting state for topic selection.
- **Human expert panel:** 5 experts (PhD-level researchers with publications in relevant subdomains), recruited for blinded evaluation. Each expert evaluates all checkpointed outputs across all subdomains.
- **Human baseline:** 1-2 human researchers (PhD students or postdocs in AI) given equivalent time (~10 hours per subdomain, matching estimated loop compute time) to brainstorm hypotheses on the same subdomains, with access to the same seed papers. Evaluated with the identical rubric by the same expert panel.

### Procedure

1. **Bootstrap (per subdomain):** Ingest 200 seed papers through E01 claim extraction. Build KG (E04). Generate initial Gap Map (E05). Initialize empty reflection memory.

2. **Per iteration (×50 per subdomain, 150 total):**
   a. Gap Map selects the highest-priority research topic/gap.
   b. ArXiv search retrieves 5-10 recent relevant papers on the selected topic.
   c. Claim extraction (E01) processes retrieved papers, updates KG.
   d. Hypothesis generation: ReAct agent generates a research hypothesis addressing the selected gap, grounded in extracted claims. Retrieves relevant reflections from memory.
   e. Critic review (E08): LLM critic evaluates the hypothesis for novelty, feasibility, and rigor. Assigns a pass/fail with justification.
   f. If the hypothesis survives critique: experiment design step generates a concrete experimental plan. If it fails: log failure reason, proceed to reflection.
   g. SUNFIRE scoring of the output (hypothesis + experiment design if applicable).
   h. Reflexion (E07): agent reflects on what worked and what failed, generates a reflection summary.
   i. Memory store: reflection is embedded and stored for future retrieval.
   j. KG and Gap Map update: new claims integrated, gap priorities adjusted based on progress.

3. **Checkpoints at iterations 1, 10, 25, 50:** Collect the best output produced up to that iteration (by SUNFIRE score) for human evaluation.

4. **Human evaluation protocol:** Each expert independently rates each checkpointed output on a 1-7 scale across three dimensions (novelty, rigor, significance). Experts also provide: (a) binary "Would you consider developing this hypothesis further?" judgment, (b) identification of the closest existing work (novelty check), (c) open-ended description of main flaws. Evaluation is fully blinded — experts do not know which outputs are from which iteration, subdomain, or whether they come from the system or the human baseline.

5. **Human baseline execution:** Human researchers receive the same seed papers and subdomain descriptions. They brainstorm hypotheses over 10 hours per subdomain. Their best outputs are included in the blinded expert evaluation pool.

### Metrics & Statistical Tests

| Metric | Description | Test |
|--------|-------------|------|
| Expert quality rating | Mean of novelty + rigor + significance (1-7) per output | Mixed-effects model: iteration -> rating, subdomain as random effect |
| Quality trajectory | Trend in expert ratings across checkpoints | Mann-Kendall trend test (p < 0.05) |
| Checkpoint comparison | Iteration 50 vs iteration 1 ratings | Wilcoxon signed-rank test (paired by expert) |
| Expert confidence intervals | Uncertainty on mean ratings at each checkpoint | Bootstrap 95% CI (10,000 resamples) |
| SUNFIRE trajectory | Automated score across all 50 iterations | Mann-Kendall trend test, Spearman correlation with iteration number |
| SUNFIRE-expert correlation | Relationship between SUNFIRE and human ratings | Spearman rank correlation with bootstrap CI |
| "Worth developing" rate | Proportion of outputs rated as worth pursuing | Exact binomial CI, comparison across checkpoints |
| Hypothesis diversity | Semantic diversity of generated hypotheses | Pairwise cosine distance of hypothesis embeddings, tracked over iterations |
| Gap Map evolution | Number of gaps closed, new gaps opened, gap priority changes | Descriptive statistics, visualization |
| Memory utilization | Frequency and relevance of retrieved reflections | Retrieval frequency distribution, relevance scoring |
| Iteration time and cost | Wall-clock time and API cost per iteration | Descriptive statistics, cumulative cost curves |
| System vs human baseline | Comparison of loop outputs vs human researcher outputs | Mann-Whitney U test on expert ratings |

## Success Criteria

- At least 2 of 3 subdomains produce at least one output rated >= 4/7 (mean across dimensions) by a majority of experts (>= 3 of 5).
- Statistically significant positive Mann-Kendall trend in expert ratings across checkpoints (p < 0.05).
- At least 3 hypotheses across all subdomains are rated "worth developing further" by a majority of experts.
- SUNFIRE scores show a positive trajectory correlated with expert ratings (Spearman rho > 0.3).

## Failure Criteria

- No output in any subdomain exceeds 3/7 mean expert rating after 50 iterations.
- Quality trajectory is flat or declining across iterations (Mann-Kendall p > 0.10 or negative trend).
- Zero hypotheses rated "worth developing further" by a majority of experts.
- SUNFIRE scores diverge from expert ratings (negative or near-zero correlation), indicating the automated metric is unreliable for guiding the loop.

## Estimated Cost & Timeline

| Component | Estimate |
|-----------|----------|
| API costs (150 iterations, KG operations, searches) | $500-1,000 |
| Expert evaluation (5 experts, ~4-6 hours each) | 20-30 hours expert time |
| Human baseline researcher time | ~30 hours (10 hours x 3 subdomains) |
| Infrastructure and engineering | 1-2 weeks setup |
| Loop execution | 2-4 weeks |
| Expert evaluation period | 1 week |
| Analysis and reporting | 1 week |
| **Total calendar time** | **4-8 weeks** |

## Dependencies

| Experiment | What E13 needs from it |
|------------|----------------------|
| E01 | Validated claim extraction pipeline with known precision/recall |
| E02 | Calibrated LLM-judge for intermediate evaluations |
| E04 | Functional knowledge graph with paper ingestion pipeline |
| E05 | Gap Map detection and priority ranking |
| E06 | SUNFIRE scoring calibration and validation |
| E07 | Reflexion memory mechanism with demonstrated improvement |
| E08 | Critic filtering with known false positive/negative rates |
| E09 | Iteration dynamics characterization (expected convergence rate, plateau timing) |

## Informs

- **Go/no-go decision** for full system build (doc 07 Phase 2+). If E13 fails, the project needs a fundamental pivot.
- **Baseline for E14:** evolutionary approach needs a concrete ReAct-Reflexion performance level to beat.
- **Baseline for E15:** loop composition experiments need single-loop performance as the control.
- **Path B and Path C implementation** (doc 11): determines whether the reflexion-based approach has sufficient ceiling.
- **SUNFIRE calibration:** real-world correlation data between SUNFIRE scores and expert ratings feeds back to improve the scoring system.

## References

- Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.
- Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *NeurIPS 2023*.
- Lu, C. et al. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery.
- Romera-Paredes, B. et al. (2023). Mathematical Discoveries from Program Search with Large Language Models. *Nature*.
- Shumailov, I. et al. (2024). AI Models Collapse When Trained on Recursively Generated Data. *Nature*.
- Gao, L. et al. (2023). Scaling Laws for Reward Model Overoptimization.
- Manheim, D. & Garrabrant, S. (2019). Categorizing Variants of Goodhart's Law.
- Sutton, R. (2019). The Bitter Lesson.
- SUNFIRE framework (internal doc 07).
- Karpathy autoresearch architecture (2024). 3-file architecture, 5-min time budget.
