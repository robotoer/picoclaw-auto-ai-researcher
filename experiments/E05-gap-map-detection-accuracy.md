# E05: Gap Map Detection Accuracy

> **Layer 2 — Component Validation**

## Status: Planned

## Hypothesis

The Gap Map (doc 06) can identify research gaps that domain experts independently confirm as genuine and important, achieving precision >= 60% and recall >= 40% against expert-identified gaps. Furthermore, the Gap Map's priority scores correlate with expert importance ratings at Spearman rho >= 0.3.

This hypothesis is falsifiable: if precision falls below 40% (majority of detected gaps are not real), or if priority scores show no correlation with expert importance, the hypothesis is rejected and the Gap Map requires fundamental redesign.

## Why This Matters

The Gap Map drives the system's research direction. Every downstream loop — ReAct-Reflexion (E09), evolutionary search (E10), full research loops (E13, E14) — selects topics based on Gap Map output. If the Gap Map identifies false gaps (areas that are not actually unexplored), the system wastes compute on solved problems. If it misses real gaps, the system ignores important open questions. A precise but low-recall Gap Map produces a narrow research agenda; a high-recall but imprecise one produces noise. Both failure modes propagate through the entire pipeline.

## Background & Prior Work

**FunSearch and automated search.** Romera-Paredes et al. (2023) demonstrated that program search with LLMs can discover mathematical results (cap set improvement from 496 to 512, a 3.2% gain), but required millions of samples and an automated evaluator as a hallucination guard. The Gap Map faces an analogous challenge: it must search a literature space and identify what is missing, where hallucinated gaps are the equivalent of invalid programs.

**Novelty measurement foundations.** Uzzi et al. (2013) showed that atypical combinations of prior work predict 2x citation impact, suggesting that genuine gaps — areas where expected combinations have not been explored — are measurable in principle. Arts & Veugelers (2020) extended this with cosine distance of embeddings to quantify novelty, providing a methodological basis for the semantic matching we use to compare Gap Map output against expert gaps. The Disruption Index offers another lens on whether a gap represents an incremental extension or a structural hole in the literature.

**LLM-as-Judge limitations.** Zheng et al. (2023) documented position bias, verbosity bias, and self-preference bias in LLM evaluations, with Cohen's kappa as low as 0.21 on specialized tasks. Since the Gap Map uses LLM-based analysis to identify gaps, these biases may cause systematic blind spots (e.g., favoring gaps described verbosely in existing literature while missing tersely noted open problems).

**AI Scientist evaluation quality.** Lu et al. (2024) found automated reviewer accuracy of 0.65 vs. 0.66 for humans, but correlation with human scores was only r=0.18. This low correlation warns that even when aggregate accuracy looks reasonable, item-level agreement can be poor — precisely the regime where Gap Map precision and recall matter most.

## Methodology

### Design

Mixed-methods evaluation comparing automated Gap Map output against independent expert gap identification on a well-defined AI subdomain. The study uses a within-domain design where both the system and experts analyze the same literature corpus, enabling direct precision/recall measurement.

Independent variable: gap identification method (automated Gap Map vs. expert survey).
Dependent variables: precision, recall, gap quality rating, priority-importance correlation.
Control: expert consensus gaps serve as ground truth.

### Data Requirements

- **Literature corpus:** 200-300 papers from a well-defined subdomain (e.g., "LLM alignment techniques" or "efficient fine-tuning methods"). Papers sourced from Semantic Scholar, filtered by venue quality (top-tier + notable workshops) and recency (2021-2025). The subdomain must be narrow enough for experts to have comprehensive knowledge but broad enough to contain meaningful gaps.
- **Expert panel:** 5 domain experts with >= 3 publications in the target subdomain. Recruited from a mix of academic and industry research positions to reduce perspective bias.
- **Expert survey instrument:** Structured questionnaire asking each expert to (a) list up to 15 gaps they perceive in the subdomain, (b) rate each gap on importance (1-5) and tractability (1-5), (c) explain why each gap exists (methodological limitation, data availability, conceptual barrier, etc.). Questionnaire piloted with 1 additional expert not in the panel.
- **Annotation budget:** 2 research assistants for semantic matching verification (~8 hours each).

### Procedure

1. **Subdomain selection.** Choose a subdomain with active research, identifiable boundaries, and available experts. Document inclusion/exclusion criteria for papers.
2. **Corpus construction.** Collect 200-300 papers using Semantic Scholar API with keyword and venue filters. Verify coverage by checking that key papers identified by at least 2 experts are included. If coverage < 90%, expand the corpus.
3. **Gap Map execution.** Run the full Gap Map pipeline (doc 06) on the corpus. Record all identified gaps, their descriptions, supporting evidence (which papers border the gap), and priority scores. Log pipeline runtime and any errors.
4. **Expert survey.** Administer the structured questionnaire to all 5 experts independently (no communication between experts during survey). Collect responses within a 2-week window.
5. **Expert gap consolidation.** Merge expert gaps using semantic clustering (cosine similarity >= 0.75 on sentence embeddings). A "consensus gap" requires identification by >= 2 experts. Record the full set (all expert gaps) and the consensus set (>= 2 experts) separately.
6. **Semantic matching.** Encode all Gap Map gaps and expert gaps using sentence-transformers (all-mpnet-base-v2, which outperforms MiniLM on semantic textual similarity tasks). Match Gap Map gaps to expert gaps using cosine similarity >= 0.75. Two research assistants independently verify all matches and non-matches; resolve disagreements by discussion.
7. **Precision calculation.** For each Gap Map gap, determine whether it matches any expert gap (true positive) or not (false positive). Precision = TP / (TP + FP).
8. **Recall calculation.** For each consensus expert gap, determine whether it is matched by any Gap Map gap (true positive) or not (false negative). Recall = TP / (TP + FN).
9. **Quality rating.** Ask experts to rate a blinded sample of 30 Gap Map gaps (mix of true positives and false positives) on importance (1-5) and tractability (1-5). Compute importance x tractability composite.
10. **False positive analysis.** Categorize all false positive Gap Map gaps: (a) gap is real but outside expert awareness, (b) gap was a real gap but has been recently addressed, (c) gap is a misunderstanding of the literature (hallucination), (d) gap is too vague to be actionable.
11. **False negative analysis.** For each expert gap missed by the Gap Map, analyze why: (a) relevant papers were missing from the corpus, (b) the gap requires cross-subdomain knowledge, (c) the gap is implicit (experts infer it from experience, not from paper content), (d) pipeline error.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| Precision | TP / (TP + FP) for Gap Map gaps matched to expert gaps | Bootstrap 95% CI (10,000 resamples) | Non-parametric, appropriate for proportion data with small n |
| Recall | TP / (TP + FN) for expert consensus gaps matched by Gap Map | Bootstrap 95% CI (10,000 resamples) | Same justification |
| F1 | Harmonic mean of precision and recall | Derived from above | Standard composite metric |
| Priority-importance correlation | Spearman rho between Gap Map priority scores and expert mean importance ratings | Spearman rank correlation with permutation test (n may be small) | Ordinal data, no distributional assumptions |
| Gap quality rating | Mean expert rating (importance x tractability) for Gap Map gaps | Descriptive (mean, SD, IQR) with Wilcoxon signed-rank comparing TP vs FP quality | Paired comparison on expert-rated items |
| Expert agreement | Fleiss' kappa across 5 experts on gap identification overlap | Point estimate with standard error | Multi-rater agreement for > 2 raters |
| Semantic match threshold sensitivity | Precision/recall at cosine thresholds 0.65, 0.70, 0.75, 0.80, 0.85 | Descriptive (precision-recall curve) | Validates robustness to matching threshold choice |

**Effect size reporting:** Report Cohen's d for quality rating differences between true positive and false positive gaps. A meaningful quality difference requires d >= 0.5.

## Success Criteria

- Precision >= 60% (bootstrap 95% CI lower bound) — the majority of Gap Map outputs correspond to genuine research gaps.
- Recall >= 40% (bootstrap 95% CI lower bound) — the Gap Map finds a meaningful fraction of expert-recognized gaps.
- Spearman rho >= 0.3 between Gap Map priority scores and expert importance ratings, with p < 0.05.
- Expert mean quality rating for true positive gaps >= 3.0 on the 1-5 importance scale.

## Failure Criteria

- Precision < 40% — the majority of detected gaps are not real, indicating the Gap Map hallucinates gaps at an unacceptable rate. This would require adding verification stages or constraining the gap detection to higher-confidence signals.
- Recall < 20% — the Gap Map misses the vast majority of real gaps, making it unsuitable as a research direction selector.
- Spearman rho < 0.1 or negative — Gap Map priority scores are unrelated to actual importance, meaning the prioritization mechanism must be redesigned.
- Fleiss' kappa among experts < 0.3 — experts themselves do not agree on what the gaps are, indicating the evaluation framework is flawed and must be revised before re-running.

**If the experiment fails:** Investigate whether failure stems from corpus coverage (papers missing), LLM comprehension (misunderstanding paper content), gap inference (inability to identify what is absent from what is present), or prioritization (gaps found but ranked incorrectly). Consider augmenting the Gap Map with citation network analysis, co-authorship network gaps, or explicit "future work" extraction from papers.

## Estimated Cost & Timeline

- **API calls:** ~$100-200 (embedding 200-300 papers, Gap Map pipeline LLM calls, sentence-transformer encoding)
- **Expert recruitment and survey:** ~$500-1,000 (honoraria for 5 experts, ~2 hours each) or in-kind collaboration
- **Research assistant time:** ~16 hours total for semantic match verification
- **Analysis and write-up:** ~8-12 hours
- **Calendar time:** 3-4 weeks (expert survey requires scheduling buffer)

## Dependencies

- **E01** (Claim Extraction Accuracy): The Gap Map ingests extracted claims to build its literature representation. If claim extraction is unreliable (E01 fails), gaps may be identified from noisy input.
- **E04** (Knowledge Graph Consistency): The Gap Map traverses the knowledge graph to find structural holes. KG inconsistencies produce spurious gaps.

## Informs

- **E09** (ReAct-Reflexion Loop): Uses Gap Map to select research topics for the agent to investigate.
- **E13** (Full Research Loop): Gap Map is the entry point for the complete research pipeline.
- **E14** (Evolutionary Research Loop): Evolutionary search explores gaps identified by the Gap Map.
- **All Layer 3-4 experiments** that use Gap Map for topic selection depend on validated gap detection.

## References

- Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A. (2023). Mathematical discoveries from program search with large language models. *Nature*, 625, 468-475.
- Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. (2013). Atypical combinations and scientific impact. *Science*, 342(6157), 468-472.
- Arts, S., & Veugelers, R. (2020). Technology familiarity, recombinant novelty, and breakthrough invention. *Industrial and Corporate Change*, 24(6), 1215-1246.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv preprint arXiv:2408.06292*.
