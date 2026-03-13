# E03: Semantic Novelty Measurement

> **Layer 1 — Foundational Capabilities**

## Status: Planned

## Hypothesis

Semantic distance metrics (embedding cosine distance and atypical reference combinations) can distinguish novel research contributions from incremental ones with AUC >= 0.70 against human novelty ratings, and the best metric achieves Spearman rho >= 0.3 with 2-year citation count.

This hypothesis is falsifiable: if all metrics yield AUC < 0.60, novelty cannot be measured automatically with these approaches and alternative signals must be explored.

## Why This Matters

The SUNFIRE framework (E06), Gap Map (E05), MAP-Elites fitness function (E10), and reward model (E11) all require an automated novelty signal to distinguish genuinely new contributions from restated or incremental work. Without reliable novelty measurement, the system cannot reward exploration over exploitation, and optimization loops risk converging on trivially safe but uninformative hypotheses. Novelty measurement is also the key differentiator between an AI research assistant and a sophisticated search engine.

## Background & Prior Work

**Atypical combinations and scientific impact.** Uzzi et al. (2013) measured novelty via co-citation pairs in the Web of Science, finding that papers combining references in atypical ways (low observed-to-expected ratio of reference pair co-occurrence) are 2x more likely to be in the top 5th percentile of citations. This operationalizes novelty as combinatorial unusualness — connecting ideas that are not typically connected. The method requires a large reference co-occurrence matrix, which can be constructed from Semantic Scholar data.

**Embedding-based novelty.** Arts & Veugelers (2020) proposed measuring novelty using cosine distance of document embeddings from the centroid of prior work in the same area. This captures semantic novelty — a paper that says something distant from what has been said before is likely more novel. The SemDis platform (Beaty & Johnson, 2021) provides validated semantic distance measures for creativity research, supporting the theoretical link between semantic distance and novelty.

**Citation as impact proxy.** Citation count is an imperfect proxy for novelty — highly cited papers may be methodological tools rather than novel contributions, and novel papers may be initially under-cited (the "sleeping beauty" phenomenon). Two-year citation count balances recency against early-recognition delay. Uzzi et al. (2013) used 8-year windows; we use 2-year as a practical compromise given the recency of our paper sample.

**Topic modeling approaches.** BERTopic (Grootendorst, 2022) provides neural topic clusters. A paper's distance from the nearest existing topic cluster captures topical novelty — work that does not fit neatly into existing categories.

## Methodology

### Design

Observational study comparing 4 automated novelty metrics against 2 ground truth signals (human novelty ratings and 2-year citation count). The 100 papers serve as the unit of analysis.

Independent variables: novelty metric type (4 levels).
Dependent variables: AUC-ROC against human labels, Spearman rho with citation count.

### Data Requirements

- **Papers:** 100 papers from cs.AI and cs.LG published in 2022-2023 (allowing 2+ years of citation accumulation by 2025). Stratified as 50 highly-cited/influential papers (top 10% by 2-year citation count within their subfield) and 50 average-impact papers (25th-75th percentile). Sampled from Semantic Scholar API.
- **Citation data:** 2-year citation count for each paper from Semantic Scholar.
- **Reference lists:** Full reference lists for all 100 papers plus a background corpus of ~10,000 papers from the same period for computing reference co-occurrence statistics.
- **Human ratings:** 3 raters classify each paper as "novel" (introduces a substantially new idea, method, or finding) vs "incremental" (extends or applies existing work without a qualitatively new contribution). Also collect a 1-7 Likert novelty score for finer-grained analysis.
- **Embeddings:** Document embeddings for all 100 papers plus the background corpus, computed using sentence-transformers (all-MiniLM-L6-v2 for efficiency; all-mpnet-base-v2 as robustness check).

### Procedure

1. **Paper sampling.** Query Semantic Scholar for cs.AI/cs.LG papers from 2022-2023. Compute 2-year citation counts. Sample 50 from the top 10% and 50 from the 25th-75th percentile. Verify no overlap in author sets exceeding 20% to reduce confounding.

2. **Human annotation.** 3 raters (graduate students or researchers in AI/ML) independently classify each paper as novel vs incremental and provide a 1-7 novelty score. Annotators read the abstract, introduction, and contributions section. Compute Fleiss' kappa for 3 raters. Resolve disagreements by majority vote for the binary label.

3. **Metric computation.**

   **(a) Embedding distance from centroid.** Compute document embeddings using sentence-transformers all-MiniLM-L6-v2 on the abstract + introduction. For each paper, compute the centroid of all papers in the same subfield published before it. Novelty score = 1 - cosine_similarity(paper_embedding, centroid).

   **(b) Atypical reference combinations (Uzzi et al. method).** Build a reference co-occurrence matrix from the background corpus. For each paper, compute the z-score of each reference pair's co-occurrence frequency. Paper novelty score = the 10th percentile z-score across all reference pairs (following Uzzi et al., who used the tail of the distribution to capture the most unusual combination).

   **(c) BERTopic-based topic novelty.** Fit BERTopic on the background corpus to identify topic clusters. For each target paper, compute the minimum cosine distance to any topic cluster centroid. Papers far from all existing clusters are scored as more novel.

   **(d) LLM novelty judgment.** Prompt each of 2 LLMs (Claude Sonnet, GPT-4o) with the abstract, introduction, and related work section. Ask for a 1-7 novelty score with structured reasoning. Average across models.

4. **Evaluation.** For each metric:
   - Compute AUC-ROC predicting the binary human novelty label (novel vs incremental).
   - Compute Spearman rho with 2-year citation count.
   - Compute Spearman rho with mean human novelty score (1-7).
   - Compute precision@10 and precision@20 for retrieving the most novel papers.

5. **Metric combination.** Train a simple logistic regression combining all 4 metrics to predict human novelty labels. Evaluate with leave-one-out cross-validation to avoid overfitting on 100 samples.

6. **Uzzi replication.** Specifically test whether papers with atypical reference combinations (bottom 10th percentile z-score) are significantly more likely to be in the top citation decile, replicating the 2x finding from Uzzi et al. (2013).

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| AUC-ROC | Area under ROC curve for each novelty metric vs human binary labels | DeLong test for comparing AUCs between metrics; bootstrap 95% CI for each AUC | DeLong test is the standard for AUC comparison on the same dataset; bootstrap handles non-normal AUC distributions |
| Spearman rho (citation) | Rank correlation between metric score and 2-year citation count | Permutation test (10,000 permutations) | Non-parametric; avoids assuming linear relationship between novelty and citation |
| Spearman rho (human score) | Rank correlation between metric score and mean human novelty score | Permutation test (10,000 permutations) | Same justification |
| Precision@k | Fraction of top-k papers by metric that are labeled novel by humans | Bootstrap 95% CI | Directly measures practical utility of ranking |
| Inter-rater reliability | Fleiss' kappa for 3 human raters | Point estimate with bootstrap SE | Standard multi-rater agreement; >= 0.4 required for meaningful ground truth |
| Uzzi replication | Odds ratio of top-citation-decile membership for atypical vs typical papers | Fisher's exact test | Small expected cell counts; Fisher's exact is appropriate |
| Combined metric | AUC of logistic regression combining all metrics | Leave-one-out CV to avoid overfitting | Only 100 samples; LOOCV maximizes training data per fold |

**Multiple comparisons:** 4 metrics are tested against the AUC >= 0.70 threshold. Apply Holm-Bonferroni correction when reporting which metrics individually pass the threshold.

## Success Criteria

- AUC >= 0.70 for at least one individual metric OR the combined metric (Holm-Bonferroni corrected).
- The best metric achieves Spearman rho >= 0.3 with 2-year citation count (permutation test p < 0.05).
- Human inter-rater Fleiss' kappa >= 0.4, confirming the ground truth is meaningful.

## Failure Criteria

- All individual metrics have AUC < 0.60 AND the combined metric has AUC < 0.65. This would indicate that novelty as perceived by humans cannot be captured by these computational approaches, at least not from the paper text and reference structure alone.
- Spearman rho with citation count is < 0.1 for all metrics. This would suggest either that novelty and impact are largely independent, or that the metrics capture neither.
- Human inter-rater kappa < 0.3. This would suggest "novelty" is too subjective for reliable annotation, requiring a revised operationalization.

**If the experiment fails:** Consider (1) richer input features (full paper text, figures, experimental results), (2) temporal novelty (was the idea new when published, even if the embedding is now close to later work?), (3) expert panel novelty assessment with detailed rubrics instead of binary labels, (4) novelty relative to a specific research question rather than the entire field, or (5) accepting that novelty must be decomposed into sub-dimensions (methodological novelty, problem novelty, result novelty) that may be independently measurable.

## Estimated Cost & Timeline

- **API calls:** ~$30-60 (embedding computation for 100 + 10,000 papers; LLM novelty judgment for 100 papers x 2 models)
- **Semantic Scholar API:** Free tier sufficient for citation data and reference lists
- **Human rating:** ~6-10 hours across 3 raters (100 papers at 3-5 min/paper reading abstract+intro)
- **Compute:** ~2-4 hours for embedding computation and BERTopic fitting on commodity hardware
- **Analysis and write-up:** ~6-10 hours
- **Calendar time:** 2-3 weeks (human rating scheduling is bottleneck; data collection can be automated)

## Dependencies

None. This is a foundational experiment with no upstream dependencies. However, access to Semantic Scholar API and a background corpus of AI papers is required.

## Informs

- **E06** (SUNFIRE): The novelty component of SUNFIRE's multi-objective scoring directly uses the best metric identified here.
- **E10** (MAP-Elites): The fitness function for MAP-Elites requires a novelty score to define quality dimensions.
- **E11** (Reward Model): The reward model must incorporate novelty; this experiment determines which novelty signal to use.
- **E13-E14** (Loop experiments): Quality measurement of generated hypotheses requires novelty assessment.

## References

- Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. (2013). Atypical Combinations and Scientific Impact. *Science*, 342(6157), 468-472.
- Arts, S., & Veugelers, R. (2020). Technology Familiarity, Recombinant Novelty, and Breakthrough Invention. *Industrial and Corporate Change*, 24(6), 1215-1246.
- Beaty, R. E., & Johnson, D. R. (2021). Automating Creativity Assessment with SemDis: An Open Platform for Computing Semantic Distance. *Behavior Research Methods*, 53, 757-780.
- Grootendorst, M. (2022). BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure. *arXiv preprint arXiv:2203.05794*.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
