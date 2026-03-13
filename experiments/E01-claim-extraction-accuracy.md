# E01: Claim Extraction Accuracy

> **Layer 1 — Foundational Capabilities**

## Status: Planned

## Hypothesis

LLMs can extract factual claims from AI research papers with >=80% precision and >=70% recall against human-annotated ground truth, using semantic similarity matching (cosine similarity >= 0.85) for claim alignment.

This hypothesis is falsifiable: if no tested model achieves these thresholds, or if the hallucinated claim rate exceeds 10%, the hypothesis is rejected.

## Why This Matters

The entire knowledge graph depends on accurate claim extraction. Downstream components — gap detection (E05), hypothesis generation (E06/SUNFIRE), hallucination prevention (E08/doc 08), and all Layer 3-4 loop experiments — consume extracted claims as input. If extraction is unreliable, every downstream system is built on noise. A false positive claim (hallucinated extraction) is particularly dangerous because it can propagate through the knowledge graph and bias hypothesis generation toward nonexistent findings.

## Background & Prior Work

**LLM-as-Judge reliability.** Zheng et al. (2023) found GPT-4 agreement with humans at 85%, exceeding human-human agreement at 81%, but also documented position bias, verbosity bias, and self-preference bias. Cohen's kappa on code evaluation was low (~0.21 for Java, ~0.10 for Python), suggesting that structured extraction tasks may have variable reliability depending on domain.

**AI Scientist evaluation.** Lu et al. (2024) reported automated reviewer accuracy of 0.65 (vs 0.66 for humans) and correlation r=0.18 with average human scores. This establishes that LLMs can approximate human judgment on scientific content, but the low correlation motivates careful measurement rather than assumption.

**Claim extraction context.** Scientific claim extraction is a subtask of information extraction. Unlike named entity recognition or relation extraction, claims are variable-granularity propositions that may be explicit ("We achieve 95% accuracy") or implicit ("This approach generalizes better"). The granularity problem — how finely to decompose a paragraph into claims — directly affects precision/recall measurements and must be controlled for.

## Methodology

### Design

Within-subjects comparison of 3 LLM extractors against human ground truth on the same 20 papers. Papers are stratified by type to ensure the results generalize across common paper formats in AI research.

Independent variable: extraction model (Claude Sonnet, GPT-4o, smaller baseline model such as Llama-3-8B or Mistral-7B).
Dependent variables: precision, recall, F1, hallucinated claim rate, claim granularity distribution.
Control: human-annotated ground truth with measured inter-annotator agreement.

### Data Requirements

- **Papers:** 20 papers from cs.AI and cs.LG on arXiv, stratified as 5 empirical, 5 theoretical, 5 survey, and 5 methods papers. Select papers published in 2023-2024 to ensure recency. Exclude papers longer than 30 pages to control annotation burden.
- **Annotations:** 2 independent human annotators extract all factual claims from each paper. Expected yield: 10-20 claims per paper, totaling approximately 200-400 annotated claims. Annotators receive a claim taxonomy (quantitative results, methodological claims, comparative claims, existence claims, causal claims) and granularity guidelines.
- **Annotation guidelines:** A claim is a single falsifiable proposition stated or directly implied by the paper. Compound claims are decomposed. Opinions, future work statements, and hedged speculations are excluded unless they contain a factual sub-claim.

### Procedure

1. **Paper selection.** Sample 20 papers from Semantic Scholar using category filters. Verify stratification. Download full text.
2. **Annotation guideline development.** Draft claim taxonomy and granularity guidelines. Pilot on 2 papers with both annotators. Revise guidelines based on disagreements. This pilot is excluded from final analysis.
3. **Human annotation.** Each annotator independently extracts claims from all 20 papers. Record time per paper.
4. **Inter-annotator agreement.** Compute Cohen's kappa on claim-level overlap (using the same cosine similarity >= 0.85 matching threshold). Resolve disagreements through discussion to produce the gold standard set. If kappa < 0.6, revise guidelines and re-annotate.
5. **LLM extraction.** Run each of the 3 models on all 20 papers using a standardized prompt that includes the same claim taxonomy and granularity guidelines given to human annotators. Use temperature 0 for reproducibility. Run each model 3 times to measure extraction stability (variance across runs).
6. **Claim matching.** Encode all claims (human and LLM) using sentence-transformers (all-MiniLM-L6-v2). Match LLM claims to gold standard claims using cosine similarity >= 0.85 threshold. A gold claim matched by at least one LLM claim counts as a true positive for recall. An LLM claim not matching any gold claim counts as a false positive for precision.
7. **Hallucination audit.** For each false positive claim (LLM claims not matching any gold claim), manually check whether the claim is (a) a valid claim missed by annotators, (b) a paraphrase below the similarity threshold, or (c) a hallucinated claim not present in the paper. Report adjusted and unadjusted hallucination rates.
8. **Granularity analysis.** Measure claim length distribution (tokens) and decomposition patterns. Compare LLM and human granularity to identify systematic over-splitting or under-splitting.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| Precision | TP / (TP + FP) at claim level | Bootstrap 95% CI (10,000 resamples) | Non-parametric, no distributional assumptions on per-paper precision values |
| Recall | TP / (TP + FN) at claim level | Bootstrap 95% CI (10,000 resamples) | Same justification |
| F1 | Harmonic mean of precision and recall | Derived from above | Standard composite metric |
| Inter-annotator kappa | Cohen's kappa on claim overlap | Point estimate with standard error | Standard agreement measure; >= 0.6 threshold per Landis & Koch (1977) |
| Hallucinated claim rate | Confirmed hallucinations / total LLM claims | Exact binomial 95% CI | Small count data, binomial CI more appropriate than normal approximation |
| Model comparison | Pairwise F1 differences | McNemar's test (paired data, same papers) | Paired design: same test set, different models; McNemar's is appropriate for paired binary outcomes |
| Extraction stability | Coefficient of variation across 3 runs | Descriptive (mean, SD) | Measures reproducibility at temperature 0 |

**Effect size reporting:** In addition to statistical significance, report Cohen's d for F1 differences between models. A meaningful improvement requires d >= 0.5 (medium effect).

## Success Criteria

- At least one model achieves precision >= 80% AND recall >= 70% (bootstrap 95% CI lower bound).
- Inter-annotator Cohen's kappa >= 0.6 (substantial agreement), confirming that the task is well-defined enough for ground truth to be meaningful.
- Hallucinated claim rate <= 10% for the best-performing model.

## Failure Criteria

- All models achieve F1 < 60%. This would indicate that LLM claim extraction is insufficiently reliable for knowledge graph construction, and manual extraction or hybrid approaches must be explored.
- Hallucinated claim rate > 10% for all models. This would mean the knowledge graph would accumulate fabricated claims at an unacceptable rate.
- Inter-annotator kappa < 0.4. This would indicate the task itself is too subjective for reliable ground truth, requiring revised operationalization of "claim."

**If the experiment fails:** Investigate whether failure is due to granularity mismatch (LLMs extracting at different resolution than humans), claim type bias (e.g., poor recall on implicit claims), or genuine comprehension failures. Consider claim type-specific extraction or retrieval-augmented extraction as mitigations.

## Estimated Cost & Timeline

- **API calls:** ~$50-100 (20 papers x 3 models x 3 runs, plus embedding computation)
- **Human annotation:** ~8-16 hours across 2 annotators (20 papers at 20-40 min/paper)
- **Hallucination audit:** ~2-4 hours (manual check of false positives)
- **Analysis and write-up:** ~4-8 hours
- **Calendar time:** 2-3 weeks (annotation can be parallelized)

## Dependencies

None. This is a foundational experiment with no upstream dependencies.

## Informs

- **E04** (Knowledge Graph Consistency): Claim extraction accuracy directly determines the quality of KG ingestion.
- **E05** (Gap Map): Gap detection requires reliable claim extraction to identify what is and is not established.
- **E06** (SUNFIRE): Hypothesis generation quality depends on accurate understanding of prior work.
- **E08** (Critic): The critic must evaluate claims, so claim extraction reliability bounds critic utility.
- **All Layer 3-4 experiments:** Every downstream loop consumes extracted claims.

## References

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv preprint arXiv:2408.06292*.
- Landis, J. R., & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159-174.
