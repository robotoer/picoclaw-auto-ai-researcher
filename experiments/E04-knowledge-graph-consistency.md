# E04: Knowledge Graph Consistency Under Continuous Ingestion

> **Layer 1 — Foundational Capabilities**

## Status: Completed

## Hypothesis

Automated knowledge graph updates from paper ingestion maintain >= 90% factual consistency (no contradictions with verified claims) after ingesting 500+ papers, when multi-layer hallucination prevention (doc 08) is applied. Specifically, the contradiction rate decreases monotonically from ~10% (no filtering) to <= 2% (Layer 1+2) to <= 0.5% (Layer 1+2+3 multi-extractor consensus).

This hypothesis is falsifiable: if the contradiction rate exceeds 5% even with all hallucination prevention layers, or if the hallucination rate exceeds 15% on spot-check, the hypothesis is rejected.

## Why This Matters

Doc 08 predicts that multi-extractor consensus reduces hallucination pass-through from 5% to 0.003% with 5 extractors, assuming independent errors. This independence assumption is almost certainly violated in practice (LLMs share training data and exhibit correlated failure modes), making empirical validation essential. If the knowledge graph accumulates errors at an unacceptable rate, Gap Map reasoning (E05), hypothesis generation (E06), and all downstream components will produce outputs based on false premises. A KG with a 10% error rate after 500 papers means ~50 false claims actively corrupting downstream reasoning.

## Background & Prior Work

**Hallucination in LLMs.** LLM hallucination is well-documented across domains. In the context of factual extraction, hallucination manifests as claims attributed to a source paper that the paper does not actually make. This is distinct from factual errors in the source paper itself.

**Multi-extractor consensus.** Doc 08 proposes a cascade of hallucination prevention layers: Layer 1 (multi-extractor voting), Layer 2 (temporal consistency checks against established knowledge), and Layer 3 (source verification with retrieval). The theoretical analysis assumes independent extractor errors, which yields hallucination pass-through of p^n for n extractors with individual error rate p. With p=0.05 and n=5, this gives 0.05^5 = 3.125e-7, or ~0.003%. In practice, correlated errors (e.g., all LLMs misinterpreting the same ambiguous passage) will produce substantially higher rates.

**Knowledge graph quality.** KG quality is typically measured along dimensions of accuracy, completeness, consistency, and timeliness (Zaveri et al., 2016). Consistency — the absence of contradictions — is critical for reasoning. A contradiction occurs when the KG simultaneously asserts claim C and a claim that entails not-C.

**ReAct and Reflexion for error correction.** Yao et al. (2023) showed ReAct's reasoning-acting loop improved task performance by +34% on ALFWorld. Shinn et al. (2023) showed Reflexion's self-correction achieved 91% pass@1 on HumanEval. These results suggest that iterative verification loops (analogous to our multi-layer prevention) can substantially reduce errors, but gains depend on the error detection mechanism's reliability.

## Methodology

### Design

Controlled experiment with 4 conditions varying the level of hallucination prevention applied during KG ingestion. All conditions process the same 500 papers, allowing direct comparison of consistency outcomes.

Independent variable: hallucination prevention level (4 levels: none, Layer 1 only, Layer 1+2, Layer 1+2+3).
Dependent variables: contradiction rate, hallucination rate, duplicate rate, temporal consistency, provenance completeness.
Control: ground truth subgraph of 100 manually verified claims.

### Data Requirements

- **Ground truth papers:** 100 well-established papers with manually verified claims. Select papers with >= 100 citations and results that have been independently replicated. Extract ~500-1,000 verified claims forming the ground truth subgraph. Sources: landmark papers in deep learning, reinforcement learning, NLP, and computer vision from 2018-2023.
- **Ingestion papers:** 500 papers from cs.AI and cs.LG, published 2022-2024. Include a mix of (a) papers that agree with ground truth claims, (b) papers that extend ground truth findings, (c) papers that present contradictory or nuanced findings, and (d) papers on unrelated topics (to test for false contradiction detection).
- **Spot-check sample:** 50 randomly selected claims from the ingested KG (per condition) for manual hallucination verification.

### Procedure

1. **Ground truth construction.** Select 100 landmark papers. Two annotators independently extract factual claims. Resolve disagreements through discussion. Verify each claim against the source paper. Store as the ground truth subgraph with full provenance (paper, section, quote).

2. **Ingestion pipeline setup.** Configure the claim extraction pipeline with toggleable hallucination prevention layers:
   - **No filtering:** Raw LLM extraction (single model, no verification).
   - **Layer 1 only:** Multi-extractor voting. Run 3 extractors (Claude Sonnet, GPT-4o, Gemini Pro). Accept claims that >= 2/3 extractors agree on (semantic similarity >= 0.85 for matching).
   - **Layer 1+2:** Add temporal consistency check. Flag claims that contradict established claims in the ground truth subgraph, using entailment detection (NLI model or LLM-based).
   - **Layer 1+2+3:** Add source verification. For flagged claims, retrieve the relevant passage from the source paper and verify the claim is supported by the text.

3. **Ingestion execution.** Process all 500 papers through each of the 4 conditions independently, producing 4 separate KGs. Log all intermediate decisions (which claims were proposed, which were filtered, which were flagged).

4. **Contradiction detection.** For each KG, check all ingested claims against the ground truth subgraph for contradictions. A contradiction is defined as: claim A entails proposition P, and the ground truth contains a claim that entails not-P. Use an NLI model (DeBERTa-v3-large fine-tuned on MNLI) with entailment probability >= 0.8 as the contradiction threshold. Manually verify all detected contradictions to compute precision of the contradiction detector.

5. **Hallucination spot-check.** Randomly sample 50 claims from each KG condition. For each claim, a human annotator checks the source paper to verify the claim is actually made. Classify as: (a) correct and supported, (b) partially supported (claim overstates or understates the finding), (c) unsupported (hallucination — claim not in paper), (d) fabricated source (paper does not exist or does not contain relevant content).

6. **Growth curve analysis.** Track the following metrics as a function of papers ingested (at 100, 200, 300, 400, 500 paper checkpoints):
   - Total nodes and edges in KG
   - Contradiction rate (contradictions / total claims)
   - Duplicate/near-duplicate rate (claims within cosine similarity >= 0.95 of existing claims)
   - Provenance completeness (% of claims with full source tracking: paper ID, section, quote)

7. **Independence analysis.** For the multi-extractor condition (Layer 1), analyze whether extractor errors are independent. Compute the correlation of errors across extractor pairs. Compare the observed consensus error rate to the theoretical rate under independence (p^n). Report the effective number of independent extractors.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| Contradiction rate | Verified contradictions / total claims in KG | Chi-squared test comparing rates across 4 conditions | 4 independent conditions with count data; chi-squared is appropriate for comparing proportions across groups |
| Pairwise contradiction comparison | Contradiction rate difference between adjacent conditions | McNemar's test (paired: same papers, different processing) | Paired design on same paper set; McNemar's for paired binary outcomes |
| Hallucination rate | Unsupported claims / spot-check sample | Exact binomial 95% CI (Clopper-Pearson) | Small sample (n=50); exact CI avoids normal approximation issues |
| Provenance completeness | Claims with full provenance / total claims | Descriptive (proportion with exact binomial CI) | Simple proportion |
| Duplicate rate | Near-duplicate claim pairs / total claims | Descriptive | Measures KG bloat rather than error |
| Growth curve trend | Contradiction rate slope over ingestion checkpoints | Linear regression on 5 checkpoints | Tests whether error accumulates (positive slope) or stabilizes (zero slope) |
| Extractor independence | Correlation of errors between extractor pairs | Tetrachoric correlation (binary error/no-error) | Measures violation of the independence assumption underlying the p^n calculation |
| Theoretical vs observed consensus error | Ratio of observed consensus error rate to p^n prediction | Descriptive comparison | Directly tests doc 08's theoretical prediction |

**Sample size justification for spot-check:** With n=50 spot-checks per condition, a hallucination rate of 5% gives an exact 95% CI of approximately [0.01, 0.14]. This is sufficient to distinguish between acceptable (< 5%) and unacceptable (> 15%) rates but not to precisely estimate rates between 5-15%. If preliminary results suggest rates in this range, increase spot-check sample to n=100.

## Success Criteria

- Contradiction rate <= 10% with no filtering (establishes baseline).
- Contradiction rate <= 2% with Layer 1+2 (chi-squared test p < 0.05 vs no-filtering).
- Contradiction rate <= 0.5% with all layers (chi-squared test p < 0.05 vs Layer 1+2).
- Hallucination rate (spot-check) <= 5% with all layers (exact binomial 95% CI upper bound < 15%).
- Provenance completeness >= 95%.
- Growth curve slope for contradiction rate is non-positive (errors do not accumulate) with all layers active.

## Failure Criteria

- Contradiction rate > 5% with all hallucination prevention layers active. This would mean the multi-layer approach is insufficient and fundamentally different error prevention strategies are needed.
- Hallucination rate > 15% on spot-check for any condition. This would indicate the extraction pipeline is unreliable at a basic level.
- Growth curve shows monotonically increasing contradiction rate. This would mean error accumulation is unbounded, making the KG unusable at scale.
- Extractor errors are highly correlated (tetrachoric r > 0.7), making the independence assumption in doc 08's analysis invalid and the p^n calculation unreliable.

**If the experiment fails:** Consider (1) more diverse extractors (different model families, different prompts, different extraction strategies) to reduce error correlation, (2) human-in-the-loop for high-uncertainty claims, (3) periodic KG auditing and pruning rather than prevention-only, (4) restricting the KG to high-confidence claims only (accepting lower recall for higher precision), or (5) using retrieval-augmented generation at query time instead of materializing a KG (verify claims on-the-fly rather than storing them).

## Estimated Cost & Timeline

- **Ground truth construction:** ~16-24 hours across 2 annotators (100 papers, higher effort than E01 because claims must be verified, not just extracted)
- **API calls:** ~$100-200 (500 papers x 3 extractors x 4 conditions, plus NLI model and verification calls)
- **Spot-check:** ~4-8 hours (200 claims total across 4 conditions at ~1-2 min/claim)
- **Compute:** ~4-8 hours for embedding computation, NLI inference, and KG construction
- **Analysis and write-up:** ~8-12 hours
- **Calendar time:** 3-5 weeks (ground truth construction is the bottleneck; can begin concurrently with E01 annotation)

## Dependencies

- **E01** (Claim Extraction Accuracy): E01 establishes the baseline accuracy of claim extraction. E04 should not begin full execution until E01 results confirm that extraction is reliable enough to make KG construction meaningful. If E01 shows F1 < 60%, the ingestion pipeline must be improved before E04 can produce interpretable results.

## Informs

- **E05** (Gap Map): Gap detection operates on the KG. Consistency directly determines whether detected gaps are real or artifacts of KG errors.
- **E08** (Critic): The critic cross-references claims against the KG. KG accuracy bounds critic reliability.
- **All Layer 3-4 experiments:** Every downstream experiment that reads from the KG inherits its error rate.
- **System architecture:** Results determine whether the KG can serve as a reliable shared knowledge store or whether claim verification must happen at query time.

## References

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *International Conference on Learning Representations (ICLR 2023)*.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Zaveri, A., Rula, A., Maurino, A., Pietrobon, R., Lehmann, J., & Auer, S. (2016). Quality Assessment for Linked Data: A Survey. *Semantic Web*, 7(1), 63-93.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv preprint arXiv:2408.06292*.
