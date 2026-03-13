# E08: Critic Agent Effectiveness

> **Layer 2 — Component Validation**

## Status: Planned

## Hypothesis

A dedicated critic agent with structured checklists catches >= 70% of factual errors and logical flaws in AI research hypotheses, with a false positive rate <= 20%, outperforming single-pass self-review by >= 25 percentage points in error detection rate.

This hypothesis is falsifiable: if the best critic condition achieves a true positive rate below 50%, or if the false positive rate exceeds 40%, the critic mechanism is producing more noise than signal and must be redesigned.

## Why This Matters

The Adversarial Peer-Review Loop (doc 02, 04) and the self-play scientific debate (doc 09, Loop 2) depend on the critic actually catching real problems. A critic that is too lenient (misses errors) allows flawed research to propagate through the pipeline, compounding errors in the knowledge graph and hypothesis generation. A critic that is too harsh (high false positive rate) rejects good work, throttling throughput and potentially biasing the system toward safe but trivial outputs. The balance between sensitivity and specificity determines whether the feedback loop is corrective or destructive.

## Background & Prior Work

**LLM-as-Judge biases and limitations.** Zheng et al. (2023) documented systematic biases in LLM evaluation: position bias (preferring the first item), verbosity bias (preferring longer responses), and self-preference bias (an LLM favoring its own outputs over competitors'). Cohen's kappa was 0.21 on specialized evaluation tasks, indicating low reliability for fine-grained judgments. A critic agent must overcome these biases to provide useful feedback — particularly self-preference bias when the critic reviews outputs from the same model family.

**AI Scientist reviewing.** Lu et al. (2024) demonstrated automated paper reviewing with accuracy 0.65 (vs. 0.66 human) and F1 0.57 for accept/reject (vs. 0.49 human). This shows LLMs can approximate the coarse-grained quality assessment of peer review. However, detecting specific errors (factual mistakes, logical flaws) is a finer-grained task than overall accept/reject, and may have substantially different reliability characteristics.

**Reflexion and self-correction.** Shinn et al. (2023) showed that verbal self-reflection improves task performance (+22% AlfWorld, +20% HotPotQA, +11% HumanEval), suggesting LLMs can identify their own mistakes when prompted to reflect. However, self-correction has known limitations: LLMs sometimes "correct" correct answers, and self-review may be systematically blind to certain error types that the model consistently produces. A dedicated critic (separate from the generator) may avoid some of these blind spots.

**FunSearch evaluation guard.** Romera-Paredes et al. (2023) used an automated mathematical evaluator as a hallucination guard during program search, filtering millions of candidate programs. Their evaluator had the advantage of formal verification (proof checking). For research hypotheses, no such formal ground truth exists, making the critic's task substantially harder and the false positive risk higher.

**Novelty assessment challenges.** Uzzi et al. (2013) and Arts & Veugelers (2020) showed that atypical combinations predict impact, but assessing whether a claim about novelty is accurate requires the critic to have comprehensive knowledge of existing work — precisely the task where LLMs may hallucinate (fabricating or missing relevant prior work).

## Methodology

### Design

Controlled experiment with 60 research hypotheses containing planted errors, evaluated by 4 critic conditions. The design uses a fully crossed structure: every hypothesis is evaluated by every critic condition, enabling paired comparisons.

Independent variable: critic condition (self-review, cross-model critic, structured checklist critic, multi-critic ensemble).
Dependent variables: true positive rate (error detection), false positive rate, error localization accuracy, feedback specificity.
Control: ground truth error labels (planted by experimenters).

### Data Requirements

- **Research hypotheses (60 total):**
  - 20 correct hypotheses: well-formed, factually accurate, logically sound research hypotheses on AI topics. Sourced from published work or carefully constructed by the experimenter.
  - 20 with planted factual errors: wrong citations (attributing results to wrong papers), incorrect claims about methods (e.g., "Transformers use recurrent connections"), incorrect numerical claims (e.g., wrong accuracy numbers), contradictions with established results. Each hypothesis contains 1-3 errors, precisely documented.
  - 20 with planted logical flaws: non-sequiturs (conclusion does not follow from premises), confounding variables ignored, overclaiming (generalizing from insufficient evidence), circular reasoning, false dichotomies. Each hypothesis contains 1-2 flaws, precisely documented.
- **Error manifest:** A detailed log of every planted error: its type, location in the text, the correct information, and the severity (minor, moderate, critical).
- **Critic implementations:**
  - **(a) Self-review:** The same LLM that generated the hypothesis reviews it with the prompt "Review this research hypothesis for errors and flaws."
  - **(b) Cross-model critic:** A different LLM family (e.g., if generator is Claude, critic is GPT-4o) reviews with the same prompt.
  - **(c) Structured checklist critic:** An LLM with a structured review protocol: Step 1 — verify each factual claim against known literature; Step 2 — check logical structure (premises, inference, conclusion); Step 3 — assess novelty claims against prior work; Step 4 — identify potential confounds or alternative explanations. Each step produces explicit findings.
  - **(d) Multi-critic ensemble:** 3 independent critics (different prompts and/or models) each review the hypothesis. An error is flagged if >= 2 of 3 critics identify it (majority vote).

### Procedure

1. **Hypothesis construction.** Create 60 hypotheses. For the 20 correct ones, adapt real research hypotheses from published papers, changing topic details to prevent memorization. For the 40 with errors, start from correct hypotheses and systematically introduce errors according to the error manifest. Have a second researcher verify that (a) correct hypotheses are indeed correct and (b) errors are unambiguous.
2. **Presentation randomization.** Shuffle all 60 hypotheses. Present to each critic condition in the same randomized order. Do not reveal the ratio of correct to flawed hypotheses.
3. **Critic execution.** Run each of the 4 conditions on all 60 hypotheses. For each, record: (a) whether the critic flags the hypothesis as containing errors (binary), (b) what specific errors the critic identifies (free text), (c) the severity rating the critic assigns, (d) the suggested correction (if any). Run each condition 3 times to measure detection stability.
4. **Error matching.** For each critic-flagged error, determine whether it matches a planted error from the manifest. Matching criteria: the critic correctly identifies the type of error (factual vs. logical) AND localizes it to the correct claim or reasoning step (within one sentence of the actual error location). A match counts as a true positive. A flagged error that does not match any planted error is a false positive (unless manual review determines it identifies a genuine unplanned error).
5. **Detection rate calculation.** True positive rate = planted errors correctly detected / total planted errors. False positive rate = false alarms on correct hypotheses / total correct hypotheses. Compute per condition and per error type (factual vs. logical).
6. **Feedback quality assessment.** For each true positive detection, rate the feedback specificity on a 3-point scale: (1) generic — "this is wrong" with no detail, (2) partially specific — identifies the type of error but not the exact issue, (3) fully specific — identifies the exact error and provides a correct alternative. Two raters independently score; resolve disagreements by discussion.
7. **Error type analysis.** Break down detection rates by error type (wrong citation, incorrect method claim, wrong number, non-sequitur, confound, overclaim, etc.). Identify which error types each critic condition is best and worst at detecting.
8. **Unplanned error discovery.** Review false positive flags on correct hypotheses. Determine whether any represent genuine errors missed during hypothesis construction (update the manifest) or represent critic hallucinations (fabricated problems).

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| True positive rate (TPR) | Planted errors correctly detected / total planted errors | Bootstrap 95% CI per condition | Non-parametric, appropriate for proportions |
| False positive rate (FPR) | Correct hypotheses falsely flagged / total correct hypotheses | Exact binomial 95% CI | Small count data, exact CI more appropriate |
| Self-review vs structured critic | TPR difference between conditions (a) and (c) | McNemar's test (paired, same hypotheses) | Paired binary outcomes on the same test set |
| Cross-condition comparison | TPR across all 4 conditions | Cochran's Q test (extension of McNemar's for 3+ conditions) | Non-parametric test for related binary outcomes across multiple conditions |
| Error type detection | TPR by error type (factual vs logical) x condition | Chi-squared test of independence | Tests whether detection rate depends on error type |
| Feedback specificity | Distribution of specificity scores (1/2/3) by condition | Mann-Whitney U comparing conditions | Ordinal data, non-parametric comparison |
| Detection stability | Agreement across 3 runs (Fleiss' kappa per condition) | Point estimate with standard error | Measures reproducibility of error detection |
| Localization accuracy | Fraction of TP where critic identifies correct location | Descriptive (proportion with 95% CI) | Measures quality beyond binary detection |

**Effect size reporting:** Report Cohen's h for TPR differences between conditions. A meaningful improvement in detection requires h >= 0.5.

## Success Criteria

- Structured checklist critic (condition c) achieves TPR >= 70% and FPR <= 20%.
- Condition (c) outperforms self-review (condition a) by >= 25 percentage points in TPR (McNemar's p < 0.05).
- Multi-critic ensemble (condition d) achieves FPR <= 10% (high precision through consensus).
- Feedback specificity: >= 60% of true positive detections in condition (c) rated as "fully specific" (score 3).
- Detection stability: Fleiss' kappa >= 0.6 across 3 runs for the best condition.

## Failure Criteria

- Best condition TPR < 50%. The critic misses the majority of planted errors, making it insufficient as a quality gate.
- Best condition FPR > 40%. The critic is essentially noise — flagging correct hypotheses nearly as often as flawed ones. This would make the feedback loop destructive rather than corrective.
- No significant difference between self-review (a) and structured critic (c). The structured checklist provides no benefit over naive self-review, suggesting the investment in critic architecture is wasted.
- Detection stability kappa < 0.3 for all conditions. Error detection is essentially random, varying across runs even with deterministic decoding.

**If the experiment fails:** Investigate whether failure stems from (a) error type (critics may catch factual errors but miss logical flaws, or vice versa — in which case, specialize critics by error type), (b) error subtlety (critics may catch obvious errors but miss subtle ones — test with varying error severity), (c) knowledge limitations (critics may lack domain knowledge to verify claims — consider retrieval-augmented critics with access to a paper database), or (d) prompt sensitivity (the critic prompt may need significant engineering). Consider hybrid approaches: automated checks for verifiable facts (citations, numbers) combined with LLM-based reasoning checks for logical structure.

## Estimated Cost & Timeline

- **Hypothesis construction:** ~12-16 hours (creating 60 hypotheses with carefully planted errors, plus verification)
- **API calls:** ~$100-200 (60 hypotheses x 4 conditions x 3 runs, plus ensemble costs)
- **Error matching and feedback scoring:** ~8-12 hours (manual matching of critic outputs to error manifest, feedback quality rating)
- **Analysis and write-up:** ~8-12 hours
- **Calendar time:** 2-3 weeks (primarily human effort for hypothesis construction and analysis)

## Dependencies

- **E01** (Claim Extraction Accuracy): The structured checklist critic's factual verification step relies on the ability to extract and verify claims. E01 establishes whether this sub-task is reliable.
- **E02** (LLM Judge Reliability): Establishes baseline reliability of LLM evaluation, which bounds critic performance. The biases documented in E02 (position bias, verbosity bias, self-preference) directly affect critic behavior.

## Informs

- **E11** (Self-Play Debate): The debate loop requires effective critics to judge arguments. If critic detection is unreliable, debate outcomes are arbitrary.
- **E09** (ReAct-Reflexion Loop): The Reflexion step depends on feedback quality. If the critic cannot identify real problems, reflections are based on noise.
- **All Layer 4 experiments:** Every full research loop includes a review/critique stage whose effectiveness depends on validated critic capabilities.

## References

- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv preprint arXiv:2408.06292*.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A. (2023). Mathematical discoveries from program search with large language models. *Nature*, 625, 468-475.
- Uzzi, B., Mukherjee, S., Stringer, M., & Jones, B. (2013). Atypical combinations and scientific impact. *Science*, 342(6157), 468-472.
- Arts, S., & Veugelers, R. (2020). Technology familiarity, recombinant novelty, and breakthrough invention. *Industrial and Corporate Change*, 24(6), 1215-1246.
