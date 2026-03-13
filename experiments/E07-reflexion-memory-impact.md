# E07: Reflexion Memory Impact on Research Quality

> **Layer 2 — Component Validation**

## Status: Planned

## Hypothesis

An LLM research agent with Reflexion-style verbal memory improves its research output quality by >= 15% (SUNFIRE score) over 10 iterations on related topics, compared to a memoryless baseline. The improvement manifests as a positive monotonic trend in quality scores across iterations.

This hypothesis is falsifiable: if no significant difference exists between memory conditions at iterations 8-10, or if the Reflexion condition shows declining or flat quality trajectories, the hypothesis is rejected.

## Why This Matters

Path B (Reflexion + Compound Memory) from doc 11 is the fastest self-improvement path, estimated at 1-2 weeks to implement. It is the foundation of the ReAct-Reflexion loop (E09) and a core mechanism in the full research pipeline (E13). If verbal reflection does not actually improve research quality — or worse, causes degenerate loops where the agent fixates on past strategies — the entire compound memory architecture is wasted complexity. This experiment isolates the reflection mechanism to determine whether it provides genuine signal before integrating it into larger loops.

## Background & Prior Work

**Reflexion.** Shinn et al. (2023, NeurIPS) demonstrated that verbal reinforcement stored in an episodic memory buffer significantly improves LLM agent performance across diverse tasks: +22% on AlfWorld (embodied reasoning), +20% on HotPotQA (multi-step QA), and +11% on HumanEval (code generation). The key insight is that natural language reflections on failures — stored and retrieved in subsequent attempts — provide a form of in-context learning without weight updates. However, these benchmarks have clear success criteria (task completion, correct answer, passing tests). Research quality is far more ambiguous, and it is unknown whether Reflexion transfers to open-ended creative tasks.

**FunSearch iterative improvement.** Romera-Paredes et al. (2023) showed iterative program improvement over millions of samples, achieving a 3.2% improvement on cap set bounds (496 to 512). This demonstrates that iterative refinement with automated evaluation can yield gains, but FunSearch used a formal evaluator (mathematical proof checker) rather than the softer SUNFIRE-style evaluation we rely on. The question is whether iterative improvement works with noisier reward signals.

**LLM-as-Judge concerns.** Zheng et al. (2023) documented that LLM judges exhibit position bias, verbosity bias, and self-preference bias. If SUNFIRE scores are computed by the same LLM that generates the research, self-preference bias could create an illusion of improvement: later outputs might score higher simply because the model learns to produce outputs it likes, not outputs that are actually better. Cohen's kappa of 0.21 on specialized tasks suggests this risk is non-trivial.

**AI Scientist iteration.** Lu et al. (2024) used iterative refinement in the AI Scientist pipeline, with automated reviewing (F1 0.57 vs. 0.49 human). Their system demonstrated that LLMs can improve papers through revision, but did not isolate the contribution of explicit reflection (as opposed to simple re-prompting with feedback).

## Methodology

### Design

Repeated-measures experiment with 3 memory conditions tested on 5 AI research subtopics, each generating 10 sequential hypotheses. This yields 50 outputs per condition, 150 total. The design is within-topic (all conditions run on all topics) with between-run independence (separate context windows per condition).

Independent variables: memory condition (no memory, episodic memory only, full Reflexion), iteration number (1-10).
Dependent variables: SUNFIRE composite score, per-dimension SUNFIRE scores, output diversity (embedding distance), repetition rate.
Controls: topic order (randomized across runs), model stochasticity (3 replications per condition), prompt template (identical base prompt across conditions).

### Data Requirements

- **Topics:** 5 AI research subtopics selected to vary in maturity and breadth: (a) a well-established area (e.g., attention mechanisms), (b) a rapidly evolving area (e.g., mixture of experts), (c) a niche area (e.g., mechanistic interpretability of specific circuits), (d) an interdisciplinary area (e.g., AI for materials science), (e) a foundational/theoretical area (e.g., scaling laws). Topic selection criteria: each must have sufficient literature for the agent to reference but enough open questions to generate 10 distinct hypotheses.
- **Model:** A single frontier LLM (e.g., Claude Sonnet or GPT-4o) used for both generation and SUNFIRE scoring. Using the same model for both is a deliberate choice that mirrors the intended system architecture, though it introduces self-preference bias risk (measured and reported).
- **Memory implementations:**
  - **(a) No memory:** Each hypothesis generated in a fresh context with only the topic description and a literature summary. No access to prior outputs.
  - **(b) Episodic memory:** Each iteration receives the topic description, literature summary, and the full text of all prior hypotheses in the sequence. No explicit reflection.
  - **(c) Full Reflexion:** Each iteration receives the topic description, literature summary, all prior hypotheses, AND a reflection on each prior hypothesis (what was good, what was weak, what strategy to try next). Reflections are generated by the same LLM immediately after each hypothesis is scored.

### Procedure

1. **Topic preparation.** For each of the 5 topics, prepare a standardized topic description (~200 words) and a literature summary (~500 words covering 10-15 key papers). These are fixed across all conditions and replications.
2. **Baseline generation (condition a).** For each topic, generate 10 hypotheses sequentially, each in a fresh context window. Record SUNFIRE scores after each generation. Repeat 3 times (different random seeds / temperature sampling).
3. **Episodic memory generation (condition b).** For each topic, generate 10 hypotheses sequentially, accumulating all prior hypotheses in context. Record SUNFIRE scores. Repeat 3 times.
4. **Reflexion generation (condition c).** For each topic, generate 10 hypotheses sequentially. After each hypothesis is scored, generate a reflection (structured prompt: "What was strong? What was weak? What should I try differently?"). Both the hypothesis and reflection are carried forward. Record SUNFIRE scores and reflection content. Repeat 3 times.
5. **Diversity measurement.** Embed all hypotheses using sentence-transformers (all-mpnet-base-v2). Compute pairwise cosine distance between consecutive hypotheses within each sequence. Compute diversity as mean pairwise distance across all hypotheses in a sequence. Identify exact or near-exact repetitions (cosine similarity > 0.9).
6. **Quality trajectory analysis.** For each condition x topic x replication, fit a linear regression of SUNFIRE score on iteration number. Record slopes and R-squared values.
7. **Cross-condition comparison.** At iterations 8-10 (the "converged" phase), compare SUNFIRE scores across conditions using Wilcoxon signed-rank test (paired by topic x replication).
8. **Reflection content analysis.** Qualitatively analyze a random sample of 30 reflections from condition (c). Categorize reflection types: strategic (changes approach), diagnostic (identifies specific weakness), generic (vague self-encouragement), repetitive (same reflection as prior iteration). Measure the proportion of actionable reflections.

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| SUNFIRE score trajectory | SUNFIRE composite score at each iteration, by condition | Mixed-effects linear regression: score ~ iteration * condition + (1\|topic) + (1\|replication) | Accounts for repeated measures, topic effects, and replication variance |
| Converged quality comparison | Mean SUNFIRE at iterations 8-10, condition (a) vs (c) | Wilcoxon signed-rank test (paired by topic x replication, n=15 pairs) | Non-parametric paired test, appropriate for small n with ordinal data |
| Improvement magnitude | (Mean SUNFIRE iterations 8-10) / (Mean SUNFIRE iterations 1-3) - 1, by condition | Bootstrap 95% CI on percentage improvement | Quantifies the practical magnitude of improvement |
| Monotonic trend | Mann-Kendall test statistic for each condition's SUNFIRE trajectory | Mann-Kendall test for monotonic trend | Standard non-parametric test for time series trend; does not assume linearity |
| Output diversity | Mean pairwise cosine distance between consecutive hypotheses | Friedman test across 3 conditions (paired by topic) | Non-parametric test for 3+ related groups |
| Repetition rate | Fraction of hypotheses with cosine similarity > 0.9 to any prior hypothesis in same sequence | Descriptive (proportions) with Fisher's exact test comparing conditions | Measures degenerate loop risk |
| Reflection quality | Proportion of reflections classified as actionable | Descriptive (proportion with exact binomial 95% CI) | Characterizes mechanism quality |

**Effect size reporting:** Report Cohen's d for condition (a) vs (c) SUNFIRE scores at iterations 8-10. Meaningful improvement requires d >= 0.5.

## Success Criteria

- Condition (c) SUNFIRE scores >= 15% higher than condition (a) at iterations 8-10, with Wilcoxon signed-rank p < 0.05.
- Positive Mann-Kendall trend in condition (c) across all 5 topics (tau > 0, p < 0.05 in at least 4/5 topics).
- Condition (c) outperforms condition (b) by >= 5% at iterations 8-10 (demonstrating that reflection adds value beyond simple episodic memory).
- Repetition rate in condition (c) <= 10% (reflection does not cause fixation).
- At least 50% of reflections classified as actionable (the reflection mechanism produces useful signal, not generic text).

## Failure Criteria

- No significant difference between conditions (a) and (c) at iterations 8-10 (Wilcoxon p > 0.10). Reflexion provides no measurable benefit for research hypothesis generation.
- Condition (c) shows declining quality after iteration 5 (degenerate loop — the agent's reflections lead it astray). This would indicate that Reflexion is actively harmful for open-ended research tasks.
- Repetition rate in condition (c) > 30% (the agent gets stuck repeating itself despite reflection).
- Condition (b) performs equal to or better than (c) (episodic memory alone suffices, making the reflection mechanism unnecessary complexity).

**If the experiment fails:** Investigate whether failure is due to (a) reflection quality (generic reflections that don't provide useful signal), (b) SUNFIRE noise (the reward signal is too noisy for iterative optimization — links to E06 results), (c) task unsuitability (open-ended hypothesis generation may not benefit from iteration the way well-defined tasks in Shinn et al. do), or (d) context window saturation (accumulated memory causes degraded performance). Consider structured reflection templates, external memory retrieval instead of full context accumulation, or human-in-the-loop reflection as mitigations.

## Estimated Cost & Timeline

- **API calls:** ~$200-400 (150 hypothesis generations + 50 reflections + 450 SUNFIRE scorings across 3 replications, plus embeddings)
- **Analysis time:** ~12-16 hours (statistical modeling, trajectory plots, reflection content analysis)
- **Calendar time:** 2-3 weeks (primarily compute time and analysis; no human subjects needed beyond the experimenter)

## Dependencies

- **E06** (SUNFIRE Scoring Validation): SUNFIRE must be validated as correlating with research quality before using it to measure improvement. If E06 fails, SUNFIRE scores in this experiment are uninterpretable.
- **E02** (LLM Judge Reliability): LLM judge must be reliable enough for SUNFIRE computation. Self-preference bias (Zheng et al., 2023) is a particular concern since the same model generates and evaluates.

## Informs

- **E09** (ReAct-Reflexion Loop): This experiment directly validates the core mechanism of E09. If Reflexion works, E09 can proceed with confidence. If it fails, E09's design must be revised.
- **E13** (Full Research Loop): The compound memory architecture depends on reflection being useful.
- **Path B implementation timeline** (doc 11): Results determine whether Path B is worth the 1-2 week implementation investment.

## References

- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A. (2023). Mathematical discoveries from program search with large language models. *Nature*, 625, 468-475.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Lu, C., Lu, C., Lange, R. T., Foerster, J., Clune, J., & Ha, D. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. *arXiv preprint arXiv:2408.06292*.
