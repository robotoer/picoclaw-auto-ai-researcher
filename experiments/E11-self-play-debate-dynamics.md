# E11: Self-Play Debate Dynamics

> **Layer 3 — Loop Mechanics**

## Status: Planned

## Hypothesis

Self-play scientific debate (generator vs critic, doc 09 Loop 2) produces research hypotheses that are rated >= 25% higher quality by human experts than single-agent generation, with the improvement driven by genuine error correction rather than surface-level polish.

This hypothesis is falsifiable: if multi-round debate does not achieve >= 25% higher quality than single-pass generation (Wilcoxon p < 0.01), or if critique accuracy falls below 60%, or if improvements are attributable to surface revision rather than substantive error correction, the hypothesis is rejected.

## Why This Matters

Self-play drove superhuman performance in games — AlphaGo Zero achieved superhuman Go play through self-play alone (Silver et al., 2017), and asymmetric self-play generated automatic curricula for multi-agent learning (Sukhbaatar et al., 2018). Doc 09 Loop 2 proposes applying the same principle to research via generator-critic co-evolution: the generator produces hypotheses, the critic identifies flaws, and the generator revises.

But scientific debate is fundamentally different from games. Games have clear win conditions, finite action spaces, and deterministic evaluation. Scientific quality has no ground truth scorer, critique can be wrong, and both generator and critic may share the same blind spots (since they may be the same model or models from the same family). The risk is that debate degenerates: critics learn to make impressive-sounding but wrong objections, generators learn to produce defensively safe but unambitious hypotheses, and the process converges on mediocrity that merely appears polished. We need to measure whether debate produces genuine improvement and identify the failure modes.

## Background & Prior Work

**Self-play in games.** Silver et al. (2017) demonstrated that AlphaGo Zero, trained purely through self-play with no human data, surpassed all previous Go programs including the original AlphaGo. The key mechanism is that self-play generates an automatic curriculum: as one player improves, the opponent must also improve, driving continuous progress. Sukhbaatar et al. (2018) extended this with asymmetric self-play, where one agent sets tasks for another, generating curricula without human design.

**LLM debate and critique.** The "LLM-as-judge" paradigm (Zheng et al., 2023) has been widely adopted but shows known biases. Self-critique — where the same model critiques its own output — can identify some errors but is limited by the model's own blind spots. Cross-model critique may be more effective because different models have partially uncorrelated error patterns.

**Reflexion and iterative refinement.** Shinn et al. (2023) showed that verbal self-reflection improved performance on HumanEval to 91% pass@1. However, Reflexion uses environmental feedback (test results) as the grounding signal, not pure debate. Without an external signal, debate may lack the grounding needed to converge on truth rather than on persuasiveness.

**Goodhart's Law in debate.** Manheim & Garrabrant (2019) identified "adversarial Goodhart" as a distinct failure mode: when an optimizer (critic) and an optimizee (generator) co-adapt, the generator may learn to satisfy the critic's specific weaknesses rather than genuinely improve. This is the core risk for self-play debate.

## Methodology

### Design

Between-subjects comparison of 5 conditions on research hypothesis generation across 10 AI topics. Total: 5 conditions x 10 topics x 10 hypotheses = 500 research hypotheses.

**Independent variable:** Generation condition:
- (a) Single-pass generation — one-shot hypothesis generation with no critique
- (b) Generate + self-critique — same model generates then critiques its own output, single revision
- (c) Generate + cross-model critique — one model generates, a different model critiques, single revision
- (d) Multi-round debate — 3 rounds of generate, critique, revise, critique. Same model for both roles.
- (e) Adversarial debate with co-trained critic — critic is specifically prompted to find the strongest possible objections; generator is prompted to produce hypotheses robust to adversarial critique. 3 rounds.

**Dependent variables:** Human expert quality rating (1-7), error rate, critique substantiveness, revision quality, debate degeneration indicators.

### Data Requirements

- **Research topics:** 10 AI research subtopics spanning subfields (e.g., efficient transformers, RLHF limitations, multi-agent coordination, continual learning, mechanistic interpretability, LLM reasoning, synthetic data, AI safety evaluation, retrieval-augmented generation, code generation). Topics selected to have established literature and identifiable open problems.
- **Hypotheses per condition-topic:** 10 independently generated hypotheses, for a total of 100 per condition, 500 overall.
- **Human evaluation:** 3 domain experts, each rating all 500 hypotheses on a 7-point Likert scale across 4 dimensions: novelty, plausibility, significance, and specificity. Experts are blinded to condition. Rating order randomized. Expected time: ~3-5 minutes per hypothesis.
- **Critique evaluation:** A separate expert panel (2 evaluators) rates a stratified sample of 150 critiques (30 per condition for conditions b-e, sampled across topics) on: specificity (1-5), accuracy (binary: does the critique identify a real problem?), and constructiveness (1-5).
- **Revision classification:** For conditions b-e, each revision is classified as: substantive (addresses a real flaw), surface (rephrasing without fixing real issues), or regressive (introduces new problems). Classification by 2 evaluators with inter-rater reliability measured.

### Procedure

1. **Topic and prompt design.** For each of the 10 topics, create a standardized prompt that includes: topic description, 5 key recent papers (title + abstract), and the instruction "Generate a novel, specific, and testable research hypothesis in this area."
2. **Condition (a): single-pass.** Generate 10 hypotheses per topic using the standardized prompt. No revision.
3. **Condition (b): self-critique.** Generate a hypothesis, then prompt the same model to critique it ("Identify the top 3 weaknesses of this hypothesis"), then prompt the model to revise given the critique. One round.
4. **Condition (c): cross-model critique.** Generate a hypothesis with model A (e.g., Claude), critique with model B (e.g., GPT-4), revise with model A given the critique. One round.
5. **Condition (d): multi-round debate.** Three rounds of: generate/revise, critique, revise, critique. The critique at round N references all prior critiques and revisions. Track whether critique points change across rounds.
6. **Condition (e): adversarial debate.** Same structure as (d) but with modified prompts: critic is instructed to "find the strongest possible objection that would invalidate this hypothesis" and generator is instructed to "produce a hypothesis that is robust to the strongest possible critiques." Three rounds.
7. **Blind evaluation.** Present all 500 hypotheses to human experts in randomized order with no condition labels. Collect ratings on 4 dimensions.
8. **Critique evaluation.** Present the stratified sample of 150 critiques to the separate critique evaluation panel. For each critique, evaluators assess specificity, accuracy, and constructiveness.
9. **Revision classification.** For each revised hypothesis (conditions b-e), present the original, critique, and revision side-by-side. Two evaluators independently classify the revision as substantive, surface, or regressive.
10. **Debate trajectory analysis.** For conditions (d) and (e), analyze quality trajectory across the 3 rounds: does each round improve quality? At what round does improvement plateau or reverse?

### Metrics & Statistical Tests

| Metric | Definition | Statistical Test | Justification |
|--------|-----------|-----------------|---------------|
| Overall quality | Mean human rating across 4 dimensions | Friedman test across 5 conditions (repeated measures on topics) | Non-parametric repeated-measures test; topics serve as blocking variable |
| Pairwise condition comparison | Quality difference between each pair of conditions | Post-hoc Wilcoxon signed-rank with Bonferroni correction (10 pairwise comparisons, adjusted alpha = 0.005) | Non-parametric paired test; Bonferroni controls family-wise error rate |
| Quality improvement CI | Bootstrap 95% CI on (condition d mean - condition a mean) / condition a mean | Bootstrap (10,000 resamples) | Non-parametric CI on the primary effect size |
| Critique accuracy | Proportion of critique points confirmed by experts | Exact binomial 95% CI per condition | Small-count proportion data |
| Critique specificity | Mean specificity score (1-5) across conditions | Kruskal-Wallis test | Non-parametric comparison across conditions |
| Revision classification | Proportion substantive vs surface vs regressive | Chi-squared test of independence (condition x classification) | Tests whether revision type distribution differs across conditions |
| Inter-rater reliability | Agreement among 3 expert raters | Intraclass Correlation Coefficient (ICC, two-way random, absolute agreement) | Standard reliability measure for multiple raters |
| Debate degeneration | Quality trend across rounds within conditions (d) and (e) | Page's trend test (ordered alternative) | Tests specifically for monotonic improvement or degradation across rounds |

**Effect size reporting:** Report rank-biserial correlation for Wilcoxon tests and Kendall's W for Friedman test.

## Success Criteria

- Multi-round debate condition (d) achieves >= 25% higher mean quality rating than single-pass (a), Wilcoxon p < 0.01.
- Critique accuracy >= 60% (at least 60% of critique points are confirmed as identifying real problems by expert evaluation).
- Revision classification shows >= 50% substantive revisions in condition (d) — improvement comes from genuine error correction, not surface polish.
- Inter-rater reliability ICC >= 0.6 (substantial agreement), confirming evaluation is meaningful.
- Adversarial debate (e) does not show degeneration: quality at round 3 >= quality at round 1 (Page's test not significant for decreasing trend).

## Failure Criteria

- No significant difference between any conditions (Friedman p >= 0.05). This would indicate that critique and debate add no value to hypothesis generation, undermining Loop 2.
- Debate condition (d) shows degeneration: quality decreases after round 2 (Page's test significant for decreasing trend after round 2). This would indicate that extended debate is counterproductive.
- Critique accuracy < 40%. This would mean the critic is mostly wrong, and any improvements from debate are accidental rather than driven by genuine error identification.
- Revision classification shows >= 60% surface revisions. This would indicate that debate produces polished but not substantively improved outputs — a Goodhart-type failure where the appearance of quality improves but not the substance.

**If the experiment fails:** Investigate whether failure is due to (1) critique quality (critics lack domain knowledge to identify real problems), (2) revision strategy (generator ignores valid critiques or makes cosmetic changes), (3) shared blind spots (same-model debate cannot find errors that the model is constitutionally unable to detect), or (4) topic difficulty (some topics may benefit from debate while others don't). Consider external grounding signals (retrieval-augmented critique) or human-in-the-loop critic training.

## Estimated Cost & Timeline

- **API calls:** ~$300-500 (500 hypotheses; conditions b-e involve 2-7 additional calls per hypothesis for critique and revision; total ~5,000-8,000 calls)
- **Human evaluation (quality):** ~40-60 hours across 3 experts (500 hypotheses at ~3-5 min each)
- **Human evaluation (critiques):** ~8-12 hours across 2 evaluators (150 critiques at ~3-5 min each)
- **Revision classification:** ~12-16 hours across 2 evaluators (400 revisions at ~2-3 min each)
- **Analysis and write-up:** ~10-15 hours
- **Calendar time:** 4-6 weeks (human evaluation is the primary bottleneck; generation can complete in 1-2 days)

## Dependencies

- **E08** (Critic Effectiveness): The critic component must produce useful feedback. If E08 shows critics are unreliable, debate conditions cannot be expected to work.
- **E02** (LLM-as-Judge Reliability): While this experiment uses human evaluation as the primary metric, understanding automated evaluation reliability is needed for interpreting any supplementary automated scores.

## Informs

- **E13** (Full Loop Architecture): Determines whether debate should be incorporated into the complete research loop, and if so, how many rounds are optimal.
- **E14** (Evolutionary Loop): Debate may serve as a fitness evaluation mechanism within evolutionary search — this experiment establishes whether it adds signal.
- **Loop 2 implementation:** Directly determines the viability of generator-critic co-evolution as a research improvement mechanism.
- **Critic design:** Mutation analysis of productive vs unproductive critiques informs how to prompt or train the critic role.

## References

- Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., Hubert, T., Baker, L., Lai, M., Bolton, A., Chen, Y., Lillicrap, T., Hui, F., Sifre, L., van den Driessche, G., Graepel, T., & Hassabis, D. (2017). Mastering the game of Go without human knowledge. *Nature*, 550, 354-359.
- Sukhbaatar, S., Lin, Z., Kober, I., Synnaeve, G., Szlam, A., & Fergus, R. (2018). Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play. *International Conference on Learning Representations (ICLR 2018)*.
- Shinn, N., Cassano, F., Gopinath, A., Shakkottai, K., Labash, A., & Karthik, B. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
- Goodhart, C. A. E. (1984). Problems of Monetary Management: The U.K. Experience. In *Monetary Theory and Practice*. Macmillan.
- Manheim, D., & Garrabrant, S. (2019). Categorizing Variants of Goodhart's Law. *arXiv preprint arXiv:1803.04585*.
- Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems (NeurIPS 2023)*.
