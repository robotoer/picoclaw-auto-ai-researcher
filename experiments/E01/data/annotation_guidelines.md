# Claim Annotation Guidelines

## Purpose

These guidelines define what constitutes a "claim" in a scientific paper for the purposes of evaluating LLM claim extraction accuracy. Annotators (human or LLM) should follow these rules to produce consistent, reproducible annotations.

---

## 1. Definition of a Claim

A **claim** is a single, falsifiable proposition asserted by the authors of the paper as a contribution or finding of their work.

A statement is a claim if and only if:
- It is a declarative sentence (or can be rephrased as one).
- It is **falsifiable** -- there exists, at least in principle, evidence that could contradict it.
- The authors **assert** it (not merely hypothesize, speculate, or attribute it to others).

---

## 2. Claim Taxonomy

Every extracted claim must be assigned exactly one of the following types.

### 2.1 Quantitative Claims

A claim that includes a specific numeric result, measurement, or statistical outcome.

**Examples:**
- "Our model achieves 94.3% accuracy on the MMLU benchmark."
- "Training converges in 12 hours on 8 A100 GPUs."
- "The method reduces inference latency by 37% compared to the baseline."

**Key test:** Does the claim contain a number (or a quantitative comparison like "doubles", "halves") tied to a measurement?

### 2.2 Methodological Claims

A claim about what a method does, how it works, or what properties it has by design.

**Examples:**
- "Our architecture processes variable-length sequences without padding."
- "The proposed loss function is invariant to label permutations."
- "We introduce a two-stage training procedure that first pre-trains on synthetic data, then fine-tunes on real data."

**Key test:** Does the claim describe a design choice, mechanism, or property of the proposed approach?

### 2.3 Comparative Claims

A claim that explicitly compares the authors' work to other methods, models, or baselines, stating a qualitative or ordinal relationship.

**Examples:**
- "Our approach outperforms all baselines on 4 out of 5 benchmarks."
- "Unlike prior work, our method does not require labeled data."
- "The proposed model is more parameter-efficient than Transformer-XL."

**Key test:** Does the claim contain a comparison between the authors' contribution and something else? (If it also contains a specific number, classify as quantitative instead.)

### 2.4 Existence Claims

A claim that something exists, occurs, or is the case -- without quantification or comparison.

**Examples:**
- "Large language models exhibit emergent reasoning capabilities at sufficient scale."
- "There is a trade-off between model size and inference speed in this setting."
- "Attention heads in the middle layers specialize for syntactic relations."

**Key test:** Does the claim assert that a phenomenon, pattern, or property exists or occurs?

### 2.5 Causal Claims

A claim that one thing causes, leads to, enables, or prevents another.

**Examples:**
- "Increasing the learning rate beyond 1e-3 causes training instability."
- "The addition of the auxiliary loss prevents mode collapse during training."
- "Pretraining on code improves downstream mathematical reasoning performance."

**Key test:** Does the claim assert a causal or mechanistic relationship (not merely a correlation)?

---

## 3. Granularity Rules

### 3.1 Decompose Compound Claims

If a sentence contains multiple independent falsifiable propositions, split them into separate claims.

**Original:** "Our model achieves 94.3% accuracy on MMLU and 88.1% on HellaSwag, while being 2x faster than GPT-4."

**Decomposed into:**
1. "Our model achieves 94.3% accuracy on MMLU." (quantitative)
2. "Our model achieves 88.1% accuracy on HellaSwag." (quantitative)
3. "Our model is 2x faster than GPT-4." (quantitative)

### 3.2 Do Not Over-Decompose

A single proposition should remain intact even if it has multiple components that are logically inseparable.

**Keep as one claim:** "Fine-tuning BERT on SQuAD 2.0 with a learning rate of 3e-5 yields 93.2 F1."

This is a single experimental result; decomposing it into "fine-tuning BERT yields 93.2 F1" and "the learning rate was 3e-5" would lose essential context.

### 3.3 Implicit Claims

If a claim is strongly implied but not explicitly stated, do **not** extract it. Only extract what the authors actually assert.

**Text:** "Table 3 shows results across all benchmarks."

This is not a claim. The numbers in Table 3 may contain claims, but the sentence itself is a reference to a table.

### 3.4 Hedged Statements

Exclude hedged speculation, unless it contains a factual sub-claim.

**Exclude:** "We believe this approach could potentially scale to billions of parameters."

**Include the factual sub-claim from:** "While we have not tested beyond 1B parameters, our experiments show linear scaling up to that point."
- Extracted claim: "The method shows linear scaling up to 1B parameters." (quantitative)

---

## 4. What to Exclude

The following categories of statements should **not** be extracted as claims.

### 4.1 Background Knowledge and Definitions

Statements that describe well-known facts, standard definitions, or established results from prior work.

- "Transformers use self-attention to process sequences." (background)
- "Precision is defined as TP / (TP + FP)." (definition)
- "Neural networks are universal function approximators." (established fact)

### 4.2 Citations of Others' Work

Statements that merely report what other authors found, unless the current paper explicitly endorses the claim as its own finding.

**Exclude:** "Brown et al. (2020) showed that GPT-3 can perform few-shot learning."

**Include:** "Consistent with Brown et al. (2020), we confirm that few-shot learning improves with model scale, observing a 12-point gain from 7B to 70B parameters."
- Extracted claim: "Few-shot learning improves with model scale, with a 12-point gain from 7B to 70B parameters." (quantitative)

### 4.3 Future Work and Aspirational Statements

- "In future work, we plan to extend this to multilingual settings."
- "We hope this will inspire further research in the area."

### 4.4 Opinions, Value Judgments, and Subjective Assessments

- "We believe this is an important contribution to the field."
- "The results are promising."
- "This is an elegant solution to the problem."

### 4.5 Methodological Descriptions That Are Not Claims

Purely procedural statements that describe what was done without asserting a result or property.

**Exclude:** "We trained the model for 100 epochs using Adam optimizer."

**Include:** "Training for 100 epochs was sufficient for convergence, with no improvement observed beyond epoch 80." (existence claim about convergence behavior)

### 4.6 Tautologies and Trivially True Statements

- "A model with more parameters has more parameters."
- "If the loss decreases, the model improves on the training objective."

---

## 5. Edge Cases and Resolution Rules

### 5.1 Ablation Results

Each row of an ablation table that the authors discuss in the text counts as a separate claim, but only if the authors explicitly assert the result (not just present it in a table without commentary).

**Include:** "Removing the attention mechanism reduces accuracy by 5.2 points."

**Exclude:** Rows in a table that are never discussed in the text.

### 5.2 Negative Results

Negative results are claims. "Our method does not outperform the baseline on the WMT dataset" is a quantitative claim.

### 5.3 Conditional Claims

Claims with conditions are valid if the condition is part of the experimental setup.

**Include:** "When trained on fewer than 1,000 examples, our method outperforms fine-tuning." (comparative)

### 5.4 Claims Spanning Multiple Sentences

If a claim is split across two consecutive sentences, combine them into one claim and note the source section.

**Text:** "The model achieves state-of-the-art results. Specifically, it reaches 96.1 F1 on the test set."

**Extracted:** "The model achieves state-of-the-art results with 96.1 F1 on the test set." (quantitative)

### 5.5 Claims in Abstracts vs. Body

Claims appearing in both the abstract and the body should be extracted **once**, using the more detailed version (usually from the body). Record the source section as the body section.

### 5.6 Statistical Significance Statements

Treat significance statements as part of the claim they modify, not as separate claims.

**Text:** "Our method outperforms the baseline by 3.2 points (p < 0.01)."

**Extracted as one claim:** "Our method outperforms the baseline by 3.2 points (p < 0.01)." (quantitative)

### 5.7 Claims About Limitations

Authors' own statements about limitations of their work are valid existence or causal claims.

**Include:** "Our method fails to generalize to out-of-domain data, with accuracy dropping from 92% to 61%." (quantitative)

### 5.8 Ambiguity Resolution

When in doubt about whether a statement is a claim, apply this test: "Could a reviewer check this statement against evidence and potentially disagree?" If yes, it is a claim. If no, it is not.

---

## 6. Annotation Procedure

1. Read the full paper once without annotating.
2. Re-read section by section, extracting claims as you go.
3. For each claim, record:
   - `text`: The claim, rephrased as a standalone falsifiable proposition if needed.
   - `type`: One of quantitative, methodological, comparative, existence, causal.
   - `source_section`: The section title or heading where the claim appears.
4. After completing all sections, review the full list for duplicates (merge them, keeping the more detailed version).
5. Verify that no excluded categories (Section 4) have slipped through.

---

## 7. Dual-Annotator LLM Protocol

For this automated experiment, we simulate inter-annotator agreement using two Claude Opus passes:

### Annotator A -- Conservative Extraction
Prompt instructs the model to extract only claims that are **explicitly and unambiguously stated**. When in doubt, exclude. This annotator has high precision but may miss implicit claims.

### Annotator B -- Comprehensive Extraction
Prompt instructs the model to extract **all plausible claims**, including those requiring minor inference. This annotator has high recall but may over-extract.

### Reconciliation
A third Claude Opus pass receives both extraction sets and these guidelines. For each disagreement, it determines whether the claim meets the guidelines and produces a final verdict. The reconciled set is the gold standard.

### Agreement Metric
Compute Cohen's kappa on claim-level decisions (after aligning claims between annotators via embedding similarity at cosine >= 0.90). Target: kappa >= 0.65 (substantial agreement). If kappa falls below 0.50, review and revise the annotation guidelines before proceeding.
