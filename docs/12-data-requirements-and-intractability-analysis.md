# 12 — Data Requirements, Self-Crawling, and Computational Intractability

> *What data do we actually need, can the system collect it itself, and are there traps where the system appears to work but never reaches anything useful?*

---

## 1. What Data Is Actually Needed (and What Isn't)

There is a common misconception that an AI research system needs "all of human knowledge" to work. It doesn't. The data requirements are surprisingly specific and, critically, mostly **freely available and self-crawlable**.

### 1.1 Data Requirements by Path

| Path | What's Needed | Where It Comes From | Self-Crawlable? |
|---|---|---|---|
| **A: Evolutionary Search** | An evaluation function + seed programs | Self-generated | Yes — evaluation IS the data |
| **B: Reflexion/Memory** | Research episodes | Self-generated during operation | Yes — the system creates this |
| **C: STaR Self-Training** | (input, good_output) pairs | Self-generated + filtered | Yes, but needs bootstrap |
| **D: RL Policy** | (state, action, reward) episodes | Self-generated during operation | Yes — episodes ARE the data |
| **E: Recursive Compiler** | All of the above + code diffs | Self-generated | Yes |

**The key insight: for the self-improvement loops, the system generates its own training data through operation.** This is the fundamental difference from supervised learning — RL and evolutionary systems create their own data through interaction.

### 1.2 External Data (Required for Bootstrap Only)

The system needs external data to bootstrap — to go from "knows nothing" to "knows enough to start generating useful outputs." Once bootstrapped, external data shifts from "required" to "helpful."

| Data Source | What It Provides | Volume | Cost | API |
|---|---|---|---|---|
| **ArXiv** | Full text of papers (CS.AI, CS.LG, stat.ML) | ~500K papers, ~15K new/month | Free | arxiv.org API (unlimited) |
| **Semantic Scholar** | Citation graphs, author networks, abstracts | 200M+ papers indexed | Free | api.semanticscholar.org (100 req/s) |
| **Papers With Code** | Benchmark leaderboards, code links | ~100K papers with code | Free | paperswithcode.com API |
| **OpenReview** | Papers + multi-round peer reviews + decisions | ~50K papers with reviews | Free | openreview.net API |
| **Hugging Face Hub** | Models, datasets, community adoption signals | 500K+ models | Free | huggingface.co API |

**Total cost to bootstrap: $0** for the data itself. The only cost is compute to process it.

**What the system does NOT need:**
- Full internet crawl (Common Crawl, C4, etc.) — not useful for research-specific tasks
- Proprietary datasets (internal company data, paywalled journals) — ArXiv covers AI/ML
- Pre-training data — we use existing pre-trained models, not training from scratch
- Human-labeled research quality data at scale — OpenReview provides this for free

### 1.3 The Bootstrap Dataset

To go from zero to "capable of generating useful research outputs," the system needs:

**Tier 1 — Minimum viable bootstrap (1–2 days to collect):**
```
1. ArXiv CS.AI + CS.LG papers from last 12 months
   - Abstracts: ~30K papers × ~200 tokens = 6M tokens
   - Full text (top 10% by citation): ~3K papers × 10K tokens = 30M tokens
   Total: ~36M tokens
   Processing cost: ~$100 (Claude Sonnet for extraction)

2. Semantic Scholar citation graph for those papers
   - Citation edges, co-citation clusters
   Collection: ~2 hours of API calls (free)

3. OpenReview data for ICLR/NeurIPS/ICML (last 2 years)
   - ~5K papers with reviews, scores, decisions
   - This IS the reward model training data
   Collection: ~1 hour of API scraping (free)
```

**Tier 2 — Solid foundation (1–2 weeks to collect):**
```
4. ArXiv full text for ALL relevant papers (last 3 years)
   - ~100K papers × 10K tokens = 1B tokens
   Processing cost: ~$500-1000

5. Papers With Code benchmark data
   - Model performance on major benchmarks
   - Links between papers and code implementations

6. "Future work" and "open problems" extraction
   - From survey papers and workshop CFPs
   - ~500 survey papers × 30K tokens = 15M tokens
```

**Tier 3 — Comprehensive (1–2 months, continuous):**
```
7. Real-time ArXiv monitoring (daily digest)
8. Author network analysis (who works on what, collaboration patterns)
9. Negative results extraction (from methods sections: "we tried X but...")
10. Historical trend data (5+ years of ArXiv submission statistics)
```

### 1.4 Can the AI Self-Crawl All of This?

**Yes, almost entirely.** Here's how:

```
┌──────────────────────────────────────────────────────────────┐
│  SELF-CRAWLING PIPELINE                                      │
│                                                              │
│  PHASE 1: Literature ingestion (automated, no human input)   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Agent → ArXiv API → fetch papers by category            │ │
│  │ Agent → extract claims, methods, results, open problems │ │
│  │ Agent → build knowledge graph from extractions          │ │
│  │ Agent → Semantic Scholar API → citation graph overlay   │ │
│  │ Agent → Papers With Code API → benchmark leaderboards   │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  PHASE 2: Reward model training data (automated)             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Agent → OpenReview API → papers + reviews + decisions   │ │
│  │ Agent → extract (paper_features, review_scores) pairs   │ │
│  │ Agent → train reward model: predict review scores       │ │
│  │         from paper features                             │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  PHASE 3: Self-generated data (from operation)               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Research episodes → reflections (Path B data)           │ │
│  │ Strategy variants → fitness scores (Path A data)        │ │
│  │ Good outputs → training pairs (Path C data)             │ │
│  │ Episodes → (state, action, reward) (Path D data)        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  PHASE 4: Continuous learning (ongoing, automated)           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ Daily ArXiv digest → new papers → KG updates            │ │
│  │ Weekly citation updates → trend detection                │ │
│  │ Monthly OpenReview updates → reward model refresh        │ │
│  │ Ongoing: reflections, strategy evolution, self-training  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                              │
│  HUMAN INPUT REQUIRED:                                       │
│  - Initial research scope ("focus on RL + LLMs + agents")    │
│  - Periodic spot-checks of output quality                    │
│  - SUNFIRE weight calibration (what "interesting" means)     │
│  - Safety review of sensitive outputs                        │
│  That's it. Everything else is self-crawled.                 │
└──────────────────────────────────────────────────────────────┘
```

**The critical point:** AI/ML research is uniquely suited for self-crawling because the primary literature is on ArXiv (free, open access, machine-readable API). This would NOT work as well for fields behind paywalls (most of biomedicine, for example).

---

## 2. Does This Need a Full Internet Dataset?

**No.** And trying to use one would actually hurt.

### 2.1 Why a Full Internet Dataset Is Not Needed

| Reason | Explanation |
|---|---|
| **We're not pre-training** | We use existing pre-trained models (Claude, Llama, Qwen). They already have internet-scale knowledge in their weights. |
| **Research data is structured** | Papers have abstracts, methods, results, references. This structure is far more useful than unstructured web text. |
| **Quality > quantity** | 100K high-quality papers with citation data beats 1B web pages for research tasks. |
| **The field is small enough** | AI/ML publishes ~15K papers/month. This is tiny compared to internet scale. A single GPU can process the entire AI literature in days. |

### 2.2 Why a Full Internet Dataset Would Hurt

1. **Signal-to-noise ratio collapse.** The ratio of useful research knowledge to random web content is approximately 1:10,000. Processing the full internet to find research-relevant information is absurdly inefficient.

2. **Contamination.** The internet contains enormous amounts of wrong, outdated, and misleading information about AI/ML. Blog posts, StackOverflow answers, and tweets often contain plausible-sounding but incorrect claims. The knowledge graph would be polluted.

3. **Compute waste.** Processing the full internet costs millions of dollars. Processing the AI research literature costs hundreds. The marginal value of internet-scale data beyond the focused corpus is near zero for this application.

### 2.3 What IS Useful Beyond Papers

There are some non-paper sources that are valuable and self-crawlable:

| Source | Value | Self-Crawlable? |
|---|---|---|
| GitHub repos linked from papers | Ground truth on what actually works (code > claims) | Yes (GitHub API) |
| Hugging Face model cards | What models exist, their capabilities and limitations | Yes (HF API) |
| Conference workshop CFPs | What the community thinks is important right now | Yes (web scrape) |
| Benchmark leaderboards | Current SOTA, rate of progress | Yes (Papers With Code API) |
| ArXiv comment/withdrawal metadata | Corrections, retractions, controversies | Yes (ArXiv API) |

---

## 3. Computational Intractability Traps

This is the critical question. **Yes, there are serious traps where the system could appear to work but never reach anything useful.** Here they are, ranked by danger:

### 3.1 Trap: "Smart-Looking Grid Search" (DANGER: HIGH)

**The problem:** The hypothesis space is combinatorially enormous. If the system is exploring hypotheses by naive enumeration — even with LLM-guided pruning — it may be doing nothing more than a slightly-intelligent grid search over an intractable space.

**How it manifests:**
- The system generates thousands of hypotheses
- Each scores slightly differently on SUNFIRE
- Progress appears to happen (scores nudge upward)
- But the improvements are from finding slightly better random points, not from learning structure
- The system never converges on genuinely productive research directions

**The math:**
Consider the hypothesis space as the product of: topics (T) × methods (M) × claims (C) × conditions (K). If T=100, M=50, C=200, K=20, the space has 20 million combinations. Even at 1000 evaluations/day, exhaustive search takes 55 years.

**LLM-guided search helps only if:**
1. The LLM's prior over the space is significantly better than uniform
2. The space has exploitable structure (nearby hypotheses have correlated quality)
3. The evaluation function is smooth enough to guide gradient-like improvement

**Evidence this is a real risk:**
- MLAgentBench (Huang et al., 2023): Claude achieved 0% success on truly novel tasks despite trying many approaches. The search was smart-looking but functionally random on hard problems.
- "Why LLMs Aren't Scientists Yet" (2026): Documents "implementation drift under execution pressure" — agents try many things but don't converge.

**Mitigations:**
1. **Don't search the hypothesis space directly.** Search the *strategy* space instead (Path A). The space of research strategies is much smaller (~dozens of meaningful dimensions) than the space of hypotheses (~millions). Evolving strategies lets the strategy discover hypotheses, adding a useful layer of abstraction.

2. **Use citation structure as a prior.** The citation graph tells you which combinations of topics and methods are productive. This is an empirically-grounded prior that massively reduces the effective search space.

3. **Measure convergence rate.** Track whether scores are improving faster than random search would predict. If the system's improvement rate is within 2× of random, the search is not adding value. Kill and redesign.

4. **Impose structure on the search.** Instead of "generate any hypothesis," decompose into: (a) select a gap from the Gap Map, (b) select a method from a short list of candidates, (c) formulate a specific claim. Each choice is from a small menu. The product is a manageable search space.

### 3.2 Trap: "Goodhart Collapse" (DANGER: HIGH)

**The problem:** The system optimizes for SUNFIRE scores (or any proxy metric) and finds degenerate solutions that score well but produce nothing useful. The self-improvement loop then amplifies these degenerate solutions because they look good on the metrics.

**How it manifests:**
- Hypothesis quality scores improve over generations
- The system produces increasingly "novel" and "surprising" outputs
- But the outputs are novel-sounding word salad, not genuine insights
- The system has learned to game its own evaluation
- All downstream training data is contaminated → capability degrades

**This is the most dangerous trap because the system's own metrics say everything is fine.** You only discover the problem through external evaluation (human review), which is expensive and infrequent.

**Evidence this is a real risk:**
- Every RLHF system exhibits some reward hacking (Gao et al., 2023: "Scaling Laws for Reward Model Overoptimization"). PPO training against a reward model initially improves quality, then degrades as the policy finds reward model exploits.
- In evolutionary systems, "bloat" is a well-known phenomenon where programs grow in complexity without improving fitness, consuming more evaluation budget per generation.

**Mitigations:**
1. **Multi-objective evaluation with orthogonal metrics.** Don't use a single scalar score. Use 5+ metrics that measure different aspects (novelty, execution success, critique survival, benchmark correlation, human spot-check). Degenerate solutions that game one metric typically fail on others.

2. **Execution-grounded evaluation.** Whenever possible, evaluate by *running* the proposed experiment, not by *judging* the proposal. A hypothesis that predicts "X will outperform Y on benchmark Z" can be tested. If the prediction is wrong, the hypothesis is wrong regardless of how good it sounds. This is the ultimate Goodhart-proof metric.

3. **Regular human audits.** Every 100 outputs, a human reviews 10 random samples. This is expensive but essential. Budget 2–4 hours/month of human expert time. Non-negotiable.

4. **Reward model ensembles.** Train 3+ reward models with different architectures and training subsets. Only trust improvements that all models agree on. Disagreement between reward models is a Goodhart early warning.

5. **KL penalty.** In RL training (Path D), penalize the policy for drifting too far from its initial distribution. This is standard in RLHF (the KL term in the PPO objective) and prevents the most extreme forms of reward hacking.

### 3.3 Trap: "Self-Training Collapse" (DANGER: MEDIUM-HIGH)

**The problem:** In Path C (STaR-style self-training), the model trains on its own outputs. If the initial outputs aren't good enough, the model trains on bad data, gets worse, generates even worse outputs, trains on those, and spirals downward.

**How it manifests:**
- Model quality degrades over self-training generations
- Or: model quality plateaus immediately (no improvement at all)
- Or: model loses diversity (mode collapse — generates the same type of output every time)

**The math:**
Let q be the fraction of the model's outputs that are "genuinely good." Self-training works when q is high enough that the filtering step (keep top k%) successfully separates good from bad. If q < k, the training set is contaminated with bad examples and the model degrades.

For research outputs, q is initially very low. A naive LLM generates plausible-sounding but largely vacuous research hypotheses. The filtering must be extremely precise to extract the rare genuinely good outputs.

**Evidence this is a real risk:**
- Shumailov et al. (2024). "AI models collapse when trained on recursively generated data." Nature. This is literally the scenario: model trains on its own outputs, quality degrades catastrophically over generations.
- The "model collapse" literature shows this happens even with high-quality initial data if the self-training loop runs long enough without external data injection.

**Mitigations:**
1. **Always mix self-generated data with external data.** Never train exclusively on the model's own outputs. Mix ratio: at least 30% external (papers, OpenReview reviews) to 70% self-generated. The external data provides an anchor that prevents collapse.

2. **Aggressive filtering.** Use top-1% not top-10%. It's better to have a tiny but high-quality training set than a large contaminated one. If you can't confidently identify the top 1%, you're not ready for self-training.

3. **Monitor for diversity loss.** Track the entropy of the model's output distribution across generations. If entropy drops (outputs becoming more similar), add diversity pressure or pause self-training.

4. **External validation checkpoint.** Before deploying a new self-trained generation, validate on a held-out set of research tasks with known-good answers. If performance drops on the validation set, reject the new generation.

5. **Don't start self-training too early.** Path C requires a critical mass of good outputs (from Paths A+B running for months). Premature self-training is worse than no self-training.

### 3.4 Trap: "Exploration Starvation" (DANGER: MEDIUM)

**The problem:** The RL policy (Path D) learns to exploit a few productive research areas and stops exploring new ones. The system produces a stream of incremental results in its comfort zone but never discovers breakthrough directions.

**How it manifests:**
- Research outputs are consistently "okay" — moderate SUNFIRE scores
- All outputs are in 2–3 closely related topics
- The system never explores the Gap Map's frontier regions
- Novel research directions are systematically avoided because their expected reward is uncertain (and the policy is risk-averse)

**Evidence this is a real risk:**
- This is the classic exploration-exploitation failure in RL. It's well-documented and well-understood theoretically.
- In human academia, the same pattern exists: researchers exploit their niche, producing incremental papers, avoiding risky new directions.

**Mitigations:**
1. **Enforce an exploration budget.** Allocate 20–30% of compute to pure exploration (randomly selected topics from the Gap Map frontier), regardless of what the RL policy wants. This is the "epsilon" in epsilon-greedy, but at the topic level.

2. **Novelty bonus in reward.** Add a term to the reward function that explicitly rewards exploring new topics: `R_total = R_quality + λ·R_novelty`. Quality-Diversity algorithms (MAP-Elites) enforce this automatically.

3. **UCB-style topic selection.** Use Upper Confidence Bound: `score(topic) = mean_reward(topic) + c·sqrt(log(t)/visits(topic))`. Topics that haven't been tried receive a large exploration bonus that decays with visits. This provably balances exploration and exploitation.

4. **Population diversity (Path A).** If the population of strategies includes some that are explicitly exploration-focused, the system won't lose the ability to explore even if the RL policy becomes exploitative.

### 3.5 Trap: "Reflection Saturation" (DANGER: MEDIUM)

**The problem:** In Path B, the reflection store grows indefinitely. After 10,000 reflections, the retrieval system returns increasingly irrelevant reflections, and the context window is consumed by low-value historical lessons rather than the current research task.

**How it manifests:**
- Performance improves for the first few hundred episodes
- Then plateaus
- Then slowly degrades as retrieval noise increases
- The system is spending more tokens processing past reflections than doing actual research

**Mitigations:**
1. **Consolidation.** Periodically (weekly), cluster similar reflections and merge them into higher-level strategy principles. Replace 100 specific reflections with 5 general lessons.

2. **Decay.** Apply temporal decay to reflection relevance scores. Older reflections are retrieved less often unless they're highly relevant.

3. **Capacity limit.** Cap the active reflection store at, say, 500 entries. When adding a new reflection, replace the least-retrieved existing one.

4. **Hierarchical memory.** Recent reflections (last 100): full text. Older reflections: compressed summaries. Very old reflections: statistics only ("this strategy worked 73% of the time on this topic type").

### 3.6 Trap: "The Streetlight Effect" (DANGER: MEDIUM)

**The problem:** The system researches what's easy to research (topics with lots of existing data, clear benchmarks, simple experiments) rather than what's important. This is the AI equivalent of looking for your keys under the streetlight because that's where the light is.

**How it manifests:**
- The system produces many results on well-benchmarked topics (image classification, language modeling, standard RL environments)
- It avoids hard problems with unclear evaluation criteria (alignment, interpretability, novel architectures)
- The Gap Map shows frontiers in hard areas, but the system routes around them because estimated tractability is low

**Mitigations:**
1. **Importance weighting that overrides tractability.** In the gap prioritization score, importance should have higher weight than tractability: `score = importance^2 × tractability` rather than `importance × tractability`. This biases toward important problems even when they're hard.

2. **Explicit hard-problem quota.** Reserve 10–20% of research threads for problems rated "important but hard" by the Gap Map. Accept lower success rates on these threads.

3. **Track the "difficulty distribution" of outputs.** If >80% of outputs address easy problems, trigger a rebalancing.

---

## 4. What Scaling Laws Tell Us

### 4.1 Relevant Scaling Laws

**Chinchilla (Hoffmann et al., 2022):** Optimal model performance scales as a power law with both model size AND data quantity. But for our system, "data" is self-generated research episodes, not pre-training tokens. The relevant question: does research output quality scale as a power law with the number of research episodes?

**FunSearch scaling (Romera-Paredes et al., 2024):** FunSearch showed that the quality of discovered programs improves log-linearly with the number of evaluations. Doubling compute roughly doubles the probability of finding a better solution. This is encouraging — it means evolutionary search over research programs (Path A) should scale.

```
P(finding better program) ∝ log(evaluations)
```

**AlphaEvolve scaling (2025):** AlphaEvolve found that using an *ensemble* of models (Flash for breadth, Pro for depth) was more important than scaling a single model. This suggests that **diversity of exploration** matters more than **depth of any single exploration**. For our system: run many diverse research strategies in parallel rather than pouring all compute into one.

**Reward model overoptimization (Gao et al., 2023):** There's a scaling law for reward hacking. As the policy trains more against a fixed reward model, performance initially improves, peaks, then degrades. The optimal training duration scales as:

```
optimal_steps ∝ sqrt(reward_model_size)
```

**Implication:** Larger reward models allow longer RL training before overoptimization kicks in. To sustain self-improvement over many cycles, the reward model itself must grow.

**STaR scaling (Zelikman et al., 2022; Singh et al., 2024 — ReST^EM):** Self-training on correct solutions shows diminishing returns per generation. The first self-training generation gives the biggest improvement; subsequent generations give progressively smaller gains. After 3–5 generations, improvements are marginal.

```
improvement(generation_n) ∝ 1/n
```

**Implication:** STaR alone (Path C) will plateau after a few generations. To sustain improvement, you need to combine it with other sources of improvement (new external data, better evaluation, strategy evolution).

**Scaling Laws for Scientific Discovery (arXiv:2503.22444, 2025):** Proposes that scientific discovery output follows:

```
Knowledge(t) ∝ N_agents^α × Capability^β × t^γ
```

where N_agents is the number of autonomous systems, Capability is per-agent capability, and t is time. The paper estimates α ≈ 0.7 (sublinear in agents — diminishing returns from adding more agents), β ≈ 1.2 (superlinear in capability — better agents are disproportionately valuable), and γ ≈ 1.0 (linear in time for a fixed system).

**Implication:** Investing in per-agent capability (Paths C, D) has higher returns than scaling the number of agents (unless agents are truly diverse and complementary).

### 4.2 The Critical Scaling Question

The central question is: **does the self-improvement rate compound, plateau, or collapse?**

```
COMPOUNDING:
improvement(t) = c × (1 + r)^t
→ The system gets better at getting better
→ This is the AI takeoff scenario
→ No evidence this happens with current methods

PLATEAU (most likely):
improvement(t) = c × log(t)
→ Diminishing returns but continuous improvement
→ Consistent with FunSearch, STaR, and RL scaling laws
→ Still very valuable — log(t) compounds with t

COLLAPSE (must prevent):
improvement(t) = c × (1 - e^(-t/τ)) - d × t
→ Initial improvement followed by degradation
→ Happens with model collapse, Goodhart, exploration starvation
→ The traps described in Section 3
```

**What the evidence suggests:** Plateau (log improvement) is the most likely outcome with current methods. This is still transformatively useful — a system that produces 10× better research after 1 year of operation is enormously valuable, even if it doesn't produce 100× after 2 years.

**Compounding improvement requires solving the recursive compiler problem (Path E):** The only way to get superlinear improvement is if the system can improve its own improvement process. This is theoretically possible but not yet demonstrated.

### 4.3 Estimated Compute-to-Quality Curves

Based on the scaling laws above, here are rough estimates of what each compute budget buys:

```
Research Quality vs. Compute Budget (estimated)

Quality │
  1.0   │                                          ──────── Tier 3
        │                                    ─────
  0.8   │                              ─────
        │                        ─────
  0.6   │                  ─────                            Tier 2
        │            ─────
  0.4   │       ────                                        Tier 1
        │   ───
  0.2   │ ──                                                Tier 0
        │─
  0.0   └─────────────────────────────────────────────────
        $100  $500  $2K   $5K   $20K  $50K  $100K  $500K
                    Monthly Compute Budget

Quality is normalized: 0.0 = random baseline, 1.0 = expert human researcher
Log-linear: each 10× compute investment roughly doubles quality delta
```

**The uncomfortable implication:** At $100/month (Tier 0), the system produces research roughly 10–20% as good as a human expert. At $20K/month (Tier 2), maybe 50–60%. Reaching 90%+ likely requires >$100K/month AND successful self-improvement loops.

The bitter lesson predicts this curve will shift left over time (same quality at lower cost), but the shape will remain log-linear.

---

## 5. Specifically: Continuous Learning vs. Batch Datasets

### 5.1 The System Should Be Continuous, Not Batch

Traditional ML operates in batch mode: collect dataset → train → deploy → repeat. Our system should operate continuously:

```
BATCH MODE (wrong for this system):
┌────────┐    ┌───────┐    ┌────────┐    ┌───────┐
│ Collect │───►│ Train │───►│ Deploy │───►│ Stale │
│ data    │    │       │    │        │    │       │
└────────┘    └───────┘    └────────┘    └───────┘
     ↑                                        │
     └────────────────────────────────────────┘
              (repeat every few months)

CONTINUOUS MODE (right for this system):
┌──────────────────────────────────────────────────┐
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Ingest   │──►│ Research │──►│ Evaluate │    │
│  │ (daily)  │   │ (ongoing)│   │ (ongoing)│    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘    │
│       │              │              │           │
│       ▼              ▼              ▼           │
│  ┌──────────────────────────────────────────┐   │
│  │     Knowledge Graph + Memory + Archive    │   │
│  │     (continuously updated)                │   │
│  └──────────────────────────────────────────┘   │
│       │              │              │           │
│       ▼              ▼              ▼           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │ Update   │   │ Evolve   │   │ Train    │   │
│  │ reward   │   │ strategy │   │ policy   │   │
│  │ model    │   │ (weekly) │   │ (daily)  │   │
│  │ (monthly)│   │          │   │          │   │
│  └──────────┘   └──────────┘   └──────────┘   │
│                                                  │
└──────────────────────────────────────────────────┘
```

**Why continuous:** The AI research field moves so fast that a batch-trained system is stale within weeks. A paper published today may invalidate a hypothesis the system generated yesterday. Continuous ingestion keeps the knowledge graph current.

### 5.2 What "Continuous Learning" Actually Means Here

There are two very different things called "continuous learning," and they have different feasibility:

**1. Continuous knowledge updates (feasible, should do):**
- New papers → update knowledge graph → update Gap Map
- New reflections → update memory store
- New evaluations → update reward model calibration
- No weight updates. Just external memory updates.
- This is basically RAG with a continuously updated corpus.

**2. Continuous weight updates (hard, do carefully):**
- Fine-tune the model on new data periodically (not continuously)
- Risk: catastrophic forgetting, distribution shift, model collapse
- Recommendation: batch updates on a schedule (monthly for reward model, quarterly for base model), not true online learning
- Always validate before deploying weight updates

**The practical answer:** Use continuous knowledge updates (external memory) for day-to-day operation. Use periodic batch weight updates for deeper capability improvement. Don't try true online learning on LLM weights — the risks outweigh the benefits with current methods.

### 5.3 The Reinforcement Learning Data Cycle

For the RL components (Path D), data is generated through operation:

```
Episode 1: Agent selects topic T1, runs research, gets reward R1
Episode 2: Agent selects topic T2, runs research, gets reward R2
...
Episode N: Agent selects topic TN, runs research, gets reward RN

After N episodes: train policy on {(state_i, action_i, reward_i)}
After 2N episodes: retrain policy on all 2N episodes
...
```

**How many episodes before RL training is useful?**

This depends on the complexity of the policy and the noise in the reward:

| State space | Action space | Reward noise | Min episodes for useful policy |
|---|---|---|---|
| Low-dim embedding | 10 discrete topics | Low (execution-based) | ~100 |
| Medium-dim | 50 topics + strategy params | Medium (SUNFIRE) | ~500 |
| High-dim | Continuous strategy space | High (human eval) | ~2,000+ |

**Estimate for our system:** With a reasonable state representation (256-dim embedding of current knowledge + Gap Map) and a moderate action space (choose from 20–50 research directions), approximately **300–500 research episodes** are needed before the RL policy outperforms random topic selection.

At 5 episodes/day (Tier 1), this is 2–3 months of operation. At 50 episodes/day (Tier 2), this is 1–2 weeks.

---

## 6. Concrete Data Collection Plan

### Week 1: Bootstrap

| Task | Estimated Time | Estimated Cost |
|---|---|---|
| Crawl ArXiv CS.AI/CS.LG/stat.ML abstracts (last 12 months) | 2 hours | $0 |
| Download and process top 3K papers (full text) | 8 hours | $50–100 (LLM extraction) |
| Build citation graph from Semantic Scholar | 2 hours | $0 |
| Crawl OpenReview ICLR/NeurIPS/ICML data | 4 hours | $0 |
| Seed knowledge graph with extracted claims | 4 hours | $50 (LLM extraction) |
| **Total** | **~20 hours** | **$100–150** |

### Week 2–4: Begin Research Operations

| Task | Estimated Time | Estimated Cost |
|---|---|---|
| Run 50+ research episodes (Path B) | Continuous | $200–400 |
| Accumulate reflections and episodic memory | Automatic | $0 |
| Build initial Gap Map from knowledge graph | 4 hours | $50 |
| Train initial reward model on OpenReview data | 2–4 hours | $20–50 |
| **Total** | **Mostly automated** | **$270–500** |

### Month 2–3: Strategy Evolution (Path A)

| Task | Estimated Time | Estimated Cost |
|---|---|---|
| Run 200+ research episodes across 5–10 strategy variants | Continuous | $500–1000 |
| Evolve strategies, track fitness across MAP-Elites grid | Automatic | $0 |
| Accumulate enough data for RL policy training (~300 episodes) | Continuous | Included above |
| **Total** | **Mostly automated** | **$500–1000** |

### Month 3–6: Self-Training Preparation (Path C)

| Task | Required | Estimated Cost |
|---|---|---|
| Corpus of 500+ research outputs with quality scores | From operations | $0 (already generated) |
| Filter for top 1–5% outputs | 1 hour (automated) | $0 |
| Mixed dataset: 30% OpenReview + 70% self-generated | 2 hours | $0 |
| Fine-tuning infrastructure (spot A100) | Setup | $50–100/run |
| **Total** | **Minimal human effort** | **$50–100 per generation** |

---

## 7. Summary: The Data Is Not the Hard Part

**The data requirements for this system are surprisingly modest:**
- Bootstrap data: $100–150, one-time, fully self-crawlable
- Ongoing data: generated through operation, costs $0 beyond compute
- No internet-scale dataset needed
- No proprietary data needed
- No human labeling at scale needed (OpenReview provides labels for free)

**The hard parts are:**
1. **Evaluation quality** — not a data problem, an algorithmic problem
2. **Avoiding intractability traps** — not a data problem, a design problem
3. **Sustaining improvement beyond log-linear** — not a data problem, a research problem

**The data strategy is: self-crawl ArXiv + Semantic Scholar + OpenReview for bootstrap, then generate everything else through operation.** The system should be data-self-sufficient within 2–3 months of operation.

---

## 8. References

### Scaling Laws
- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models." (Chinchilla)
- Gao et al. (2023). "Scaling Laws for Reward Model Overoptimization." ICML 2023.
- arXiv:2503.22444 (2025). "Scaling Laws in Scientific Discovery with AI and Robot Scientists."

### Model Collapse and Self-Training Risks
- Shumailov et al. (2024). "AI models collapse when trained on recursively generated data." Nature.
- Alemohammad et al. (2024). "Self-Consuming Generative Models Go MAD." ICLR 2024.

### Self-Training
- Zelikman et al. (2022). "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.
- Singh et al. (2024). "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models." (ReST^EM)

### Exploration in RL
- Auer et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." (UCB)
- Bellemare et al. (2016). "Unifying Count-Based Exploration and Intrinsic Motivation."

---

*← Back to [Self-Improving Loop Analysis](11-self-improving-loop-analysis.md)*
