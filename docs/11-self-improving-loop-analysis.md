# 11 — Most Promising Paths to a Self-Improving AI Research Loop

> *What actually works, what's closest to working, and what's the fastest path to a system that recursively improves its own research capability?*

---

## 0. What "Self-Improving" Actually Means

Before choosing a path, we need to be precise about what improves, because there are distinct levels and they have very different difficulty:

```
Level 0: Better OUTPUTS (produce better papers/hypotheses per run)
         Mechanism: search over output space
         Difficulty: ★☆☆☆☆ — Demonstrated (Karpathy autoresearch, AI Scientist)

Level 1: Better KNOWLEDGE (understand the field more deeply over time)
         Mechanism: accumulate structured knowledge, episodic memory
         Difficulty: ★★☆☆☆ — Demonstrated (RAG systems, knowledge graphs)

Level 2: Better STRATEGIES (learn which research approaches work)
         Mechanism: RL on research outcomes, reflection accumulation
         Difficulty: ★★★☆☆ — Partially demonstrated (Reflexion, STaR)

Level 3: Better CAPABILITIES (the agent itself becomes more capable)
         Mechanism: fine-tuning on own outputs, distillation, RLHF
         Difficulty: ★★★★☆ — Early demonstrations (STaR, self-play)

Level 4: Better SELF-IMPROVEMENT (the improvement process itself improves)
         Mechanism: meta-learning, recursive self-modification
         Difficulty: ★★★★★ — Theoretical, not demonstrated
```

**The key insight: you don't need Level 4 to build something transformatively useful.** Level 2 alone — an agent that learns which research strategies work and which don't — would be groundbreaking. Level 3 — where the agent's actual capabilities grow — is the real prize.

**The recursive opportunity:** Since our agent's research *topic* is self-improving AI, it has a unique property: its research outputs can be directly applied to improving itself. A discovery about better RL reward shaping can be applied to its own reward model. A finding about better exploration strategies can be applied to its own search. This creates a potential flywheel that no other research domain offers.

---

## 1. The Five Most Promising Concrete Paths

### Path A: Evolutionary Program Search (Highest confidence, near-term)

**What:** Use LLMs as mutation operators in an evolutionary loop over research *programs* — not individual outputs, but the code/prompts that generate outputs.

**Why this is the most promising near-term path:**
- **It's been proven to produce real discoveries.** FunSearch found new cap set constructions (Nature, 2024). AlphaEvolve improved on Strassen's 1969 algorithm and saved 0.7% of Google's compute (2025). These aren't toy results.
- **It requires no weight updates.** The LLM is used as-is (frozen API model). Self-improvement happens at the population level — the archive of programs gets better, not the model's weights. This means you can start TODAY with Claude or GPT-4.
- **It has a natural verification loop.** Programs can be *run* and their outputs *measured*. No need for fuzzy reward models — either the program produces better results or it doesn't.
- **It scales with compute** (bitter lesson). More evaluations per generation = faster improvement. More parallel populations = more diversity.

**Concrete implementation for our system:**

```
┌──────────────────────────────────────────────────────────────┐
│  WHAT EVOLVES: Research strategy programs                     │
│                                                              │
│  A "research strategy program" is a structured prompt/code   │
│  that defines:                                               │
│  - How to select which papers to read                        │
│  - How to generate hypotheses from literature                │
│  - How to design experiments                                 │
│  - How to evaluate results                                   │
│  - How to decide what to do next                             │
│                                                              │
│  THE LOOP:                                                   │
│  1. Population of N strategy programs (start: N=10)          │
│  2. Each strategy runs a research episode (1-4 hours)        │
│  3. Evaluate: SUNFIRE score of outputs + experiment success  │
│  4. Select top-k strategies                                  │
│  5. LLM proposes mutations of top-k strategies:              │
│     - "What if this strategy also did X?"                    │
│     - "What if it skipped step Y?"                           │
│     - "Combine the best part of strategy A with strategy B"  │
│  6. Replace bottom strategies with mutations                 │
│  7. Repeat                                                   │
│                                                              │
│  ARCHIVE: MAP-Elites grid                                    │
│    X-axis: exploration vs. exploitation emphasis              │
│    Y-axis: research domain                                   │
│    Cell: best strategy for that (style, domain) combination  │
└──────────────────────────────────────────────────────────────┘
```

**Self-improvement mechanism:** The population of strategies improves over generations. After 100 generations, the best strategies in the archive will be radically different from the initial hand-written ones. The system discovers research methodologies that humans wouldn't design.

**This is exactly Karpathy's autoresearch, generalized.** Autoresearch evolves `train.py` (a training program). We evolve `research_strategy.py` (a research methodology program). The structure is identical:
- Fixed evaluation (SUNFIRE score ↔ val_bpb)
- Agent modifies the program
- Run, evaluate, keep or discard
- Repeat overnight

**Time to first results:** 2–4 weeks. This can be built with existing infrastructure.

**Key citations:**
- Romera-Paredes et al. (2024). FunSearch. Nature.
- AlphaEvolve (2025). arXiv:2506.13131.
- Lehman et al. (2022). Evolution through Large Models. arXiv:2206.08896.
- Karpathy (2026). autoresearch.

---

### Path B: Reflexion + Compound Memory (Fastest to implement, no training)

**What:** The agent accumulates a growing library of *reflections* — verbal lessons learned from past research episodes — and retrieves relevant ones before each new episode. No weight updates, no fine-tuning. Self-improvement happens through memory accumulation.

**Why this is promising:**
- **Works with frozen API models.** No fine-tuning infrastructure needed. Start immediately.
- **Demonstrated to improve performance across episodes.** Reflexion (Shinn et al., 2023) showed consistent improvement on coding, QA, and reasoning tasks.
- **The "memory" is inspectable and debuggable.** Unlike weight changes, you can read the reflections and understand *why* the system is making different choices.
- **Compounds over time.** After 1000 research episodes, the reflection library contains 1000 lessons about what works and what doesn't. This is an empirically-derived research methodology.

**Concrete implementation:**

```
┌──────────────────────────────────────────────────────────────┐
│  THREE MEMORY STORES:                                        │
│                                                              │
│  1. EPISODIC MEMORY (what happened)                          │
│     - Research episode logs                                  │
│     - Experiments run and their outcomes                     │
│     - Papers read and key findings                           │
│     Tagged: [topic, date, outcome, strategy_used]            │
│                                                              │
│  2. REFLECTION MEMORY (what I learned)                       │
│     After each episode, agent generates:                     │
│     - "What worked well and why?"                            │
│     - "What failed and why?"                                 │
│     - "What would I do differently next time?"               │
│     - "What surprised me?"                                   │
│     Tagged: [topic, lesson_type, confidence]                 │
│                                                              │
│  3. STRATEGY MEMORY (how to do things)                       │
│     Periodically consolidated from reflections:              │
│     - "When researching [topic_type], start with [approach]" │
│     - "Hypothesis quality improves when I [technique]"       │
│     - "Avoid [pattern] because [reason]"                     │
│     Tagged: [applicability_conditions, success_rate]          │
│                                                              │
│  BEFORE EACH EPISODE:                                        │
│  1. Retrieve relevant reflections (semantic search)          │
│  2. Retrieve relevant strategies                             │
│  3. Inject into context: "Based on past experience..."       │
│  4. Run research episode                                     │
│  5. Reflect and store new memories                           │
│                                                              │
│  CONSOLIDATION (weekly):                                     │
│  - Cluster similar reflections                               │
│  - Promote consistent lessons to strategy memory             │
│  - Demote strategies that stopped working                    │
│  - Compress old episodic memories to summaries               │
└──────────────────────────────────────────────────────────────┘
```

**Self-improvement mechanism:** The agent gets better because it remembers what works. It's "self-improving" in the same way a human researcher improves — through accumulated experience, not through changing their neurons.

**The ceiling:** This approach has a ceiling because the base model's capabilities are fixed. It can learn *when* to apply its existing capabilities, but it can't acquire new ones. For true capability growth, you need Path C or D.

**Time to first results:** 1–2 weeks. Minimal infrastructure needed.

**Key citations:**
- Shinn et al. (2023). Reflexion. NeurIPS 2023.
- Madaan et al. (2023). Self-Refine. NeurIPS 2023.
- Park et al. (2023). Generative Agents. (Memory architecture for LLM agents.)

---

### Path C: STaR-Style Self-Training on Own Research Outputs (The capability growth path)

**What:** The agent generates research outputs, filters for the best ones (using external evaluation — peer review scores, experiment success, critic survival), and fine-tunes itself on its own best work. Each generation of the model is trained on the best outputs of the previous generation.

**Why this is promising:**
- **STaR (Zelikman et al., 2022) proved this works** for mathematical reasoning — models bootstrapped from their own correct solutions improve significantly.
- **It produces actual capability growth** — not just better strategies, but better underlying reasoning.
- **It naturally focuses training data on the hardest problems** — easy problems are solved immediately; the training signal comes from problems the model struggled with but eventually got right.
- **It leverages the recursive property:** improvements in research capability produce better research outputs, which produce better training data, which produce further capability improvements.

**Concrete implementation:**

```
┌──────────────────────────────────────────────────────────────┐
│  THE SELF-TRAINING LOOP:                                     │
│                                                              │
│  GENERATION PHASE (1-4 weeks):                               │
│  1. Current model runs research episodes                     │
│  2. Generates: hypotheses, experiment designs, analyses,     │
│     critiques, syntheses                                     │
│  3. Each output is evaluated:                                │
│     - By critic agents (does it survive debate?)             │
│     - By execution (does the experiment actually work?)      │
│     - By external metrics (SUNFIRE score)                    │
│     - By peer review simulation                              │
│                                                              │
│  FILTERING PHASE:                                            │
│  4. Partition outputs into:                                  │
│     - GOLD: Top 10% by composite score                       │
│     - SILVER: Top 10-30%                                     │
│     - DISCARD: Bottom 70%                                    │
│                                                              │
│  TRAINING PHASE:                                             │
│  5. Fine-tune next-generation model on GOLD outputs          │
│     Method: DPO or RLHF                                     │
│     - Positive examples: GOLD outputs                        │
│     - Negative examples: DISCARD outputs                     │
│     - This teaches the model to distinguish good from bad    │
│       research                                               │
│                                                              │
│  6. Also fine-tune on (input, GOLD_output) pairs:            │
│     - "Given this knowledge state + research question,       │
│        produce an output like this GOLD example"             │
│     - This directly improves generation capability           │
│                                                              │
│  VALIDATION PHASE:                                           │
│  7. Run new model on held-out research tasks                 │
│  8. Compare against previous generation                      │
│  9. If improved: deploy as new base model                    │
│  10. If regressed: rollback, investigate why                 │
│                                                              │
│  REPEAT: Each cycle produces a better researcher.            │
│                                                              │
│  CRUCIAL: Start with an open-source model (Llama, Qwen)     │
│  that you can actually fine-tune. API models can't be        │
│  self-trained.                                               │
└──────────────────────────────────────────────────────────────┘
```

**The verification problem:** STaR works for math because you can verify answers. Research is harder — how do you know a hypothesis is "correct"? Strategies:
1. **Use experiment execution as the verifier.** If the agent designs an experiment and it produces the predicted result, that's verified.
2. **Use adversarial critique as a proxy.** Hypotheses that survive multiple rounds of aggressive critique from a separate model are more likely correct.
3. **Use downstream utility.** If a research output enables further successful research, it was useful regardless of "correctness."
4. **Use community feedback (delayed).** Eventually, citations and real peer review provide ground truth.

**Self-improvement mechanism:** The model's weights actually change. Each generation is measurably better at research tasks than the previous one. This is true capability growth, not just strategy improvement.

**Time to first results:** 2–3 months (need data collection period + fine-tuning infrastructure).

**Key citations:**
- Zelikman et al. (2022). STaR: Self-Taught Reasoner. NeurIPS 2022.
- Singh et al. (2024). Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models. (ReST^EM)
- Yuan et al. (2024). Self-Rewarding Language Models. (Meta)
- Chen et al. (2024). Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. (SPIN)

---

### Path D: RL on Research Outcomes (The strategy optimization path)

**What:** Train an RL policy that makes high-level research decisions (what to study, which hypothesis to pursue, when to pivot, how much compute to allocate) using research outcomes as the reward signal.

**Why this is promising:**
- **Research decision-making is a sequential decision problem** — exactly what RL is designed for.
- **The action space is tractable.** You're not training the LLM via RL (too expensive); you're training a small policy network that *directs* the LLM. Actions: "investigate topic X," "pursue hypothesis Y," "allocate Z compute to experiment W."
- **Compound effects.** A small improvement in research direction selection can yield large improvements in output quality — choosing the right problem to work on matters more than how well you work on it.
- **It's the direct optimization of what we care about** — research outcomes.

**Concrete implementation:**

```
┌──────────────────────────────────────────────────────────────┐
│  RL FORMULATION:                                             │
│                                                              │
│  STATE (what the policy observes):                           │
│  - Current knowledge graph summary (embedding)              │
│  - Gap Map frontier positions                                │
│  - Active research threads and their status                  │
│  - Recent reflection summaries                               │
│  - Compute budget remaining                                  │
│  - Time since last successful output                         │
│                                                              │
│  ACTIONS (what the policy decides):                          │
│  - Which research thread to work on next                     │
│  - Whether to start a new thread or continue existing        │
│  - Whether to explore (new topic) or exploit (deepen)        │
│  - How much compute to allocate to the next experiment       │
│  - Whether to pivot (abandon a thread that's not working)    │
│  - When to synthesize across threads                         │
│                                                              │
│  REWARD (what the policy optimizes):                         │
│  Immediate (per action):                                     │
│  - Knowledge graph growth (new entities, resolved conflicts) │
│  - Compression progress (model of the field improved)        │
│  Delayed (per episode):                                      │
│  - SUNFIRE score of research outputs                         │
│  - Critic survival rate of hypotheses                        │
│  - Experiment success rate                                   │
│  Very delayed (per month):                                   │
│  - Human expert evaluation of output portfolio               │
│  - Actual citation/usage of published outputs                │
│                                                              │
│  POLICY ARCHITECTURE:                                        │
│  Small transformer (~100M params) that takes state embedding │
│  and outputs action distribution. Trained with PPO.          │
│  Updated daily based on accumulated research outcomes.       │
│                                                              │
│  THE KEY: This policy is CHEAP to train. It's not the LLM   │
│  — it's a small network that directs the LLM. RL training   │
│  on a 100M param model costs ~$10/day on spot A100.          │
└──────────────────────────────────────────────────────────────┘
```

**Self-improvement mechanism:** The policy gets better at directing research. The LLM's capabilities don't change, but they're applied more effectively. This is like having an increasingly skilled research director managing a team of capable but undirected researchers.

**The sparse reward problem:** Research rewards are delayed by months (citations). Mitigation:
1. **Dense proxy rewards** (SUNFIRE score, critic survival) for daily training signal
2. **Hindsight relabeling** — when a delayed reward arrives, propagate it back to all the decisions that led to it
3. **Simulated environments** — train the policy in simulation first using historical data (which papers led to which outcomes?)

**Time to first results:** 3–6 months (need baseline data collection + RL infrastructure).

**Key citations:**
- Duan et al. (2016). RL². arXiv:1611.02779.
- Finn et al. (2017). MAML. ICML 2017.
- Ouyang et al. (2022). InstructGPT / RLHF.

---

### Path E: The Recursive Research Compiler (Most ambitious, highest ceiling)

**What:** The agent's research topic IS self-improving AI. Every research output it produces is a potential improvement to its own system. It literally researches how to make itself better, implements the findings, evaluates the improvement, and repeats.

**Why this is the highest-ceiling path:**
- **It's the only path with unbounded potential.** All other paths hit ceilings (memory size, base model capability, policy capacity). This path can in principle break through every ceiling because the agent can research how to break through ceilings.
- **It exploits the unique recursive structure of this project.** An agent researching protein folding can't apply its findings to itself. An agent researching self-improving AI can.
- **Every component of the system is a potential research target.** The reward model, the memory architecture, the exploration strategy, the agent coordination protocol — all of these are active research areas where the agent can make contributions *and* apply them.

**Concrete implementation:**

```
┌──────────────────────────────────────────────────────────────┐
│  THE RECURSIVE RESEARCH COMPILER                             │
│                                                              │
│  CAPABILITY REGISTRY:                                        │
│  The system maintains a registry of its own components       │
│  with current performance metrics:                           │
│                                                              │
│  ┌─────────────────────────────────────────────────┐         │
│  │ Component          │ Metric          │ Current  │         │
│  │ Hypothesis quality │ Critic survival │ 34%      │         │
│  │ Literature search  │ Recall@10       │ 72%      │         │
│  │ Experiment design  │ Info gain/cost  │ 0.41     │         │
│  │ Reward model       │ Correlation w/  │ 0.63     │         │
│  │                    │ human judgment  │          │         │
│  │ Topic selection    │ SUNFIRE score   │ 0.58     │         │
│  │ Memory retrieval   │ Relevance@5     │ 0.69     │         │
│  └─────────────────────────────────────────────────┘         │
│                                                              │
│  THE LOOP:                                                   │
│                                                              │
│  1. IDENTIFY BOTTLENECK                                      │
│     Which component, if improved, would most increase        │
│     overall research output quality?                         │
│     (Use sensitivity analysis: perturb each component        │
│      and measure impact on downstream metrics)               │
│                                                              │
│  2. RESEARCH THE BOTTLENECK                                  │
│     Search literature for improvements to that component.    │
│     Generate hypotheses about how to improve it.             │
│     Design experiments to test the hypotheses.               │
│                                                              │
│  3. IMPLEMENT THE IMPROVEMENT                                │
│     Write code. Modify the relevant component.               │
│     This is where the agent modifies ITSELF.                 │
│                                                              │
│  4. EVALUATE                                                 │
│     A/B test: run the improved system alongside the          │
│     current system on the same research tasks.               │
│     Measure: did overall research quality improve?           │
│                                                              │
│  5. DEPLOY OR ROLLBACK                                       │
│     If improved: deploy the new version.                     │
│     If regressed: rollback, log the failure, learn from it.  │
│                                                              │
│  6. UPDATE CAPABILITY REGISTRY                               │
│     Record the new performance metrics.                      │
│     Recalculate the bottleneck.                              │
│     The bottleneck has shifted — a different component       │
│     is now the weakest link.                                 │
│                                                              │
│  7. REPEAT                                                   │
│     Each cycle improves the weakest component.               │
│     The overall system improves monotonically.               │
│     Over time, every component has been improved             │
│     multiple times.                                          │
│                                                              │
│  SAFETY CONSTRAINT:                                          │
│  The agent can modify its own components but NOT:            │
│  - Its safety constraints                                    │
│  - Its evaluation criteria (SUNFIRE weights)                 │
│  - Its rollback mechanism                                    │
│  - Its human oversight interface                             │
│  These are the "constitutional" elements that remain         │
│  human-controlled.                                           │
└──────────────────────────────────────────────────────────────┘
```

**Why this is hard:**
- Requires the agent to be good enough at research to improve its own components — a bootstrap problem
- Self-modification is risky — bad changes can compound
- The evaluation must be bulletproof — if the agent can game its own metrics, it will
- Requires a sophisticated understanding of its own architecture

**The bootstrap strategy:** Don't start here. Start with Paths A+B (evolutionary search + memory accumulation). Use those to build up to Path C (self-training). Use C to build capability for Path D (RL policy). Only then attempt Path E, when the system is already a competent researcher.

**Time to first results:** 12–18 months (requires Paths A–D as prerequisites).

---

## 2. The Recommended Stacking Order

These paths are not alternatives — they stack. The recommended order maximizes the probability of success at each stage by building on demonstrated capabilities:

```
MONTH 1-2: Path B (Reflexion + Memory)
           ↓ Produces: accumulated research experience
           ↓ Enables: knowing what strategies work

MONTH 2-4: Path A (Evolutionary Program Search)
           ↓ Produces: optimized research strategy programs
           ↓ Enables: systematic research methodology improvement

MONTH 4-8: Path C (STaR Self-Training)
           ↓ Requires: corpus of good research outputs (from A+B)
           ↓ Produces: a model with improved research capabilities
           ↓ Enables: higher-quality outputs → better training data → faster improvement

MONTH 6-12: Path D (RL on Research Outcomes)
            ↓ Requires: enough research episodes for RL signal (from A+B+C)
            ↓ Produces: optimized research direction policy
            ↓ Enables: efficient compute allocation, strategic research planning

MONTH 12+: Path E (Recursive Research Compiler)
           ↓ Requires: competent researcher (from A+B+C+D)
           ↓ Produces: a system that improves its own components
           ↓ Enables: unbounded improvement (in principle)
```

**The critical transition is C → D.** Once the model can self-train on its own outputs AND an RL policy directs its research, you have a genuine flywheel:
- Better research direction (D) → better outputs → better training data (C) → more capable model → even better research direction → ...

This is the self-improving loop. Everything before this is scaffolding to reach it.

---

## 3. The Minimal Viable Self-Improving Loop

If you want the **absolute simplest** thing that deserves the name "self-improving AI research loop," here it is:

```python
# The Minimal Self-Improving Research Loop
# This can run TODAY with an API model + a vector DB

while True:
    # 1. Choose what to research (starts random, gets smarter)
    topic = select_topic(
        knowledge_graph,
        past_reflections,  # <-- self-improvement signal
        gap_map
    )

    # 2. Do research
    papers = search_literature(topic)
    knowledge = extract_claims(papers)
    hypothesis = generate_hypothesis(knowledge, past_reflections)

    # 3. Test it
    critique = run_critic(hypothesis)
    if critique.survived:
        experiment = design_experiment(hypothesis)
        result = run_experiment(experiment)
    else:
        result = CritiqueResult(failed_at="critique", reasons=critique.reasons)

    # 4. Evaluate
    score = sunfire_score(result)

    # 5. REFLECT (this is where self-improvement happens)
    reflection = generate_reflection(
        topic=topic,
        hypothesis=hypothesis,
        result=result,
        score=score,
        prompt="""
        What worked? What didn't? What would you do differently?
        What did you learn about researching this type of topic?
        What strategy should you use next time for similar topics?
        """
    )

    # 6. Store reflection
    memory.store(reflection, tags=[topic, score, strategy_used])

    # 7. Update knowledge
    knowledge_graph.update(knowledge)
    gap_map.update(result)

    # The next iteration retrieves relevant reflections
    # and uses them to make better decisions.
    # Over hundreds of iterations, the reflection store
    # becomes a rich, empirically-derived research methodology.
```

**This is ~200 lines of real code.** It requires:
- An LLM API ($100–300/month)
- A vector database (free, local)
- An ArXiv API connection (free)
- A simple experiment sandbox (local Python)

**What makes it "self-improving":** The reflection store. Each cycle adds lessons that inform future cycles. After 100 cycles, the system is making decisions informed by 100 prior experiences. After 1000, it has more empirical research experience than most PhD students.

**What it DOESN'T do (yet):**
- Doesn't change its own weights (Paths C/D)
- Doesn't evolve its strategy programs (Path A)
- Doesn't modify its own code (Path E)

But it's a real, running self-improving loop that you can start with TODAY.

---

## 4. The Biggest Unsolved Problems

### 4.1 The Evaluation Problem

Every self-improving loop depends on a reliable evaluation signal. If the evaluation is wrong, the system improves at the wrong thing. The fundamental challenge:

**Research quality is hard to measure automatically.** SUNFIRE is a proxy. Critic survival is a proxy. Even peer review is a noisy signal. The system needs to get better at evaluating itself, but evaluating evaluation is even harder.

**Most promising mitigations:**
1. **Use execution as ground truth** where possible. An experiment that produces the predicted result is verified. An experiment that fails tells you something too.
2. **Ensemble evaluation.** Multiple evaluation criteria (SUNFIRE, critic, execution, human spot-check). Only trust improvements that show gains across multiple criteria.
3. **Red-team the evaluator.** Periodically try to fool the evaluation system. If you can, it's not reliable enough.
4. **Slow down to be right.** It's better to make 10 high-confidence improvements than 100 that might be noise.

### 4.2 The Cold Start Problem

The system needs good research outputs to train on (Path C), but it can't produce good research outputs until it's been trained. How to bootstrap?

**Most promising mitigations:**
1. **Start with Paths A+B** (no training needed). Accumulate outputs.
2. **Seed with human-written examples.** Use existing high-quality papers as the initial training set.
3. **Use OpenReview data.** Thousands of papers with expert reviews — perfect training data for the reward model.
4. **Lower the bar initially.** The first generation doesn't need to produce Nobel-worthy research. It needs to produce research that is *slightly better than random*. That's enough to start the self-training flywheel.

### 4.3 The Degenerate Loop Problem

A self-improving system can converge to a degenerate fixed point where it produces outputs that score well on its own metrics but are actually useless. This is Goodhart's Law applied to self-improvement.

**Most promising mitigations:**
1. **External anchoring.** Periodically evaluate against metrics the system can't game: human expert judgment, real-world experimental outcomes, actual citations.
2. **Diversity pressure.** The MAP-Elites archive (Path A) prevents convergence to a single strategy. Quality-diversity ensures the system maintains a portfolio of approaches.
3. **Adversarial auditing.** A separate "red team" agent specifically tries to find degenerate outputs that score well. The main system is penalized for producing outputs the red team can expose.
4. **Slow outer loop.** The evaluation criteria (SUNFIRE weights) are updated infrequently and only based on external (human) judgment, not the system's own assessment.

### 4.4 The Alignment Problem (Specific to Self-Improvement)

A system that modifies itself could modify its own goals. Standard alignment concerns apply, but with extra urgency:

**Hard constraints (not modifiable by the system):**
- Constitutional research principles (no dual-use, no fraud, transparency)
- Human oversight of strategic direction
- Rollback capability always available
- Evaluation criteria only modifiable by humans
- Rate limits on self-modification (max 1 component change per day)

---

## 5. What Makes Our Approach Different from Existing Systems

| Property | autoresearch | AI Scientist | FunSearch | **Our system** |
|---|---|---|---|---|
| Scope | Single training file | Single paper | Single function | Full research lifecycle |
| Self-improves | No | No | Population level | Multiple levels (A–E) |
| Multi-agent | No | No | No (ensemble) | Yes (population) |
| Knowledge accumulates | No (fresh/session) | No | Archive only | KG + memory + reflections |
| Researches self-improvement | No | No | No | **Yes (recursive)** |
| Learns from failures | Discards | Limited | Archive pruning | AKDC (explicit failure training) |
| Adapts strategy | No | Fixed pipeline | Fixed evolution | RL policy + evolution |

**Our unique advantage: the recursive property.** Because the research target IS self-improving AI, every output is a potential self-upgrade. No other system has this structure.

---

## 6. Concrete First Steps (This Week)

1. **Build the minimal loop** (Section 3). Get it running. Doesn't need to be good — needs to exist.
2. **Run 50 research episodes.** Accumulate reflections and episodic memories.
3. **Analyze the reflections.** What patterns emerge? What strategies does the system converge on?
4. **Build the evolutionary loop** (Path A). Create 5 variant strategies. Let them compete.
5. **Instrument everything.** Log every decision, every evaluation, every outcome. This data is the fuel for Paths C and D.

---

## 7. References

- Zelikman et al. (2022). "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.
- Singh et al. (2024). "Beyond Human Data: Scaling Self-Training for Problem-Solving." (ReST^EM)
- Yuan et al. (2024). "Self-Rewarding Language Models." Meta.
- Chen et al. (2024). "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models." (SPIN)
- Shinn et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
- Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior."
- Romera-Paredes et al. (2024). "FunSearch." Nature.
- AlphaEvolve (2025). arXiv:2506.13131.
- Lehman et al. (2022). "Evolution through Large Models." arXiv:2206.08896.
- Karpathy (2026). autoresearch.
- Duan et al. (2016). "RL²." arXiv:1611.02779.
- Finn et al. (2017). "MAML." ICML 2017.
- Ouyang et al. (2022). "InstructGPT / RLHF."

---

*← Back to [Bitter Lesson, Loops & Novel Paths](09-bitter-lesson-loops-and-novel-paths.md)*
