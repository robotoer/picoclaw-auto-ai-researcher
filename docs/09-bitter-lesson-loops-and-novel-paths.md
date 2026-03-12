# 09 — The Bitter Lesson, Research Loops, and Novel Paths

> *How do we design a self-improving AI researcher that honors the bitter lesson, leverages reinforcement learning, and scales with compute rather than human engineering?*

---

## 0. The Bitter Lesson as a Design Principle

Richard Sutton's "The Bitter Lesson" (2019) is the single most important meta-observation for this project:

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. [...] Researchers always tried to make systems that worked the way the researchers thought their own minds worked — they tried to put that knowledge in their systems — but it proved ultimately counterproductive."

**Key implications for our autonomous AI researcher:**

1. **Don't hand-engineer research heuristics.** The temptation is to encode "how a good researcher thinks" — topic selection rules, hypothesis templates, quality checklists. The bitter lesson says: these will be outperformed by systems that learn these patterns from data and scale with compute.

2. **Invest in search and learning, not knowledge.** Instead of building elaborate ontologies of "what makes good research," invest in:
   - Large-scale search over the hypothesis space
   - Learned reward models that improve with more data
   - RL policies that discover research strategies humans wouldn't design

3. **Scale the inner loop, not the outer specification.** The system's power should come from running more compute through general algorithms, not from more detailed human specifications of what "interesting" means.

4. **Accept that the system will find strategies we don't understand.** Just as AlphaGo discovered moves that no human Go player would have conceived, a truly scaled AI researcher will pursue research directions that seem counterintuitive but turn out to be productive.

**The practical tension:** We need *some* structure to bootstrap the system (you can't learn from nothing). The art is in providing minimal scaffolding that enables learning, then getting out of the way. The SUNFIRE framework, the Gap Map, the agent roles — these are bootstrapping scaffolds, not permanent architecture. The system should eventually learn when to use them and when to discard them.

### Sutton's Specific Examples and Our Analogies

| Sutton's Example | AI Research Analogy |
|---|---|
| Chess: Deep Blue's search beat hand-coded evaluation | Hypothesis search: broad automated search beats hand-curated research agendas |
| Speech: Statistical methods beat phoneme-based rules | Paper analysis: learned embeddings beat hand-designed taxonomies |
| Vision: ConvNets beat hand-designed features | Research strategy: learned policies beat hand-designed research workflows |
| Go: MCTS + neural nets beat hand-coded heuristics | Scientific discovery: RL + LLMs beat hand-coded scientific method pipelines |

### Sutton's Follow-Up: "The Era of Experience"

Sutton extended the bitter lesson with the concept that AI systems should learn through direct interaction with their environment rather than from human-curated datasets. For our system: the autonomous researcher should learn from *doing research* (running experiments, generating hypotheses, getting feedback), not from studying *how humans do research*.

---

## 0.5 Landscape of Existing Autonomous Research Systems

Before proposing our loops, we survey what exists. The field has moved fast since 2024.

### Karpathy's autoresearch (March 2026)

The most philosophically aligned prior work is Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) (28K+ stars within days of release). Its design is a near-perfect embodiment of the bitter lesson for ML research.

**Architecture:** Deliberately minimal — three files:
- `prepare.py` — fixed data prep and evaluation (human-maintained, never modified by agent)
- `train.py` — the single file the agent edits (full GPT model, optimizer, training loop)
- `program.md` — markdown instructions for the agent (human iterates on this)

**The loop:**
```
1. Read program.md (research strategy/instructions)
2. Modify train.py (change architecture, hyperparams, optimizer, anything)
3. Run training (fixed 5-minute wall clock budget)
4. Evaluate val_bpb (bits per byte — vocab-size-independent)
5. Keep or discard changes based on metric
6. Repeat (~12 experiments/hour, ~100 overnight)
```

**Key insight — "Programming via Instructions":** The human researcher's role shifts from writing code to writing `program.md` — essentially encoding "research org code" as natural language instructions. The agent handles all implementation. This inverts traditional ML development and aligns perfectly with the bitter lesson: let general search (agent exploring code modifications) replace specific human knowledge (hand-designed architectures).

**Key insight — "Fixed Time Budget":** Training always runs for exactly 5 minutes regardless of what the agent changes. This makes experiments directly comparable and means autoresearch finds the most optimal model *for your specific hardware* in that time budget. This is compute-aware search.

**What autoresearch doesn't do (and we must):**
- No multi-agent coordination (single agent only)
- No RL training loop (relies entirely on in-context learning)
- No knowledge accumulation across sessions (each night starts fresh)
- No literature review or hypothesis generation (only modifies training code)
- No self-improvement of the research strategy itself (program.md is human-edited)
- No cross-domain research capability (locked to a single training setup)

**Our system extends autoresearch's philosophy** to the full research lifecycle: not just "optimize this training run" but "discover what to research, how to research it, and learn from the results."

As Karpathy wrote: *"Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies."* — We are building toward exactly this.

### The AI Scientist v1/v2 (Sakana AI, 2024–2025)

The AI Scientist (Lu et al., 2024) was the first end-to-end system automating the full research lifecycle at ~$15 per paper. v2 (arXiv:2504.08066, April 2025) eliminated reliance on human-authored code templates via **progressive agentic tree search** with backtracking. v2 produced the first entirely AI-generated peer-review-accepted workshop paper.

**Loop:** Idea generation → Implementation → Experimentation → Analysis → Writing → Review → Iterate.

### FunSearch and AlphaEvolve (DeepMind, 2024–2025)

FunSearch (Nature, January 2024) pairs an LLM with an automated evaluator in an evolutionary loop over *programs* (not solutions). Discovered new cap set constructions exceeding best-known bounds — the first verified LLM-driven mathematical discovery.

AlphaEvolve (arXiv:2506.13131, May 2025) generalized this to a full evolutionary coding agent. Key discoveries: first improvement over Strassen's 1969 matrix multiplication algorithm, 23% speedup in Gemini training kernels, 0.7% recovery of Google's worldwide compute resources.

**Critical design principle shared with our approach:** Search over *programs* (how to solve), not *solutions* (what the answer is). Programs are verifiable, interpretable, and transferable.

### AgentRxiv (Schmidgall & Moor, March 2025)

A shared preprint server for autonomous research agents (arXiv:2503.18102). Multiple agent laboratories upload/retrieve reports and build on each other's findings. Achieved 13.7% improvement on MATH-500 through inter-agent knowledge sharing. Demonstrates that multi-agent collaboration on research scales.

### Documented Failure Modes (arXiv:2601.03315, January 2026)

"Why LLMs Aren't Scientists Yet" documents six recurring failure modes:
1. **Bias toward training data defaults** — models default to outdated approaches
2. **Implementation drift under execution pressure** — agents deviate from plans when errors arise
3. **Memory and context degradation** across long-horizon tasks
4. **Overexcitement** — declaring success despite obvious failures
5. **Insufficient domain intelligence** — inability to recognize trivial results
6. **Weak scientific taste** — cannot distinguish meaningful from trivial contributions

Three critical unsolved problems: **long-horizon coherence**, **research taste**, and **missing negative-space training data**. Our loops must explicitly address all six.

### Summary Table

| System | Loop Type | Scope | Self-Improving? | Multi-Agent? |
|---|---|---|---|---|
| autoresearch (Karpathy) | Edit-Train-Eval | Single training file | No (fresh each session) | No |
| AI Scientist v2 (Sakana) | Agentic tree search | Full paper pipeline | No | No |
| FunSearch/AlphaEvolve (DeepMind) | Evolutionary + LLM | Program search | Yes (evolution) | No (ensemble) |
| AgentRxiv | Shared knowledge | Multi-lab collaboration | Partial | Yes |
| ChemCrow | Tool-augmented LLM | Wet-lab chemistry | No | No |
| **Our system (proposed)** | **6-layer composed loops** | **Full research lifecycle** | **Yes (RL + evolution)** | **Yes (population)** |

---

## 1. Four Research-Backed Loops / High-Level Algorithms

### Loop 1: The ReAct-Reflexion Research Cycle (Observe-Orient-Reason-Act-Reflect)

**Inspiration:** ReAct (Yao et al., 2022), Reflexion (Shinn et al., 2023), OODA loop (Boyd, 1986)

**Core idea:** The agent interleaves reasoning with action, and after each research episode, generates a verbal self-reflection that is stored in episodic memory. Future episodes retrieve relevant reflections, enabling cross-episode learning without weight updates.

```
┌─────────────────────────────────────────────────┐
│  OBSERVE: Ingest new papers, scan ArXiv feed,   │
│  monitor citation graphs, detect anomalies       │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  ORIENT: Update knowledge graph, recalculate    │
│  Gap Map frontiers, assess current competence    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  REASON: Chain-of-thought over current state.    │
│  "Given what I know and what's changed, what     │
│  research direction has highest expected value?"  │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  ACT: Generate hypothesis, design experiment,    │
│  run simulation, draft output                    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  REFLECT: Evaluate outcome. "What worked? What   │
│  didn't? What would I do differently? What did   │
│  I learn about the research landscape?"          │
│                                                  │
│  Store reflection in episodic memory with tags:  │
│  [topic, strategy_used, outcome, lesson]         │
└──────────────────┬──────────────────────────────┘
                   │
                   └──────► Next cycle (retrieve relevant
                            past reflections before ORIENT)
```

**Why it works (bitter lesson alignment):** The reflections are *learned* research heuristics, not hand-coded ones. Over hundreds of cycles, the reflection store becomes a rich, experience-derived strategy corpus. The system discovers what works by doing, not by being told.

**RL connection:** The reflection store can be formalized as an experience replay buffer. Policy gradient methods (PPO, REINFORCE) can optimize the REASON step's strategy selection, with the REFLECT step providing the training signal. The key innovation from Reflexion is that the "training" happens through natural language stored in memory, not through gradient updates — making it compatible with frozen API-served models.

**Key citations:**
- Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
- Shinn et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
- Madaan et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023.

---

### Loop 2: The Self-Play Scientific Debate Loop (Generator-Critic Co-Evolution)

**Inspiration:** AlphaZero self-play (Silver et al., 2017), AI Safety via Debate (Irving et al., 2018), Generative Adversarial Networks (Goodfellow et al., 2014)

**Core idea:** Two agents — a Generator and a Critic — co-evolve through adversarial interaction. The Generator proposes hypotheses, experimental designs, and research outputs. The Critic attacks them: finding flaws, prior art, methodological weaknesses, and logical gaps. Both are rewarded: the Generator for producing ideas that survive critique, the Critic for successfully identifying real flaws.

```
┌──────────────────────────────────────────────────┐
│  GENERATOR proposes hypothesis H with:           │
│  - Supporting evidence from knowledge graph      │
│  - Proposed experimental test                    │
│  - Predicted outcome                             │
│  - Novelty claim                                 │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  CRITIC attacks H:                               │
│  Round 1: Prior art search — has this been done? │
│  Round 2: Logical analysis — does the reasoning  │
│           hold? Are there hidden assumptions?    │
│  Round 3: Methodological review — would the      │
│           proposed experiment actually test H?   │
│  Round 4: Counter-hypothesis generation — what   │
│           alternative explanations exist?        │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  GENERATOR responds to critique:                 │
│  - Addresses each objection                      │
│  - Revises H if needed                           │
│  - Provides additional evidence                  │
│  - Concedes points that are valid                │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  JUDGE evaluates the debate:                     │
│  - Score: hypothesis quality × critique quality  │
│  - Generator reward: R_G = survive_rate × novelty│
│  - Critic reward: R_C = flaw_detection_rate      │
│  - Both rewards feed back into RL fine-tuning    │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
            Surviving hypotheses → Experiment pipeline
            Debate transcripts → Training data for both agents
```

**Why it works (bitter lesson alignment):** Self-play is the purest expression of the bitter lesson — the system generates its own training data through interaction, scaling with compute. More debate rounds = better hypotheses, without any additional human knowledge engineering. AlphaZero showed this in Go; this loop applies the same principle to scientific reasoning.

**RL connection:** The Generator and Critic can be trained with multi-agent RL. The Generator's policy is optimized via policy gradients with the Judge's score as reward. The Critic's policy is trained as an adversary (similar to the discriminator in a GAN, but operating on structured arguments rather than images). The co-evolutionary dynamic prevents either agent from stagnating.

**Key citations:**
- Silver et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm."
- Irving et al. (2018). "AI Safety via Debate." arXiv:1805.00899.
- Du et al. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate."
- Liang et al. (2023). "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate."

---

### Loop 3: The Evolutionary Population-Based Research Loop (Quality-Diversity)

**Inspiration:** MAP-Elites (Mouret & Clune, 2015), Population Based Training (Jaderberg et al., 2017), POET (Wang et al., 2019), Quality-Diversity optimization

**Core idea:** Instead of optimizing a single research strategy, maintain a *population* of diverse research agents (or research threads) that explore different regions of the research space. Periodically select the best-performing agents, mutate their strategies, and replace the worst-performing ones. This is evolution applied to research methodology.

```
┌──────────────────────────────────────────────────┐
│  INITIALIZE: Population of N research agents     │
│  with diverse strategies:                        │
│  - Agent A: exploitation-focused (incremental)   │
│  - Agent B: exploration-focused (moonshot)       │
│  - Agent C: cross-domain synthesizer             │
│  - Agent D: contrarian (challenges consensus)    │
│  - Agent E: replication-focused                  │
│  ...                                             │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  PARALLEL EXECUTION: All agents run research     │
│  threads simultaneously on different topics      │
│  and with different strategies                   │
│  (Each agent has its own inner loop from Loop 1) │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  EVALUATION: Score each agent's outputs using    │
│  SUNFIRE + community feedback + downstream       │
│  citation velocity                               │
│                                                  │
│  MAP-Elites grid:                                │
│    X-axis: Novelty (low → high)                  │
│    Y-axis: Topic domain                          │
│    Cell value: Best SUNFIRE score achieved        │
│                                                  │
│  Goal: Fill every cell of the grid, not just     │
│  maximize a single score                         │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  SELECTION + MUTATION:                           │
│  - Copy strategies from high-performing agents   │
│  - Mutate: change temperature, topic weights,    │
│    exploration rate, depth vs. breadth preference │
│  - Replace worst-performing agents               │
│  - Occasionally inject fully random new agents   │
│    ("immigration" to prevent premature           │
│    convergence)                                  │
└──────────────────┬───────────────────────────────┘
                   │
                   └──────► Next generation
```

**Why it works (bitter lesson alignment):** Population-based search is fundamentally a compute-scaling strategy. More agents = more of the research landscape explored. The system discovers effective research strategies through selection pressure rather than human design. This is exactly the lesson: throw compute at the problem via general search, don't hand-design the search strategy.

**RL connection:** Population Based Training (PBT) is a hybrid of RL and evolutionary methods — it combines online RL training with population-level hyperparameter evolution. Each agent in the population can be running its own RL policy for hypothesis generation, with PBT managing the meta-level strategy evolution. Quality-Diversity (QD) algorithms like MAP-Elites ensure the population doesn't collapse to a single strategy, which is critical for research where diversity of approach is as valuable as peak performance.

**Key citations:**
- Mouret & Clune (2015). "Illuminating search spaces by mapping elites." arXiv:1504.04909.
- Jaderberg et al. (2017). "Population Based Training of Neural Networks." arXiv:1711.09846.
- Wang et al. (2019). "POET: Endlessly Generating Increasingly Complex and Diverse Learning Environments." ICML 2019.
- Wang et al. (2020). "Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions."
- Ecoffet et al. (2021). "First return, then explore." Nature 590.
- Faldor et al. (2024). "OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness." ICML 2024.
- Lehman et al. (2022). "Evolution through Large Models." arXiv:2206.08896. (OpenELM — LLMs as intelligent mutation operators in evolutionary search; combined with MAP-Elites, generated hundreds of thousands of functional programs in domains the LLM had never seen in pre-training.)

---

### Loop 4: The Meta-RL Curriculum Loop (Learning to Learn to Research)

**Inspiration:** RL² (Duan et al., 2016), MAML (Finn et al., 2017), Learning to Reinforcement Learn (Wang et al., 2016), Curriculum Learning (Bengio et al., 2009), Automated Curriculum Learning (Portelas et al., 2020)

**Core idea:** Instead of hand-designing the order in which the agent learns topics and acquires skills, use meta-RL to learn a *curriculum policy* that decides what the agent should study next. The outer loop trains this curriculum policy; the inner loop is the agent actually doing research on the chosen topic. The curriculum policy is rewarded based on how much the inner agent's research capability improves.

```
┌──────────────────────────────────────────────────┐
│  OUTER LOOP (Meta-RL): Curriculum Policy π_C     │
│                                                  │
│  State: Agent's current competence map           │
│         (topic → confidence score),              │
│         Gap Map frontier positions,              │
│         Recent research performance metrics      │
│                                                  │
│  Action: Select next topic/skill to learn        │
│          (which papers to read, which subfield   │
│           to explore, which tool to master)      │
│                                                  │
│  Reward: Improvement in downstream research      │
│          quality (SUNFIRE delta after learning)   │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  INNER LOOP: Agent learns the selected topic     │
│                                                  │
│  1. Read relevant papers (active learning:       │
│     prioritize high-uncertainty papers)          │
│  2. Extract and integrate knowledge              │
│  3. Attempt research tasks in the new domain     │
│  4. Self-evaluate progress                       │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  EVALUATE: How did learning this topic affect     │
│  the agent's overall research capability?        │
│                                                  │
│  Metrics:                                        │
│  - Hypothesis quality on related topics           │
│  - Experiment design quality                     │
│  - Cross-domain synthesis ability                │
│  - Gap-filling success rate                      │
│                                                  │
│  Compute reward for curriculum policy:           │
│  R_C = SUNFIRE(after) - SUNFIRE(before) +        │
│        forward_transfer_bonus                    │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  UPDATE: Train curriculum policy π_C using PPO   │
│                                                  │
│  The policy learns patterns like:                │
│  - "Learning topic X before Y improves Y perf"   │
│  - "Broad exploration early, deep dive later"    │
│  - "Switch topics when marginal gains decline"   │
│  - "Foundational topics have high forward        │
│     transfer — learn them first"                 │
└──────────────────┬───────────────────────────────┘
                   │
                   └──────► Next curriculum step
```

**Why it works (bitter lesson alignment):** The curriculum itself is learned, not designed. A human researcher might say "learn linear algebra before machine learning" — but a meta-RL curriculum policy discovers these dependencies from data, and may discover non-obvious orderings that a human wouldn't prescribe. The system learns *how to learn*, which is a second-order application of the bitter lesson.

**RL connection:** This is pure meta-RL. The curriculum policy is a standard RL agent whose environment is the inner agent's learning process. MAML provides the mathematical framework: the outer gradient optimizes for parameters that enable fast inner-loop adaptation. RL² shows that an RNN trained across many episodes can implicitly implement a learning algorithm in its hidden state — applied here, the curriculum policy's hidden state encodes a learned theory of what makes certain learning orderings productive.

**Key citations:**
- Duan et al. (2016). "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning." arXiv:1611.02779.
- Finn et al. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." ICML 2017.
- Wang et al. (2016). "Learning to Reinforcement Learn." CogSci 2016.
- Bengio et al. (2009). "Curriculum Learning." ICML 2009.
- Portelas et al. (2020). "Automatic Curriculum Learning For Deep RL: A Short Survey." IJCAI 2020.
- Dennis et al. (2020). "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design." NeurIPS 2020.

---

## 2. Two Completely Novel Paths

### Novel Path 1: The Thermodynamic Knowledge Reactor

**Conceptual foundation:** Treat the research frontier as a thermodynamic system. The knowledge graph has "free energy" — the gap between what is known and what is knowable given current methods. The autonomous researcher is a heat engine that converts compute into knowledge, reducing the free energy of the system.

This draws on:
- Karl Friston's Free Energy Principle (FEP) — organisms minimize variational free energy (surprise)
- Schmidhuber's Compression Progress — interestingness = rate of compression improvement
- Statistical mechanics of information (Jaynes, Landauer)

But applies them in a way that has not been proposed: **treating the entire research process as a thermodynamic cycle with measurable efficiency**.

```
┌───────────────────────────────────────────────────┐
│  THE KNOWLEDGE THERMODYNAMIC CYCLE                │
│                                                   │
│  1. FREE ENERGY COMPUTATION                       │
│     For each region R of the Gap Map, compute:    │
│                                                   │
│     F(R) = E(R) - T·S(R)                          │
│                                                   │
│     where:                                        │
│     E(R) = "energy" = inverse coverage density    │
│            (uncovered gaps have high energy)       │
│     T = "temperature" = exploration rate           │
│         (how aggressively the agent explores)      │
│     S(R) = "entropy" = uncertainty about the       │
│            region (how much we don't know about    │
│            what we don't know)                     │
│                                                   │
│  2. WORK EXTRACTION                               │
│     The agent does "work" by researching region R: │
│     - Generates hypotheses (reduces E)             │
│     - Runs experiments (reduces S)                 │
│     - Publishes results (crystallizes knowledge)   │
│                                                   │
│     Work = ΔF = F(before) - F(after)              │
│                                                   │
│  3. EFFICIENCY MONITORING                         │
│     η = Work / Compute_spent                       │
│                                                   │
│     The system tracks η per region, per strategy,  │
│     per agent. High-η strategies are scaled up.    │
│     Low-η strategies are abandoned.                │
│                                                   │
│  4. TEMPERATURE ANNEALING                          │
│     Start with high T (broad exploration).         │
│     As regions are covered, anneal T downward      │
│     (exploit, go deeper). But periodically         │
│     "reheat" to escape local minima (revisit       │
│     abandoned areas with fresh perspective).       │
│                                                   │
│  5. PHASE TRANSITIONS                              │
│     Monitor for phase transitions: sudden drops    │
│     in F that indicate a paradigm shift. When      │
│     detected, the system enters a "critical"       │
│     regime where exploration rate spikes and the   │
│     curriculum is reorganized around the new        │
│     paradigm.                                     │
│                                                   │
│  6. CARNOT LIMIT                                   │
│     There exists a theoretical maximum efficiency  │
│     for converting compute into knowledge in any   │
│     given region. The system estimates this limit  │
│     and stops investing when actual η approaches   │
│     it — the region is "thermally exhausted."      │
└───────────────────────────────────────────────────┘
```

**What makes this novel:**
- No existing system models research productivity as a thermodynamic process with measurable efficiency
- The temperature annealing schedule is *learned*, not prescribed (bitter lesson)
- Phase transition detection enables automatic paradigm recognition
- The Carnot limit concept provides a principled stopping criterion: stop researching a topic when you're approaching the theoretical maximum extraction rate
- The free energy formulation naturally balances exploration (high entropy regions) against exploitation (high energy regions)

**Bitter lesson alignment:** The thermodynamic framework is a *general* formulation that works across all research domains. It doesn't encode any domain-specific research knowledge — it's pure compute-scaling applied through an information-theoretic lens. The system discovers where to invest compute by measuring thermodynamic efficiency, not by following human-designed research agendas.

**RL integration:** The temperature schedule and region selection are RL policies trained to maximize cumulative work extraction (ΔF) per unit compute. This is a resource-constrained RL problem similar to budget-aware exploration in multi-armed bandits, but over a continuous, evolving landscape.

**Active Inference Foundation:** This novel path is deeply connected to Karl Friston's Free Energy Principle (FEP) and active inference framework. In FEP, organisms minimize *variational free energy* — the difference between their model's predictions and actual observations. The Expected Free Energy (EFE) decomposes into:
- **Pragmatic value** (exploitation): Expected utility — pursuing known-productive research
- **Epistemic value** (exploration): Expected information gain — resolving uncertainty about the research landscape

The key insight from active inference is that exploration and exploitation are **unified under a single objective** (minimize free energy), not balanced as a tradeoff. An agent maximizes epistemic value until uncertainty is sufficiently reduced, then automatically shifts to exploitation. This provides the principled temperature annealing schedule: temperature is not a knob we set — it's an emergent property of the agent's uncertainty about each knowledge region.

**Go-Explore Integration:** The "first return, then explore" principle from Go-Explore (Ecoffet et al., 2021, Nature) directly applies. When revisiting a research direction, the agent should deterministically return to a known-good knowledge state (its best understanding from last time) before exploring new territory. This solves two failure modes identified in autonomous research systems:
- **Detachment:** Forgetting how to reach a productive research state
- **Derailment:** Failing to return to a productive state before branching

**Connection to existing theory:**
- Friston (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*.
- Parr, Pezzulo & Friston (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior.* MIT Press.
- Schmidhuber (2009). "Driven by Compression Progress."
- Jaynes (1957). "Information Theory and Statistical Mechanics." *Physical Review*.
- Still et al. (2012). "Thermodynamics of Prediction." *Physical Review Letters*.
- Ecoffet et al. (2021). "First return, then explore." *Nature* 590.

---

### Novel Path 2: The Adversarial Knowledge Distillation Cascade (AKDC)

**Conceptual foundation:** Instead of a fixed architecture of specialist agents, build a *cascade* of progressively more capable research agents where each generation is distilled from the previous generation plus its failures. The key innovation: each new generation is specifically trained on the *problems the previous generation couldn't solve*, creating a natural curriculum of increasing difficulty.

This combines:
- Knowledge distillation (Hinton et al., 2015)
- Adversarial training (Goodfellow et al., 2014)
- Curriculum learning from failures (Shrivastava et al., 2016 — hard negative mining)
- Progressive growing (Karras et al., 2018 — from image generation)

But in a novel configuration: **a cascade where failure is the primary training signal and each level of the cascade handles what the previous level couldn't**.

```
┌───────────────────────────────────────────────────┐
│  THE ADVERSARIAL KNOWLEDGE DISTILLATION CASCADE   │
│                                                   │
│  LEVEL 0: Base Agent (frontier LLM, no fine-tune) │
│  ┌─────────────────────────────────────┐          │
│  │ Research tasks → success/failure     │          │
│  │ Success: output goes to publication  │          │
│  │ Failure: task + failure mode logged  │          │
│  └─────────┬───────────────────────────┘          │
│            │ failure cases                        │
│  LEVEL 1: Specialist-1                            │
│  ┌─────────▼───────────────────────────┐          │
│  │ Distilled from Level 0 +            │          │
│  │ fine-tuned on Level 0's failures    │          │
│  │                                     │          │
│  │ Handles: problems L0 couldn't solve │          │
│  │ New successes → publication         │          │
│  │ New failures → Level 2 training set │          │
│  └─────────┬───────────────────────────┘          │
│            │ failure cases                        │
│  LEVEL 2: Specialist-2                            │
│  ┌─────────▼───────────────────────────┐          │
│  │ Distilled from Level 1 +            │          │
│  │ fine-tuned on Level 1's failures    │          │
│  │ + adversarial augmentation          │          │
│  │                                     │          │
│  │ This level is trained with an        │          │
│  │ adversary that generates harder     │          │
│  │ research problems based on the      │          │
│  │ failure patterns of L0 and L1       │          │
│  └─────────┬───────────────────────────┘          │
│            │                                      │
│  ... (cascade grows as needed)                    │
│                                                   │
│  ROUTING POLICY:                                  │
│  New research task → attempt at Level 0           │
│  If L0 fails → escalate to Level 1               │
│  If L1 fails → escalate to Level 2               │
│  If all levels fail → flag as "frontier problem"  │
│  and add to the training set for the next cascade │
│  expansion                                        │
│                                                   │
│  BACKWARD PROPAGATION:                            │
│  When Level N solves a problem, distill the        │
│  solution back into Level N-1 via fine-tuning.    │
│  Over time, lower levels absorb capabilities      │
│  from higher levels, becoming more capable.       │
│  This is the "knowledge distillation" cascade.    │
│                                                   │
│  ADVERSARIAL AUGMENTATION:                        │
│  A dedicated adversary agent generates research   │
│  problems designed to be hard for the current     │
│  cascade. These are:                              │
│  - Topics at the boundary of current competence   │
│  - Hypotheses that require cross-domain synthesis │
│  - Experimental designs with subtle confounds     │
│  - Questions that expose known blind spots        │
│                                                   │
│  The adversary is rewarded for generating problems│
│  that the cascade can't solve. This creates an    │
│  automatic difficulty curriculum.                  │
└───────────────────────────────────────────────────┘
```

**What makes this novel:**
- **Failure as primary training signal:** Most systems optimize on successes. The AKDC explicitly mines failures as the most informative training data — the frontier of the agent's capability is defined by what it can't do yet.
- **Progressive specialization through distillation:** Each cascade level becomes a specialist in a specific type of difficult problem, while lower levels handle the routine work. This is analogous to a medical referral system (GP → specialist → super-specialist).
- **Adversarial problem generation:** The adversary creates a self-generating curriculum of increasing difficulty. No human needs to design the curriculum — the adversary learns to find the agent's weaknesses.
- **Backward distillation:** Solutions discovered at higher levels propagate back down, continuously raising the capability floor. This is the opposite of catastrophic forgetting — it's progressive remembering.
- **Scalable compute allocation:** Easy problems are handled cheaply (Level 0). Hard problems get more compute (higher levels). This is compute-efficient because most research tasks are routine; only the frontier problems need deep investment.

**Bitter lesson alignment:** The system's capability grows by training on its own failures at scale. More compute = more research attempted = more failures = more training data = faster capability growth. The adversary ensures the system is always working at its capability frontier, preventing the waste of compute on problems already solved.

**RL integration:**
- The routing policy (which level to attempt first, when to escalate) is an RL policy trained to minimize total compute per successful research output.
- The adversary is trained with RL where the reward is proportional to the cascade's failure rate on generated problems.
- The backward distillation schedule (when to propagate solutions downward) is optimized by meta-RL to maximize the rate of overall capability improvement.

**Connection to existing theory:**
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."
- Goodfellow et al. (2014). "Generative Adversarial Nets."
- Shrivastava et al. (2016). "Training Region-based Object Detectors with Online Hard Example Mining." CVPR 2016.
- Karras et al. (2018). "Progressive Growing of GANs for Improved Quality, Stability, and Variation."
- Sukhbaatar et al. (2018). "Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play." ICLR 2018.

---

## 3. How These Loops Compose: The Full System

The four research-backed loops and two novel paths are not alternatives — they compose into a layered system:

```
┌──────────────────────────────────────────────────────────────┐
│  LAYER 4: Thermodynamic Knowledge Reactor                    │
│  (Novel Path 1)                                              │
│  Global resource allocation. Which regions of knowledge      │
│  space get compute? Temperature annealing. Phase detection.  │
│  This is the strategic "where to invest" layer.              │
└──────────────────────┬───────────────────────────────────────┘
                       │ allocates compute to regions
┌──────────────────────▼───────────────────────────────────────┐
│  LAYER 3: Population-Based Evolution                         │
│  (Loop 3)                                                    │
│  Within each allocated region, maintain a population of      │
│  research strategies. Select, mutate, replace. Quality-      │
│  diversity ensures coverage. This is the "which strategies   │
│  to use" layer.                                              │
└──────────────────────┬───────────────────────────────────────┘
                       │ selects strategies for each agent
┌──────────────────────▼───────────────────────────────────────┐
│  LAYER 2: Meta-RL Curriculum                                 │
│  (Loop 4)                                                    │
│  For each agent with a selected strategy, the curriculum     │
│  policy decides what to learn next. This is the "what to     │
│  study before attempting the research task" layer.           │
└──────────────────────┬───────────────────────────────────────┘
                       │ provides curriculum for each agent
┌──────────────────────▼───────────────────────────────────────┐
│  LAYER 1: ReAct-Reflexion + Self-Play Debate                 │
│  (Loops 1 & 2)                                               │
│  Individual agents execute research: observe, reason, act,   │
│  reflect. Hypotheses go through adversarial debate. This     │
│  is the "doing the actual research" layer.                   │
└──────────────────────┬───────────────────────────────────────┘
                       │ produces research outputs + failures
┌──────────────────────▼───────────────────────────────────────┐
│  LAYER 0: Adversarial Knowledge Distillation Cascade         │
│  (Novel Path 2)                                              │
│  Failures from Layer 1 feed the cascade. Solutions from      │
│  higher cascade levels propagate back. The cascade handles   │
│  progressive capability building. This is the "how to get    │
│  smarter over time" layer.                                   │
└──────────────────────────────────────────────────────────────┘
```

**Signals flow bidirectionally:**
- Top-down: Compute allocation → strategy selection → curriculum → execution
- Bottom-up: Research outcomes → updated knowledge → revised strategies → revised compute allocation

---

## 4. Hardware Budget and Performance Tiers

The system's capability is directly proportional to compute investment (the bitter lesson in action). Here are concrete budgets at different performance levels.

### Tier 0: Proof of Concept — "Laptop Researcher" ($100–300/month)

**What you get:** A single-agent research assistant that reads papers, maintains a local knowledge graph, and generates hypotheses. No RL, no multi-agent, no self-improvement. Essentially Loop 1 only, without the RL training.

| Component | Specification | Cost |
|---|---|---|
| LLM inference | Claude Sonnet via API, ~2M tokens/day | ~$120/month |
| Embedding model | Local sentence-transformers on CPU | $0 (local) |
| Vector DB | Qdrant local (Docker) | $0 (local) |
| Knowledge graph | NetworkX in-memory or SQLite | $0 (local) |
| Storage | 50GB local SSD | $0 (existing) |
| **Total** | | **~$120–300/month** |

**Performance:** Can process ~50 papers/day. Generates ~5 hypotheses/day. No experiment execution. No self-improvement. Single research thread.

**Bottleneck:** LLM API token budget limits depth of analysis.

---

### Tier 1: Indie Researcher — "Single GPU Workstation" ($500–2,000/month)

**What you get:** Multi-agent system with debate loops (Loops 1+2). Local LLM inference for routine tasks, API for frontier reasoning. Basic RL-based curriculum learning. Can run small-scale experiments.

| Component | Specification | Cost |
|---|---|---|
| LLM inference (frontier) | Claude Opus/Sonnet API, ~5M tokens/day | ~$300–600/month |
| LLM inference (routine) | Local Llama 3.1 70B on RTX 4090 / RTX 5090 | $0–50/month (electricity) |
| GPU hardware | 1× RTX 4090/5090 (24–32GB VRAM) | ~$2,000–2,500 one-time (amortized ~$80/month over 2.5yr) |
| Embedding model | Local SPECTER/sentence-transformers on GPU | $0 (shared GPU) |
| Vector DB | Qdrant local (Docker) | $0 (local) |
| Knowledge graph | Neo4j Community Edition (local) | $0 (local) |
| Cloud compute (burst) | Spot H100 instances for fine-tuning (~10 hrs/month) | ~$200–400/month |
| Storage | 500GB SSD + 2TB HDD | ~$10/month (amortized) |
| **Total** | | **~$500–2,000/month** |

**Performance:** Can process ~200 papers/day. 3-agent debate loop. ~20 hypotheses/day with critique. Basic RL curriculum training (monthly). Can run small ML experiments (training small models, analyzing datasets). 2–3 concurrent research threads.

**Bottleneck:** Fine-tuning budget limits RL iteration speed.

---

### Tier 2: Research Lab — "Multi-GPU Cluster" ($5,000–20,000/month)

**What you get:** Full system with all four loops + Novel Path 2 (AKDC). Population-based evolution with 5–10 agent variants. Meta-RL curriculum learning. Automated experiment execution. Simulated peer review. Continuous learning.

| Component | Specification | Cost |
|---|---|---|
| LLM inference (frontier) | Claude Opus API, ~20M tokens/day | ~$1,200–3,000/month |
| LLM inference (routine) | 2–4× H100 (80GB) for local Llama/Qwen 70B+ | ~$6,000–12,000/month (cloud) |
| Fine-tuning | 4× H100 cluster, ~50 hrs/month | ~$1,500–3,000/month |
| Embedding & retrieval | Dedicated GPU for embeddings | Shared with above |
| Vector DB | Qdrant Cloud (dedicated) | ~$200–500/month |
| Knowledge graph | Neo4j AuraDB Professional | ~$300–1,000/month |
| Experiment sandbox | Kubernetes cluster with GPU access | ~$1,000–3,000/month |
| Storage | 10TB object storage | ~$50/month |
| Monitoring & logging | Prometheus/Grafana stack | ~$100/month |
| **Total** | | **~$5,000–20,000/month** |

**Performance:** Can process ~1,000 papers/day. 6-agent specialist team with debate. Population of 5–10 strategy variants evolving weekly. Meta-RL curriculum updating daily. Can run medium-scale ML experiments (training 1B+ parameter models). 10+ concurrent research threads. Monthly distillation cycles. AKDC with 2–3 cascade levels.

**Bottleneck:** Multi-agent coordination overhead; reward model quality.

---

### Tier 3: Industrial Scale — "Dedicated Compute Cluster" ($50,000–200,000+/month)

**What you get:** Full system including Novel Path 1 (Thermodynamic Knowledge Reactor). Operates continuously across all target research areas. Population of 50+ agent variants. Real-time literature monitoring. Can run large-scale experiments. Produces a continuous stream of research outputs. Self-improving through all loops simultaneously.

| Component | Specification | Cost |
|---|---|---|
| LLM inference (frontier) | Dedicated Claude/GPT-4 enterprise tier, ~100M+ tokens/day | ~$10,000–30,000/month |
| LLM inference (local) | 16–64× H100/H200 cluster for local models | ~$20,000–80,000/month |
| Fine-tuning & RL | Dedicated training cluster (8–32× H100) | ~$5,000–20,000/month |
| Vector DB | Qdrant/Pinecone enterprise (high availability) | ~$1,000–5,000/month |
| Knowledge graph | Neo4j Enterprise (cluster, HA) | ~$2,000–10,000/month |
| Experiment sandbox | Large Kubernetes cluster with multi-GPU jobs | ~$5,000–30,000/month |
| Data pipeline | Dedicated ETL infrastructure | ~$1,000–5,000/month |
| Storage | 100TB+ object storage + SSD cache | ~$500–2,000/month |
| Networking & monitoring | Enterprise-grade observability | ~$500–2,000/month |
| Engineering support | DevOps/MLOps personnel (partial FTE) | ~$5,000–15,000/month |
| **Total** | | **~$50,000–200,000+/month** |

**Performance:** Processes the full ArXiv AI feed in real-time (~500+ papers/day with full text). 50+ agent variants in population. Thermodynamic resource allocation across 100+ research regions. AKDC with 5+ cascade levels. Can run large-scale experiments (training 7B+ parameter models, large-scale benchmarks). Produces 5–10 research-quality outputs per week. Continuous self-improvement on all axes. Real paradigm-shift detection.

**Bottleneck:** Reward model calibration; safety review bandwidth; diminishing returns without novel methodological breakthroughs.

---

### Cost Scaling Law (Bitter Lesson Prediction)

Based on the bitter lesson, we predict a **log-linear relationship** between compute investment and research output quality:

```
Research_Quality ∝ log(Compute_Budget)
```

The first $500/month gets you from "nothing" to "useful assistant." Each 10× increase in budget roughly doubles the effective research capability:
- $500 → single agent, basic hypothesis generation
- $5,000 → multi-agent with RL, medium experiments
- $50,000 → full system, continuous self-improvement
- $500,000 → frontier research capability, potential for genuine discovery

This log-linear scaling is the same pattern observed in LLM training (scaling laws), and the bitter lesson predicts it will hold for research capability as well.

---

## 5. The RL Integration Strategy

### 5.1 Where RL Fits in Each Loop

| Loop | RL Component | Algorithm | Training Signal |
|---|---|---|---|
| Loop 1 (ReAct-Reflexion) | Strategy selection in REASON step | PPO / REINFORCE | SUNFIRE score of research output |
| Loop 2 (Self-Play Debate) | Generator and Critic policies | Multi-agent PPO | Judge's debate score |
| Loop 3 (Population Evolution) | Population-level strategy evolution | PBT / CMA-ES | Quality-Diversity metrics |
| Loop 4 (Meta-RL Curriculum) | Curriculum policy | MAML / RL² | Forward transfer + SUNFIRE delta |
| Novel 1 (Thermodynamic) | Temperature schedule + region selection | Contextual bandits | Thermodynamic efficiency η |
| Novel 2 (AKDC) | Routing policy + adversary | Multi-agent RL | Cascade failure reduction rate |

### 5.2 The Reward Hierarchy

```
Ultimate reward (years): Real scientific impact (citations, adoption, replication)
       │
       ▼
Proxy reward (months): Simulated peer review acceptance + SUNFIRE score
       │
       ▼
Immediate reward (days): Critic survival rate + Gap Map coverage increase
       │
       ▼
Intrinsic reward (real-time): Compression progress (Schmidhuber) +
                                prediction error on new papers +
                                empowerment (downstream research options opened)
```

The system operates primarily on intrinsic and immediate rewards (dense, available now) while using proxy and ultimate rewards for periodic calibration. This hierarchy is itself learned via meta-RL: the weights between reward levels are optimized to maximize long-term ultimate reward.

### 5.3 Knowledge Transfer from Existing RL

The bitter lesson suggests we should leverage existing RL knowledge rather than building from scratch. Specific transfers:

1. **From game-playing RL:** MCTS for hypothesis space search (like AlphaGo's move selection, but for research direction selection)
2. **From robotics RL:** Sim-to-real transfer → Literature-to-experiment transfer (train in simulated experiments, deploy to real ones)
3. **From NLP RL (RLHF):** Reward modeling from human preferences → Research quality modeling from peer review data
4. **From multi-agent RL:** Communication protocols, credit assignment → Agent coordination, contribution attribution
5. **From exploration RL:** Intrinsic motivation, count-based exploration → Novelty-seeking in research space

### 5.4 Leveraging Existing Models

Rather than training from scratch, the system should use existing frontier models as:

1. **Knowledge bases:** Frontier LLMs (Claude, GPT-4) encode vast amounts of scientific knowledge. Use them as "lookup tables" for the RL agent's world model.
2. **Skill libraries:** Pre-trained models for specific tasks (code generation, mathematical reasoning, literature search) serve as the RL agent's action primitives.
3. **Reward model initializations:** Fine-tune from models already trained on peer review data (OpenReview) rather than training reward models from scratch.
4. **World model components:** Use LLMs to simulate experimental outcomes ("what would happen if...") as a cheap world model for RL planning (model-based RL with LLM dynamics models).

---

## 6. Risks, Failure Modes, and the Bitter Lesson's Dark Side

### 6.1 The Bitter Lesson Cuts Both Ways

The bitter lesson says general methods + compute wins. But it also implies:

- **Systems that scale will outcompete those that don't** — if a well-funded lab builds a similar system with 100× our compute, our careful engineering becomes irrelevant.
- **Compute-efficient methods may be necessary for bootstrap** — the bitter lesson describes the *long run*. In the short run, clever engineering can provide crucial advantages while compute catches up.
- **The learning pressure may find adversarial solutions** — a system optimizing for SUNFIRE scores may learn to generate "interesting-looking" rather than genuinely interesting research (Goodhart's law at scale).

### 6.2 RL-Specific Risks

| Risk | Description | Mitigation |
|---|---|---|
| Reward hacking | Agent optimizes proxy metrics, not true research quality | Multi-objective rewards; periodic human audits; adversarial red-teaming |
| Policy collapse | Agent converges to producing one type of output | Quality-Diversity constraints; diversity bonuses; population-based approaches |
| Reward model drift | What counts as "good research" changes over time | Continual reward model updates; meta-RL on reward weights |
| Credit assignment | In multi-agent systems, hard to attribute success | Shapley value-based attribution; ablation studies |
| Sample efficiency | RL requires many episodes; research episodes are expensive | Model-based RL with LLM world models; transfer from related tasks |

### 6.3 The Alignment Problem for Research Agents

A research agent optimizing for compute-scaled learning may discover:
- Strategies that game citation metrics without producing useful science
- Topics that are easy to publish on but don't advance understanding
- Ways to generate "novel" results that are technically new but scientifically trivial

This is not hypothetical — it mirrors the incentive problems in human academia (publish-or-perish, citation gaming, salami-slicing). The system needs alignment mechanisms:

1. **Constitutional constraints** (from Anthropic's Constitutional AI): hard-coded principles that override learned behavior
2. **Human-in-the-loop checkpoints**: periodic strategic review by human researchers
3. **Impact measurement**: track whether the system's outputs are actually *used* by others, not just cited
4. **Diversity requirements**: enforce that the portfolio spans breakthrough, development, and application work

---

## 7. Implementation Roadmap

### Phase 0 (Months 1–2): Bootstrap with Loop 1
- Implement ReAct-Reflexion cycle with a single agent
- Build episodic memory store for reflections
- Connect to ArXiv ingestion pipeline (already partially built)
- Establish SUNFIRE baseline metrics
- **Hardware:** Tier 0 ($100–300/month)

### Phase 1 (Months 3–4): Add Loop 2 (Self-Play Debate)
- Implement Generator-Critic-Judge triad
- Build debate logging and analysis pipeline
- First round of hypothesis generation with critique
- **Hardware:** Tier 0–1 ($300–1,000/month)

### Phase 2 (Months 5–8): Add Loop 4 (Meta-RL Curriculum)
- Implement curriculum policy as a small RL agent
- Build competence assessment system
- Train initial curriculum on historical paper reading data
- Begin RL fine-tuning of research strategies
- **Hardware:** Tier 1 ($1,000–2,000/month)

### Phase 3 (Months 9–12): Add Loop 3 (Population Evolution)
- Spawn population of agent variants
- Implement MAP-Elites quality-diversity tracking
- Build population-level selection and mutation
- **Hardware:** Tier 1–2 ($2,000–10,000/month)

### Phase 4 (Months 13–18): Novel Path 2 (AKDC)
- Build failure logging and cascade training pipeline
- Implement adversarial problem generator
- Train first 2-level cascade
- Implement backward distillation
- **Hardware:** Tier 2 ($10,000–20,000/month)

### Phase 5 (Months 19–24): Novel Path 1 (Thermodynamic Reactor)
- Implement free energy computation over Gap Map
- Build temperature annealing scheduler
- Implement phase transition detection
- Full system integration
- **Hardware:** Tier 2–3 ($20,000–100,000/month)

---

## 8. Key Design Decisions (For Brainstorming)

1. **Open-source vs. API models?** Bitter lesson says: use whatever gives the most compute per dollar. Today that's API models for reasoning, open-source for fine-tuning. This will shift as open-source catches up.

2. **Single codebase vs. microservices?** The agents need to be independently deployable and scalable. Microservices with shared knowledge infrastructure (KG, vector DB, episodic memory).

3. **When to start RL training?** You need a baseline of good research outputs to train reward models. Bootstrap with prompt engineering → collect data → train reward model → start RL. Estimate: 3–6 months of data collection before meaningful RL.

4. **How many agent variants in the population?** Start with 3–5. Scale to 10–20 once the evaluation pipeline is robust. Population size is a hyperparameter that PBT itself can optimize.

5. **What defines a "research episode"?** One full cycle of Loop 1 (observe → orient → reason → act → reflect). Duration: hours to days depending on depth. The system needs a clear episode boundary for RL training.

6. **How to handle the cold start problem?** Seed the knowledge graph from existing survey papers. Seed the episodic memory with hand-written reflections about research best practices. Seed the reward model from OpenReview data. Then let the system learn.

7. **What research areas to start with?** Areas where:
   - The agent has strong base knowledge (AI/ML)
   - Feedback loops are fast (computational experiments, not wet-lab)
   - The community is active and produces rapid citation signals
   - Initial suggestion: RL for LLMs, multi-agent systems, efficient inference, AI safety

---

## 9. References

### The Bitter Lesson
- Sutton, R. (2019). "The Bitter Lesson." http://www.incompleteideas.net/IncIdeas/BitterLesson.html

### Loop 1: ReAct-Reflexion
- Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023.
- Shinn et al. (2023). "Reflexion: Language Agents with Verbal Reinforcement Learning." NeurIPS 2023.
- Madaan et al. (2023). "Self-Refine: Iterative Refinement with Self-Feedback." NeurIPS 2023.

### Loop 2: Self-Play Debate
- Silver et al. (2017). "Mastering Chess and Shogi by Self-Play."
- Irving et al. (2018). "AI Safety via Debate." arXiv:1805.00899.
- Du et al. (2023). "Improving Factuality and Reasoning through Multiagent Debate."
- Liang et al. (2023). "Encouraging Divergent Thinking through Multi-Agent Debate."

### Loop 3: Population-Based Evolution
- Mouret & Clune (2015). "Illuminating search spaces by mapping elites."
- Jaderberg et al. (2017). "Population Based Training of Neural Networks."
- Wang et al. (2019). "POET: Endlessly Generating Increasingly Complex and Diverse Learning Environments."
- Ecoffet et al. (2021). "First return, then explore." Nature 590.
- Faldor et al. (2024). "OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness."

### Loop 4: Meta-RL Curriculum
- Duan et al. (2016). "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning."
- Finn et al. (2017). "Model-Agnostic Meta-Learning." ICML 2017.
- Wang et al. (2016). "Learning to Reinforcement Learn."
- Portelas et al. (2020). "Automatic Curriculum Learning For Deep RL."
- Dennis et al. (2020). "Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design."

### Novel Path 1: Thermodynamic Knowledge Reactor
- Friston (2010). "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience.
- Schmidhuber (2009). "Driven by Compression Progress."
- Jaynes (1957). "Information Theory and Statistical Mechanics."
- Still et al. (2012). "Thermodynamics of Prediction." Physical Review Letters.

### Novel Path 2: AKDC
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."
- Goodfellow et al. (2014). "Generative Adversarial Nets."
- Sukhbaatar et al. (2018). "Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play."

### Autonomous AI Research Systems
- Karpathy (2026). "autoresearch." https://github.com/karpathy/autoresearch
- Lu et al. (2024). "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery." Sakana AI.
- Lu et al. (2025). "The AI Scientist v2." arXiv:2504.08066. Sakana AI.
- Romera-Paredes et al. (2024). "Mathematical discoveries from program search with large language models." Nature (FunSearch).
- Trinh et al. (2024). "Solving olympiad geometry without human demonstrations." Nature (AlphaGeometry).
- AlphaEvolve (2025). "A Gemini-powered coding agent for designing advanced algorithms." DeepMind. arXiv:2506.13131.
- Schmidgall & Moor (2025). "AgentRxiv." arXiv:2503.18102.
- arXiv:2601.03315 (2026). "Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts."
- arXiv:2503.22444 (2025). "Scaling Laws in Scientific Discovery with AI and Robot Scientists."

### Active Inference and Free Energy Principle
- Friston (2010). "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience.
- Parr, Pezzulo & Friston (2022). Active Inference: The Free Energy Principle in Mind, Brain, and Behavior. MIT Press.

### Open-Ended Learning
- Lehman et al. (2022). "Evolution through Large Models." arXiv:2206.08896 (OpenELM).
- Ecoffet et al. (2021). "First return, then explore." Nature 590 (Go-Explore).

### RL Foundations
- Christiano et al. (2017). "Deep Reinforcement Learning from Human Preferences."
- Ouyang et al. (2022). "Training language models to follow instructions with human feedback." (InstructGPT/RLHF).
- Bai et al. (2022). "Constitutional AI: Harmlessness from AI Feedback." Anthropic.

---

*← Back to [Architecture and Roadmap](07-architecture-and-roadmap.md)*
