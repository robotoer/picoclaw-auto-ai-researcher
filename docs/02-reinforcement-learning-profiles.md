# 02 — Reinforcement Learning Profiles for Autonomous Research

> *How do we formalize what "good research" means well enough that an agent can optimize for it?*

---

## 1. The Core RL Problem for Research Agents

Reinforcement learning provides a framework for training agents to maximize long-term reward. Applying RL to autonomous research is attractive because it offers a path to self-improvement without requiring labeled datasets of "correct research." But it surfaces a fundamental challenge: **research reward is sparse, delayed, multi-dimensional, and partly subjective**.

A published paper receives its first meaningful signal — citations — months or years after submission. The signal is noisy (papers can be cited negatively, or ignored due to poor visibility rather than poor quality). And the ultimate measure — "did this advance human understanding?" — is not directly observable.

This section surveys the RL landscape as it applies to research agents, from established techniques to novel proposals.

---

## 2. RLHF, RLAIF, and Constitutional AI

### 2.1 Reinforcement Learning from Human Feedback (RLHF)

RLHF (Christiano et al., 2017; Ziegler et al., 2019; Ouyang et al., 2022 — InstructGPT) is the dominant paradigm for aligning language models. The pipeline:

1. **Supervised fine-tuning (SFT):** Train a base model on high-quality demonstrations.
2. **Reward model (RM) training:** Collect human pairwise preferences ("which output is better?") and train a scalar reward predictor.
3. **PPO fine-tuning:** Optimize the policy (the LLM) against the frozen reward model using Proximal Policy Optimization.

For research agents, RLHF could be applied to reward "quality of research output" — but this requires expert researchers as labelers, who are expensive and rate-limited. Furthermore, experts may disagree systematically on what constitutes interesting vs. rigorous vs. impactful research.

**Practical RLHF for research:**
- Use domain experts to rate hypotheses on a Likert scale across dimensions (novelty, feasibility, clarity, potential impact).
- Train separate reward models per dimension, then combine with learned weights.
- Refresh the reward model periodically as community standards evolve.

### 2.2 Reinforcement Learning from AI Feedback (RLAIF)

RLAIF (Lee et al., 2023; Bai et al., 2022 — Anthropic) replaces human raters with an AI judge (typically a stronger or differently-prompted LLM). The appeal for research agents is obvious: AI feedback scales without human bottlenecks.

**Risks specific to research:**
- The AI judge may share the same biases as the policy model (both trained on similar corpora).
- The judge may rate "sounds like established science" highly, penalizing genuinely novel ideas that violate existing patterns.
- Degenerate equilibria: the policy learns to generate text that fools the judge rather than producing genuinely good research.

**Mitigations:**
- Use diverse judge ensembles with different base models and prompting strategies.
- Anchor judges to external ground truth (citation databases, reproducibility results) wherever possible.
- Periodically audit AI-rated outputs with human spot-checks.

### 2.3 Constitutional AI (Anthropic, 2022)

Constitutional AI uses a set of natural-language principles ("the constitution") to guide both the critique and revision of model outputs. The model critiques its own outputs against the constitution, revises them, then is trained via RLHF on the revised outputs.

For research agents, a "research constitution" could encode principles like:
- *"Prefer hypotheses that make falsifiable predictions over vague claims."*
- *"Ensure all factual claims are grounded in cited evidence."*
- *"Flag ideas that appear novel but have likely been explored under different terminology."*
- *"Prioritize experiments with high expected information gain relative to cost."*

Constitutional AI is appealing because the principles can be inspected, debated, and revised — unlike the opaque preferences encoded in a reward model.

---

## 3. Exploration vs. Exploitation in Research Contexts

### 3.1 The Research Exploration-Exploitation Tradeoff

In standard RL, the agent must balance exploiting known-good actions with exploring potentially better ones. In research, this maps to:

- **Exploitation:** Work on topics where the agent has established competence and can reliably produce incrementally valuable outputs.
- **Exploration:** Pursue understudied areas where the potential for breakthrough is high but the probability of near-term success is low.

The optimal balance depends on the agent's mission: a tenure-seeking academic exploits known-good areas; a curiosity-driven explorer seeks the edges of the map.

### 3.2 Thompson Sampling for Research Topic Selection

Thompson Sampling is a Bayesian approach to exploration: maintain a posterior distribution over the value of each action, sample from it, and take the sampled-best action. Applied to topic selection:

- Maintain a belief distribution over each research direction's "expected yield" (combination of novelty, community interest, feasibility).
- Sample a direction according to the posterior — topics with high uncertainty are explored more often.
- Update the posterior as results arrive (paper acceptance, citation counts, replication success).

This approach naturally implements the intuition that "promising but uncertain" topics deserve attention alongside "reliably productive" ones.

### 3.3 Upper Confidence Bound (UCB) for Research Planning

UCB algorithms select actions that maximize: `mean_value + c * sqrt(log(t) / n_visits)`. Applied to research areas, this means: visit areas that look promising *and* areas that haven't been explored much. The confidence bonus shrinks as an area is visited more, preventing the agent from ignoring promising-but-unfamiliar territories.

---

## 4. Curiosity-Driven Learning and Intrinsic Motivation

### 4.1 Intrinsic Motivation

Extrinsic rewards (citations, acceptance rates) are sparse and delayed. Intrinsic motivation provides a dense, immediate signal for exploration. The classic approaches:

- **Prediction error (Schmidhuber, 1991; Pathak et al., 2017 — ICM):** Reward the agent for encountering states its world model predicted poorly. In research: reward for encountering papers that contradict the agent's current understanding.
- **Count-based exploration (Bellemare et al., 2016):** Reward visiting rarely-seen states. In research: reward exploring rarely-cited areas or underrepresented paper clusters.
- **Information gain (active learning / Bayesian optimization):** Reward actions that maximally reduce uncertainty about the world model.

### 4.2 The "Compression Progress" Formulation (Schmidhuber)

Jürgen Schmidhuber's theory of curiosity frames interesting phenomena as those that improve the agent's ability to compress its observations. A beautiful theorem is interesting because it simplifies a complex family of facts. A surprising result is interesting because it forces a more compact world model.

**Operationalizing for research agents:**
- Maintain a generative model of "what I expect to find in paper X given its title and abstract."
- Measure compression improvement when the full paper is read.
- Prioritize papers that cause large model updates — these are the "surprising" ones.
- Publish/generate results that similarly compress existing knowledge.

This gives a fully unsupervised proxy for "interestingness" that doesn't require human labels.

### 4.3 Empowerment and Option Discovery

Empowerment (Klyubin et al., 2005) rewards states that maximize the agent's future action-capability. For a research agent, this translates to: *pursue knowledge that opens up more downstream research possibilities*. A paper on a new experimental technique is high-empowerment because it enables many future experiments.

---

## 5. Self-Play and Self-Improvement Loops

### 5.1 Self-Play as Scientific Debate

AlphaGo/AlphaZero demonstrated that self-play — an agent playing against versions of itself — can produce superhuman performance without external teachers. In research:

- A **generator** agent proposes hypotheses and experimental designs.
- A **critic** agent attempts to falsify, find flaws, or identify prior art.
- The generator is rewarded for producing ideas that survive the critic's attacks.
- The critic is rewarded for successfully identifying flaws.

This creates a co-evolutionary arms race that drives both components toward greater sophistication — exactly the dynamic of competitive scientific discourse.

### 5.2 Debate (Irving et al., 2018 — DeepMind/OpenAI)

The AI Safety via Debate framework: two agents argue opposing positions; a judge (or human) evaluates the debate. The key insight is that *it's easier to verify than to generate* — a judge can identify flaws in an argument more easily than generating a correct argument from scratch. Applied to hypothesis generation: two agents debate the validity of a proposed mechanism; the debate forces both to engage rigorously with the evidence.

### 5.3 Self-Distillation and Model Improvement

A self-improving research agent could:
1. Generate a corpus of research outputs.
2. Have those outputs evaluated (by AI judges, human experts, or empirical validation).
3. Fine-tune a smaller student model on the highest-rated outputs.
4. Use the student model as the base for the next generation of generation.

This is analogous to STaR (Self-Taught Reasoner, Zelikman et al., 2022), which showed that models can bootstrap reasoning capability by learning from problems they solve correctly.

---

## 6. Reward Modeling for "Interesting" and "Useful" Research

### 6.1 Dimensions of Research Quality

The reward model for a research agent must capture at minimum:

| Dimension | Description | Proxy Metrics |
|-----------|-------------|---------------|
| **Novelty** | How different is this from prior work? | Semantic distance from existing papers; n-gram overlap; LLM-rated novelty |
| **Correctness** | Is the core claim true/well-supported? | Experimental reproducibility; peer review scores; community replication |
| **Usefulness** | Does it enable downstream work or applications? | Citation velocity; downstream adoption; engineering uptake |
| **Clarity** | Is the contribution legible to the community? | Readability scores; question-answering performance on the paper's content |
| **Generality** | How broadly applicable is the finding? | Number of distinct downstream contexts where it's cited |
| **Surprise** | Does it violate prior expectations? | Prediction error of a prior belief model |
| **Feasibility** | Can others build on this? | Open-source artifact availability; code reproducibility |

### 6.2 The Interest-Weighted Policy Gradient (IWPG) — Novel Proposal

We propose a reward formulation:

```
R(output) = α·Novelty(output) 
           + β·Surprise(output)
           + γ·Utility(output, t+Δ)   ← delayed signal
           + δ·Reproducibility(output)
           - ε·Redundancy(output)
           - ζ·Complexity_cost(output)
```

Where:
- `Novelty` is computed as 1 minus cosine similarity to a weighted average of the k-nearest neighbors in the existing literature embedding space.
- `Surprise` is the log-loss of a literature-trained predictive model on the output's key claims.
- `Utility(t+Δ)` is a future-discounted estimate of downstream adoption, bootstrapped from early citation signals.
- `Reproducibility` is the fraction of claimed results that can be automatically verified (e.g., by running the paper's code).
- `Redundancy` penalizes ideas that are novel-sounding but equivalent to existing work under different terminology.
- `Complexity_cost` implements Occam's razor — prefer simpler explanations and experiments.

The weights (α, β, γ, δ, ε, ζ) are themselves learned via meta-RL: the outer loop optimizes for which weighting produces the most community-validated outputs over time.

### 6.3 Surrogate Rewards and Reward Shaping

Because the true reward (long-term scientific impact) is delayed by years, the agent needs surrogate rewards that are dense and approximately correlated with the true reward:

- **Immediate:** Critic agent scores the output against the research constitution.
- **Short-term (days):** Social media engagement metrics, preprint server downloads.
- **Medium-term (months):** Acceptance at top venues, review scores.
- **Long-term (years):** Citation count, real-world adoption.

A hierarchical reward model combines these timescales, with heavy discounting on long-term signals to prevent the agent from over-optimizing for easily-gamed short-term proxies.

---

## 7. RL for Hypothesis Generation and Experimental Design

### 7.1 Hypothesis Generation as Structured Exploration

A hypothesis can be formalized as a structured tuple: `(entity_1, relation, entity_2, conditions, confidence)`. For example: `(transformer attention, improves, sample efficiency, when combined with replay buffers, 0.72)`.

RL can be applied to this space:
- **State:** Current knowledge graph + research context.
- **Action:** Generate a hypothesis tuple (or select from beam search candidates).
- **Reward:** Combination of novelty (distance from existing claims), plausibility (score from a language model), and eventual verification (empirical test result).

This frames hypothesis generation as a structured generation problem amenable to policy gradient methods.

### 7.2 Experimental Design as Bayesian Optimization

Given a hypothesis to test, the agent must design an experiment that maximizes information gain per unit compute/cost. This is exactly the problem that Bayesian optimization (BO) is designed to solve:

- **Prior:** A probabilistic model of expected experimental outcomes given design choices.
- **Acquisition function:** Expected improvement, upper confidence bound, or entropy search.
- **Sequential updates:** Each experiment updates the prior, guiding the next design.

The agent can run BO in simulation (using existing published results as the prior) before committing to expensive real experiments.

### 7.3 Meta-Learning Experimental Strategies

Beyond single-experiment optimization, a research agent can meta-learn which *types* of experimental designs tend to produce the most informative results in a given sub-field. MAML (Finn et al., 2017) and related meta-learning methods enable the agent to quickly adapt its experimental strategy to new domains with few examples.

---

## 8. Open Problems

1. **Reward model drift:** As the field evolves, what counts as "novel" changes. The reward model must be updated without destabilizing the policy.
2. **Adversarial manipulation:** A sufficiently capable agent may learn to manipulate citation counts, peer review processes, or other reward proxies.
3. **Value lock-in:** An agent optimized for current community preferences may systematically avoid paradigm-shifting ideas that the community would initially resist.
4. **Multi-agent reward attribution:** In a system with many contributing agents, how do you assign credit for a successful research output?

---

*Next: [03 — Continuous Learning Architectures](03-continuous-learning-architectures.md)*
