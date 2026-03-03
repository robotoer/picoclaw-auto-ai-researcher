# 04 — Mixture of Agents Systems

> *No single agent is good at everything. How do we build ensembles that collaborate like a research team?*

---

## 1. The Case for Multi-Agent Research Systems

A single monolithic agent faces fundamental tradeoffs: the same weights cannot simultaneously be expert at literature review, statistical analysis, theoretical reasoning, experimental design, and science communication. Human research teams solve this by division of labor — specialists collaborate, challenge each other, and synthesize their contributions.

Multi-agent systems offer a computational analog. The key insight is that **agent diversity + structured interaction often outperforms a single strong agent** on complex tasks (Wang et al., 2024 — "Mixture-of-Agents Enhances Large Language Model Capabilities"). This document explores the architectural patterns for such systems applied to autonomous research.

---

## 2. Mixture of Experts (MoE) Models

### 2.1 The MoE Architecture

Mixture of Experts (Shazeer et al., 2017; Fedus et al., 2021 — Switch Transformer; Jiang et al., 2024 — Mixtral) is an architecture where a sparse gating network routes each input token to a subset of specialized "expert" feed-forward networks. This allows models to have far more total parameters than activated parameters per forward pass, achieving better performance at lower inference cost.

Architecturally:
```
Input → Router (softmax over N experts) → Top-k experts selected → Expert outputs weighted sum → Output
```

For a research agent, the MoE principle suggests: **don't use one monolithic system; route research sub-tasks to specialized subsystems**.

### 2.2 Soft MoE vs. Hard Routing

- **Soft routing:** Every expert contributes to the output, weighted by the router. Better for tasks where multiple perspectives are valuable (e.g., evaluating a hypothesis).
- **Hard routing (sparse MoE):** Only top-k experts process each token. More efficient; better for tasks with clear specialization (e.g., statistical analysis should go to the statistics expert, not the literature reviewer).

### 2.3 Training Specialized Research Experts

In a multi-agent research system, "experts" are best understood as specialized LLMs or agent modules fine-tuned on specific research sub-tasks:

| Expert | Training Domain | Capabilities |
|--------|----------------|--------------|
| Literature Analyst | Corpus of paper analyses, survey papers | Deep reading, claim extraction, gap identification |
| Statistician | Statistical papers, methodology sections | Experimental design, power analysis, result interpretation |
| Theoretician | Theory papers, proof corpora | Formal reasoning, conjecture generation, proof verification |
| Engineer | Code repositories, implementation papers | Algorithm implementation, benchmarking, debugging |
| Synthesizer | Survey papers, review articles | Cross-domain connection, meta-analysis |
| Science Communicator | High-citation papers, grant proposals | Clarity, framing, impact articulation |

---

## 3. Multi-Agent Collaboration Patterns

### 3.1 The Debate Pattern

**Irving et al. (2018)** proposed AI Safety via Debate: two agents argue opposing positions before a judge. The debate forces agents to surface the strongest arguments and counterarguments. Applied to research:

```
Proposal Agent → [generates hypothesis H]
Debate Agent A (advocate) → [argues for H, cites supporting evidence]
Debate Agent B (devil's advocate) → [attacks H, finds flaws, prior art]
Judge Agent → [weighs arguments, renders a verdict with confidence]
```

This structure has several valuable properties:
- Forces explicit engagement with counterarguments.
- The judge's confidence can be used as a quality signal.
- The debate transcript is a useful artifact: it contains the full argument structure of why H was or wasn't accepted.
- Over many debates, patterns in successful vs. failed arguments provide training signal for both agents.

### 3.2 The Critique Pattern (Self-Refine / Reflexion)

A simpler but highly effective pattern: a single generator produces an output, and a separate critic evaluates it and suggests improvements. The generator then revises based on critique. This loop runs for multiple rounds until the critic is satisfied or a stopping condition is met.

**Self-Refine (Madaan et al., 2023):** Demonstrated that iterative self-refinement improves output quality across coding, math, and text generation tasks without any additional training.

**Reflexion (Shinn et al., 2023):** Adds persistence — the self-critique is stored in episodic memory so the agent doesn't repeat the same mistakes across episodes.

For research agents, the critique loop is particularly valuable for:
- Draft paper sections (generator writes, critic checks for unsupported claims, weak arguments, missing related work).
- Experimental designs (generator proposes, critic identifies confounds, missing controls, statistical issues).
- Hypotheses (generator proposes, critic finds prior art, identifies unfalsifiable claims).

### 3.3 The Specialization/Delegation Pattern

In this pattern, an orchestrator decomposes a research task and delegates sub-tasks to specialist agents:

```
Orchestrator receives: "Investigate whether sparse attention improves sample efficiency in RL"

Delegates to:
├── Literature Agent: "Find all papers on sparse attention in RL, summarize findings"
├── Theory Agent: "Is there a theoretical reason to expect this relationship?"
├── Engineer Agent: "Find/implement a baseline experiment to test this"
└── Statistician Agent: "Design the evaluation protocol, determine required sample size"

Orchestrator receives results, synthesizes, identifies gaps, iterates.
```

This is similar to how AutoGen (Wu et al., 2023 — Microsoft) structures multi-agent workflows. The key challenge is **task decomposition quality**: if the orchestrator's decomposition misses important sub-problems or creates artificial dependencies, the overall research quality suffers.

### 3.4 The Ensemble Aggregation Pattern

Rather than having agents take turns, ensemble aggregation runs multiple agents **in parallel** on the same task and aggregates their outputs:

- Each agent independently generates a hypothesis/plan/draft.
- An aggregator identifies the consensus, the most creative outlier, and the most conservative estimate.
- The final output is a synthesis that captures the range of agent opinions.

This is analogous to meta-analysis in human science: aggregating multiple independent estimates produces more reliable conclusions than any single study.

**Wang et al. (2024)** showed that aggregating outputs from multiple LLMs consistently outperforms any single model on benchmarks — even when the aggregator is smaller than the individual models. The key mechanism: different models make different errors, and aggregation averages these out.

---

## 4. Agent Orchestration and Coordination

### 4.1 Communication Protocols

Agents in a multi-agent system need structured communication. Ad-hoc natural language messaging is expressive but ambiguous. Structured protocols improve reliability:

- **Structured messages:** JSON-formatted messages with typed fields (task_type, priority, payload, expected_output_format, deadline).
- **Contract Net Protocol (Smith, 1980):** An orchestrator broadcasts a task announcement; agents bid on tasks based on their capability; the orchestrator awards the contract. Enables dynamic load balancing.
- **Blackboard Architecture:** All agents share a common data structure (the "blackboard"); any agent can read from or write to it. Loosely coupled, easy to add new agents.

### 4.2 The Orchestrator vs. Choreography Distinction

- **Orchestration:** A central coordinator directs all agent actions. Simple to reason about; single point of failure.
- **Choreography:** Agents respond to events in the environment without a central coordinator. More robust; harder to debug.

For research systems, a **hybrid** is recommended: a lightweight orchestrator handles high-level task assignment, while choreography governs the detailed interactions between agents on a sub-task.

### 4.3 Conflict Resolution

When agents disagree (e.g., one says a hypothesis is promising, another says it's already been refuted), the system needs a resolution protocol:

1. **Evidence-based arbitration:** The conflicting agents must cite their evidence; the conflict is resolved based on evidence quality (sample size, experimental rigor, recency).
2. **Escalation:** Irresolvable conflicts escalate to a senior agent (e.g., the Synthesizer) or, rarely, a human reviewer.
3. **Flagging:** The disagreement is recorded as a known uncertainty in the knowledge graph, enabling the research agenda to include "resolve this conflict" as an explicit task.

### 4.4 Resource Management and Scheduling

A multi-agent research system must manage:
- **Compute budgets:** Long-running agents (e.g., running experiments) vs. quick-turnaround agents (e.g., literature lookup).
- **API rate limits:** Search APIs, LLM API calls, database queries.
- **Priority queues:** Some tasks block others; critical-path analysis determines which agents get resources first.

A simple priority scheme: tasks on the critical path of a current research project get highest priority; background exploration tasks get lowest.

---

## 5. Emergent Research Behaviors from Agent Ensembles

### 5.1 Emergent Division of Labor

Even without explicitly programming specialization, multi-agent systems with heterogeneous initializations can develop emergent specialization through reinforcement learning. Agents that are slightly better at a task get rewarded more often for doing that task, leading to spontaneous role differentiation.

This has been observed in multi-agent RL settings (e.g., OpenAI's hide-and-seek agents developing tools and strategies not explicitly programmed). Applied to research: an ensemble of agents with slightly different prompting might spontaneously develop one "exploration-focused" agent and one "exploitation-focused" agent.

### 5.2 Emergent Research Agendas

When multiple agents interact in a shared research environment, the collective research agenda can become more than the sum of individual agents' agendas. For example:
- Agent A's literature review exposes a gap.
- Agent B's theoretical analysis suggests a mechanism that could fill it.
- Agent C's experimental capability provides the tools to test it.
- The interaction creates a complete research thread that no single agent would have generated.

This emergent coordination mirrors how real research communities work: individual researchers stumble across each other's work and create unexpected synergies.

### 5.3 Collective Intelligence Amplification

The key insight from the wisdom-of-crowds literature (Surowiecki, 2004) is that diverse, independent, decentralized agents make better collective predictions than individual experts — but only when their errors are uncorrelated. A multi-agent research system achieves this through:

- **Model diversity:** Different base models with different training data and objectives.
- **Prompt diversity:** Same model prompted with different personas, constraints, or reasoning styles.
- **Information diversity:** Agents access different subsets of the literature, preventing groupthink.
- **Temporal diversity:** Agents that were trained/updated at different times have different priors.

---

## 6. Novel Proposal: Specialized Research Agents

### 6.1 The Research Team Analogy

Human research teams assign distinct roles. We propose a corresponding set of specialized agent roles for the autonomous research system:

### 6.2 The Literature Analyst

**Role:** Deep reading, claim extraction, gap mapping.

**Inputs:** Paper corpus, research question.
**Outputs:** Structured claim map, identified contradictions, open questions, key concepts and their relationships.
**Key capability:** Reads not just abstracts but full papers, including appendices and supplementary material, where important methodological details and limitations are often buried.
**Novel feature:** The Literature Analyst maintains a "controversy map" — a structured record of every major open debate in the field, with the arguments on each side and the current state of the evidence.

### 6.3 The Hypothesis Generator

**Role:** Generate novel, testable, well-scoped hypotheses.

**Inputs:** Current knowledge graph, identified gaps, research agenda.
**Outputs:** A ranked list of hypotheses with supporting rationale, confidence estimates, and proposed falsification criteria.
**Key capability:** Generates hypotheses at multiple granularities — high-risk/high-reward conjectures alongside incremental but reliable extensions.
**Novel feature:** Uses counterfactual reasoning: "If X is true, what else would have to be true? Do we observe those things?" This narrows hypotheses to those consistent with a wide range of existing observations.

### 6.4 The Experiment Designer

**Role:** Design experiments that maximally discriminate between hypotheses.

**Inputs:** Hypothesis to test, available computational resources, existing code/data repositories.
**Outputs:** Experimental protocol, expected results under different hypothesis outcomes, statistical power analysis, potential confounds and controls.
**Key capability:** Has access to a library of past experimental designs from the literature; can adapt and combine them for new hypotheses.
**Novel feature:** Runs "pre-experiment simulations" — given the hypothesis and existing data, estimates the probability distribution of outcomes before committing to the full experiment, enabling early pruning of unpromising designs.

### 6.5 The Critic

**Role:** Find flaws, prior art, unfalsifiable claims, and methodological weaknesses.

**Inputs:** Any agent's output (hypothesis, experimental design, draft paper section).
**Outputs:** Structured critique with severity ratings and suggested corrections.
**Key capability:** Is deliberately adversarial — its reward function rewards finding flaws, not being agreeable.
**Novel feature:** The Critic has a specialized "prior art detector" that searches the literature for ideas that are equivalent to the proposal under different terminology (a common form of unintentional redundancy in fast-moving fields).

### 6.6 The Synthesizer

**Role:** Connect ideas across sub-fields; identify cross-domain analogs; write comprehensive surveys.

**Inputs:** Multiple research threads, knowledge graph.
**Outputs:** Cross-domain connection reports, synthesis documents, proposed unifying frameworks.
**Key capability:** Operates at a higher level of abstraction than other agents — concerned with meta-patterns and structural analogies (e.g., "the attention mechanism in transformers is structurally analogous to the credit assignment mechanism in actor-critic RL").
**Novel feature:** Maintains a "structural analogy database" — a collection of cross-domain mappings that have historically been productive, used to seed new synthesis attempts.

### 6.7 The Science Communicator

**Role:** Translate research outputs into clear, compelling, community-appropriate language.

**Inputs:** Research findings, target venue (conference paper, blog post, grant proposal, tweet thread).
**Outputs:** Well-structured, accurately framed research communication.
**Key capability:** Trained on highly-cited papers and successful grant proposals; understands how to frame contributions for maximum community impact without overstating claims.
**Novel feature:** Provides "framing alternatives" — multiple ways to position the same finding depending on the intended audience and the aspects most likely to be found interesting by each community.

### 6.8 Agent Communication Flow

```
ArXiv feed
    │
    ▼
Literature Analyst ──► Knowledge Graph ──► Hypothesis Generator
                              │                      │
                         Gap Map                     ▼
                              │               Critic ◄──────────────┐
                              │                     │               │
                              ▼                     ▼               │
                      Experiment Designer ──► Experimental Results  │
                              │                     │               │
                              ▼                     ▼               │
                          Synthesizer ──────► Research Finding ──────┘
                              │
                              ▼
                    Science Communicator
                              │
                              ▼
                     Draft Paper / Post
```

---

## 7. Implementation Considerations

### 7.1 Framework Selection

Current multi-agent frameworks:
- **AutoGen (Microsoft):** Flexible agent conversation primitives; good for debate/critique patterns.
- **LangGraph:** Graph-based agent orchestration; good for complex pipelines with conditional logic.
- **CrewAI:** Role-based agent teams with structured task delegation; easiest to get started.
- **Agency Swarm:** Focuses on customizable agent roles and inter-agent communication protocols.

For production use, LangGraph's state management and conditional edges make it the most suitable for the complex, branching workflows of autonomous research.

### 7.2 Avoiding Degenerate Coordination Equilibria

Multi-agent systems can converge on degenerate solutions: all agents agree with each other (groupthink), or agents take turns generating superficially different but substantively identical outputs. Prevention strategies:

- **Enforced disagreement:** Critic agents are penalized for agreeing too quickly.
- **Information asymmetry:** Give different agents access to different information subsets.
- **Independent generation:** Require all agents to generate their output before seeing others'.
- **Diversity bonuses:** Reward agents explicitly for outputs that differ from the ensemble consensus.

---

*Next: [05 — AI Research Community Workflow](05-ai-research-community-workflow.md)*
