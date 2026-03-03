# picoclaw-auto-ai-researcher

> **An autonomous AI research system that explores the intelligence space, identifies research gaps, and produces novel, useful scientific contributions — without human-in-the-loop bottlenecks.**

---

## Vision

The frontier of AI is moving faster than any individual or team can track. Thousands of papers appear on arXiv every week; entire research directions emerge, converge, and fork in the span of months. Meanwhile, the most capable autonomous agents — systems like AutoGPT, BabyAGI, and clawdbot — have demonstrated that LLM-backed agents can plan, use tools, and iterate toward goals without constant human supervision.

**picoclaw-auto-ai-researcher** bridges these two trends. It is a proposed architecture and research program for building a fully automated AI researcher: an agent (or ensemble of agents) that can independently navigate the scientific literature, detect unexplored regions of idea-space, generate and evaluate hypotheses, design and simulate experiments, synthesize results, and ultimately produce contributions that the broader community finds interesting and useful.

The system is guided by three core principles:

1. **Radical autonomy** — minimize human intervention at every step of the research lifecycle.
2. **Directed curiosity** — optimize not just for correctness but for *interestingness*: novelty, surprise, generality, and downstream impact.
3. **Continuous improvement** — the system learns from its own outputs, community feedback, and the evolving literature, compounding its effectiveness over time.

---

## Why Now?

- **Frontier LLMs** (GPT-4o, Claude 3.5, Gemini 1.5) have demonstrated strong scientific reasoning, code generation, and literature synthesis.
- **Tool-use and function-calling** APIs make it practical to give agents access to search engines, code interpreters, and paper databases.
- **Vector databases and RAG** enable efficient grounding in large corpora.
- **RL from AI feedback (RLAIF)** provides a path to self-improving reward models without requiring a human rater for every judgment.
- **Multi-agent frameworks** (AutoGen, CrewAI, LangGraph) make it increasingly practical to coordinate specialized agent roles.

The pieces exist. This project proposes how to assemble them.

---

## Repository Structure

```
picoclaw-auto-ai-researcher/
├── README.md                              ← You are here
└── docs/
    ├── 01-autonomous-agent-foundations.md    ← What makes an agent truly autonomous
    ├── 02-reinforcement-learning-profiles.md ← RL approaches for research agents
    ├── 03-continuous-learning-architectures.md ← Lifelong learning without forgetting
    ├── 04-mixture-of-agents-systems.md       ← Multi-agent collaboration patterns
    ├── 05-ai-research-community-workflow.md  ← Modeling how science actually works
    ├── 06-research-gap-filling.md            ← Automated gap identification & filling
    └── 07-architecture-and-roadmap.md        ← Full system design and milestones
```

---

## Table of Contents

| # | Document | Core Question |
|---|----------|---------------|
| 1 | [Autonomous Agent Foundations](docs/01-autonomous-agent-foundations.md) | What does it take for an agent to truly self-direct? |
| 2 | [Reinforcement Learning Profiles](docs/02-reinforcement-learning-profiles.md) | How do we reward "interesting" and "useful" research? |
| 3 | [Continuous Learning Architectures](docs/03-continuous-learning-architectures.md) | How does the agent grow without forgetting what it learned? |
| 4 | [Mixture of Agents Systems](docs/04-mixture-of-agents-systems.md) | How do specialized agents collaborate to do better science? |
| 5 | [AI Research Community Workflow](docs/05-ai-research-community-workflow.md) | How does science actually work, and what can be automated? |
| 6 | [Research Gap Filling](docs/06-research-gap-filling.md) | How do we find and fill the white spaces on the map? |
| 7 | [Architecture and Roadmap](docs/07-architecture-and-roadmap.md) | What does the full system look like, and how do we build it? |

---

## Quick-Start Reading Path

**Newcomers to autonomous agents:** Start with [01](docs/01-autonomous-agent-foundations.md) → [04](docs/04-mixture-of-agents-systems.md) → [07](docs/07-architecture-and-roadmap.md).

**ML practitioners:** Start with [02](docs/02-reinforcement-learning-profiles.md) → [03](docs/03-continuous-learning-architectures.md) → [07](docs/07-architecture-and-roadmap.md).

**AI researchers:** Start with [05](docs/05-ai-research-community-workflow.md) → [06](docs/06-research-gap-filling.md) → [07](docs/07-architecture-and-roadmap.md).

---

## Key Novel Proposals (Summary)

- **Interest-Weighted Policy Gradient (IWPG):** A reward formulation that combines novelty, surprise, community uptake (citation velocity), and reproducibility into a single optimizable signal.
- **The Gap Map:** A continuously updated topology of the intelligence-space, rendered as a graph where nodes are concepts and edge-absence signals opportunity.
- **Recursive Self-Curriculum:** The agent generates its own learning syllabus from areas where its self-assessed confidence is low relative to field importance.
- **Adversarial Peer-Review Loop:** A dedicated critic agent attacks every generated hypothesis; the generator is rewarded for surviving critique.
- **Modular Expert Spawning:** When the system detects a persistent blind spot, it spins up a new specialist agent fine-tuned on that subdomain and integrates it into the ensemble.

---

## Contributing

This is an early-stage research initiative. The documents here represent a research program, not a deployed system. Contributions, critiques, and extensions are welcome via issues and pull requests.

---

## License

MIT
