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
├── CLAUDE.md                              ← AI assistant instructions
├── pyproject.toml                         ← Project config, dependencies, tool settings
├── config.example.yaml                    ← System configuration reference
├── docker-compose.yml                     ← Docker services (Neo4j, Qdrant, etc.)
├── src/auto_researcher/                   ← Python implementation
│   ├── agents/                            ← Specialized research agents
│   ├── evaluation/                        ← IWPG, peer review, SUNFIRE scoring
│   ├── infrastructure/                    ← Knowledge graph, vector store, gap map
│   ├── ingestion/                         ← arXiv monitoring, claim extraction, PDF parsing
│   ├── learning/                          ← Curriculum planning, reward model, consolidation
│   ├── models/                            ← Data models (claims, papers, hypotheses, etc.)
│   ├── orchestrator/                      ← Task routing and resource management
│   ├── utils/                             ← LLM helpers and logging
│   └── verification/                      ← Claim verification and provenance tracking
├── tests/                                 ← Unit and integration tests
├── experiments/                           ← Experiment designs (E01-E15) and implementations (E01-E04)
├── experiments-completed/                 ← Completed experiment write-ups
└── docs/
    ├── 01-autonomous-agent-foundations.md    ← What makes an agent truly autonomous
    ├── 02-reinforcement-learning-profiles.md ← RL approaches for research agents
    ├── 03-continuous-learning-architectures.md ← Lifelong learning without forgetting
    ├── 04-mixture-of-agents-systems.md       ← Multi-agent collaboration patterns
    ├── 05-ai-research-community-workflow.md  ← Modeling how science actually works
    ├── 06-research-gap-filling.md            ← Automated gap identification & filling
    ├── 07-architecture-and-roadmap.md        ← Full system design and milestones
    ├── 08-hallucination-cascade-prevention.md ← Multi-layered hallucination defense
    ├── 09-bitter-lesson-loops-and-novel-paths.md ← Research loops, novel algorithms, hardware budget
    ├── 10-hardware-budget-and-scaling.md     ← Detailed cost analysis and tier specifications
    ├── 11-self-improving-loop-analysis.md    ← Most promising paths to recursive self-improvement
    └── 12-data-requirements-and-intractability-analysis.md ← Data needs, scaling laws, and intractability traps
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
| 8 | [Hallucination Cascade Prevention](docs/08-hallucination-cascade-prevention.md) | How do we prevent false claims from corrupting the knowledge graph? |
| 9 | [Bitter Lesson, Loops & Novel Paths](docs/09-bitter-lesson-loops-and-novel-paths.md) | What are the core algorithms, and how does the bitter lesson shape design? |
| 10 | [Hardware Budget & Scaling](docs/10-hardware-budget-and-scaling.md) | How much does it cost, and how should compute be allocated? |
| 11 | [Self-Improving Loop Analysis](docs/11-self-improving-loop-analysis.md) | What are the most promising paths to recursive self-improvement? |
| 12 | [Data Requirements & Intractability](docs/12-data-requirements-and-intractability-analysis.md) | What data is needed, what traps exist, and what do scaling laws predict? |

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
- **Thermodynamic Knowledge Reactor:** Treats the research frontier as a thermodynamic system; the agent maximizes knowledge extraction efficiency per unit compute, with learned temperature annealing and phase transition detection.
- **Adversarial Knowledge Distillation Cascade (AKDC):** A progressive cascade of increasingly capable agents where each level is trained on the failures of the previous level, with adversarial problem generation and backward distillation.

---

## Contributing

This is an early-stage research initiative. The documents here represent a research program, not a deployed system. Contributions, critiques, and extensions are welcome via issues and pull requests.

---

## License

MIT
